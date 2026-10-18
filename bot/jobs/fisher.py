"""Fisher job — full Majestic RP fishing loop (6 stages)."""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot.capture import GameCapture
from bot.config_loader import LOGS_DIR, ROUTES_DIR, fishing_region, scale_region
from bot.detectors.cast_timing import CastTimingDetector
from bot.detectors.catch_modal import CatchModalDetector
from bot.detectors.fish_fight import FishFightDetector
from bot.detectors.fish_phase import FishPhase, FishPhaseDetector
from bot.detectors.fish_prompt import FishPromptDetector
from bot.detectors.fishing_setup import FishingSetupDetector
from bot.services.sustenance import SustenanceResult, SustenanceService
from bot.state_machine import BotState, StateMachine
from bot.telemetry.fishing_session import FishingSessionTelemetry


class FisherJob:
    def __init__(
        self,
        config: dict[str, Any],
        capture: GameCapture,
        input_handler,
        state_machine: StateMachine,
        sustenance_service: SustenanceService | None = None,
    ) -> None:
        self._config = config
        self._capture = capture
        self._input = input_handler
        self._sm = state_machine
        self._sustenance = sustenance_service

        regions = config.get("regions", {})
        self._timing_region = fishing_region(regions, "timing_bar")
        self._prompt_region = fishing_region(regions, "fishing_prompt")
        self._bite_text_region = fishing_region(regions, "bite_text")
        self._bobber_region = fishing_region(regions, "bobber_widget")
        self._hook_prompt_region = fishing_region(regions, "hook_prompt")
        self._fish_icon_region = fishing_region(regions, "fish_icon")
        self._fight_bar_region = fishing_region(regions, "fight_bar")
        self._modal_region = fishing_region(regions, "catch_modal")
        # Everything _tick_waiting_bite() reads from a frame each call — used
        # with grab_union() so ONE mss call per tick covers all of it. On this
        # kind of capture path the cost is dominated by a large *fixed*
        # per-call overhead (measured ~100-110ms regardless of region size),
        # not pixel area — so it's better to fold catch_modal into the same
        # single grab_union() call than to spend a second separate capture on
        # it, even though area-wise it's the biggest region here.
        self._bite_poll_regions = [
            self._timing_region,
            self._bite_text_region,
            self._bobber_region,
            self._hook_prompt_region,
            self._modal_region,
        ]

        fishing_cfg = config.get("fishing", {})
        self._cast_settle = fishing_cfg.get("cast_settle_ms", 600) / 1000.0
        self._post_timing_grace = max(
            0.0,
            float(fishing_cfg.get("post_timing_grace_ms", 1200)) / 1000.0,
        )
        self._bite_timeout = fishing_cfg.get("bite_timeout_sec", 35)
        self._min_bite_delay = fishing_cfg.get("min_bite_delay_sec", 1.0)
        self._baseline_settle = fishing_cfg.get("baseline_settle_sec", 1.5)
        self._bite_confirm_frames = max(1, int(fishing_cfg.get("bite_confirm_frames", 2)))
        prompt_cfg = config.get("prompt", {})
        self._hook_delta_min = prompt_cfg.get("hook_delta_min", 0.35)
        self._text_delta_min = prompt_cfg.get("text_delta_min", 0.20)
        self._bobber_change_min = prompt_cfg.get("bobber_change_min", 0.055)
        self._hook_match_threshold = prompt_cfg.get("hook_match_threshold", 0.80)
        self._bite_text_threshold = prompt_cfg.get("bite_text_threshold", 0.60)
        self._bite_absolute_threshold = prompt_cfg.get("bite_absolute_threshold", 0.72)
        self._bite_keyword_threshold = prompt_cfg.get("bite_keyword_threshold", 0.72)
        self._bite_debug_interval = fishing_cfg.get("bite_debug_interval_sec", 5.0)
        self._last_bite_debug = 0.0
        self._hook_delay = fishing_cfg.get("hook_delay_ms", [0, 15])
        self._after_collect = fishing_cfg.get("after_collect_ms", 700) / 1000.0
        self._keep_click_delay = fishing_cfg.get("keep_click_delay_ms", 200) / 1000.0
        self._reel_modal_wait = fishing_cfg.get("reel_modal_wait_sec", 5.0)
        self._cast_key = fishing_cfg.get("cast_key", "space")
        self._auto_cast = bool(fishing_cfg.get("auto_cast", False))
        self._cast_start_max_attempts = max(
            1,
            int(fishing_cfg.get("cast_start_max_attempts", 3)),
        )
        self._cast_start_retry_delay = max(
            0.0,
            float(fishing_cfg.get("cast_start_retry_delay_sec", 1.0)),
        )
        self._hook_trigger = fishing_cfg.get("hook_trigger", "bite_text")
        self._hook_keys: list[str] = list(fishing_cfg.get("hook_keys", ["lmb", "enter"]))
        if not self._hook_keys:
            legacy = fishing_cfg.get("hook_key", "enter")
            self._hook_keys = [legacy]

        self._setup_enabled = bool(
            fishing_cfg.get("auto_start_activity", True)
        )
        self._setup_key = str(fishing_cfg.get("setup_key", "e")).lower()
        self._setup_after_start = float(
            fishing_cfg.get("setup_after_start_press_ms", 2000)
        ) / 1000.0
        self._setup_after_equipment = float(
            fishing_cfg.get("setup_after_equipment_press_ms", 2000)
        ) / 1000.0
        self._setup_start_timeout = float(
            fishing_cfg.get("setup_start_timeout_sec", 30)
        )
        self._setup_equipment_timeout = float(
            fishing_cfg.get("setup_equipment_timeout_sec", 20)
        )
        self._setup_prompt_search_delay = max(
            0.25,
            float(fishing_cfg.get("setup_prompt_search_delay_sec", 1.0)),
        )
        self._setup_prompt_first_step = str(
            fishing_cfg.get("setup_prompt_first_step", "s")
        ).lower()
        if self._setup_prompt_first_step not in ("w", "s"):
            self._setup_prompt_first_step = "s"
        self._setup_prompt_step_hold_ms = max(
            20,
            int(fishing_cfg.get("setup_prompt_step_hold_ms", 500)),
        )
        self._setup_prompt_step_wait = max(
            0.25,
            float(fishing_cfg.get("setup_prompt_step_wait_sec", 1.0)),
        )
        self._setup_prompt_cycle_pause = max(
            self._setup_prompt_step_wait,
            float(fishing_cfg.get("setup_prompt_cycle_pause_sec", 2.0)),
        )
        self._setup_prompt_max_steps = max(
            0,
            int(fishing_cfg.get("setup_prompt_max_steps", 2)),
        )
        self._sustenance_enabled = bool(
            fishing_cfg.get("sustenance_enabled", True)
        ) and sustenance_service is not None
        self._sustenance_exit_key = str(
            fishing_cfg.get("sustenance_exit_key", "esc")
        ).lower()
        self._sustenance_exit_settle = float(
            fishing_cfg.get("sustenance_exit_settle_ms", 700)
        ) / 1000.0
        self._sustenance_exit_attempts = max(
            1,
            int(fishing_cfg.get("sustenance_exit_attempts", 3)),
        )
        self._sustenance_failure_retry = max(
            0.25,
            float(fishing_cfg.get("sustenance_failure_retry_ms", 1000))
            / 1000.0,
        )
        self._sustenance_max_failures = max(
            1,
            int(fishing_cfg.get("sustenance_max_failures", 3)),
        )
        self._sustenance_after_action = float(
            fishing_cfg.get("sustenance_after_action_ms", 700)
        ) / 1000.0
        self._target_fish_count = max(
            0,
            int(fishing_cfg.get("target_fish_count", 0)),
        )

        debug = config.get("debug", {})
        self._debug = debug.get("enabled", False)
        self._save_screenshots = debug.get("save_screenshots", False)
        self._show_overlay = debug.get("show_overlay", False)
        self._debug_live_window = debug.get("live_window", False)
        screenshot_format = str(
            debug.get("screenshot_format", "jpg")
        ).lower().lstrip(".")
        self._screenshot_format = (
            screenshot_format
            if screenshot_format in ("jpg", "jpeg", "webp")
            else "jpg"
        )
        self._screenshot_quality = max(
            1,
            min(100, int(debug.get("screenshot_quality", 82))),
        )
        self._screenshot_max_per_session = max(
            1,
            int(debug.get("screenshot_max_per_session", 120)),
        )
        self._screenshot_max_per_tag = max(
            1,
            int(debug.get("screenshot_max_per_tag", 30)),
        )

        self._timing = CastTimingDetector(config)
        self._fight = FishFightDetector(config)
        self._prompt = FishPromptDetector(config)
        self._modal = CatchModalDetector(config)
        self._phase_det = FishPhaseDetector(config)
        self._setup_detector = FishingSetupDetector(config)
        self._telemetry = FishingSessionTelemetry(config)

        self._mode = config.get("mode", "cast_and_reel")
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0
        self._stats = {
            "fish_caught": 0,
            "casts": 0,
            "hooks": 0,
            "timing_hits": 0,
            "timeouts": 0,
            "sustenance_pauses": 0,
            "drinks": 0,
            "meals": 0,
            "unidentified_catches": 0,
        }
        self._bite_wait_start = 0.0
        self._baseline_hook = 0.0
        self._baseline_text = 0.0
        self._baseline_bobber: np.ndarray | None = None
        self._baseline_ready = False
        self._bite_streak = 0
        self._debug_frame_counter = 0
        self._cast_pressed = False
        self._cast_start_attempts = 0
        self._sustenance_checked_for_cast = False
        self._sustenance_pending_needs: SustenanceResult | None = None
        self._sustenance_retry_at = 0.0
        self._sustenance_last_deferred_reason = ""
        self._sustenance_failure_count = 0
        self._sustenance_fishing_exited = False
        self._timing_session = False
        self._timing_solve_start = 0.0
        self._timing_no_slider_streak = 0
        self._timing_ignore_until = 0.0
        self._last_reel_log = 0.0
        self._last_timing_debug = 0.0
        self._timing_debug_interval = fishing_cfg.get("timing_debug_interval_sec", 0.25)
        self._baseline_text_samples: list[float] = []
        self._last_bite_block_reason = ""
        self._overlay_interval = debug.get("overlay_interval_sec", 0.5)
        self._last_overlay_save = 0.0
        self._last_timing_crop_save = 0.0
        self._paused_cleanup_done = False
        self._setup_complete = not self._setup_enabled
        self._setup_stage_started = 0.0
        self._setup_settle_until = 0.0
        self._setup_nudge_direction = self._setup_prompt_first_step
        self._setup_nudge_next_at = 0.0
        self._setup_nudge_steps = 0
        self._setup_recovery_passive = False
        self._normal_cycle_ready_at = 0.0
        self._last_telemetry_sample = 0.0
        self._last_telemetry_reel_sample = 0.0
        self._telemetry_finalized = False
        self._input_focus_blocked = False
        self._screenshot_count = 0
        self._screenshot_tag_counts: dict[str, int] = {}

    @property
    def stats(self) -> dict[str, int]:
        return self._stats

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def target_fish_count(self) -> int:
        return self._target_fish_count

    def record_performance(
        self,
        average_tick_ms: float,
        maximum_tick_ms: float,
    ) -> None:
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            telemetry.performance(average_tick_ms, maximum_tick_ms, self._fps)

    def record_unhandled_error(self, exc: BaseException) -> None:
        self._failure(
            "unhandled_exception",
            exception_type=type(exc).__name__,
            message=str(exc),
        )

    def tick(self) -> None:
        if not self._sm.active:
            return

        self._observe_state()

        if not self._sm.running:
            # Once per pause, not every 16ms tick: release held keys and drop
            # the timing session — resuming into CAST_TIMING with a stale
            # session start used to fire an instant bogus "timing timeout".
            if not self._paused_cleanup_done:
                self._input.release_all_keys()
                self._timing_session = False
                self._paused_cleanup_done = True
            return
        self._paused_cleanup_done = False

        if not self._input_focus_allowed():
            if not self._input_focus_blocked:
                self._input.release_all_keys()
                self._input_focus_blocked = True
                self._failure("game_window_not_focused")
                print(
                    "\n[BOT] ввод приостановлен: окно GTA не активно",
                    flush=True,
                )
            return
        if self._input_focus_blocked:
            self._input_focus_blocked = False
            self._event("game_window_focus_restored")
            print("\n[BOT] окно GTA снова активно -> продолжаю", flush=True)

        self._update_fps()

        # A failed inventory/food attempt must take precedence over setup and
        # casting, including after an F9 pause that reset the public state to
        # IDLE.  Keep retrying the pending need; never silently fish again.
        if (
            self._sustenance_pending_needs is not None
            or self._sm.state == BotState.FISHING_SUSTENANCE
        ):
            self._sm.state = BotState.FISHING_SUSTENANCE
            self._tick_sustenance_pending()
            self._observe_state()
            return

        if self._setup_enabled and not self._setup_complete:
            self._tick_setup()
            self._observe_state()
            return

        handlers = {
            BotState.WAITING: self._tick_waiting,
            BotState.CASTING: self._tick_casting,
            BotState.CAST_TIMING: self._tick_cast_timing,
            BotState.WAITING_BITE: self._tick_waiting_bite,
            BotState.HOOKING: self._tick_hooking,
            BotState.REELING: self._tick_reeling,
            BotState.COLLECT: self._tick_collect,
        }
        handler = handlers.get(self._sm.state)
        if handler:
            handler()

        self._observe_state()

        # A hotkey callback may pause/stop the bot while a handler is sleeping
        # or solving a minigame. Do not perform debug captures or any further
        # work after that asynchronous state change.
        if not self._sm.running:
            return

        if self._debug and self._show_overlay:
            now = time.perf_counter()
            if now - self._last_overlay_save >= self._overlay_interval:
                self._last_overlay_save = now
                self._show_debug_overlay()

    def _grab(self, region: list[int]):
        return self._capture.grab_region(region)

    def _observe_state(self) -> None:
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            telemetry.observe_state(self._sm.state.name)

    def _input_focus_allowed(self) -> bool:
        checker = getattr(self._input, "is_game_focused", None)
        return True if not callable(checker) else bool(checker())

    def _event(self, name: str, **data: Any) -> None:
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            telemetry.event(name, **data)

    def _failure(self, reason: str, **data: Any) -> None:
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            telemetry.failure(reason, **data)

    def _pause_for_manual_recovery(self, reason: str) -> None:
        """Stop all input until F9 and a real start prompt are both present."""
        self._input.release_all_keys()
        self._cast_pressed = False
        self._timing_session = False
        self._timing_ignore_until = 0.0
        self._sustenance_checked_for_cast = False
        self._normal_cycle_ready_at = 0.0
        self._setup_enabled = True
        self._setup_complete = False
        self._setup_recovery_passive = True
        self._event(
            "fishing_recovery_paused",
            reason=reason,
            requires_start_prompt=True,
        )
        if self._sm.running:
            self._sm.toggle()

    def _begin_setup_start(self, reason: str) -> None:
        self._sm.state = BotState.FISHING_START
        self._cast_start_attempts = 0
        self._timing_ignore_until = 0.0
        now = time.perf_counter()
        self._setup_stage_started = now
        self._setup_settle_until = 0.0
        self._setup_nudge_direction = getattr(
            self,
            "_setup_prompt_first_step",
            "s",
        )
        self._setup_nudge_steps = 0
        self._setup_nudge_next_at = now + getattr(
            self,
            "_setup_prompt_search_delay",
            1.0,
        )
        self._event("setup_start_waiting", reason=reason)

    def _nudge_for_setup_prompt(self, now: float) -> bool:
        """Briefly step forward, then back, while looking for the start E."""
        if getattr(self, "_setup_recovery_passive", False):
            return False
        steps = int(getattr(self, "_setup_nudge_steps", 0))
        maximum = int(getattr(self, "_setup_prompt_max_steps", 2))
        if steps >= maximum:
            return False
        next_at = getattr(self, "_setup_nudge_next_at", float("inf"))
        if now < next_at:
            return False

        first_key = getattr(self, "_setup_prompt_first_step", "s")
        second_key = "w" if first_key == "s" else "s"
        key = getattr(self, "_setup_nudge_direction", first_key)
        hold_ms = int(getattr(self, "_setup_prompt_step_hold_ms", 500))
        self._input.tap_key(key, hold_ms)
        self._setup_nudge_steps = steps + 1
        self._event(
            "setup_prompt_search_step",
            key=key,
            hold_ms=hold_ms,
        )
        print(
            f"\n[BOT] подсказки начала нет -> короткий шаг {key.upper()} "
            f"({hold_ms} мс)",
            flush=True,
        )

        if key == first_key:
            self._setup_nudge_direction = second_key
            wait = getattr(self, "_setup_prompt_step_wait", 1.0)
        else:
            self._setup_nudge_direction = first_key
            wait = getattr(self, "_setup_prompt_cycle_pause", 2.0)
        self._setup_nudge_next_at = now + max(0.25, float(wait))
        return True

    def _tick_setup(self) -> None:
        if self._sm.state not in (
            BotState.FISHING_START,
            BotState.FISHING_EQUIPMENT,
        ):
            self._begin_setup_start("session_started")

        now = time.perf_counter()
        if now < self._setup_settle_until:
            return

        frame = self._capture.grab()
        if not self._sm.running:
            return
        scores = self._setup_detector.scores(frame)

        # Recovery takes precedence over generic fishing phase detection: the
        # equipment bar overlaps some old bottom-HUD ROIs and can otherwise be
        # mistaken for an already active fishing session.
        if self._sm.state == BotState.FISHING_START:
            passive_recovery = getattr(
                self,
                "_setup_recovery_passive",
                False,
            )
            if (
                not passive_recovery
                and scores.equipment
                >= self._setup_detector.equipment_threshold
            ):
                self._sm.state = BotState.FISHING_EQUIPMENT
                self._setup_stage_started = now
                self._event(
                    "setup_equipment_found",
                    recovered=True,
                    score=round(scores.equipment, 4),
                )
                return
            if scores.start >= self._setup_detector.start_threshold:
                self._setup_recovery_passive = False
                self._input.press_key_named(self._setup_key)
                self._event(
                    "setup_start_pressed",
                    key=self._setup_key,
                    score=round(scores.start, 4),
                )
                print(
                    f"\n[BOT] приглашение рыбалки найдено "
                    f"({scores.start:.2f}) -> {self._setup_key.upper()}",
                    flush=True,
                )
                self._sm.state = BotState.FISHING_EQUIPMENT
                self._setup_stage_started = now
                self._setup_settle_until = now + self._setup_after_start
                return

            if self._nudge_for_setup_prompt(now):
                return

            if now - self._setup_stage_started >= self._setup_start_timeout:
                self._failure(
                    "setup_start_prompt_timeout",
                    start_score=round(scores.start, 4),
                    equipment_score=round(scores.equipment, 4),
                    nudge_steps=int(getattr(self, "_setup_nudge_steps", 0)),
                    passive_recovery=bool(passive_recovery),
                )
                print(
                    "\n[BOT] не вижу приглашение начать рыбалку — "
                    "ставлю бота на безопасную паузу",
                    flush=True,
                )
                self._pause_for_manual_recovery(
                    "setup_start_prompt_timeout"
                )
            return

        # The first E may have gone to the control panel if the user had not
        # switched focus back to GTA yet.  Do not assume that it succeeded:
        # while the start prompt is still visible, retry E at the configured
        # interval and keep waiting for the actual equipment screen.
        if scores.start >= self._setup_detector.start_threshold:
            self._input.press_key_named(self._setup_key)
            self._event(
                "setup_start_retried",
                key=self._setup_key,
                score=round(scores.start, 4),
            )
            print(
                f"\n[BOT] приглашение всё ещё видно ({scores.start:.2f}) "
                f"-> повторяю {self._setup_key.upper()}",
                flush=True,
            )
            self._setup_stage_started = now
            self._setup_settle_until = now + self._setup_after_start
            return

        if scores.equipment >= self._setup_detector.equipment_threshold:
            self._setup_recovery_passive = False
            self._input.press_key_named(self._setup_key)
            self._event(
                "setup_equipment_confirmed",
                key=self._setup_key,
                score=round(scores.equipment, 4),
            )
            print(
                f"\n[BOT] снаряжение готово ({scores.equipment:.2f}) "
                f"-> {self._setup_key.upper()}, начинаю рыбалку",
                flush=True,
            )
            self._setup_complete = True
            self._sustenance_checked_for_cast = False
            self._normal_cycle_ready_at = now + self._setup_after_equipment
            self._sm.state = BotState.WAITING
            return

        if now - self._setup_stage_started >= self._setup_equipment_timeout:
            self._failure(
                "setup_equipment_prompt_timeout",
                start_score=round(scores.start, 4),
                equipment_score=round(scores.equipment, 4),
            )
            print(
                "\n[BOT] экран снаряжения не найден — возвращаюсь к "
                "поиску приглашения",
                flush=True,
            )
            self._begin_setup_start("equipment_timeout")

    def _recover_setup_if_visible(
        self,
        frame: np.ndarray,
        *,
        source: str,
    ) -> bool:
        """Re-enter setup if GTA resets the activity during a live cycle."""
        detector = getattr(self, "_setup_detector", None)
        if detector is None:
            return False
        scores = detector.scores(frame)
        if scores.equipment >= detector.equipment_threshold:
            self._setup_enabled = True
            self._setup_complete = False
            self._cast_pressed = False
            self._timing_session = False
            self._sustenance_checked_for_cast = False
            self._sm.state = BotState.FISHING_EQUIPMENT
            self._setup_stage_started = time.perf_counter()
            self._setup_settle_until = 0.0
            self._event(
                "setup_equipment_recovered",
                source=source,
                score=round(scores.equipment, 4),
            )
            print(
                f"\n[BOT] игра вернула экран снаряжения "
                f"({scores.equipment:.2f}) -> восстанавливаю рыбалку",
                flush=True,
            )
            return True

        if scores.start >= detector.start_threshold:
            self._setup_enabled = True
            self._setup_complete = False
            self._cast_pressed = False
            self._timing_session = False
            self._sustenance_checked_for_cast = False
            self._begin_setup_start("runtime_start_prompt")
            self._event(
                "setup_start_recovered",
                source=source,
                score=round(scores.start, 4),
            )
            print(
                f"\n[BOT] рыбалка завершилась ({scores.start:.2f}) "
                f"-> запускаю её заново",
                flush=True,
            )
            return True

        return False

    def _grab_timing(self) -> np.ndarray:
        return self._grab(self._timing_region)

    def _crop(self, frame: np.ndarray, region: list[int]) -> np.ndarray:
        return self._capture.crop_from_frame(frame, region)

    def _bite_signals(self, frame: np.ndarray) -> tuple[float, float, float]:
        hook = self._crop(frame, self._hook_prompt_region)
        text = self._crop(frame, self._bite_text_region)
        bobber = self._crop(frame, self._bobber_region)
        return (
            self._prompt.hook_match_score(hook),
            self._prompt.bite_text_score(text),
            self._prompt.bobber_bite_score(bobber),
        )

    def _detect_phase(self, frame: np.ndarray) -> FishPhase:
        return self._phase_det.detect(frame)

    def _timing_active_in(self, frame: np.ndarray) -> bool:
        """Fresh, stateless probe — must NOT be affected by the locked zone
        used during an active CAST_TIMING solve session (see CastTimingDetector)."""
        crop = self._crop(frame, self._timing_region)
        return crop.size > 0 and self._timing.is_active_fresh(crop)

    def _caught_modal_visible_in(self, frame: np.ndarray) -> bool:
        crop = self._crop(frame, self._modal_region)
        return crop.size > 0 and self._modal.is_visible(crop)

    def _uses_cast(self) -> bool:
        return self._mode in ("cast_and_reel", "full")

    def _after_timing_miss(self) -> None:
        """Bar gone / timeout — never auto-press Space unless auto_cast."""
        self._cast_pressed = False
        self._sm.state = self._next_cycle_state()

    def _handle_missing_timing_bar(self) -> None:
        """Cap failed cast starts and recover without an endless Space loop."""
        attempts = int(getattr(self, "_cast_start_attempts", 0))
        maximum = int(getattr(self, "_cast_start_max_attempts", 3))
        self._cast_pressed = False
        self._sustenance_checked_for_cast = False

        if attempts < maximum:
            retry_delay = float(
                getattr(self, "_cast_start_retry_delay", 1.0)
            )
            self._normal_cycle_ready_at = time.perf_counter() + retry_delay
            self._sm.state = BotState.WAITING
            return

        self._failure(
            "cast_start_retry_limit_reached",
            attempts=attempts,
            maximum=maximum,
        )
        self._event("cast_start_recovery", action="safe_pause")
        if getattr(self, "_save_screenshots", False):
            self._save_debug_screenshot("cast_start_retry_limit")
        self._pause_for_manual_recovery("cast_start_retry_limit")
        print(
            f"\n[BOT] полоса заброса не появилась {attempts} раза -> "
            "бот поставлен на паузу; продолжение только по F9 и подсказке E",
            flush=True,
        )

    def _next_cycle_state(self) -> BotState:
        self._sustenance_checked_for_cast = False
        if self._auto_cast and self._uses_cast():
            return BotState.CASTING
        return BotState.WAITING

    def _check_sustenance_before_cast(self) -> bool:
        """Pause fishing for food/water once before each cast timing stage."""
        if not getattr(self, "_sustenance_enabled", False):
            return False
        if getattr(self, "_sustenance_checked_for_cast", False):
            return False

        self._sustenance_checked_for_cast = True
        service = self._sustenance
        if service is None:
            return False

        needs = service.check_needs(force=True)
        self._record_sustenance_check(needs, source="before_cast")
        if not (needs.water_needed or needs.food_needed):
            return False

        self._start_sustenance_pending(needs)
        block_reason = service.action_block_reason()
        if block_reason is not None:
            self._event("sustenance_deferred", reason=block_reason)
            self._sustenance_last_deferred_reason = block_reason
            self._sustenance_retry_at = (
                time.perf_counter() + self._sustenance_failure_retry
            )
            return True

        self._attempt_pending_sustenance(needs)
        return True

    def _record_sustenance_check(
        self,
        needs: SustenanceResult,
        *,
        source: str,
    ) -> None:
        service = self._sustenance
        if service is None:
            return
        self._event(
            "sustenance_check",
            source=source,
            reason=needs.reason,
            water_detected=needs.water_detected,
            hunger_detected=needs.hunger_detected,
            water_needed=needs.water_needed,
            food_needed=needs.food_needed,
            thirst_score=round(service.last_thirst_score, 4),
            hunger_score=round(service.last_hunger_score, 4),
        )

    def _start_sustenance_pending(self, needs: SustenanceResult) -> None:
        first_pause = self._sustenance_pending_needs is None
        self._sustenance_pending_needs = needs
        self._sustenance_retry_at = 0.0
        self._sustenance_last_deferred_reason = ""
        if first_pause:
            self._sustenance_failure_count = 0
            self._sustenance_fishing_exited = False
        self._sm.state = BotState.FISHING_SUSTENANCE
        self._timing_session = False
        self._cast_pressed = False
        self._setup_enabled = True
        # Setup remains complete while food/water is pending so its E presses
        # can never run ahead of the sustenance workflow.
        self._setup_complete = True
        self._normal_cycle_ready_at = 0.0
        self._input.release_all_keys()
        if first_pause:
            self._stats["sustenance_pauses"] += 1

    def _exit_fishing_for_sustenance(self) -> bool:
        """Leave the fishing equipment state and verify it before pressing I."""
        # A pending sustenance workflow can make several inventory attempts.
        # Once ESC has already taken us out of fishing, another ESC may open the
        # GTA pause menu and make I look broken.  Keep the confirmed exit for
        # the whole pending workflow.
        if getattr(self, "_sustenance_fishing_exited", False):
            self._event(
                "sustenance_fishing_exit_confirmed",
                attempt=0,
                reason="already_confirmed",
            )
            return True

        # After a failed inventory attempt we may already be back at the
        # top-left E invitation. Pressing ESC there would open GTA's pause
        # menu, so recognise that safe state before sending another key.
        frame = self._capture.grab()
        scores = self._setup_detector.scores(frame)
        if (
            scores.start >= self._setup_detector.start_threshold
            and scores.equipment < self._setup_detector.equipment_threshold
        ):
            self._event(
                "sustenance_fishing_exit_confirmed",
                attempt=0,
                start_score=round(scores.start, 4),
                equipment_score=round(scores.equipment, 4),
                reason="already_outside",
            )
            self._sustenance_fishing_exited = True
            return True

        for attempt in range(1, self._sustenance_exit_attempts + 1):
            self._input.press_key_named(self._sustenance_exit_key)
            self._event(
                "sustenance_fishing_exit",
                attempt=attempt,
                key=self._sustenance_exit_key,
                water_needed=self._sustenance_pending_needs.water_needed,
                food_needed=self._sustenance_pending_needs.food_needed,
            )
            time.sleep(self._sustenance_exit_settle)
            if not self._sm.running:
                return False

            frame = self._capture.grab()
            scores = self._setup_detector.scores(frame)
            equipment_visible = (
                scores.equipment
                >= self._setup_detector.equipment_threshold
            )
            self._event(
                "sustenance_fishing_exit_probe",
                attempt=attempt,
                start_score=round(scores.start, 4),
                equipment_score=round(scores.equipment, 4),
                equipment_visible=equipment_visible,
            )
            if not equipment_visible:
                self._event(
                    "sustenance_fishing_exit_confirmed",
                    attempt=attempt,
                    start_score=round(scores.start, 4),
                    equipment_score=round(scores.equipment, 4),
                    reason="equipment_hidden",
                )
                self._sustenance_fishing_exited = True
                return True

        self._failure(
            "sustenance_fishing_exit_not_confirmed",
            attempts=self._sustenance_exit_attempts,
            equipment_score=round(scores.equipment, 4),
        )
        return False

    def _attempt_pending_sustenance(self, needs: SustenanceResult) -> None:
        service = self._sustenance
        if service is None or not self._sm.running:
            return

        self._sm.state = BotState.FISHING_SUSTENANCE
        print(
            "\n[BOT] нужна еда или вода -> ESC, приостанавливаю рыбалку",
            flush=True,
        )
        if not self._exit_fishing_for_sustenance():
            if self._sm.running:
                self._schedule_sustenance_retry("fishing_exit_not_confirmed")
            return

        result = service.consume_detected_needs(
            needs,
            allow_action=self._sm.running,
        )
        if result.drank:
            self._stats["drinks"] += 1
        if result.ate:
            self._stats["meals"] += 1
        self._event(
            "sustenance_result",
            reason=result.reason,
            drank=result.drank,
            ate=result.ate,
            water_found=result.water_found,
            food_found=result.food_found,
            inventory_attempts=result.inventory_attempts,
            water_score=round(
                getattr(service, "last_item_scores", {}).get("water", 0.0),
                4,
            ),
            food_score=round(
                getattr(service, "last_item_scores", {}).get("food", 0.0),
                4,
            ),
        )
        if result.reason == "inventory_not_open":
            self._failure(
                f"sustenance_{result.reason}",
                water_found=result.water_found,
                food_found=result.food_found,
            )
        else:
            if needs.water_needed and not result.water_found:
                self._failure("sustenance_water_not_found")
            if needs.food_needed and not result.food_found:
                self._failure("sustenance_food_not_found")
            if result.reason not in ("consumed", "levels_ok"):
                self._failure(f"sustenance_{result.reason}")

        water_done = not needs.water_needed or result.drank
        food_done = not needs.food_needed or result.ate
        if not (water_done and food_done):
            # Preserve only the needs that were not actually consumed.  A
            # later HUD frame saying no_need is not proof of success: the
            # warning can be hidden by inventory/transition frames.
            if result.drank or result.ate:
                self._sustenance_pending_needs = SustenanceResult(
                    checked=True,
                    water_detected=needs.water_detected,
                    hunger_detected=needs.hunger_detected,
                    water_needed=needs.water_needed and not result.drank,
                    food_needed=needs.food_needed and not result.ate,
                    reason="need",
                )
            else:
                # Preserve the original object as well as its flags when the
                # attempt made no progress at all.
                self._sustenance_pending_needs = needs
            self._sustenance_failure_count = (
                getattr(self, "_sustenance_failure_count", 0) + 1
            )
            maximum_failures = getattr(
                self,
                "_sustenance_max_failures",
                3,
            )
            self._event(
                "sustenance_attempt_failed",
                reason=result.reason,
                attempt=self._sustenance_failure_count,
                maximum=maximum_failures,
                water_pending=self._sustenance_pending_needs.water_needed,
                food_pending=self._sustenance_pending_needs.food_needed,
            )
            if self._sustenance_failure_count >= maximum_failures:
                self._failure(
                    "sustenance_retry_limit_reached",
                    attempts=self._sustenance_failure_count,
                    water_pending=self._sustenance_pending_needs.water_needed,
                    food_pending=self._sustenance_pending_needs.food_needed,
                )
                self._event(
                    "sustenance_manual_intervention_required",
                    water_pending=self._sustenance_pending_needs.water_needed,
                    food_pending=self._sustenance_pending_needs.food_needed,
                )
                self._pause_for_manual_recovery(
                    "sustenance_retry_limit_reached"
                )
                print(
                    "[BOT] еда/вода не найдены после нескольких попыток -> "
                    "безопасная пауза без движения; пополните инвентарь и "
                    "нажмите F9",
                    flush=True,
                )
                return
            self._schedule_sustenance_retry(result.reason)
            return

        self._sustenance_failure_count = 0
        self._event(
            "sustenance_completed",
            drank=result.drank,
            ate=result.ate,
        )
        time.sleep(self._sustenance_after_action)
        self._finish_sustenance_cycle()

    def _schedule_sustenance_retry(self, reason: str) -> None:
        self._sm.state = BotState.FISHING_SUSTENANCE
        self._sustenance_retry_at = (
            time.perf_counter() + self._sustenance_failure_retry
        )
        self._sustenance_last_deferred_reason = ""
        self._event(
            "sustenance_retry_scheduled",
            reason=reason,
            retry_after_sec=round(self._sustenance_failure_retry, 3),
        )
        print(
            "[BOT] еда/вода не завершены -> рыбалка заблокирована, повторяю проверку",
            flush=True,
        )

    def _tick_sustenance_pending(self) -> None:
        needs = self._sustenance_pending_needs
        service = self._sustenance
        if needs is None or service is None:
            return
        now = time.perf_counter()
        if now < self._sustenance_retry_at:
            return

        # F9 after a safe sustenance pause starts a fresh bounded retry batch.
        maximum_failures = getattr(self, "_sustenance_max_failures", 3)
        if (
            getattr(self, "_sustenance_failure_count", 0)
            >= maximum_failures
        ):
            self._sustenance_failure_count = 0
            self._event("sustenance_retry_batch_resumed")

        block_reason = service.action_block_reason()
        if block_reason is not None:
            if block_reason != self._sustenance_last_deferred_reason:
                self._event("sustenance_deferred", reason=block_reason)
                self._sustenance_last_deferred_reason = block_reason
            self._sustenance_retry_at = now + self._sustenance_failure_retry
            return

        self._sustenance_last_deferred_reason = ""
        current = service.check_needs(force=True)
        self._record_sustenance_check(current, source="retry")
        if not (current.water_needed or current.food_needed):
            self._event(
                "sustenance_pending_retained",
                reason=current.reason,
                source="hud_recheck",
                water_pending=needs.water_needed,
                food_pending=needs.food_needed,
            )
            self._attempt_pending_sustenance(needs)
            return

        merged = SustenanceResult(
            checked=True,
            water_detected=needs.water_detected or current.water_detected,
            hunger_detected=needs.hunger_detected or current.hunger_detected,
            water_needed=needs.water_needed or current.water_needed,
            food_needed=needs.food_needed or current.food_needed,
            reason="need",
        )
        self._sustenance_pending_needs = merged
        self._attempt_pending_sustenance(merged)

    def _finish_sustenance_cycle(self) -> None:
        self._sustenance_pending_needs = None
        self._sustenance_retry_at = 0.0
        self._sustenance_last_deferred_reason = ""
        self._sustenance_failure_count = 0
        self._sustenance_fishing_exited = False

        self._cast_pressed = False
        self._timing_session = False
        self._setup_enabled = True
        self._setup_complete = False
        self._normal_cycle_ready_at = 0.0
        self._sustenance_checked_for_cast = False
        if self._sm.running:
            self._begin_setup_start("sustenance_completed")
        print(
            "[BOT] еда/вода завершены -> заново запускаю рыбалку",
            flush=True,
        )

    def _tick_waiting(self) -> None:
        if time.perf_counter() < getattr(self, "_normal_cycle_ready_at", 0.0):
            return
        frame = self._capture.grab()
        if not self._sm.running:
            return
        if self._recover_setup_if_visible(frame, source="waiting"):
            return
        if self._timing_active_in(frame):
            self._sm.state = BotState.CAST_TIMING
            return

        phase = self._detect_phase(frame)

        if phase == FishPhase.CAUGHT:
            self._sm.state = BotState.COLLECT
        elif phase == FishPhase.CAST_TIMING:
            self._sm.state = BotState.CAST_TIMING
        elif phase == FishPhase.REELING:
            # No cast_pressed gate: if the bot bailed out of a live fight
            # (ui_never_appeared / timeout) the flag is already False, and the
            # gate used to leave the fish to escape. Phase REELING is confirmed
            # by either the strict horizontal fill or icon + fixed-bar signal.
            self._sm.state = BotState.REELING
        elif phase == FishPhase.BITE:
            # Recovery path: the bot may have been enabled after the cast or
            # returned to WATCHING just as the bite appeared.
            self._press_hook(frame)
            return
        elif phase == FishPhase.WAITING:
            # Recover an already active session even if this process did not
            # observe the original cast/timing press.
            self._start_waiting_bite()
            return
        elif self._uses_cast() and self._auto_cast and not self._cast_pressed:
            self._sm.state = BotState.CASTING

    def _tick_casting(self) -> None:
        frame = self._capture.grab()
        if not self._sm.running:
            return
        if self._recover_setup_if_visible(frame, source="casting"):
            return
        phase = self._detect_phase(frame)
        if phase == FishPhase.CAUGHT:
            self._sm.state = BotState.COLLECT
            return

        if self._check_sustenance_before_cast():
            return

        self._sm.state = BotState.CAST_TIMING

    def _tick_cast_timing(self) -> None:
        if self._check_sustenance_before_cast():
            return

        def get_frame():
            return self._grab_timing()

        def press_key(key: str):
            self._input.press_key_named(key)
            self._stats["timing_hits"] += 1
            if not self._cast_pressed:
                self._cast_pressed = True
                self._stats["casts"] += 1

        frame = get_frame()

        if not self._timing_session:
            self._timing_session = True
            self._timing_solve_start = time.perf_counter()
            self._timing.reset()
            self._timing_no_slider_streak = 0
            already_active = self._timing.is_active(frame)

            if not already_active:
                if self._auto_cast:
                    self._cast_start_attempts = int(
                        getattr(self, "_cast_start_attempts", 0)
                    ) + 1
                    self._input.press_key_named(self._cast_key)
                    self._event(
                        "cast_started",
                        key=self._cast_key,
                        attempt=self._cast_start_attempts,
                        maximum=self._cast_start_max_attempts,
                    )
                    if self._debug:
                        print(f"\n[BOT] cast start ({self._cast_key})", flush=True)
                    time.sleep(0.05)
                deadline = time.perf_counter() + self._timing._appear_timeout
                while time.perf_counter() < deadline:
                    if not self._sm.running or not self._input_focus_allowed():
                        self._timing_session = False
                        return
                    frame = get_frame()
                    if self._timing.is_active(frame):
                        break
                    time.sleep(self._timing.poll_sec)
                if not self._timing.is_active(frame):
                    self._timing_session = False
                    self._failure(
                        "timing_bar_not_seen",
                        attempt=self._cast_start_attempts,
                        maximum=self._cast_start_max_attempts,
                    )
                    if self._debug:
                        print(
                            "\n[BOT] timing bar not seen — waiting (no Space)",
                            flush=True,
                        )
                    self._handle_missing_timing_bar()
                    return

            # A real timing bar proves that GTA accepted the cast-start input.
            self._cast_start_attempts = 0

            analysis = self._timing.analyze(frame)
            if self._debug:
                slider_label = analysis.slider_x if analysis.slider_found else -1
                print(
                    f"\n[BOT] timing bar seen={analysis.active} "
                    f"zone={analysis.green_left}-{analysis.green_right} "
                    f"center={analysis.green_center} slider={slider_label} "
                    f"slider_found={analysis.slider_found} in_zone={analysis.in_zone}",
                    flush=True,
                )
                if self._show_overlay:
                    self._save_timing_debug()
        else:
            analysis = self._timing.analyze(frame)

        if not self._sm.running:
            self._timing_session = False
            return

        if not analysis.active:
            self._timing_session = False
            self._failure("timing_bar_disappeared")
            if self._debug:
                msg = "timing bar gone — recast" if self._auto_cast else "timing bar gone — waiting"
                print(f"\n[BOT] {msg}", flush=True)
            self._after_timing_miss()
            return

        if time.perf_counter() - self._timing_solve_start >= self._timing._timeout:
            self._timing_session = False
            self._failure("timing_timeout")
            if self._debug:
                msg = "timing timeout — recast" if self._auto_cast else "timing timeout — waiting"
                print(f"\n[BOT] {msg}", flush=True)
            self._after_timing_miss()
            return

        # Green zone without a slider is a false positive (HUD tint, leftover
        # pixels) — burns the full timing timeout if we treat it as active.
        if analysis.active and not analysis.slider_found:
            self._timing_no_slider_streak += 1
            if self._timing_no_slider_streak >= 12:
                self._timing_session = False
                self._failure("timing_ghost_bar")
                if self._debug:
                    msg = "timing ghost bar (no slider) — waiting"
                    print(f"\n[BOT] {msg}", flush=True)
                self._after_timing_miss()
                return
        else:
            self._timing_no_slider_streak = 0

        hit, reason = self._timing.evaluate(analysis, frame.shape[1])

        if self._debug and time.perf_counter() - self._last_timing_debug >= self._timing_debug_interval:
            self._last_timing_debug = time.perf_counter()
            if analysis.slider_found:
                prev = self._timing._prev_slider_x
                prev_label = prev if prev is not None else -1
                print(
                    f"\n[BOT] timing track zone={analysis.green_left}-{analysis.green_right} "
                    f"slider={analysis.slider_x} prev={prev_label} "
                    f"in_zone={analysis.in_zone} FPS={self.fps:.0f}",
                    flush=True,
                )
            if self._show_overlay:
                self._save_timing_debug_frame(frame)

        if hit:
            timing_hit_at = time.perf_counter()
            self._event(
                "timing_hit",
                reason=reason,
                slider_x=int(analysis.slider_x),
                green_left=int(analysis.green_left),
                green_right=int(analysis.green_right),
            )
            if self._debug:
                print(
                    f"\n[BOT] Space PRESS ({reason}) slider={analysis.slider_x} "
                    f"zone={analysis.green_left}-{analysis.green_right}",
                    flush=True,
                )
            press_key(self._cast_key)
            # GTA keeps the completed bar visible for roughly 0.7-1.0 seconds.
            # During that animation it still contains a real slider and green
            # zone, so the stateless phase probe used by WAITING_BITE used to
            # mistake it for a second cast and start a Space/ghost-bar loop.
            # Keep detecting bites during this grace period, but do not allow
            # the just-completed bar to pull the state back to CAST_TIMING.
            self._timing_ignore_until = (
                timing_hit_at + getattr(self, "_post_timing_grace", 1.2)
            )
            self._timing_session = False
            self._timing.reset()
            if not self._sm.running:
                return
            time.sleep(self._cast_settle)
            if not self._sm.running:
                return
            self._bite_wait_start = time.perf_counter()
            self._cast_pressed = True
            self._reset_bite_baseline()
            self._sm.state = BotState.WAITING_BITE
            return

        if reason == "lost":
            self._timing_session = False
            self._failure("timing_lost")
            if self._debug:
                msg = "timing lost — recast" if self._auto_cast else "timing lost — waiting"
                print(f"\n[BOT] {msg}", flush=True)
            self._after_timing_miss()

    def _start_waiting_bite(self) -> None:
        self._cast_start_attempts = 0
        self._timing_ignore_until = 0.0
        self._bite_wait_start = time.perf_counter()
        self._cast_pressed = True
        self._reset_bite_baseline()
        self._sm.state = BotState.WAITING_BITE

    def _reset_bite_baseline(self) -> None:
        self._baseline_ready = False
        self._baseline_hook = 0.0
        self._baseline_text = 0.0
        self._baseline_bobber = None
        self._bite_streak = 0
        self._baseline_text_samples = []

    def _maybe_capture_baseline(self, frame: np.ndarray) -> None:
        if self._baseline_ready:
            return
        timing_crop = self._crop(frame, self._timing_region)
        if self._timing.is_active_fresh(timing_crop):
            return
        text = self._crop(frame, self._bite_text_region)
        txt_sc = self._prompt.bite_text_score(text)
        self._baseline_text_samples.append(txt_sc)
        if time.perf_counter() - self._bite_wait_start < self._baseline_settle:
            return
        hook = self._crop(frame, self._hook_prompt_region)
        bobber = self._crop(frame, self._bobber_region)
        self._baseline_hook = self._prompt.hook_match_score(hook)
        self._baseline_text = min(self._baseline_text_samples) if self._baseline_text_samples else txt_sc
        self._baseline_bobber = bobber.copy() if bobber.size else None
        self._baseline_ready = True
        if self._debug:
            print(
                f"\n[BOT] baseline wait hook={self._baseline_hook:.2f} "
                f"text={self._baseline_text:.2f} (min of {len(self._baseline_text_samples)} samples, "
                f"last={txt_sc:.2f})",
                flush=True,
            )

    def _bobber_change(self, current: np.ndarray) -> float:
        if self._baseline_bobber is None or current.size == 0:
            return 0.0
        base = self._baseline_bobber
        if current.shape != base.shape:
            current = cv2.resize(current, (base.shape[1], base.shape[0]))
        return float(np.mean(cv2.absdiff(current, base))) / 255.0

    def _tick_waiting_bite(self) -> None:
        # One capture, one pass of checks per call — the old version looped
        # this up to bite_poll_burst (8) times per tick with an 8ms sleep
        # between iterations, on the assumption each capture was cheap
        # (~8-10ms). Measured on a real run: each mss capture actually costs
        # ~100-110ms *fixed*, regardless of region size or count — so the
        # burst was really 8 x ~110ms back-to-back (~900ms per tick) with the
        # 8ms sleeps being negligible in comparison. Shrinking the captured
        # area (grab_union) barely helped because area wasn't the dominant
        # cost. Doing a single check per call and letting the outer main
        # loop's own cadence (tick() + 16ms sleep, see main_fishing.py)
        # provide the polling rate fixes this: ~1 capture ≈ 110-150ms per
        # outer tick instead of ~900ms, and bite_confirm_frames/baseline
        # logic (all persisted on self, not on this loop) behaves identically
        # either way — it just accumulates across outer ticks instead of
        # inner-loop iterations.
        if not self._sm.running:
            return

        frame = self._capture.grab_union(self._bite_poll_regions)
        if not self._sm.running:
            return

        if self._caught_modal_visible_in(frame):
            self._sm.state = BotState.COLLECT
            return

        timing_probe_allowed = time.perf_counter() >= getattr(
            self,
            "_timing_ignore_until",
            0.0,
        )
        if timing_probe_allowed and self._timing_active_in(frame):
            self._sm.state = BotState.CAST_TIMING
            return
        if self._should_hook(frame):
            hook_sc, txt_sc, bob_sc = self._bite_signals(frame)
            if self._debug:
                keys = "+".join(self._hook_keys)
                if self._hook_trigger == "bite_text":
                    print(
                        f"\n[BOT] КЛЁВ! text={txt_sc:.2f} → {keys}",
                        flush=True,
                    )
                else:
                    print(
                        f"\n[BOT] КЛЁВ! hook={hook_sc:.2f} text={txt_sc:.2f} "
                        f"bobber={bob_sc:.2f} → {keys}",
                        flush=True,
                    )
            self._press_hook(frame)
            return

        telemetry = getattr(self, "_telemetry", None)
        now = time.perf_counter()
        sample_interval = (
            telemetry.sample_interval_sec if telemetry is not None else 2.0
        )
        if (
            telemetry is not None
            and telemetry.enabled
            and now - getattr(self, "_last_telemetry_sample", 0.0)
            >= sample_interval
        ):
            self._last_telemetry_sample = now
            hook_sc, txt_sc, bob_sc = self._bite_signals(frame)
            text_frame = self._crop(frame, self._bite_text_region)
            scores = self._prompt.text_scores(text_frame)
            self._event(
                "bite_sample",
                elapsed_sec=round(now - self._bite_wait_start, 3),
                baseline_ready=bool(self._baseline_ready),
                text_score=round(txt_sc, 4),
                keyword_score=round(scores.keyword, 4),
                wait_score=round(scores.wait, 4),
                hook_score=round(hook_sc, 4),
                bobber_score=round(bob_sc, 4),
                block_reason=self._last_bite_block_reason,
            )

        if self._debug and time.perf_counter() - self._last_bite_debug >= self._bite_debug_interval:
            self._last_bite_debug = time.perf_counter()
            hook_sc, txt_sc, bob_sc = self._bite_signals(frame)
            text_frame = self._crop(frame, self._bite_text_region)
            scores = self._prompt.text_scores(text_frame)
            kw_sc, wait_sc = scores.keyword, scores.wait
            bob_ch = self._bobber_change(self._crop(frame, self._bobber_region))
            ready = "ok" if self._baseline_ready else "settling"
            need_delta = self._baseline_text + self._text_delta_min
            print(
                f"\n[BOT] жду клёв [{ready}] trigger={self._hook_trigger} "
                f"text={txt_sc:.2f}(base={self._baseline_text:.2f} need≥{need_delta:.2f} "
                f"or abs≥{self._bite_absolute_threshold:.2f}) "
                f"kw={kw_sc:.2f} wait={wait_sc:.2f} "
                f"hook={hook_sc:.2f} bobΔ={bob_ch:.3f} "
                f"block={self._last_bite_block_reason}",
                flush=True,
            )

        elapsed = time.perf_counter() - self._bite_wait_start
        if self._bite_timeout > 0 and elapsed >= self._bite_timeout:
            text_frame = self._crop(frame, self._bite_text_region)
            hook_frame = self._crop(frame, self._hook_prompt_region)
            scores = self._prompt.text_scores(text_frame)
            session_alive = (
                scores.waiting
                or self._prompt.is_fishing_control_visible(hook_frame)
            )
            if session_alive:
                # A fixed 35-second timeout used to abandon valid casts while
                # the game explicitly still said to wait. Keep the timeout as
                # a lost-session watchdog, but renew it while fishing UI lives.
                self._bite_wait_start = time.perf_counter()
                self._event(
                    "bite_wait_extended",
                    wait_score=round(scores.wait, 4),
                )
                if self._debug:
                    print(
                        "\n[BOT] bite timeout reached, fishing UI still active "
                        "-> keep waiting",
                        flush=True,
                    )
                return
            if self._debug:
                print("\n[BOT] bite timeout → recast", flush=True)
            self._failure(
                "bite_timeout",
                wait_score=round(scores.wait, 4),
            )
            if self._save_screenshots:
                self._save_debug_screenshot("bite_timeout")
                if not self._sm.running:
                    return
            self._stats["timeouts"] += 1
            self._cast_pressed = False
            self._sm.state = self._next_cycle_state()

    def _should_hook(self, frame: np.ndarray) -> bool:
        if time.perf_counter() - self._bite_wait_start < self._min_bite_delay:
            self._last_bite_block_reason = "min_delay"
            return False
        self._maybe_capture_baseline(frame)
        if not self._baseline_ready:
            self._last_bite_block_reason = "baseline_settling"
            return False

        text_frame = self._crop(frame, self._bite_text_region)
        scores = self._prompt.text_scores(text_frame)
        txt_sc, kw_sc, wait_sc, waiting = scores.bite, scores.keyword, scores.wait, scores.waiting

        if self._hook_trigger == "bite_text":
            if waiting:
                self._last_bite_block_reason = f"waiting(wait={wait_sc:.2f})"
                self._bite_streak = 0
                return False

            need_delta = self._baseline_text + self._text_delta_min
            delta_hit = txt_sc >= self._bite_text_threshold and txt_sc >= need_delta
            abs_hit = txt_sc >= self._bite_absolute_threshold
            kw_hit = kw_sc >= self._bite_keyword_threshold
            text_hit = delta_hit or abs_hit or kw_hit

            if not text_hit:
                self._last_bite_block_reason = (
                    f"text={txt_sc:.2f} need≥{self._bite_text_threshold:.2f} "
                    f"and ≥{need_delta:.2f}(Δ{self._text_delta_min:.2f}) "
                    f"OR abs≥{self._bite_absolute_threshold:.2f} "
                    f"OR kw≥{self._bite_keyword_threshold:.2f}(kw={kw_sc:.2f}) "
                    f"wait={wait_sc:.2f}"
                )
                self._bite_streak = 0
                return False

            self._bite_streak += 1
            if self._debug:
                mode = "delta" if delta_hit else ("absolute" if abs_hit else "keyword")
                self._last_bite_block_reason = f"HIT({mode})"
            return self._bite_streak >= self._bite_confirm_frames

        hook_sc, _, _ = self._bite_signals(frame)
        bob_ch = self._bobber_change(self._crop(frame, self._bobber_region))
        hook_hit = (
            hook_sc >= self._hook_match_threshold
            and (hook_sc - self._baseline_hook) >= self._hook_delta_min
        )
        text_hit = (
            txt_sc >= self._bite_text_threshold
            and (txt_sc - self._baseline_text) >= self._text_delta_min
        )
        bob_hit = bob_ch >= self._bobber_change_min

        if hook_hit or text_hit or bob_hit:
            self._bite_streak += 1
        else:
            self._bite_streak = 0

        return self._bite_streak >= self._bite_confirm_frames

    def _press_hook(self, detected_frame: np.ndarray | None = None) -> None:
        if not self._sm.running:
            return
        delay = random.uniform(self._hook_delay[0], self._hook_delay[1]) / 1000.0
        if delay > 0:
            time.sleep(delay)
        if not self._sm.running:
            return

        for key in self._hook_keys:
            if not self._sm.running:
                return
            self._input.press_key_named(key)

        self._stats["hooks"] += 1
        hook_data: dict[str, Any] = {
            "keys": list(self._hook_keys),
            "delay_ms": round(delay * 1000.0, 2),
        }
        if detected_frame is not None:
            hook_sc, txt_sc, bob_sc = self._bite_signals(detected_frame)
            hook_data.update(
                hook_score=round(hook_sc, 4),
                text_score=round(txt_sc, 4),
                bobber_score=round(bob_sc, 4),
                block_reason=self._last_bite_block_reason,
            )
        self._event("hook_pressed", **hook_data)
        if self._debug:
            keys = "+".join(self._hook_keys)
            print(f"\n[BOT] подсечка ({keys})", flush=True)

        # Use the frame that already triggered detection, and save it only
        # after the key press. A second full capture plus PNG compression before
        # Space introduced a large and unnecessary hook delay.
        if self._save_screenshots and detected_frame is not None:
            self._save_debug_screenshot("bite_detected", detected_frame)

        if not self._sm.running:
            return
        time.sleep(0.08)
        if self._sm.running:
            self._sm.state = BotState.REELING

    def _tick_hooking(self) -> None:
        self._press_hook()

    def _continue_after_fish_escape(
        self,
        frame: np.ndarray,
        *,
        reel_duration: float,
    ) -> None:
        """Continue an already-visible next cast instead of stopping the bot."""
        duration = round(reel_duration, 3)
        self._failure("fish_escaped", duration_sec=duration)
        self._event(
            "reel_recovery",
            action="continue_cast_timing",
            reason="fish_escaped",
            duration_sec=duration,
        )
        if self._save_screenshots:
            self._save_debug_screenshot("fish_escaped", frame)

        # The game has already opened the next timing bar. CAST_TIMING must
        # solve that bar as-is; it must not press Space to start another one.
        self._cast_pressed = False
        self._cast_start_attempts = 0
        self._sustenance_checked_for_cast = False
        self._timing_session = False
        self._timing_ignore_until = 0.0
        self._sm.state = BotState.CAST_TIMING
        print(
            "\n[BOT] рыба сорвалась, следующая полоса уже открыта -> продолжаю заброс",
            flush=True,
        )

    def _continue_after_unidentified_catch(
        self,
        frame: np.ndarray,
        *,
        reel_duration: float,
    ) -> None:
        """Continue when GTA ends reeling without showing a fish modal.

        Majestic does not always show the normal fish dialog: junk can be
        collected directly.  The screen is therefore a valid terminal result,
        not a reason to leave the bot paused for the rest of the night.
        """
        duration = round(reel_duration, 3)
        self._stats["unidentified_catches"] = (
            self._stats.get("unidentified_catches", 0) + 1
        )
        self._event(
            "catch_without_modal",
            classification="junk_or_unidentified",
            count=self._stats["unidentified_catches"],
            duration_sec=duration,
        )
        self._event(
            "reel_recovery",
            action="continue_after_unidentified_catch",
            reason="junk_or_no_modal",
            duration_sec=duration,
        )
        if self._save_screenshots:
            self._save_debug_screenshot("catch_without_modal", frame)

        self._cast_pressed = False
        self._cast_start_attempts = 0
        self._timing_session = False
        self._timing_ignore_until = 0.0
        self._sm.state = self._next_cycle_state()
        print(
            "\n[BOT] окно рыбы не появилось — возможно пойман мусор; "
            "продолжаю следующий цикл",
            flush=True,
        )

    def _recover_known_post_reel_state(
        self,
        frame: np.ndarray,
        *,
        reel_duration: float,
    ) -> bool:
        """Recover only from a positively identified screen after reeling."""
        if self._timing_active_in(frame):
            self._continue_after_fish_escape(
                frame,
                reel_duration=reel_duration,
            )
            return True

        if self._recover_setup_if_visible(frame, source="post_reel"):
            self._event("reel_recovery", action="resume_setup")
            return True

        phase_detector = getattr(self, "_phase_det", None)
        phase = (
            phase_detector.detect(frame)
            if phase_detector is not None
            else FishPhase.UNKNOWN
        )
        if phase == FishPhase.CAUGHT:
            self._sm.state = BotState.COLLECT
            return True
        if phase == FishPhase.CAST_TIMING:
            self._continue_after_fish_escape(
                frame,
                reel_duration=reel_duration,
            )
            return True
        if phase == FishPhase.REELING:
            self._event("reel_recovery", action="resume_reeling")
            self._sm.state = BotState.REELING
            return True
        if phase == FishPhase.WAITING:
            self._event("reel_recovery", action="resume_waiting_bite")
            self._start_waiting_bite()
            return True
        if phase == FishPhase.BITE:
            self._event("reel_recovery", action="hook_visible_bite")
            self._press_hook(frame)
            return True
        return False

    def _tick_reeling(self) -> None:
        modal_frame = self._grab(self._modal_region)
        if not self._sm.running:
            return
        if self._modal.is_visible(modal_frame):
            self._input.set_reel_direction(None)
            self._sm.state = BotState.COLLECT
            return

        # One mss capture per loop iteration (~40ms) shared by full/bar/prompt/modal
        # checks instead of 2-3 separate captures (~90ms+) — was the main reason
        # reeling felt sluggish.
        last_frame: list[np.ndarray] = []

        def get_full():
            frame = self._capture.grab()
            last_frame[:] = [frame]
            return frame

        def get_bar():
            frame = last_frame[0]
            crop = self._fight.crop_bar(frame)
            if self._fight.is_sane_bar_crop(crop, frame.shape[1]):
                return crop
            return self._capture.crop_from_frame(frame, self._fight_bar_region)

        def get_prompt():
            return self._capture.crop_from_frame(last_frame[0], self._prompt_region)

        def set_steer(direction: str | None):
            self._input.set_reel_direction(direction)

        def is_caught(full_frame: np.ndarray) -> bool:
            modal_crop = self._capture.crop_from_frame(full_frame, self._modal_region)
            return self._modal.is_visible(modal_crop)

        def on_steer(analysis):
            telemetry_now = time.perf_counter()
            if (
                telemetry_now
                - getattr(self, "_last_telemetry_reel_sample", 0.0)
                >= 2.0
            ):
                self._last_telemetry_reel_sample = telemetry_now
                self._event(
                    "reel_sample",
                    active=bool(analysis.active),
                    source=str(analysis.source),
                    direction=analysis.direction.name,
                    progress=(
                        round(float(analysis.progress), 4)
                        if analysis.progress is not None
                        else None
                    ),
                    marker_x=int(analysis.marker_x),
                    delta_x=int(analysis.delta_x),
                )
            if not self._debug:
                return
            if not analysis.active:
                print("\r[BOT] reel ждём начало вываживания...".ljust(70), end="", flush=True)
                return
            key = "-" if analysis.direction.name == "CENTER" else ("A" if analysis.direction.name == "LEFT" else "D")
            prog_s = f"{analysis.progress:.2f}" if analysis.progress is not None else "?"
            now = time.perf_counter()
            if now - self._last_reel_log >= 2.0:
                print(
                    f"\n[BOT] reel {analysis.source} prog={prog_s} "
                    f"x={analysis.marker_x} Δ{analysis.delta_x:+d}px → hold {key}",
                    flush=True,
                )
                self._last_reel_log = now
            else:
                # \r line padded to a fixed width — shorter reprints used to
                # leave tails of the previous line ("hold D   d D") in logs.
                msg = (
                    f"[BOT] reel {analysis.source} prog={prog_s} "
                    f"Δ{analysis.delta_x:+d}px → hold {key}"
                )
                print("\r" + msg.ljust(70), end="", flush=True)
            self._debug_frame_counter += 1
            if self._show_overlay and self._debug_frame_counter % 10 == 0:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
                # Pass the tick's analysis — draw_debug must not re-analyze
                # (analyze() consumes pan measurements and can quiet-flip).
                dbg = self._fight.draw_debug(last_frame[0], get_bar(), analysis)
                cv2.imwrite(str(LOGS_DIR / "reel_live.png"), dbg)

        reel_started = time.perf_counter()
        success = self._fight.solve(
            get_full,
            get_bar,
            get_prompt,
            set_steer,
            is_caught=is_caught,
            should_abort=lambda: (
                not self._sm.running or not self._input_focus_allowed()
            ),
            on_steer=on_steer,
        )
        reel_duration = time.perf_counter() - reel_started
        self._input.set_reel_direction(None)

        # Preserve IDLE/STOPPED set asynchronously by F9/F10. Previously the
        # tail below overwrote STOPPED with WAITING and left a dead process
        # running forever after the stop hotkey.
        if not self._sm.running:
            return

        final_frame = self._capture.grab()
        if not self._sm.running:
            return
        if is_caught(final_frame):
            self._event(
                "reel_finished",
                success=True,
                reason="caught_modal",
                duration_sec=round(reel_duration, 3),
            )
            self._sm.state = BotState.COLLECT
            return

        reason = self._fight.last_solve_reason
        self._event(
            "reel_finished",
            success=bool(success),
            reason=reason or "unknown",
            duration_sec=round(reel_duration, 3),
        )
        if reason == "timeout":
            self._failure("reel_timeout", duration_sec=round(reel_duration, 3))
        elif reason == "ui_never_appeared":
            self._failure(
                "reel_ui_never_appeared",
                duration_sec=round(reel_duration, 3),
            )
        if success or reason == "ui_gone":
            post_reel_frame = final_frame
            timing_streak = 0
            deadline = time.perf_counter() + self._reel_modal_wait
            while time.perf_counter() < deadline:
                if not self._sm.running:
                    return
                post_reel_frame = self._capture.grab()
                if not self._sm.running:
                    return
                modal_frame = self._capture.crop_from_frame(
                    post_reel_frame,
                    self._modal_region,
                )
                if self._modal.is_visible(modal_frame):
                    self._sm.state = BotState.COLLECT
                    return

                # A real timing bar with both the green zone and moving slider
                # proves that the fish escaped and GTA has already advanced to
                # the next cast. Confirm it twice to reject a transition frame.
                if self._timing_active_in(post_reel_frame):
                    timing_streak += 1
                    if timing_streak >= 2:
                        self._continue_after_fish_escape(
                            post_reel_frame,
                            reel_duration=reel_duration,
                        )
                        return
                else:
                    timing_streak = 0
                time.sleep(0.1)

            if self._recover_known_post_reel_state(
                post_reel_frame,
                reel_duration=reel_duration,
            ):
                return

            self._continue_after_unidentified_catch(
                post_reel_frame,
                reel_duration=reel_duration,
            )
            return

        if self._save_screenshots and not success:
            self._save_debug_screenshot("reel_timeout")
            if not self._sm.running:
                return

        if self._debug:
            messages = {
                "timeout": f"таймаут вываживания ({self._fight.timeout_sec:.0f}с) — перепроверяю интерфейс",
                "ui_never_appeared": "миниигра A/D не появилась после подсечки",
                "ui_gone": "вываживание кончилось, но окно улова не появилось",
                "aborted": "остановлено пользователем",
            }
            if not success or reason == "ui_gone":
                print(
                    f"\n[BOT] рыба не поймана: {messages.get(reason, reason or 'unknown')}",
                    flush=True,
                )

        if self._sm.running and reason == "timeout":
            # A hard timeout does not prove that the fish escaped. Long real
            # fights reached the timeout with 65-92% progress, and jumping
            # directly to CASTING then produced a cascade of false timing
            # errors while the A/D UI was still visible. Route through
            # WAITING: it resumes REELING when that UI remains, collects a
            # late modal, or casts again only after the screen is clear.
            self._cast_pressed = False
            self._sustenance_checked_for_cast = False
            self._sm.state = BotState.WAITING
            return

        if self._sm.running:
            self._cast_pressed = False
            self._sm.state = self._next_cycle_state()

    def _tick_collect(self) -> None:
        modal_frame = self._grab(self._modal_region)
        if not self._sm.running:
            return
        if not self._modal.is_visible(modal_frame):
            self._cast_pressed = False
            self._sm.state = self._next_cycle_state()
            return

        # Small settle delay before clicking — the modal/button can still be
        # mid-animation the instant it's first detected as visible.
        time.sleep(self._keep_click_delay)
        if not self._sm.running:
            return

        btn = fishing_region(self._config.get("regions", {}), "keep_button")
        x, y, w, h = self._capture.scaled_region(btn)
        rect = self._capture.rect
        sx = rect.left + x + w // 2
        sy = rect.top + y + h // 2
        self._input.click_at(sx, sy)
        # Settle delay before restarting the cycle (casting again) — gives
        # the "kept fish" UI time to fully close instead of the next cast
        # potentially firing while it's still on screen.
        time.sleep(self._after_collect)
        if not self._sm.running:
            return
        # Count the fish only after the modal actually closed — a missed
        # click used to inflate the counter; if it's still up, stay in
        # COLLECT and the next tick clicks again.
        modal_frame = self._grab(self._modal_region)
        if not self._sm.running:
            return
        if self._modal.is_visible(modal_frame):
            self._failure("keep_click_missed")
            if self._debug:
                print("\n[BOT] keep click missed — retrying", flush=True)
            return
        self._stats["fish_caught"] += 1
        self._event(
            "fish_collected",
            fish_number=self._stats["fish_caught"],
        )
        if self._debug:
            print(f"\n[BOT] kept fish #{self._stats['fish_caught']}", flush=True)
        if self._stop_if_fish_target_reached():
            return
        self._cast_pressed = False
        self._sm.state = self._next_cycle_state()

    def _stop_if_fish_target_reached(self) -> bool:
        target = max(0, int(getattr(self, "_target_fish_count", 0)))
        caught = int(self._stats.get("fish_caught", 0))
        if target == 0 or caught < target:
            return False

        self._event(
            "fish_target_reached",
            target=target,
            fish_caught=caught,
        )
        print(
            f"\n[BOT] цель выполнена: поймано {caught} из {target} рыб",
            flush=True,
        )
        self._input.release_all_keys()
        self._sm.stop()
        return True

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now

    def _save_debug_screenshot(
        self,
        tag: str,
        frame: np.ndarray | None = None,
    ) -> bool:
        total = int(getattr(self, "_screenshot_count", 0))
        per_tag = getattr(self, "_screenshot_tag_counts", {})
        tag_count = int(per_tag.get(tag, 0))
        if total >= int(getattr(self, "_screenshot_max_per_session", 120)):
            return False
        if tag_count >= int(getattr(self, "_screenshot_max_per_tag", 30)):
            return False

        regions = self._debug_screenshot_regions(tag)
        if frame is None:
            frame = self._grab(self._bounding_region(regions))
        else:
            frame = self._crop_debug_frame(frame, regions)
        if frame is None or frame.size == 0:
            return False

        telemetry = getattr(self, "_telemetry", None)
        session_id = str(
            getattr(
                telemetry,
                "session_id",
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        )
        directory = LOGS_DIR / "fishing_screenshots" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        extension = str(getattr(self, "_screenshot_format", "jpg"))
        if extension == "jpeg":
            extension = "jpg"
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        path = directory / f"{tag}_{tag_count + 1:03d}_{ts}.{extension}"
        quality = int(getattr(self, "_screenshot_quality", 82))
        if extension == "webp":
            options = [cv2.IMWRITE_WEBP_QUALITY, quality]
        else:
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]
        if not cv2.imwrite(str(path), frame, options):
            return False

        self._screenshot_count = total + 1
        per_tag[tag] = tag_count + 1
        self._screenshot_tag_counts = per_tag
        self._event(
            "debug_screenshot_saved",
            tag=tag,
            path=str(path),
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
        )
        return True

    def _debug_screenshot_regions(self, tag: str) -> list[list[int]]:
        if tag in ("cast_start_retry_limit", "reel_modal_missing"):
            width, height = self._config.get("screen", {}).get(
                "resolution", [1920, 1080]
            )
            return [[0, 0, int(width), int(height)]]
        if tag.startswith("bite"):
            return [
                self._bite_text_region,
                self._bobber_region,
                self._hook_prompt_region,
            ]
        if tag.startswith("reel"):
            return [
                self._fight_bar_region,
                self._fish_icon_region,
                self._modal_region,
            ]
        return [self._timing_region]

    @staticmethod
    def _bounding_region(regions: list[list[int]]) -> list[int]:
        x1 = min(region[0] for region in regions)
        y1 = min(region[1] for region in regions)
        x2 = max(region[0] + region[2] for region in regions)
        y2 = max(region[1] + region[3] for region in regions)
        return [x1, y1, x2 - x1, y2 - y1]

    def _crop_debug_frame(
        self,
        frame: np.ndarray,
        regions: list[list[int]],
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        resolution = self._config.get("screen", {}).get(
            "resolution", [1920, 1080]
        )
        scaled = [
            scale_region(region, resolution, width, height)
            for region in regions
        ]
        x, y, w, h = self._bounding_region(scaled)
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        return frame[y : y + h, x : x + w]

    def _save_timing_debug(self) -> None:
        self._save_timing_debug_frame(self._grab_timing())

    def _save_timing_debug_frame(self, roi: np.ndarray) -> None:
        if roi.size == 0:
            return
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        dbg = self._timing.draw_debug(roi)
        cv2.imwrite(str(LOGS_DIR / "timing_bar_live.png"), dbg)

    def _put_label(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        detail: str,
        color: tuple[int, int, int],
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, title, (x + 2, max(y + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        if detail:
            cv2.putText(
                frame,
                detail,
                (x + 2, min(y + h - 4, frame.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
            )

    def _show_debug_overlay(self) -> None:
        raw = self._capture.grab()
        frame = np.ascontiguousarray(raw.copy())
        fh, fw = frame.shape[:2]

        # --- Этап 1: полоса тайминга ---
        tx, ty, tw, th = self._capture.scaled_region(self._timing_region)
        timing_crop = self._crop(raw, self._timing_region)
        # Fresh probe — the overlay must never touch the session's zone lock.
        ta = self._timing.analyze_fresh(timing_crop)
        if ta.active and tw > 0 and th > 0:
            gl, gr = ta.green_left, ta.green_right
            zone_color = (0, 255, 255) if ta.in_zone else (0, 255, 0)
            cv2.rectangle(frame, (tx + gl, ty), (tx + gr, ty + th), zone_color, 2)
            if ta.slider_found:
                sx = ta.slider_x
                cv2.line(frame, (tx + sx, ty), (tx + sx, ty + th), (255, 255, 255), 2)
                timing_detail = f"slider={sx} zone={gl}-{gr} in_zone={ta.in_zone}"
            else:
                timing_detail = f"zone={gl}-{gr} no slider"
        else:
            timing_detail = "no green zone"
        self._put_label(
            frame, tx, ty, tw, th,
            "1 TIMING (Space)",
            timing_detail,
            (0, 255, 0),
        )

        # --- Этап 2: текст «Клюет!» ---
        bx, by, bw, bh = self._capture.scaled_region(self._bite_text_region)
        bite_crop = self._crop(raw, self._bite_text_region)
        scores = self._prompt.text_scores(bite_crop)
        txt_sc, kw_sc, wait_sc = scores.bite, scores.keyword, scores.wait
        need_delta = self._baseline_text + self._text_delta_min
        bite_detail = (
            f"text={txt_sc:.2f} kw={kw_sc:.2f} wait={wait_sc:.2f} "
            f"need>={self._bite_absolute_threshold:.2f} or >={need_delta:.2f}"
        )
        self._put_label(frame, bx, by, bw, bh, "2 BITE TEXT (Space)", bite_detail, (255, 0, 255))

        # --- Прочие ROI ---
        for name, region, color in [
            ("bobber", self._bobber_region, (0, 255, 255)),
            ("hook UI", self._hook_prompt_region, (0, 165, 255)),
            ("fish", self._fish_icon_region, (255, 255, 0)),
            ("modal", self._modal_region, (255, 128, 0)),
        ]:
            x, y, w, h = self._capture.scaled_region(region)
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(frame, name, (x, max(y - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        phase = self._phase_det.phase_label(self._detect_phase(raw))
        hook_sc, _, bob_sc = self._bite_signals(raw)
        legend = [
            "LEGEND: green=zone yellow=center white=slider",
            f"{self._sm.state_label()} | {phase} | block={self._last_bite_block_reason}",
        ]
        for i, line in enumerate(legend):
            cv2.putText(frame, line, (10, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"hook={hook_sc:.2f} bob={bob_sc:.2f} | win {fw}x{fh}",
            (10, fh - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        live_path = LOGS_DIR / "live_debug_fishing.png"
        cv2.imwrite(str(live_path), frame)

        if timing_crop.size > 0:
            cv2.imwrite(str(LOGS_DIR / "timing_bar_live.png"), self._timing.draw_debug(timing_crop))
        if bite_crop.size > 0:
            bite_dbg = bite_crop.copy()
            cv2.putText(
                bite_dbg,
                f"text={txt_sc:.2f} kw={kw_sc:.2f}",
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            cv2.imwrite(str(LOGS_DIR / "bite_text_live.png"), bite_dbg)

        if self._debug_live_window:
            cv2.imshow("Fishing Bot Detection", frame)
            cv2.waitKey(1)

    def cleanup(self) -> None:
        try:
            self._input.release_all_keys()
        finally:
            telemetry = getattr(self, "_telemetry", None)
            if (
                telemetry is not None
                and not getattr(self, "_telemetry_finalized", False)
            ):
                self._telemetry_finalized = True
                summary = telemetry.finish(self._stats)
                if summary is not None:
                    print(
                        f"\n[BOT] отчёт сессии: {telemetry.summary_path}",
                        flush=True,
                    )
                    for conclusion in summary.get("conclusions", []):
                        print(
                            f"[АНАЛИЗ] {conclusion['title']}: "
                            f"{conclusion['recommendation']}",
                            flush=True,
                        )
