"""Automation loop for the Majestic outdoor gym exercise."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from bot.capture import GameCapture
from bot.detectors.gym_prompt import GymPromptDetector, GymRecognition
from bot.input_handler import InputHandler
from bot.state_machine import StateMachine


class GymJob:
    def __init__(
        self,
        config: dict[str, Any],
        capture: GameCapture,
        input_handler: InputHandler,
        state_machine: StateMachine,
        before_cycle: Callable[[], None] | None = None,
    ) -> None:
        self._capture = capture
        self._input = input_handler
        self._sm = state_machine
        self._before_cycle = before_cycle
        self._detector = GymPromptDetector(config)

        gym = config.get("gym", {})
        self._region = config.get("regions", {}).get("prompt", [904, 946, 114, 114])
        self._start_key = str(gym.get("start_key", "e")).lower()
        self._start_settle = float(gym.get("start_settle_sec", 0.45))
        self._start_retry = float(gym.get("start_retry_sec", 8.0))
        self._restart_absent = float(gym.get("restart_after_absent_sec", 7.0))
        self._confirm_frames = int(gym.get("confirm_frames", 2))
        self._release_frames = int(gym.get("release_confirm_frames", 2))
        self._same_key_retry = float(gym.get("retry_same_key_sec", 0.70))
        self._reaction_delay = gym.get("reaction_delay_ms", [45, 85])
        self._key_hold = float(gym.get("key_hold_sec", 0.055))

        debug = config.get("debug", {})
        self._save_live_crop = bool(debug.get("save_live_crop", False))
        self._debug_path = Path(__file__).resolve().parents[2] / "logs" / "gym_live.png"
        self._last_debug_save = 0.0

        self._was_running = False
        self._start_retry_after = 0.0
        self._waiting_since = 0.0
        self._absent_since: float | None = None
        self._had_prompt = False
        self._candidate: str | None = None
        self._candidate_frames = 0
        self._absent_frames = 0
        self._armed = True
        self._last_pressed_key: str | None = None
        self._last_press_time = 0.0
        self._stats = {
            "cycles": 0,
            "prompts": 0,
            "w": 0,
            "d": 0,
            "s": 0,
            "a": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def tick(self) -> None:
        if not self._sm.running:
            self._reset_when_paused()
            return

        now = time.perf_counter()
        if not self._was_running:
            if now < self._start_retry_after:
                return
            self._was_running = True
            self._begin_cycle(now)
            return

        if now - self._waiting_since < self._start_settle:
            return

        roi = self._capture.grab_region(self._region)
        recognition = self._detector.detect(roi)
        self._save_debug(roi, recognition, now)

        if recognition.key is None:
            self._candidate = None
            self._candidate_frames = 0
            self._absent_frames += 1
            if self._absent_frames >= self._release_frames:
                self._armed = True
                if self._absent_since is None:
                    self._absent_since = now

            timeout = self._restart_absent if self._had_prompt else self._start_retry
            if self._absent_since is not None and now - self._absent_since >= timeout:
                self._begin_cycle(now)
            return

        self._had_prompt = True
        self._absent_since = None
        self._absent_frames = 0
        key = recognition.key
        if key == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = key
            self._candidate_frames = 1

        if self._candidate_frames < self._confirm_frames:
            return

        changed = key != self._last_pressed_key
        retry_due = now - self._last_press_time >= self._same_key_retry
        if not (self._armed or changed or retry_due):
            return

        delay_ms = random.uniform(float(self._reaction_delay[0]), float(self._reaction_delay[1]))
        time.sleep(delay_ms / 1000.0)
        if not self._sm.running:
            return
        if self._input.press_key(key, self._key_hold) is False:
            return
        self._stats["prompts"] += 1
        self._stats[key] = self._stats.get(key, 0) + 1
        self._last_pressed_key = key
        self._last_press_time = time.perf_counter()
        self._armed = False
        self._candidate_frames = 0
        print(
            f"\n[BOT] Подсказка {key.upper()} "
            f"(score={recognition.score:.2f}, всего={self._stats['prompts']})",
            flush=True,
        )

    def _begin_cycle(self, now: float) -> None:
        if self._before_cycle is not None:
            self._before_cycle()
        if not self._sm.running:
            return
        if self._input.press_key(self._start_key, self._key_hold) is False:
            self._was_running = False
            self._start_retry_after = time.perf_counter() + max(self._start_settle, 0.1)
            return
        now = time.perf_counter()
        self._stats["cycles"] += 1
        self._waiting_since = now
        self._absent_since = now
        self._had_prompt = False
        self._candidate = None
        self._candidate_frames = 0
        self._absent_frames = 0
        self._armed = True
        self._last_pressed_key = None
        self._last_press_time = 0.0
        print(
            f"\n[BOT] Нажал {self._start_key.upper()}, цикл #{self._stats['cycles']}",
            flush=True,
        )

    def _reset_when_paused(self) -> None:
        if self._was_running:
            self._input.release_all_keys()
        self._was_running = False
        self._start_retry_after = 0.0
        self._candidate = None
        self._candidate_frames = 0
        self._absent_since = None

    def _save_debug(self, roi, recognition: GymRecognition, now: float) -> None:
        if not self._save_live_crop or now - self._last_debug_save < 0.25:
            return
        self._debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(self._debug_path),
            self._detector.draw_debug(roi, recognition),
        )
        self._last_debug_save = now

    def cleanup(self) -> None:
        self._input.release_all_keys()
