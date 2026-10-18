"""Miner job logic — orchestrates mining loop."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot.capture import GameCapture
from bot.config_loader import LOGS_DIR, ROUTES_DIR
from bot.detectors.carry import CarryDetector
from bot.detectors.mine_phase import MinePhase, MinePhaseDetector
from bot.detectors.minigame import MinigameDetector
from bot.detectors.notification import NotificationDetector
from bot.input_handler import InputHandler
from bot.navigation.minimap_nav import MinimapNavigator
from bot.navigation.route_player import RoutePlayer
from bot.state_machine import BotState, StateMachine


class MinerJob:
    def __init__(
        self,
        config: dict[str, Any],
        capture: GameCapture,
        input_handler: InputHandler,
        state_machine: StateMachine,
    ) -> None:
        self._config = config
        self._capture = capture
        self._input = input_handler
        self._sm = state_machine

        regions = config.get("regions", {})
        self._notif_region = regions.get("notification", [20, 20, 400, 120])
        self._minigame_region = regions.get("minigame", [520, 340, 880, 420])
        self._progress_region = regions.get("minigame_progress", [700, 680, 520, 60])
        self._rocks_region = regions.get("rocks_mask", [580, 380, 760, 280])
        self._carry_region = regions.get("carry_indicator", [20, 140, 400, 80])
        self._minimap_region = regions.get("minimap", [0, 780, 300, 300])
        self._config_resolution = config.get("screen", {}).get("resolution", [1920, 1080])

        nav_cfg = config.get("navigation", {})
        self._navigation_enabled = nav_cfg.get("enabled", False)
        self._nav_mode = nav_cfg.get("mode", "routes").lower()
        route_names = nav_cfg.get("routes", {})
        self._route_name_to_rock = route_names.get("to_rock", "fork_tunnel2")
        self._route_name_to_drop = route_names.get("to_drop", "mine_to_drop")
        self._route_name_after_drop = route_names.get("after_drop", "drop_to_fork")

        debug = config.get("debug", {})
        self._debug = debug.get("enabled", False)
        self._save_screenshots = debug.get("save_screenshots", False)
        self._show_overlay = debug.get("show_overlay", False)

        routes_cfg = config.get("routes", {})
        self._routes_dir = Path(routes_cfg.get("directory", str(ROUTES_DIR)))
        if not self._routes_dir.is_absolute():
            from bot.config_loader import ROOT_DIR

            self._routes_dir = ROOT_DIR / self._routes_dir

        self._minigame = MinigameDetector(config)
        self._notification = NotificationDetector(config)
        self._carry = CarryDetector(config)
        self._route_player = RoutePlayer(input_handler)
        self._minimap = MinimapNavigator(config, input_handler)
        self._phase_detector = MinePhaseDetector(config)

        self._route_to_drop = self._routes_dir / f"{self._route_name_to_drop}.yaml"
        self._route_back = self._routes_dir / f"{self._route_name_after_drop}.yaml"
        self._route_to_rock = self._routes_dir / f"{self._route_name_to_rock}.yaml"

        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0
        self._stats = {
            "cycles": 0,
            "minigames": 0,
            "e_presses": 0,
            "veins_clicked": 0,
            "nav_steps": 0,
        }
        self._mode = config.get("mode", "minigame_only")
        self._e_intent = "mine"
        self._debug_frame_counter = 0
        self._sync_minigame_scale()

    def _sync_minigame_scale(self) -> None:
        rect = self._capture.rect
        self._minigame.sync_window_scale(rect.width, rect.height)

    @property
    def stats(self) -> dict[str, int]:
        return self._stats

    @property
    def fps(self) -> float:
        return self._fps

    def tick(self) -> None:
        if not self._sm.active:
            return

        if not self._sm.running:
            self._input.release_all_keys()
            if self._route_player.playing:
                self._route_player.stop()
            self._minimap.stop()
            return

        self._update_fps()
        self._sync_minigame_scale()

        if self._sm.state == BotState.WAITING:
            self._tick_waiting()
        elif self._sm.state == BotState.NAVIGATE:
            self._tick_navigate()
        elif self._sm.state == BotState.ROUTE_TO_ROCK:
            self._tick_route_playback("rock")
        elif self._sm.state == BotState.PRESS_E:
            self._tick_press_e()
        elif self._sm.state == BotState.MINING:
            self._tick_mining()
        elif self._sm.state == BotState.MINIGAME:
            self._tick_minigame()
        elif self._sm.state == BotState.CARRY:
            self._tick_carry()
        elif self._sm.state == BotState.ROUTE_TO_DROP:
            self._tick_route_playback("drop")
        elif self._sm.state == BotState.ROUTE_BACK:
            self._tick_route_playback("back")

        if self._show_overlay and self._debug:
            self._show_debug_overlay()

    def _grab_notif(self):
        return self._capture.grab_region(self._notif_region)

    def _grab_minigame(self):
        return self._capture.grab_region(self._minigame_region)

    def _grab_progress(self):
        return self._capture.grab_region(self._progress_region)

    def _grab_carry(self):
        return self._capture.grab_region(self._carry_region)

    def _grab_minimap(self):
        return self._capture.grab_region(self._minimap_region)

    def _uses_navigation(self) -> bool:
        return self._navigation_enabled and self._mode in ("full", "walk_minigame")

    def _uses_routes(self) -> bool:
        return self._uses_navigation() and self._nav_mode == "routes"

    def _uses_minimap(self) -> bool:
        return self._uses_navigation() and self._nav_mode == "minimap"

    def _route_path(self, name: str) -> Path:
        return self._routes_dir / f"{name}.yaml"

    def _start_route_file(self, path: Path) -> bool:
        if not path.exists():
            if self._debug:
                print(f"\n[BOT] route missing: {path}", flush=True)
            return False
        if not self._route_player.load(path):
            return False
        self._route_player.start()
        if self._debug:
            print(f"\n[BOT] playing route: {path.stem}", flush=True)
        return True

    def _detect_phase(self) -> MinePhase:
        return self._phase_detector.detect(
            self._capture.grab(),
            self._capture.grab_region,
        )

    def _tick_waiting(self) -> None:
        phase = self._detect_phase()

        if phase == MinePhase.MINIGAME:
            self._sm.state = BotState.MINIGAME
            return

        if self._mode == "minigame_only":
            return

        if phase == MinePhase.CARRY:
            self._sm.state = BotState.CARRY
            return

        if phase == MinePhase.AT_DROP:
            if self._mode == "full" and self._notification.should_press_e():
                self._e_intent = "drop"
                self._sm.state = BotState.PRESS_E
            elif self._uses_navigation():
                self._sm.state = BotState.NAVIGATE
            return

        if phase == MinePhase.AT_ROCK:
            if self._mode == "full" and self._notification.should_press_e():
                self._e_intent = "mine"
                self._sm.state = BotState.PRESS_E
            else:
                self._sm.state = BotState.MINING
            return

        if phase == MinePhase.MINING:
            self._sm.state = BotState.MINING
            return

        if self._uses_routes() and self._should_start_rock_route(phase):
            if self._start_route_file(self._route_to_rock):
                self._sm.state = BotState.ROUTE_TO_ROCK
            return

        if self._uses_minimap() and phase == MinePhase.NAVIGATE:
            self._sm.state = BotState.NAVIGATE
            return

        if self._mode == "full":
            notif_frame = self._grab_notif()
            if self._notification.is_visible(notif_frame) and self._notification.should_press_e():
                self._sm.state = BotState.PRESS_E
                return

            if self._carry.is_carrying(self._grab_carry()):
                self._sm.state = BotState.CARRY
                return

            self._input.spam_lmb_if_ready()

    def _should_start_rock_route(self, phase: MinePhase) -> bool:
        if phase in (MinePhase.AT_ROCK, MinePhase.MINING, MinePhase.MINIGAME, MinePhase.CARRY):
            return False
        if phase == MinePhase.AT_DROP:
            return False
        return self._route_to_rock.exists()

    def _tick_route_playback(self, purpose: str) -> None:
        phase = self._detect_phase()

        if phase == MinePhase.MINIGAME:
            self._route_player.stop()
            self._sm.state = BotState.MINIGAME
            return

        if purpose == "rock" and phase in (MinePhase.AT_ROCK, MinePhase.MINING):
            self._route_player.stop()
            if (
                phase == MinePhase.AT_ROCK
                and self._mode == "full"
                and self._notification.should_press_e()
            ):
                self._e_intent = "mine"
                self._sm.state = BotState.PRESS_E
            else:
                self._sm.state = BotState.MINING
            return

        if purpose in ("drop", "back") and phase == MinePhase.AT_DROP:
            self._route_player.stop()
            if self._mode == "full" and self._notification.should_press_e():
                self._e_intent = "drop"
                self._sm.state = BotState.PRESS_E
            else:
                self._sm.state = BotState.WAITING
            return

        if purpose == "drop" and phase == MinePhase.CARRY:
            pass
        elif phase == MinePhase.CARRY and purpose == "rock":
            self._route_player.stop()
            self._sm.state = BotState.CARRY
            return

        finished = self._route_player.tick()
        if not finished:
            return

        if purpose == "rock":
            notif = self._grab_notif()
            if self._notification.is_mine_visible(notif) and self._mode == "full":
                self._e_intent = "mine"
                self._sm.state = BotState.PRESS_E
            else:
                self._sm.state = BotState.MINING if self._mode == "full" else BotState.WAITING
        elif purpose == "drop":
            if self._mode == "full":
                self._input.press_e()
                time.sleep(0.4)
            if self._route_back.exists():
                self._start_route_file(self._route_back)
                self._sm.state = BotState.ROUTE_BACK
            else:
                self._sm.state = BotState.WAITING
        else:
            if self._route_to_rock.exists():
                self._start_route_file(self._route_to_rock)
                self._sm.state = BotState.ROUTE_TO_ROCK
            else:
                self._sm.state = BotState.WAITING

    def _tick_navigate(self) -> None:
        phase = self._detect_phase()

        if phase == MinePhase.MINIGAME:
            self._minimap.stop()
            self._sm.state = BotState.MINIGAME
            return

        if phase == MinePhase.AT_DROP:
            self._minimap.stop()
            if self._mode == "full" and self._notification.should_press_e():
                self._e_intent = "drop"
                self._sm.state = BotState.PRESS_E
            else:
                self._sm.state = BotState.WAITING
            return

        if phase == MinePhase.AT_ROCK:
            self._minimap.stop()
            if self._mode == "full" and self._notification.should_press_e():
                self._e_intent = "mine"
                self._sm.state = BotState.PRESS_E
            else:
                self._sm.state = BotState.MINING
            return

        if phase == MinePhase.MINING:
            self._minimap.stop()
            self._sm.state = BotState.MINING
            return

        if phase == MinePhase.CARRY:
            self._minimap.stop()
            self._sm.state = BotState.CARRY
            return

        frame = self._grab_minimap()
        arrived = self._minimap.tick(frame)
        self._stats["nav_steps"] += 1

        if arrived:
            self._minimap.stop()
            if self._debug:
                print("\n[BOT] arrived at minimap blip", flush=True)
            notif = self._grab_notif()
            if self._notification.is_drop_visible(notif):
                if self._mode == "full" and self._notification.should_press_e():
                    self._e_intent = "drop"
                    self._sm.state = BotState.PRESS_E
                else:
                    self._sm.state = BotState.WAITING
            elif phase == MinePhase.CARRY or self._carry.is_carrying(self._grab_carry()):
                self._sm.state = BotState.CARRY
            elif self._notification.is_mine_visible(notif):
                if self._mode == "full" and self._notification.should_press_e():
                    self._e_intent = "mine"
                    self._sm.state = BotState.PRESS_E
                else:
                    self._sm.state = BotState.WAITING
            elif self._mode == "full":
                self._sm.state = BotState.MINING
            else:
                self._sm.state = BotState.WAITING

    def _tick_press_e(self) -> None:
        self._input.press_e()
        self._notification.mark_pressed()
        self._stats["e_presses"] += 1
        if self._debug:
            print(f"\n[BOT] pressed E ({self._e_intent})", flush=True)
        time.sleep(0.3)
        if self._e_intent == "drop":
            if self._uses_routes() and self._route_back.exists():
                self._start_route_file(self._route_back)
                self._sm.state = BotState.ROUTE_BACK
            else:
                self._sm.state = BotState.WAITING
        else:
            self._sm.state = BotState.MINING
        self._e_intent = "mine"

    def _tick_mining(self) -> None:
        minigame_frame = self._grab_minigame()
        progress_frame = self._grab_progress()

        if self._minigame.is_active(minigame_frame, progress_frame):
            self._input.reset_lmb_timer()
            self._sm.state = BotState.MINIGAME
            return

        if self._carry.is_carrying(self._grab_carry()):
            self._sm.state = BotState.CARRY
            return

        self._input.spam_lmb_if_ready()

    def _tick_minigame(self) -> None:
        region = self._minigame_region

        def get_frame():
            return self._capture.grab_region(region)

        def get_progress_frame():
            return self._capture.grab_region(self._progress_region)

        def to_screen(x, y):
            return self._capture.to_screen_coords(x, y, region)

        def click(sx, sy):
            if self._debug:
                print(f"\n[BOT] click vein at screen ({sx}, {sy})", flush=True)
            self._input.click_at(sx, sy)
            self._stats["veins_clicked"] += 1

        def should_abort():
            return not self._sm.running

        cleared = self._minigame.solve(
            get_frame,
            to_screen,
            click,
            get_progress_frame,
            should_abort=should_abort,
        )
        vein_count = len(self._minigame.find_veins(self._grab_minigame()))
        if self._debug:
            print(
                f"\n[BOT] minigame done: cleared={cleared}, veins_visible={vein_count}", flush=True
            )
        self._stats["minigames"] += 1

        if cleared:
            self._stats["cycles"] += 1

        if not cleared and self._save_screenshots:
            self._save_debug_screenshot("minigame_timeout")

        if self._uses_routes():
            if self._route_to_drop.exists():
                self._start_route_file(self._route_to_drop)
                self._sm.state = BotState.ROUTE_TO_DROP
            else:
                self._sm.state = BotState.WAITING
        elif self._uses_minimap():
            phase = self._detect_phase()
            if phase == MinePhase.CARRY:
                self._sm.state = BotState.NAVIGATE
            elif phase == MinePhase.NAVIGATE:
                self._sm.state = BotState.NAVIGATE
            else:
                self._sm.state = BotState.WAITING
        else:
            self._sm.state = BotState.WAITING

    def _tick_carry(self) -> None:
        phase = self._detect_phase()

        if phase == MinePhase.AT_DROP:
            if self._mode == "full" and self._notification.should_press_e():
                self._minimap.stop()
                self._e_intent = "drop"
                self._sm.state = BotState.PRESS_E
            return

        if self._uses_routes():
            if self._route_to_drop.exists():
                self._start_route_file(self._route_to_drop)
                self._sm.state = BotState.ROUTE_TO_DROP
            return

        if self._uses_minimap() and phase in (MinePhase.NAVIGATE, MinePhase.CARRY):
            self._sm.state = BotState.NAVIGATE
            return

        if not self._carry.is_carrying(self._grab_carry()) and phase != MinePhase.CARRY:
            self._sm.state = BotState.WAITING
            return

        notif_frame = self._grab_notif()
        if self._notification.is_drop_visible(notif_frame) and self._notification.should_press_e():
            self._e_intent = "drop"
            self._sm.state = BotState.PRESS_E

    def _tick_route(self, drop_off: bool) -> None:
        """Legacy helper — use _tick_route_playback instead."""
        self._tick_route_playback("drop" if drop_off else "back")

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now

    def _save_debug_screenshot(self, tag: str) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        frame = self._capture.grab()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOGS_DIR / f"{tag}_{ts}.png"
        cv2.imwrite(str(path), frame)

    def _show_debug_overlay(self) -> None:
        """Save debug frames to disk only — cv2.imshow captures itself and causes recursion."""
        self._debug_frame_counter += 1
        if self._debug_frame_counter % 15 != 0:
            return

        frame = np.ascontiguousarray(self._capture.grab().copy())
        minigame_roi = self._grab_minigame()
        progress_roi = self._grab_progress()

        region_overlays = [
            ("minigame", self._minigame_region, (0, 255, 0)),
            ("rocks", self._rocks_region, (0, 255, 255)),
            ("progress", self._progress_region, (255, 0, 255)),
        ]
        if self._uses_navigation() or self._mode != "minigame_only":
            region_overlays.append(("minimap", self._minimap_region, (255, 255, 0)))
        if self._mode != "minigame_only":
            region_overlays.extend(
                [
                    ("notif", self._notif_region, (255, 0, 0)),
                    ("carry", self._carry_region, (0, 165, 255)),
                ]
            )

        for name, region, color in region_overlays:
            scaled = self._capture.scaled_region(region)
            x, y, w, h = scaled
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, name, (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        debug_mini = np.ascontiguousarray(self._minigame.draw_debug(minigame_roi, progress_roi))
        mg_scaled = self._capture.scaled_region(self._minigame_region)
        mg_x, mg_y, mg_w, mg_h = mg_scaled
        if mg_w > 0 and mg_h > 0:
            resized = cv2.resize(debug_mini, (mg_w, mg_h), interpolation=cv2.INTER_LINEAR)
            frame[mg_y : mg_y + mg_h, mg_x : mg_x + mg_w] = resized

        fill = self._minigame.get_progress_fill(progress_roi)
        vein_count = len(self._minigame.find_veins(minigame_roi))
        phase = self._phase_detector.phase_label(self._detect_phase())
        cv2.putText(
            frame,
            f"{self._sm.state_label()} | {phase} | veins: {vein_count} | progress: {fill:.0%}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = LOGS_DIR / "live_debug.png"
        cv2.imwrite(str(out_path), frame)

    def cleanup(self) -> None:
        self._minimap.stop()
        self._input.release_all_keys()
