"""Minimap navigation — follow walkable paths, not straight lines to blip."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import cv2
import numpy as np

from bot.input_handler import InputHandler


@dataclass
class BlipTarget:
    x: int
    y: int
    area: float
    distance: float
    angle_deg: float


@dataclass
class SteerTarget:
    x: int
    y: int
    distance: float
    angle_deg: float
    on_path: bool


class RecoveryState(Enum):
    NONE = auto()
    BACKUP = auto()
    TURN = auto()


class MinimapNavigator:
    def __init__(self, config: dict[str, Any], input_handler: InputHandler | None = None) -> None:
        self._config = config
        self._input = input_handler
        regions = config.get("regions", {})
        self._region = list(regions.get("minimap", [0, 780, 300, 300]))

        nav_cfg = config.get("navigation", {})
        center = nav_cfg.get("player_center", None)
        if center:
            self._center = tuple(center)
        else:
            self._center = (self._region[2] // 2, int(self._region[3] * 0.55))

        self._arrival_distance = nav_cfg.get("arrival_distance_px", 22)
        self._turn_threshold_deg = nav_cfg.get("turn_threshold_deg", 12)
        self._turn_only_deg = nav_cfg.get("turn_only_deg", 32)
        self._mouse_turn_deg = nav_cfg.get("mouse_turn_deg", 50)
        self._steer_mode = nav_cfg.get("steer_mode", "mouse").lower()
        self._stop_walk_angle = nav_cfg.get("stop_walk_angle_deg", 75)
        self._max_mouse_step = nav_cfg.get("max_mouse_step_px", 120)
        self._walk_keys = nav_cfg.get("walk_forward_key", "w")
        self._turn_left_key = nav_cfg.get("turn_left_key", "a")
        self._turn_right_key = nav_cfg.get("turn_right_key", "d")
        self._use_shift = nav_cfg.get("sprint", True)
        self._sprint_min_angle = nav_cfg.get("sprint_max_angle_deg", 14)
        self._stuck_timeout_sec = nav_cfg.get("stuck_timeout_sec", 2.8)
        self._stuck_distance_px = nav_cfg.get("stuck_distance_gain_px", 4)
        self._recovery_backup_ms = nav_cfg.get("recovery_backup_ms", 450)
        self._recovery_turn_ms = nav_cfg.get("recovery_turn_ms", 550)
        self._mouse_pixels_per_deg = nav_cfg.get("mouse_pixels_per_deg", 9)
        self._path_lookahead = nav_cfg.get("path_lookahead_px", 48)
        self._path_gray_min = nav_cfg.get("path_gray_min", 72)
        self._path_gray_max = nav_cfg.get("path_gray_max", 210)
        self._path_scan_deg = nav_cfg.get("path_scan_half_angle_deg", 55)

        blip_hsv = nav_cfg.get("blip_hsv", {})
        self._blip_low = np.array(blip_hsv.get("low", [18, 120, 160]))
        self._blip_high = np.array(blip_hsv.get("high", [38, 255, 255]))
        self._min_blip_area = nav_cfg.get("min_blip_area", 8)
        self._max_blip_area = nav_cfg.get("max_blip_area", 500)
        self._exclude_green = nav_cfg.get("exclude_green_icons", True)

        self._active_keys: set[str] = set()
        self._last_distance = 9999.0
        self._last_progress_time = time.perf_counter()
        self._recovery = RecoveryState.NONE
        self._recovery_until = 0.0
        self._last_steer: SteerTarget | None = None
        self._last_mouse_turn = 0.0
        self._mouse_turn_interval = nav_cfg.get("mouse_turn_interval_ms", 45) / 1000.0

    @property
    def region(self) -> list[int]:
        return self._region

    def stop(self) -> None:
        if self._input:
            for key in list(self._active_keys):
                self._input.key_up(key)
            if self._use_shift:
                self._input.key_up("shift")
        self._active_keys.clear()
        self._last_distance = 9999.0
        self._recovery = RecoveryState.NONE

    def find_target_blip(self, frame: np.ndarray) -> BlipTarget | None:
        if frame.size == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blip_low, self._blip_high)
        if self._exclude_green:
            green = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
            mask &= ~green

        mask = self._apply_circle_mask(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = self._center
        best: BlipTarget | None = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_blip_area or area > self._max_blip_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            bx = int(moments["m10"] / moments["m00"])
            by = int(moments["m01"] / moments["m00"])
            dx, dy = bx - cx, cy - by
            distance = math.hypot(dx, dy)
            if distance < 8:
                continue
            angle = math.degrees(math.atan2(dx, dy))
            candidate = BlipTarget(bx, by, area, distance, angle)
            if best is None or distance < best.distance:
                best = candidate

        return best

    def _extract_path_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        path = cv2.inRange(gray, self._path_gray_min, self._path_gray_max)
        path = self._apply_circle_mask(path)
        path = cv2.morphologyEx(path, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return path

    def _steer_target(self, frame: np.ndarray, blip: BlipTarget) -> SteerTarget:
        cx, cy = self._center
        path_mask = self._extract_path_mask(frame)
        target_angle = blip.angle_deg
        best: SteerTarget | None = None
        best_score = float("inf")

        angles: list[float] = []
        for delta in range(0, self._path_scan_deg + 1, 5):
            if delta == 0:
                angles.append(target_angle)
            else:
                angles.extend([target_angle + delta, target_angle - delta])

        for angle in angles:
            rad = math.radians(angle)
            for dist in (
                self._path_lookahead,
                self._path_lookahead - 12,
                self._path_lookahead + 12,
                self._path_lookahead - 24,
            ):
                if dist <= 10:
                    continue
                px = int(cx + math.sin(rad) * dist)
                py = int(cy - math.cos(rad) * dist)
                if not self._point_on_path(path_mask, px, py):
                    continue
                dx, dy = px - cx, cy - py
                distance = math.hypot(dx, dy)
                steer_angle = math.degrees(math.atan2(dx, dy))
                score = abs(steer_angle - target_angle) + abs(dist - self._path_lookahead) * 0.15
                if score >= best_score:
                    continue
                best = SteerTarget(px, py, distance, steer_angle, True)
                best_score = score

        if best is not None:
            return best

        return SteerTarget(blip.x, blip.y, blip.distance, blip.angle_deg, False)

    @staticmethod
    def _point_on_path(path_mask: np.ndarray, x: int, y: int, radius: int = 4) -> bool:
        h, w = path_mask.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        patch = path_mask[y0:y1, x0:x1]
        return patch.size > 0 and np.count_nonzero(patch) >= max(3, patch.size // 6)

    def has_arrived(self, frame: np.ndarray) -> bool:
        blip = self.find_target_blip(frame)
        if blip is None:
            return True
        return blip.distance <= self._arrival_distance

    def tick(self, frame: np.ndarray) -> bool:
        if self._input is None:
            blip = self.find_target_blip(frame)
            return blip is None or blip.distance <= self._arrival_distance

        now = time.perf_counter()
        if self._recovery != RecoveryState.NONE and now < self._recovery_until:
            return False
        if self._recovery != RecoveryState.NONE and now >= self._recovery_until:
            self._recovery = RecoveryState.NONE
            self._last_progress_time = now

        blip = self.find_target_blip(frame)
        if blip is None:
            self.stop()
            return True

        if blip.distance <= self._arrival_distance:
            self.stop()
            return True

        steer = self._steer_target(frame, blip)
        self._last_steer = steer
        self._track_stuck(blip)

        if self._is_stuck():
            self._start_recovery(steer)
            return False

        self._apply_steering(steer)
        return False

    def _apply_steering(self, steer: SteerTarget) -> None:
        if self._steer_mode == "mouse":
            self._apply_mouse_steering(steer)
        else:
            self._apply_keyboard_steering(steer)

    def _apply_mouse_steering(self, steer: SteerTarget) -> None:
        """Hold W and steer with mouse — same as manual play in the reference video."""
        angle = steer.angle_deg
        abs_angle = abs(angle)

        self._set_key(self._turn_left_key, False)
        self._set_key(self._turn_right_key, False)

        if abs_angle > self._stop_walk_angle:
            self._set_forward(False)
        else:
            sprint = abs_angle <= self._sprint_min_angle
            self._set_forward(True, sprint=sprint)

        if self._input is not None:
            now = time.perf_counter()
            if now - self._last_mouse_turn >= self._mouse_turn_interval:
                gain = self._mouse_pixels_per_deg
                if abs_angle > 45:
                    gain *= 1.35
                dx = int(max(-self._max_mouse_step, min(self._max_mouse_step, angle * gain)))
                if dx != 0:
                    self._input.turn_camera(dx)
                    self._last_mouse_turn = now

    def _apply_keyboard_steering(self, steer: SteerTarget) -> None:
        angle = steer.angle_deg
        abs_angle = abs(angle)

        if abs_angle >= self._mouse_turn_deg and self._input is not None:
            self._set_forward(False)
            self._set_key(self._turn_left_key, False)
            self._set_key(self._turn_right_key, False)
            now = time.perf_counter()
            if now - self._last_mouse_turn >= self._mouse_turn_interval:
                dx = int(
                    max(
                        -self._max_mouse_step,
                        min(self._max_mouse_step, angle * self._mouse_pixels_per_deg),
                    )
                )
                self._input.turn_camera(dx)
                self._last_mouse_turn = now
            return

        if abs_angle > self._turn_only_deg:
            self._set_forward(False)
            if angle < 0:
                self._set_key(self._turn_left_key, True)
                self._set_key(self._turn_right_key, False)
            else:
                self._set_key(self._turn_right_key, True)
                self._set_key(self._turn_left_key, False)
            return

        if abs_angle > self._turn_threshold_deg:
            self._set_forward(True, sprint=False)
            if angle < 0:
                self._set_key(self._turn_left_key, True)
                self._set_key(self._turn_right_key, False)
            else:
                self._set_key(self._turn_right_key, True)
                self._set_key(self._turn_left_key, False)
            return

        self._set_key(self._turn_left_key, False)
        self._set_key(self._turn_right_key, False)
        self._set_forward(True, sprint=True)

    def _set_forward(self, enabled: bool, sprint: bool = True) -> None:
        self._set_key(self._walk_keys, enabled)
        if self._use_shift:
            use_sprint = enabled and sprint
            self._set_key("shift", use_sprint)

    def _set_key(self, key: str, down: bool) -> None:
        if self._input is None:
            return
        if down:
            if key not in self._active_keys:
                self._input.key_down(key)
                self._active_keys.add(key)
        elif key in self._active_keys:
            self._input.key_up(key)
            self._active_keys.discard(key)

    def _track_stuck(self, blip: BlipTarget) -> None:
        now = time.perf_counter()
        gained = self._last_distance - blip.distance
        walking = self._walk_keys in self._active_keys

        if gained >= self._stuck_distance_px or (walking and gained > 0.5):
            self._last_progress_time = now
            self._last_distance = blip.distance
        elif self._last_distance == 9999.0:
            self._last_distance = blip.distance
            self._last_progress_time = now

    def _is_stuck(self) -> bool:
        walking = self._walk_keys in self._active_keys
        if not walking:
            return False
        return time.perf_counter() - self._last_progress_time >= self._stuck_timeout_sec

    def _start_recovery(self, steer: SteerTarget) -> None:
        if self._input is None:
            return
        self.stop()
        self._input.tap_key("s", self._recovery_backup_ms)
        dx = int(
            max(
                -self._max_mouse_step,
                min(self._max_mouse_step, steer.angle_deg * self._mouse_pixels_per_deg * 1.6),
            )
        )
        self._input.turn_camera(dx)
        if self._steer_mode != "mouse":
            turn_key = self._turn_left_key if steer.angle_deg < 0 else self._turn_right_key
            self._input.tap_key(turn_key, self._recovery_turn_ms)
        self._recovery = RecoveryState.TURN
        self._recovery_until = time.perf_counter() + (
            (self._recovery_backup_ms + self._recovery_turn_ms) / 1000.0 + 0.15
        )
        self._last_progress_time = time.perf_counter()
        self._last_distance = 9999.0

    def _apply_circle_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        cx, cy = self._center
        radius = int(min(w, h) * 0.46)
        circle = np.zeros_like(mask)
        cv2.circle(circle, (cx, cy), radius, 255, -1)
        return cv2.bitwise_and(mask, circle)

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        debug = frame.copy()
        cx, cy = self._center
        path_mask = self._extract_path_mask(frame)
        debug[path_mask > 0] = (debug[path_mask > 0] * 0.5 + np.array((60, 60, 60))).astype(
            np.uint8
        )

        cv2.circle(debug, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(debug, (cx, cy), self._arrival_distance, (80, 80, 80), 1)

        blip = self.find_target_blip(frame)
        if blip:
            cv2.circle(debug, (blip.x, blip.y), 5, (0, 200, 255), -1)
            steer = self._steer_target(frame, blip)
            color = (0, 255, 120) if steer.on_path else (0, 120, 255)
            cv2.circle(debug, (steer.x, steer.y), 4, color, -1)
            cv2.line(debug, (cx, cy), (steer.x, steer.y), color, 1)
            cv2.line(debug, (cx, cy), (blip.x, blip.y), (60, 60, 60), 1)
            cv2.putText(
                debug,
                f"d={blip.distance:.0f} a={steer.angle_deg:.0f} path={steer.on_path}",
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        return debug
