"""Reel minigame bar — green zone + white indicator (A/D steering)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import cv2
import numpy as np


class SteerDirection(Enum):
    LEFT = auto()  # press A
    RIGHT = auto()  # press D
    CENTER = auto()  # release both


@dataclass
class ReelAnalysis:
    active: bool
    direction: SteerDirection
    green_center_x: int
    indicator_x: int
    green_ratio: float


class ReelBarDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        reel_cfg = config.get("reel", {})
        self._green_low = np.array(reel_cfg.get("green_low", [35, 40, 60]))
        self._green_high = np.array(reel_cfg.get("green_high", [95, 255, 255]))
        self._indicator_gray_min = reel_cfg.get("indicator_gray_min", 200)
        self._steer_margin = reel_cfg.get("steer_margin_px", 12)
        self._green_ratio_min = reel_cfg.get("green_ratio_min", 0.04)
        self._inactive_frames = reel_cfg.get("inactive_frames_to_finish", 5)
        self._timeout = reel_cfg.get("timeout_sec", 45)
        self._tick_ms = reel_cfg.get("tick_ms", 16)

        self._consecutive_inactive = 0

    def reset(self) -> None:
        self._consecutive_inactive = 0

    @property
    def timeout_sec(self) -> float:
        return self._timeout

    @property
    def tick_sec(self) -> float:
        return self._tick_ms / 1000.0

    def green_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self._green_low, self._green_high)

    def analyze(self, frame: np.ndarray) -> ReelAnalysis:
        if frame.size == 0:
            return ReelAnalysis(False, SteerDirection.CENTER, 0, 0, 0.0)

        green = self.green_mask(frame)
        green_ratio = float(np.count_nonzero(green)) / green.size
        if green_ratio < self._green_ratio_min:
            return ReelAnalysis(False, SteerDirection.CENTER, 0, 0, green_ratio)

        cols = np.where(np.any(green, axis=0))[0]
        if len(cols) < 5:
            return ReelAnalysis(False, SteerDirection.CENTER, 0, 0, green_ratio)

        green_left, green_right = int(cols[0]), int(cols[-1])
        green_center = (green_left + green_right) // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = gray >= self._indicator_gray_min
        indicator_mask = bright & (green == 0)

        indicator_cols = np.where(np.any(indicator_mask, axis=0))[0]
        if len(indicator_cols) < 2:
            # Fallback: brightest column outside green core
            row_means = np.mean(gray, axis=0)
            masked = row_means.copy()
            masked[green_left : green_right + 1] = 0
            if masked.max() < self._indicator_gray_min - 30:
                return ReelAnalysis(
                    True, SteerDirection.CENTER, green_center, green_center, green_ratio
                )
            indicator_x = int(np.argmax(masked))
        else:
            indicator_x = int(np.mean(indicator_cols))

        offset = indicator_x - green_center
        if offset < -self._steer_margin:
            direction = SteerDirection.RIGHT
        elif offset > self._steer_margin:
            direction = SteerDirection.LEFT
        else:
            direction = SteerDirection.CENTER

        return ReelAnalysis(True, direction, green_center, indicator_x, green_ratio)

    def is_active(self, frame: np.ndarray) -> bool:
        return self.analyze(frame).active

    def track_completion(self, frame: np.ndarray) -> bool:
        """Return True when reel bar disappeared long enough."""
        if self.is_active(frame):
            self._consecutive_inactive = 0
            return False
        self._consecutive_inactive += 1
        return self._consecutive_inactive >= self._inactive_frames

    def solve(
        self,
        get_frame: Callable[[], np.ndarray],
        set_steer: Callable[[str | None], None],
        should_abort: Callable[[], bool] | None = None,
    ) -> bool:
        """Steer with A/D until reel bar disappears. Returns True on success."""
        self.reset()
        start = time.perf_counter()
        while time.perf_counter() - start < self._timeout:
            if should_abort and should_abort():
                set_steer(None)
                return False

            frame = get_frame()
            if self.track_completion(frame):
                set_steer(None)
                return True

            analysis = self.analyze(frame)
            if analysis.active:
                if analysis.direction == SteerDirection.LEFT:
                    set_steer("left")
                elif analysis.direction == SteerDirection.RIGHT:
                    set_steer("right")
                else:
                    set_steer(None)

            time.sleep(self.tick_sec)

        set_steer(None)
        return False

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        debug = np.ascontiguousarray(frame.copy())
        analysis = self.analyze(frame)

        green = self.green_mask(frame)
        overlay = debug.copy()
        overlay[green > 0] = (0, 180, 0)
        cv2.addWeighted(overlay, 0.35, debug, 0.65, 0, debug)

        if analysis.active:
            cv2.line(
                debug,
                (analysis.green_center_x, 0),
                (analysis.green_center_x, debug.shape[0]),
                (0, 255, 255),
                1,
            )
            cv2.line(
                debug,
                (analysis.indicator_x, 0),
                (analysis.indicator_x, debug.shape[0]),
                (255, 255, 255),
                2,
            )
            color = {
                SteerDirection.LEFT: (0, 0, 255),
                SteerDirection.RIGHT: (255, 0, 0),
                SteerDirection.CENTER: (0, 255, 0),
            }
            cv2.putText(
                debug,
                f"{analysis.direction.name} ({analysis.indicator_x - analysis.green_center_x:+d}px)",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color.get(analysis.direction, (255, 255, 255)),
                1,
            )
        else:
            cv2.putText(debug, "inactive", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        return debug
