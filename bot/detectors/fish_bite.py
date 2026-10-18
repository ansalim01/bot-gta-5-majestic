"""Fish bite detection — bubbles and splash in water area."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


class FishBiteDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        bite_cfg = config.get("bite", {})
        self._bubble_low = np.array(bite_cfg.get("bubble_low", [0, 0, 180]))
        self._bubble_high = np.array(bite_cfg.get("bubble_high", [180, 60, 255]))
        self._min_area = bite_cfg.get("min_bubble_area", 8)
        self._max_area = bite_cfg.get("max_bubble_area", 2500)
        self._splash_delta = bite_cfg.get("splash_brightness_delta", 25)
        self._splash_ratio_min = bite_cfg.get("splash_ratio_min", 0.015)
        self._debounce_sec = bite_cfg.get("debounce_sec", 0.4)

        self._last_bite_time = 0.0
        self._baseline_brightness: float | None = None
        self._baseline_frames = 0

    def reset_baseline(self) -> None:
        self._baseline_brightness = None
        self._baseline_frames = 0

    def _update_baseline(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if self._baseline_brightness is None:
            self._baseline_brightness = mean_brightness
            self._baseline_frames = 1
            return
        alpha = 0.08
        self._baseline_brightness = (
            1 - alpha
        ) * self._baseline_brightness + alpha * mean_brightness
        self._baseline_frames += 1

    def _bubble_count(self, frame: np.ndarray) -> int:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._bubble_low, self._bubble_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self._min_area <= area <= self._max_area:
                count += 1
        return count

    def _splash_detected(self, frame: np.ndarray) -> bool:
        if self._baseline_brightness is None or self._baseline_frames < 5:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = gray > (self._baseline_brightness + self._splash_delta)
        ratio = float(np.count_nonzero(bright)) / bright.size
        return ratio >= self._splash_ratio_min

    def is_bite(self, frame: np.ndarray, update_baseline: bool = True) -> bool:
        if frame.size == 0:
            return False

        if update_baseline:
            self._update_baseline(frame)

        bubbles = self._bubble_count(frame)
        splash = self._splash_detected(frame)

        if bubbles >= 2 or splash:
            now = time.perf_counter()
            if now - self._last_bite_time >= self._debounce_sec:
                self._last_bite_time = now
                return True
        return False

    def bubble_positions(self, frame: np.ndarray) -> list[tuple[int, int]]:
        if frame.size == 0:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._bubble_low, self._bubble_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points: list[tuple[int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self._min_area <= area <= self._max_area:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    points.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
        return points

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        debug = np.ascontiguousarray(frame.copy())
        for x, y in self.bubble_positions(frame):
            cv2.circle(debug, (x, y), 8, (0, 255, 255), 2)
        bite = self.is_bite(frame, update_baseline=False)
        label = "BITE!" if bite else f"bubbles: {len(self.bubble_positions(frame))}"
        cv2.putText(
            debug,
            label,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if bite else (200, 200, 200),
            1,
        )
        return debug
