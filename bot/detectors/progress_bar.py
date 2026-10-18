"""Progress bar detector for ore collection minigame."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class ProgressBarDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        minigame_cfg = config.get("minigame", {})
        self._fill_threshold = minigame_cfg.get("progress_fill_threshold", 0.85)
        self._white_threshold = minigame_cfg.get("progress_white_threshold", 0.08)
        self._inner_margin = minigame_cfg.get("progress_inner_margin", 0.15)

    def is_visible(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_pixels = np.count_nonzero(gray > 200)
        white_ratio = white_pixels / gray.size

        dark_pixels = np.count_nonzero(gray < 80)
        dark_ratio = dark_pixels / gray.size

        return white_ratio > self._white_threshold and dark_ratio > 0.02

    def fill_ratio(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0

        h, w = frame.shape[:2]
        margin_x = int(w * self._inner_margin)
        margin_y = int(h * self._inner_margin)
        inner = frame[margin_y : h - margin_y, margin_x : w - margin_x]

        if inner.size == 0:
            return 0.0

        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)

        # Полоса на бумаге: белый фон + чёрная штриховка (не зелёная заливка)
        dark = gray < 95
        mid = (gray >= 95) & (gray < 200)
        row_fill = np.mean(dark, axis=1)
        filled_rows = np.count_nonzero(row_fill > 0.08)
        if filled_rows > inner.shape[0] * 0.15:
            return float(np.count_nonzero(dark)) / dark.size

        # Fallback: цветная заливка (старый вариант UI)
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        fill_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([130, 255, 255]))
        fill_mask = cv2.bitwise_and(fill_mask, cv2.inRange(gray, 60, 250))
        color_ratio = float(np.count_nonzero(fill_mask)) / fill_mask.size
        if color_ratio > 0.02:
            return color_ratio

        return float(np.count_nonzero(dark)) / max(np.count_nonzero(mid | dark), 1)

    def is_complete(self, frame: np.ndarray) -> bool:
        if not self.is_visible(frame):
            return False
        return self.fill_ratio(frame) >= self._fill_threshold

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        debug = np.ascontiguousarray(frame.copy())
        visible = self.is_visible(frame)
        fill = self.fill_ratio(frame)
        color = (0, 255, 0) if visible else (0, 0, 255)
        cv2.putText(
            debug,
            f"progress: {fill:.0%}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        return debug
