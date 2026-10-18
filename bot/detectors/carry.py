"""Carry ore indicator detector."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR


class CarryDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        carry_cfg = config.get("carry", {})
        self._match_threshold = carry_cfg.get("match_threshold", 0.75)
        self._brightness_threshold = carry_cfg.get("brightness_threshold", 35)
        self._templates = self._load_templates()

    def _load_templates(self) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        carry_dir = TEMPLATES_DIR / "carry"
        search_paths = list(carry_dir.glob("carry*")) if carry_dir.exists() else []
        search_paths += list(TEMPLATES_DIR.glob("carry*"))

        for path in sorted(set(search_paths)):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                img = cv2.imread(str(path))
                if img is not None:
                    templates.append(img)
        return templates

    def reload_templates(self) -> None:
        self._templates = self._load_templates()

    def is_carrying(self, frame: np.ndarray) -> bool:
        if self._templates:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for template in self._templates:
                t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                th, tw = t_gray.shape[:2]
                if th >= frame.shape[0] or tw >= frame.shape[1]:
                    continue
                result = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
                if float(result.max()) >= self._match_threshold:
                    return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([25, 255, 255]))
        orange_ratio = np.count_nonzero(orange_mask) / orange_mask.size
        mean_brightness = float(np.mean(gray))
        return orange_ratio > 0.015 and mean_brightness > self._brightness_threshold + 60
