"""Bottom-center carry prompt: 'Отнесите добытую руду в кучу'."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR


class CarryBannerDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        carry_cfg = config.get("carry_banner", {})
        self._match_threshold = carry_cfg.get("match_threshold", 0.62)
        self._templates = self._load_templates()

    def _load_templates(self) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        carry_dir = TEMPLATES_DIR / "carry"
        for path in sorted(carry_dir.glob("carry*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                img = cv2.imread(str(path))
                if img is not None:
                    templates.append(img)
        return templates

    def is_visible(self, frame: np.ndarray) -> bool:
        if frame.size == 0 or not self._templates:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for template in self._templates:
            t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            th, tw = t_gray.shape[:2]
            if th >= frame.shape[0] or tw >= frame.shape[1]:
                continue
            result = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
            if float(result.max()) >= self._match_threshold:
                return True
        return False
