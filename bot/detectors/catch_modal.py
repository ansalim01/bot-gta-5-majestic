"""Catch modal — click «Забрать себе»."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR, fishing_region


def _best_template_match(frame: np.ndarray, templates: list[np.ndarray]) -> float:
    if frame.size == 0 or not templates:
        return 0.0
    best = 0.0
    for tpl in templates:
        if tpl.shape[0] > frame.shape[0] or tpl.shape[1] > frame.shape[1]:
            scale = min(frame.shape[0] / tpl.shape[0], frame.shape[1] / tpl.shape[1]) * 0.95
            tpl = cv2.resize(tpl, (int(tpl.shape[1] * scale), int(tpl.shape[0] * scale)))
        res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    return best


class CatchModalDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("catch_modal", {})
        regions = config.get("regions", {})
        self._modal_region = fishing_region(regions, "catch_modal")
        self._keep_button = fishing_region(regions, "keep_button")
        self._match_threshold = cfg.get("match_threshold", 0.85)
        self._title_templates = self._load_title_templates()
        if not self._title_templates:
            # Without templates is_visible() is a hard False: the COLLECT
            # state becomes unreachable and no fish is ever picked up — say
            # so instead of failing silently.
            print(
                f"[BOT] WARNING: no catch-modal templates (modal*.png) in "
                f"{TEMPLATES_DIR / 'fishing'} — fish collection is disabled",
                flush=True,
            )

    def _load_title_templates(self) -> list[np.ndarray]:
        fish_dir = TEMPLATES_DIR / "fishing"
        templates: list[np.ndarray] = []
        if fish_dir.exists():
            for path in fish_dir.glob("modal*.png"):
                img = cv2.imread(str(path))
                if img is not None:
                    templates.append(img)
        return templates

    def match_score(self, modal_frame: np.ndarray) -> float:
        return _best_template_match(modal_frame, self._title_templates)

    def is_visible(self, modal_frame: np.ndarray) -> bool:
        if not self._title_templates:
            return False
        return self.match_score(modal_frame) >= self._match_threshold

    def keep_button_center(self) -> tuple[int, int]:
        x, y, w, h = self._keep_button
        return x + w // 2, y + h // 2

    def draw_debug(self, modal_frame: np.ndarray) -> np.ndarray:
        debug = np.ascontiguousarray(modal_frame.copy())
        score = self.match_score(modal_frame)
        visible = score >= self._match_threshold
        x, y, w, h = self._keep_button
        mx, my, _, _ = self._modal_region
        rel_x = x - mx
        rel_y = y - my
        if 0 <= rel_x < debug.shape[1] and 0 <= rel_y < debug.shape[0]:
            cv2.rectangle(
                debug,
                (rel_x, rel_y),
                (rel_x + w, rel_y + h),
                (0, 255, 0) if visible else (0, 0, 255),
                2,
            )
        cv2.putText(
            debug,
            f"modal {score:.2f}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0) if visible else (0, 0, 255),
            1,
        )
        return debug
