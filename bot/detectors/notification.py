"""Top-left notification detector for 'press E' prompts."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR


class NotificationDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        notif_cfg = config.get("notification", {})
        self._brightness_threshold = notif_cfg.get("brightness_threshold", 40)
        self._match_threshold = notif_cfg.get("match_threshold", 0.75)
        self._drop_match_threshold = notif_cfg.get("drop_match_threshold", 0.75)
        self._mine_match_threshold = notif_cfg.get("mine_match_threshold", 0.75)
        self._roi_corr_min = notif_cfg.get("roi_correlation_min", 0.75)
        self._roi_corr_margin = notif_cfg.get("roi_correlation_margin", 0.18)
        self._debounce_sec = notif_cfg.get("debounce_sec", 2.0)
        self._last_press_time = 0.0
        self._templates, self._drop_templates, self._mine_templates = self._load_templates()

    def _load_templates(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        all_templates: list[np.ndarray] = []
        drop_templates: list[np.ndarray] = []
        mine_templates: list[np.ndarray] = []
        notif_dir = TEMPLATES_DIR / "notification"
        search_dirs = [notif_dir, TEMPLATES_DIR] if notif_dir.exists() else [TEMPLATES_DIR]

        for directory in search_dirs:
            for path in sorted(directory.glob("notif*")):
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    continue
                name = path.stem.lower()
                all_templates.append(img)
                if "drop" in name or "dump" in name or "высып" in name:
                    drop_templates.append(img)
                elif "mine" in name or "добыт" in name:
                    mine_templates.append(img)

        return all_templates, drop_templates, mine_templates

    def reload_templates(self) -> None:
        self._templates, self._drop_templates, self._mine_templates = self._load_templates()

    def is_visible(self, frame: np.ndarray) -> bool:
        return self.is_drop_visible(frame) or self.is_mine_visible(frame)

    def is_drop_visible(self, frame: np.ndarray) -> bool:
        drop_score, mine_score = self._roi_scores(frame)
        if drop_score >= self._roi_corr_min and drop_score >= mine_score + self._roi_corr_margin:
            return True
        if self._drop_templates and self._match_templates(
            frame, self._drop_templates, self._drop_match_threshold
        ):
            return True
        return False

    def is_mine_visible(self, frame: np.ndarray) -> bool:
        drop_score, mine_score = self._roi_scores(frame)
        if mine_score >= self._roi_corr_min and mine_score >= drop_score + self._roi_corr_margin:
            return True
        if self._mine_templates and self._match_templates(
            frame, self._mine_templates, self._mine_match_threshold
        ):
            return True
        if self._templates and not self._drop_templates and not self._mine_templates:
            if self._match_templates(frame, self._templates, self._match_threshold):
                return True
        if not self._mine_templates and not self._drop_templates:
            return self._detect_by_brightness(frame)
        return False

    def _roi_scores(self, frame: np.ndarray) -> tuple[float, float]:
        drop_score = 0.0
        mine_score = 0.0
        if self._drop_templates:
            drop_score = max(self._correlate_roi(frame, t) for t in self._drop_templates)
        if self._mine_templates:
            mine_score = max(self._correlate_roi(frame, t) for t in self._mine_templates)
        return drop_score, mine_score

    @staticmethod
    def _correlate_roi(frame: np.ndarray, template: np.ndarray) -> float:
        if frame.size == 0 or template.size == 0:
            return 0.0
        if frame.shape != template.shape:
            template = cv2.resize(template, (frame.shape[1], frame.shape[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= gray.mean()
        ref -= ref.mean()
        gray_std = float(gray.std())
        ref_std = float(ref.std())
        if gray_std < 1e-3 or ref_std < 1e-3:
            return 0.0
        gray /= gray_std
        ref /= ref_std
        return float(np.mean(gray * ref))

    def _match_templates(
        self,
        frame: np.ndarray,
        templates: list[np.ndarray],
        threshold: float,
    ) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for template in templates:
            t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            th, tw = t_gray.shape[:2]
            if th >= frame.shape[0] or tw >= frame.shape[1]:
                continue
            result = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
            if float(result.max()) >= threshold:
                return True
        return False

    def _detect_by_brightness(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.count_nonzero(bright) / bright.size

        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size

        mean_brightness = float(np.mean(gray))
        return (
            mean_brightness > self._brightness_threshold + 80
            and bright_ratio > 0.02
            and edge_ratio > 0.03
        )

    def should_press_e(self) -> bool:
        return time.perf_counter() - self._last_press_time >= self._debounce_sec

    def mark_pressed(self) -> None:
        self._last_press_time = time.perf_counter()

    def draw_debug(self, frame: np.ndarray, detected: bool) -> np.ndarray:
        debug = frame.copy()
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(
            debug,
            f"notif: {detected}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        return debug
