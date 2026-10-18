"""Recognition of the W/D/S/A gym prompt shown at the bottom of the screen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot.config_loader import ROOT_DIR


@dataclass(frozen=True)
class GymRecognition:
    key: str | None
    score: float
    margin: float
    circle_visible: bool
    glyph: np.ndarray | None = None
    source: str = "none"


class GymPromptDetector:
    """Extract and classify one of the four fixed handwritten glyphs."""

    NORMALIZED_ROI_SIZE = 114
    GLYPH_SIZE = 64

    def __init__(self, config: dict[str, Any]) -> None:
        gym = config.get("gym", {})
        templates_dir = Path(gym.get("templates_dir", "config/templates/gym"))
        if not templates_dir.is_absolute():
            templates_dir = ROOT_DIR / templates_dir

        self._min_score = float(gym.get("min_match_score", 0.78))
        self._min_margin = float(gym.get("min_match_margin", 0.10))
        self._min_brightness = float(gym.get("min_circle_brightness_ratio", 0.35))
        self._templates: dict[str, list[np.ndarray]] = {}
        for key in gym.get("keys", ["w", "d", "s", "a"]):
            key = str(key).lower()
            paths = sorted(templates_dir.glob(f"prompt_{key}*.png"))
            if not paths:
                raise FileNotFoundError(
                    f"Gym prompt templates not found: {templates_dir} (expected prompt_{key}*.png)"
                )
            glyphs: list[np.ndarray] = []
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read gym template: {path}")
                glyph, _ = self._extract_glyph(image)
                if glyph is None:
                    raise ValueError(f"Cannot extract glyph from template: {path}")
                glyphs.append(glyph)
            self._templates[key] = glyphs

    def detect(self, roi: np.ndarray) -> GymRecognition:
        glyph, brightness = self._extract_glyph(roi)
        circle_visible = brightness >= self._min_brightness
        if glyph is None or not circle_visible:
            return GymRecognition(None, 0.0, 0.0, circle_visible, glyph)

        scores = {
            key: max(self._glyph_similarity(glyph, template) for template in templates)
            for key, templates in self._templates.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        key, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = score - runner_up
        source = "template"
        if score < self._min_score or margin < self._min_margin:
            # A generated CAPTCHA may use a new font/background combination.
            # A generated CAPTCHA may use an unseen font. Fall back to
            # topology of the cleaned black strokes instead of accepting the
            # nearest-looking template with poor confidence.
            key = self._shape_fallback(glyph)
            source = "shape"
        return GymRecognition(key, score, margin, True, glyph, source)

    def _extract_glyph(self, roi: np.ndarray) -> tuple[np.ndarray | None, float]:
        if roi is None or roi.size == 0:
            return None, 0.0

        size = self.NORMALIZED_ROI_SIZE
        interpolation = (
            cv2.INTER_AREA if roi.shape[0] >= size and roi.shape[1] >= size else cv2.INTER_LINEAR
        )
        normalized = cv2.resize(roi, (size, size), interpolation=interpolation)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

        yy, xx = np.ogrid[:size, :size]
        circle = (xx - 57) ** 2 + (yy - 57) ** 2 <= 50**2
        circle_pixels = max(int(np.count_nonzero(circle)), 1)
        brightness = float(np.count_nonzero((gray >= 145) & circle)) / circle_pixels

        dark = np.zeros_like(gray)
        dark[10:104, 10:104] = (gray[10:104, 10:104] < 72).astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)

        candidates: list[tuple[float, int]] = []
        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            center_x = x + width / 2
            center_y = y + height / 2
            valid = (
                80 <= area <= 2500
                and 20 <= center_x <= 94
                and 18 <= center_y <= 92
                and width >= 12
                and height >= 20
                and x > 10
                and y > 10
                and x + width < 104
                and y + height < 104
            )
            if not valid:
                continue
            distance = float(np.hypot(center_x - 57, center_y - 56))
            candidates.append((area - 3.0 * distance, index))

        if brightness < self._min_brightness or not candidates:
            return None, brightness

        component = max(candidates)[1]
        mask = (labels == component).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        glyph = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

        target = 52
        scale = min(target / glyph.shape[1], target / glyph.shape[0])
        width = max(1, round(glyph.shape[1] * scale))
        height = max(1, round(glyph.shape[0] * scale))
        glyph = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_NEAREST)
        result = np.zeros((self.GLYPH_SIZE, self.GLYPH_SIZE), dtype=np.uint8)
        x = (self.GLYPH_SIZE - width) // 2
        y = (self.GLYPH_SIZE - height) // 2
        result[y : y + height, x : x + width] = glyph
        return result, brightness

    @staticmethod
    def _glyph_similarity(first: np.ndarray, second: np.ndarray) -> float:
        kernel = np.ones((3, 3), dtype=np.uint8)
        first_dilated = cv2.dilate(first, kernel)
        second_dilated = cv2.dilate(second, kernel)
        first_count = max(int(np.count_nonzero(first)), 1)
        second_count = max(int(np.count_nonzero(second)), 1)
        first_covered = np.count_nonzero((first > 0) & (second_dilated > 0)) / first_count
        second_covered = np.count_nonzero((second > 0) & (first_dilated > 0)) / second_count
        return float((first_covered + second_covered) / 2.0)

    @staticmethod
    def _shape_fallback(glyph: np.ndarray) -> str | None:
        ys, xs = np.where(glyph > 0)
        if len(xs) == 0:
            return None
        mask = glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 0
        height, width = mask.shape
        if height < 10 or width < 8:
            return None

        def run_count(line: np.ndarray) -> int:
            padded = np.pad(line.astype(np.int8), (1, 1))
            return int(np.count_nonzero(np.diff(padded) == 1))

        top_end = max(1, round(height * 0.65))
        top_runs = float(np.mean([run_count(row) for row in mask[:top_end]]))
        column_runs = float(np.mean([run_count(column) for column in mask.T]))
        band = max(1, round(height * 0.20))

        def horizontal_span(rows: np.ndarray) -> float:
            columns = np.where(rows)[1]
            if len(columns) == 0:
                return 0.0
            return float(columns.max() - columns.min() + 1) / width

        top_span = horizontal_span(mask[:band])
        bottom_span = horizontal_span(mask[-band:])
        half = max(1, width // 2)
        left_density = float(np.mean(mask[:, :half]))
        right_density = float(np.mean(mask[:, half:]))

        # W: 3-4 separate strokes in its upper portion converge downwards.
        if top_span - bottom_span >= 0.20 and top_runs >= column_runs * 1.10:
            return "w"

        # A: either a narrow cap above broad feet, or a strongly italic
        # right-heavy diagonal/crossbar construction.
        if (top_span <= 0.50 and bottom_span >= 0.85) or right_density >= left_density * 1.38:
            return "a"

        # S is crossed more often by vertical scanlines than horizontal ones.
        # The remaining loop/arch construction is D.
        if column_runs >= top_runs * 1.14:
            return "s"
        return "d"

    @staticmethod
    def draw_debug(roi: np.ndarray, recognition: GymRecognition) -> np.ndarray:
        debug = roi.copy()
        label = recognition.key.upper() if recognition.key else "-"
        color = (0, 220, 0) if recognition.key else (0, 0, 255)
        cv2.putText(
            debug,
            f"{label} {recognition.score:.2f} m={recognition.margin:.2f} {recognition.source}",
            (3, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        return debug
