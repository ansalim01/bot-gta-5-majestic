"""Detect the two screens that precede the normal fishing loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR, scale_region


@dataclass(frozen=True)
class FishingSetupScores:
    start: float
    equipment: float


class FishingSetupDetector:
    """Template detector for "start fishing" and equipment confirmation."""

    def __init__(self, config: dict[str, Any]) -> None:
        regions = config.get("regions", {})
        self._resolution = config.get("screen", {}).get("resolution", [1920, 1080])
        self.start_region = list(regions.get("fishing_start_prompt", [0, 0, 400, 90]))
        self.equipment_region = list(regions.get("fishing_equipment_prompt", [1500, 990, 420, 90]))

        setup = config.get("fishing_setup", {})
        # The start banner is translucent, so a grayscale template also
        # compares the scenery visible through it. Day/night changes pushed a
        # real fishing prompt down to 0.67 while unrelated E prompts reached
        # 0.82. Match only the stable bright text/key foreground instead.
        self.start_threshold = float(setup.get("start_foreground_match_threshold", 0.90))
        self._start_foreground_min = int(setup.get("start_foreground_min_gray", 215))
        self.equipment_threshold = float(setup.get("equipment_match_threshold", 0.76))
        template_dir = TEMPLATES_DIR / "fishing"
        self._start_template = self._load(
            template_dir / str(setup.get("start_template", "start_fishing_prompt.png"))
        )
        self._equipment_template = self._load(
            template_dir / str(setup.get("equipment_template", "equipment_ready_prompt.png"))
        )

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read fishing setup template: {path}")
        return image

    def _crop(self, frame: np.ndarray, region: list[int]) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.empty((0, 0), dtype=np.uint8)
        height, width = frame.shape[:2]
        x, y, rw, rh = scale_region(
            region,
            self._resolution,
            width,
            height,
        )
        x, y = max(0, min(x, width)), max(0, min(y, height))
        rw, rh = max(0, min(rw, width - x)), max(0, min(rh, height - y))
        crop = frame[y : y + rh, x : x + rw]
        if crop.size == 0:
            return np.empty((0, 0), dtype=np.uint8)
        reference_width, reference_height = region[2], region[3]
        if crop.shape[1] != reference_width or crop.shape[0] != reference_height:
            crop = cv2.resize(
                crop,
                (reference_width, reference_height),
                interpolation=cv2.INTER_CUBIC,
            )
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return crop

    @staticmethod
    def _score(crop: np.ndarray, template: np.ndarray) -> float:
        if crop.size == 0 or crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
            return 0.0
        result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        return float(cv2.minMaxLoc(result)[1])

    def _start_score(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        minimum = max(0, min(self._start_foreground_min, 255))
        crop_foreground = cv2.inRange(crop, minimum, 255)
        template_foreground = cv2.inRange(
            self._start_template,
            minimum,
            255,
        )
        return self._score(crop_foreground, template_foreground)

    def scores(self, frame: np.ndarray) -> FishingSetupScores:
        return FishingSetupScores(
            start=self._start_score(self._crop(frame, self.start_region)),
            equipment=self._score(
                self._crop(frame, self.equipment_region),
                self._equipment_template,
            ),
        )

    def is_start_prompt(self, frame: np.ndarray) -> bool:
        return self.scores(frame).start >= self.start_threshold

    def is_equipment_prompt(self, frame: np.ndarray) -> bool:
        return self.scores(frame).equipment >= self.equipment_threshold
