"""Detect current fishing phase from HUD."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np

from bot.config_loader import fishing_region
from bot.detectors.cast_timing import CastTimingDetector
from bot.detectors.catch_modal import CatchModalDetector
from bot.detectors.fish_fight import FishFightDetector
from bot.detectors.fish_prompt import FishPromptDetector


class FishPhase(Enum):
    UNKNOWN = auto()
    CAST_TIMING = auto()
    WAITING = auto()
    BITE = auto()
    REELING = auto()
    CAUGHT = auto()


class FishPhaseDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        regions = config.get("regions", {})
        self._cfg_resolution = config.get("screen", {}).get("resolution", [1920, 1080])
        self._timing_region = fishing_region(regions, "timing_bar")
        self._bite_text_region = fishing_region(regions, "bite_text")
        self._bobber_region = fishing_region(regions, "bobber_widget")
        self._modal_region = fishing_region(regions, "catch_modal")

        self._timing = CastTimingDetector(config)
        self._fight = FishFightDetector(config)
        self._prompt = FishPromptDetector(config)
        self._modal = CatchModalDetector(config)

    def _crop(self, frame: np.ndarray, region: list[int]) -> np.ndarray:
        from bot.config_loader import scale_region

        h, w = frame.shape[:2]
        x, y, rw, rh = scale_region(region, self._cfg_resolution, w, h)
        x, y = max(0, min(x, w)), max(0, min(y, h))
        rw, rh = max(0, min(rw, w - x)), max(0, min(rh, h - y))
        return frame[y : y + rh, x : x + rw]

    def detect(self, frame: np.ndarray) -> FishPhase:
        """Single full-window frame in, phase out — every sub-region is sliced
        from it in-memory (no extra screen captures)."""
        modal = self._crop(frame, self._modal_region)
        if self._modal.is_visible(modal):
            return FishPhase.CAUGHT

        timing_frame = self._crop(frame, self._timing_region)
        timing_active = timing_frame.size > 0 and self._timing.is_active_fresh(timing_frame)
        if timing_active:
            return FishPhase.CAST_TIMING

        # Recover a live fight even when the floating green/white icon is
        # hidden behind the player model. The content-located yellow fill is
        # already sufficient to distinguish reeling from the green cast bar.
        bar_crop = self._fight.crop_bar(frame)
        if self._fight.bar_fight_active(frame, bar_crop, timing_active=False):
            return FishPhase.REELING

        bite_text = self._crop(frame, self._bite_text_region)
        bobber = self._crop(frame, self._bobber_region)

        # text_scores() computes bite/keyword/wait in one pass instead of the
        # ~17 redundant matchTemplate calls that is_biting()+is_waiting() would
        # otherwise trigger (keyword templates overlap bite templates, and
        # is_waiting() used to recompute both wait and bite scores again).
        text = self._prompt.text_scores(bite_text)
        # Waiting takes precedence over generic white controls. On the current
        # HUD "hook fish [Space]" is present throughout the entire wait, so
        # treating that control as a bite made every waiting frame look like
        # BITE. The configured bite-text detector remains the authoritative
        # trigger in FisherJob.
        if text.waiting:
            return FishPhase.WAITING

        if text.bite >= self._prompt.bite_text_threshold or self._prompt.is_bobber_biting(bobber):
            return FishPhase.BITE

        return FishPhase.UNKNOWN

    def phase_label(self, phase: FishPhase) -> str:
        return phase.name
