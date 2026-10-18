"""Detect current mining job phase from HUD / minimap."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import cv2
import numpy as np

from bot.detectors.carry_banner import CarryBannerDetector
from bot.detectors.minigame import MinigameDetector
from bot.detectors.notification import NotificationDetector
from bot.detectors.progress_bar import ProgressBarDetector
from bot.navigation.minimap_nav import MinimapNavigator


class MinePhase(Enum):
    UNKNOWN = auto()
    NAVIGATE = auto()
    AT_ROCK = auto()
    AT_DROP = auto()
    MINING = auto()
    MINIGAME = auto()
    CARRY = auto()


class MinePhaseDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        regions = config.get("regions", {})
        self._notif_region = regions.get("notification", [20, 20, 400, 120])
        self._minigame_region = regions.get("minigame", [520, 340, 880, 420])
        self._progress_region = regions.get("minigame_progress", [700, 680, 520, 60])
        self._carry_banner_region = regions.get("carry_banner", [550, 1005, 820, 45])
        self._mining_hud_region = regions.get("mining_hud", [1550, 900, 350, 140])

        phase_cfg = config.get("mine_phase", {})
        self._minigame_pink_min = phase_cfg.get("minigame_pink_ratio", 0.08)
        self._mining_white_min = phase_cfg.get("mining_white_ratio", 0.008)
        self._mining_blue_min = phase_cfg.get("mining_blue_ratio", 0.006)

        self._minigame = MinigameDetector(config)
        self._notification = NotificationDetector(config)
        self._progress_bar = ProgressBarDetector(config)
        self._carry_banner = CarryBannerDetector(config)
        self._minimap = MinimapNavigator(config)

    def detect(
        self,
        full_frame: np.ndarray,
        grab_region,
    ) -> MinePhase:
        if self._carry_banner.is_visible(grab_region(self._carry_banner_region)):
            return MinePhase.CARRY

        minigame_frame = grab_region(self._minigame_region)
        progress_frame = grab_region(self._progress_region)
        if self._is_minigame_phase(minigame_frame, progress_frame):
            return MinePhase.MINIGAME

        notif_frame = grab_region(self._notif_region)
        if self._notification.is_drop_visible(notif_frame):
            return MinePhase.AT_DROP
        if self._notification.is_mine_visible(notif_frame):
            return MinePhase.AT_ROCK

        mining_frame = grab_region(self._mining_hud_region)
        if self._is_mining_hud(mining_frame):
            return MinePhase.MINING

        minimap_frame = grab_region(self._minimap.region)
        if self._minimap.find_target_blip(minimap_frame) is not None:
            return MinePhase.NAVIGATE

        return MinePhase.UNKNOWN

    def _pink_ratio(self, frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pink = cv2.inRange(hsv, np.array([140, 40, 100]), np.array([175, 255, 255]))
        return float(np.count_nonzero(pink)) / pink.size

    def _is_minigame_phase(self, minigame_frame: np.ndarray, progress_frame: np.ndarray) -> bool:
        if self._pink_ratio(minigame_frame) < self._minigame_pink_min:
            return False
        if self._progress_bar.is_visible(progress_frame):
            return True
        return len(self._minigame.find_veins(minigame_frame)) > 0

    def _is_mining_hud(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_ratio = float(np.count_nonzero(gray > 195)) / gray.size
        blue_mask = cv2.inRange(hsv, np.array([85, 40, 70]), np.array([135, 255, 255]))
        blue_ratio = float(np.count_nonzero(blue_mask)) / blue_mask.size
        return white_ratio >= self._mining_white_min or blue_ratio >= self._mining_blue_min

    def phase_label(self, phase: MinePhase) -> str:
        return phase.name
