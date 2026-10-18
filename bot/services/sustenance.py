"""Reusable hunger/thirst service for Majestic bots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot.capture import GameCapture
from bot.config_loader import ROOT_DIR
from bot.input_handler import InputHandler


@dataclass(frozen=True)
class TemplateMatch:
    score: float
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass(frozen=True)
class SustenanceResult:
    checked: bool = False
    drank: bool = False
    ate: bool = False
    water_detected: bool = False
    hunger_detected: bool = False
    water_needed: bool = False
    food_needed: bool = False
    water_found: bool = True
    food_found: bool = True
    inventory_attempts: int = 0
    reason: str = "idle"


class SustenanceDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        settings = config.get("sustenance", {})
        self.thirst_threshold = float(settings.get("thirst_match_threshold", 0.78))
        self.inventory_threshold = float(settings.get("inventory_match_threshold", 0.78))
        self.item_threshold = float(settings.get("item_match_threshold", 0.72))
        templates = config.get("templates", {})
        self._thirst = self._load_one(templates.get("thirst_warning"))
        self._inventory_header = self._load_one(templates.get("inventory_header"))
        self._items = {
            "water": self._load_many(templates.get("water", [])),
            "food": self._load_many(templates.get("food", [])),
        }

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        result = Path(path)
        return result if result.is_absolute() else ROOT_DIR / result

    def _load_one(self, path: str | Path | None) -> np.ndarray:
        if not path:
            raise ValueError("Required sustenance template path is missing")
        resolved = self._resolve(path)
        image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read template: {resolved}")
        return image

    def _load_many(self, paths: list[str] | tuple[str, ...]) -> list[np.ndarray]:
        return [self._load_one(path) for path in paths]

    @staticmethod
    def _best_match(image: np.ndarray, template: np.ndarray) -> TemplateMatch:
        if (
            image is None
            or image.size == 0
            or image.shape[0] < template.shape[0]
            or image.shape[1] < template.shape[1]
        ):
            return TemplateMatch(0.0, 0, 0, template.shape[1], template.shape[0])
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        return TemplateMatch(
            float(score),
            int(location[0]),
            int(location[1]),
            int(template.shape[1]),
            int(template.shape[0]),
        )

    def thirst_score(self, hud_crop: np.ndarray) -> float:
        # Keep the exact template as a strong signal, but do not make thirst
        # detection depend on the scenery captured around the translucent HUD
        # tile.  Its surrounding pixels change significantly between daytime,
        # nighttime and inventory transitions.
        template_score = self._best_match(hud_crop, self._thirst).score
        tile_score = self._colored_tile_score(
            hud_crop,
            np.array((92, 45, 65), dtype=np.uint8),
            np.array((122, 230, 235), dtype=np.uint8),
        )
        return max(template_score, tile_score)

    @staticmethod
    def _colored_tile_score(
        hud_crop: np.ndarray,
        color_low: np.ndarray,
        color_high: np.ndarray,
    ) -> float:
        """Score a compact coloured HUD tile containing a white glyph."""
        if hud_crop is None or hud_crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(hud_crop, cv2.COLOR_BGR2HSV)
        colored = cv2.inRange(hsv, color_low, color_high)
        colored = cv2.morphologyEx(
            colored,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(colored, 8)
        best = 0.0
        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            aspect = width / max(height, 1)
            if not (
                18 <= width <= 75 and 18 <= height <= 75 and 0.65 <= aspect <= 1.40 and area >= 250
            ):
                continue
            patch = hud_crop[y : y + height, x : x + width]
            white = cv2.inRange(
                patch,
                np.array((175, 175, 175), dtype=np.uint8),
                np.array((255, 255, 255), dtype=np.uint8),
            )
            white_pixels = int(np.count_nonzero(white))
            area_score = min(1.0, area / 650.0)
            glyph_score = min(1.0, white_pixels / 24.0)
            fill_score = min(
                1.0,
                area / max(width * height * 0.55, 1.0),
            )
            best = max(
                best,
                (area_score + glyph_score + fill_score) / 3.0,
            )
        return float(best)

    @staticmethod
    def hunger_tile_score(hud_crop: np.ndarray) -> float:
        """Find the orange rounded HUD tile without depending on its glyph."""
        return SustenanceDetector._colored_tile_score(
            hud_crop,
            np.array((5, 120, 140), dtype=np.uint8),
            np.array((28, 255, 255), dtype=np.uint8),
        )

    def inventory_header_match(self, crop: np.ndarray) -> TemplateMatch:
        return self._best_match(crop, self._inventory_header)

    def find_item(self, inventory_crop: np.ndarray, kind: str) -> TemplateMatch:
        templates = self._items.get(kind, [])
        if not templates:
            return TemplateMatch(0.0, 0, 0, 1, 1)
        return max(
            (self._best_match(inventory_crop, template) for template in templates),
            key=lambda match: match.score,
        )


class SustenanceService:
    """Check and replenish food/water when called from a safe bot state.

    The HUD is sampled at a configured interval. Inventory is opened only
    after a hunger/thirst warning is actually visible. Other jobs should call
    ``tick(allow_action=True)`` only between actions where opening inventory
    cannot ruin a minigame or held movement key.
    """

    def __init__(
        self,
        config: dict[str, Any],
        capture: GameCapture,
        input_handler: InputHandler,
    ) -> None:
        self._config = config
        self._capture = capture
        self._input = input_handler
        self._detector = SustenanceDetector(config)
        self._regions = config.get("regions", {})
        settings = config.get("sustenance", {})
        self._inventory_key = str(settings.get("inventory_key", "i"))
        self._hud_interval = float(settings.get("hud_check_interval_sec", 60))
        self._water_enabled = bool(settings.get("water_enabled", True))
        self._food_enabled = bool(settings.get("food_enabled", False))
        self._hunger_score_threshold = float(settings.get("hunger_tile_score_threshold", 0.72))
        self._cooldown = float(settings.get("action_cooldown_sec", 20))
        self._failed_retry = float(settings.get("failed_retry_sec", 8))
        self._open_wait = float(settings.get("inventory_open_wait_sec", 0.85))
        self._open_attempts = max(
            1,
            int(settings.get("inventory_open_attempts", 3)),
        )
        self._open_poll = max(
            0.05,
            float(settings.get("inventory_open_poll_sec", 0.15)),
        )
        self._open_retry = max(
            0.0,
            float(settings.get("inventory_open_retry_sec", 0.35)),
        )
        self._item_wait = float(settings.get("after_item_wait_sec", 0.70))
        self._close_wait = float(settings.get("inventory_close_wait_sec", 0.35))
        self._double_click_interval = float(settings.get("double_click_interval_sec", 0.12))
        debug = config.get("debug", {})
        self._save_debug = bool(debug.get("save_inventory_frame", False))
        self._debug_path = ROOT_DIR / "logs" / "sustenance_inventory.png"

        self._last_hud_check = float("-inf")
        self._last_action = float("-inf")
        self._last_attempt = float("-inf")
        self._force_check = False
        self._busy = False
        self._stats = {
            "checks": 0,
            "drinks": 0,
            "meals": 0,
            "failures": 0,
        }
        self.last_result = SustenanceResult()
        self.last_thirst_score = 0.0
        self.last_hunger_score = 0.0
        self.last_item_scores = {"water": 0.0, "food": 0.0}

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def busy(self) -> bool:
        return self._busy

    def request_check(self) -> None:
        self._force_check = True

    def action_block_reason(self) -> str | None:
        """Explain why consuming an item should be deferred right now."""
        now = time.perf_counter()
        if now - self._last_action < self._cooldown:
            return "cooldown"
        if now - self._last_attempt < self._failed_retry:
            return "retry_wait"
        return None

    def check_needs(self, *, force: bool = False) -> SustenanceResult:
        """Inspect only the HUD; never open inventory or press a key."""
        if self._busy:
            return SustenanceResult(reason="not_safe")

        now = time.perf_counter()
        if not force and now - self._last_hud_check < self._hud_interval:
            return SustenanceResult(reason="hud_wait")
        self._last_hud_check = now
        hud = self._capture.grab_region(self._regions.get("thirst_warning", [250, 785, 190, 180]))
        self.last_thirst_score = self._detector.thirst_score(hud)
        self.last_hunger_score = self._detector.hunger_tile_score(hud)
        water_detected = self.last_thirst_score >= self._detector.thirst_threshold
        hunger_detected = self.last_hunger_score >= self._hunger_score_threshold
        need_water = water_detected and self._water_enabled
        need_food = hunger_detected and self._food_enabled
        if need_water or need_food:
            reason = "need"
        elif hunger_detected and not self._food_enabled:
            reason = "food_not_configured"
            print(
                "\n[BOT] Индикатор голода найден; предмет еды пока не настроен.",
                flush=True,
            )
        else:
            reason = "no_need"
        result = SustenanceResult(
            checked=True,
            water_detected=water_detected,
            hunger_detected=hunger_detected,
            water_needed=need_water,
            food_needed=need_food,
            reason=reason,
        )
        self.last_result = result
        return result

    def tick(self, *, allow_action: bool = True) -> SustenanceResult:
        if self._busy or not allow_action:
            return SustenanceResult(reason="not_safe")

        requested = self._force_check
        self._force_check = False
        needs = self.check_needs(force=requested)
        if needs.reason != "need":
            return needs

        return self.consume_detected_needs(needs, allow_action=allow_action)

    def consume_detected_needs(
        self,
        needs: SustenanceResult,
        *,
        allow_action: bool = True,
    ) -> SustenanceResult:
        """Consume needs found by ``check_needs`` without sampling HUD again.

        This is useful for another job that must first inspect the HUD, leave
        its current activity, and only then allow inventory interaction.
        """
        if self._busy or not allow_action:
            return SustenanceResult(reason="not_safe")
        if not needs.checked or not (needs.water_needed or needs.food_needed):
            return needs

        need_water = needs.water_needed
        need_food = needs.food_needed
        block_reason = self.action_block_reason()
        if block_reason is not None:
            result = SustenanceResult(
                checked=True,
                water_detected=needs.water_detected,
                hunger_detected=needs.hunger_detected,
                water_needed=need_water,
                food_needed=need_food,
                reason=block_reason,
            )
            self.last_result = result
            return result

        self._last_attempt = time.perf_counter()
        self._busy = True
        try:
            result = self._inspect_and_consume(
                need_water=need_water,
                need_food=need_food,
                water_detected=needs.water_detected,
                hunger_detected=needs.hunger_detected,
            )
            self.last_result = result
            if result.drank or result.ate:
                self._last_action = time.perf_counter()
            if result.reason == "input_blocked" or not result.water_found or not result.food_found:
                self._stats["failures"] += 1
            return result
        finally:
            self._busy = False

    def _inspect_and_consume(
        self,
        *,
        need_water: bool,
        need_food: bool,
        water_detected: bool,
        hunger_detected: bool,
    ) -> SustenanceResult:
        opened_by_service = False
        inventory_attempts = 0
        frame = self._capture.grab()
        offset = self._inventory_offset(frame)
        if offset is None:
            for attempt in range(1, self._open_attempts + 1):
                # Re-check before every key press. If the previous I was
                # handled late by GTA, another I here would close the window
                # before the detector ever saw it.
                if attempt > 1:
                    frame = self._capture.grab()
                    offset = self._inventory_offset(frame)
                    if offset is not None:
                        break

                if self._input.press_key(self._inventory_key) is False:
                    return SustenanceResult(
                        checked=False,
                        water_detected=water_detected,
                        hunger_detected=hunger_detected,
                        water_needed=need_water,
                        food_needed=need_food,
                        water_found=False,
                        food_found=False,
                        inventory_attempts=attempt,
                        reason="input_blocked",
                    )
                opened_by_service = True
                inventory_attempts = attempt
                deadline = time.perf_counter() + self._open_wait
                while True:
                    frame = self._capture.grab()
                    offset = self._inventory_offset(frame)
                    if offset is not None:
                        break
                    if time.perf_counter() >= deadline:
                        break
                    time.sleep(
                        min(
                            self._open_poll,
                            max(0.0, deadline - time.perf_counter()),
                        )
                    )
                if offset is not None:
                    break
                if attempt < self._open_attempts:
                    time.sleep(self._open_retry)
        if offset is None:
            # Do not send a blind "closing" I here: the window was never
            # confirmed open, and that extra key can actually open it just as
            # the method gives up.
            return SustenanceResult(
                checked=False,
                water_detected=water_detected,
                hunger_detected=hunger_detected,
                water_needed=need_water,
                food_needed=need_food,
                water_found=False,
                food_found=False,
                inventory_attempts=inventory_attempts,
                reason="inventory_not_open",
            )

        try:
            self._stats["checks"] += 1
            if self._save_debug:
                self._debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self._debug_path), frame)

            self.last_item_scores = {"water": 0.0, "food": 0.0}
            # Search the first item in the exact frame whose inventory header
            # was confirmed above. Capturing again here used to race a window
            # refresh/transition: the saved debug frame contained Cola with a
            # 1.00 match while the immediately following frame reported it as
            # missing.
            drank, water_found, water_blocked = self._consume_if_needed(
                "water", need_water, offset, frame=frame
            )
            ate, food_found, food_blocked = self._consume_if_needed(
                "food",
                need_food,
                offset,
                frame=None if drank else frame,
            )
            if drank:
                self._stats["drinks"] += 1
            if ate:
                self._stats["meals"] += 1
            if water_blocked or food_blocked:
                reason = "input_blocked"
            else:
                reason = "consumed" if drank or ate else "levels_ok"
            return SustenanceResult(
                checked=True,
                drank=drank,
                ate=ate,
                water_detected=water_detected,
                hunger_detected=hunger_detected,
                water_needed=need_water,
                food_needed=need_food,
                water_found=water_found,
                food_found=food_found,
                inventory_attempts=inventory_attempts,
                reason=reason,
            )
        finally:
            if opened_by_service:
                self._input.press_key(self._inventory_key)
                time.sleep(self._close_wait)

    def _consume_if_needed(
        self,
        kind: str,
        needed: bool,
        offset: tuple[int, int],
        *,
        frame: np.ndarray | None = None,
    ) -> tuple[bool, bool, bool]:
        """Return consumed, found and input-blocked flags for one item."""
        if not needed:
            return False, True, False
        if frame is None:
            frame = self._capture.grab()
        region = self._shifted_region("inventory_items", offset)
        crop = self._crop_actual(frame, region)
        match = self._detector.find_item(crop, kind)
        self.last_item_scores[kind] = float(match.score)
        if match.score < self._detector.item_threshold:
            print(
                f"\n[BOT] {kind}: предмет не найден (score={match.score:.2f})",
                flush=True,
            )
            return False, False, False

        center_x, center_y = match.center
        rect = self._capture.rect
        screen_x = rect.left + region[0] + center_x
        screen_y = rect.top + region[1] + center_y
        if self._input.double_click_at(screen_x, screen_y, self._double_click_interval) is False:
            return False, True, True
        time.sleep(self._item_wait)
        print(
            f"\n[BOT] {'Выпил воду' if kind == 'water' else 'Поел'} (item score={match.score:.2f})",
            flush=True,
        )
        return True, True, False

    def _inventory_offset(self, frame: np.ndarray) -> tuple[int, int] | None:
        search = self._scaled_region("inventory_header_search")
        crop = self._crop_actual(frame, search)
        match = self._detector.inventory_header_match(crop)
        if match.score < self._detector.inventory_threshold:
            return None
        anchor = self._scaled_region("inventory_header_anchor")
        actual_x = search[0] + match.x
        actual_y = search[1] + match.y
        return actual_x - anchor[0], actual_y - anchor[1]

    def _scaled_region(self, name: str) -> list[int]:
        defaults = {
            "inventory_header_search": [1040, 25, 520, 125],
            "inventory_header_anchor": [1165, 72, 110, 34],
            "inventory_items": [1150, 105, 350, 555],
        }
        return self._capture.scaled_region(self._regions.get(name, defaults[name]))

    def _shifted_region(self, name: str, offset: tuple[int, int]) -> list[int]:
        x, y, width, height = self._scaled_region(name)
        return [x + offset[0], y + offset[1], width, height]

    @staticmethod
    def _crop_actual(frame: np.ndarray, region: list[int]) -> np.ndarray:
        x, y, width, height = region
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width))
        y = max(0, min(y, frame_height))
        width = max(0, min(width, frame_width - x))
        height = max(0, min(height, frame_height - y))
        return frame[y : y + height, x : x + width]
