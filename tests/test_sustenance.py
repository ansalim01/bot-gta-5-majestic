from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from bot.services.sustenance import (
    SustenanceDetector,
    SustenanceResult,
    SustenanceService,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "config" / "templates" / "sustenance"


def load_config():
    with open(ROOT / "config" / "sustenance.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class SustenanceDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = SustenanceDetector(load_config())

    def test_thirst_warning_rejects_normal_hud(self) -> None:
        alert = cv2.imread(str(ASSETS / "validation_hud_alert.png"))
        clear = cv2.imread(str(ASSETS / "validation_hud_clear.png"))
        self.assertGreaterEqual(
            self.detector.thirst_score(alert),
            self.detector.thirst_threshold,
        )
        self.assertLess(
            self.detector.thirst_score(clear),
            self.detector.thirst_threshold,
        )

    def test_finds_water_anywhere_in_inventory_grid(self) -> None:
        grid = cv2.imread(str(ASSETS / "validation_inventory_grid.png"))
        water = self.detector.find_item(grid, "water")
        food = self.detector.find_item(grid, "food")
        self.assertGreaterEqual(water.score, self.detector.item_threshold)
        self.assertGreaterEqual(food.score, self.detector.item_threshold)
        self.assertGreater(water.center[0], 250)
        self.assertGreater(food.center[1], 300)

    def test_detects_orange_hunger_tile_by_shape_and_color(self) -> None:
        hud = cv2.imread(str(ASSETS / "validation_hud_clear.png"))
        threshold = load_config()["sustenance"]["hunger_tile_score_threshold"]
        self.assertLess(
            self.detector.hunger_tile_score(hud),
            threshold,
        )
        cv2.rectangle(hud, (5, 40), (42, 77), (0, 160, 245), -1)
        cv2.line(hud, (15, 49), (15, 68), (255, 255, 255), 2)
        cv2.line(hud, (27, 49), (27, 68), (255, 255, 255), 2)
        self.assertGreaterEqual(
            self.detector.hunger_tile_score(hud),
            threshold,
        )

    def test_detects_blue_thirst_tile_without_matching_background(self) -> None:
        hud = np.zeros((180, 190, 3), dtype=np.uint8)
        # Blue tile colour taken from the real Majestic warning, but place it
        # on a completely different background so template matching cannot be
        # the reason this passes.
        cv2.rectangle(hud, (80, 70), (116, 106), (150, 105, 70), -1)
        cv2.rectangle(hud, (94, 80), (102, 98), (245, 245, 245), 2)
        cv2.line(hud, (95, 86), (101, 86), (245, 245, 245), 2)
        self.assertGreaterEqual(
            self.detector.thirst_score(hud),
            self.detector.thirst_threshold,
        )


class SustenanceServiceTests(unittest.TestCase):
    @staticmethod
    def _inventory_frame():
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        header = cv2.imread(str(ASSETS / "inventory_header.png"))
        grid = cv2.imread(str(ASSETS / "validation_inventory_grid.png"))
        stats = cv2.imread(str(ASSETS / "validation_stats.png"))
        frame[72:106, 1165:1275] = header
        frame[105:660, 1150:1500] = grid
        frame[497:548, 814:1096] = stats
        return frame

    def test_urgent_thirst_double_clicks_water(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "inventory_open_wait_sec": 0,
                "after_item_wait_sec": 0,
                "inventory_close_wait_sec": 0,
            }
        )
        alert = cv2.imread(str(ASSETS / "validation_hud_alert.png"))
        full = self._inventory_frame()

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)

            @staticmethod
            def grab_region(_region):
                return alert.copy()

            @staticmethod
            def grab():
                return full.copy()

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.keys = []
                self.double_clicks = []

            def press_key(self, key):
                self.keys.append(key)

            def double_click_at(self, x, y, interval):
                self.double_clicks.append((x, y, interval))

        input_handler = Input()
        service = SustenanceService(config, Capture(), input_handler)
        result = service.tick()
        self.assertTrue(result.drank)
        self.assertFalse(result.ate)
        self.assertEqual(len(input_handler.double_clicks), 1)
        x, y, _interval = input_handler.double_clicks[0]
        self.assertTrue(1430 <= x <= 1490)
        self.assertTrue(120 <= y <= 220)

    def test_no_hud_warning_never_opens_inventory(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "hud_check_interval_sec": 60,
            }
        )
        clear = cv2.imread(str(ASSETS / "validation_hud_clear.png"))

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)
            hud_grabs = 0

            @classmethod
            def grab_region(cls, _region):
                cls.hud_grabs += 1
                return clear.copy()

            @staticmethod
            def grab():
                raise AssertionError("inventory must not be captured")

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.keys = []

            def press_key(self, key):
                self.keys.append(key)

            @staticmethod
            def double_click_at(_x, _y, _interval):
                raise AssertionError("nothing should be clicked")

        input_handler = Input()
        service = SustenanceService(config, Capture(), input_handler)
        first = service.tick()
        second = service.tick()
        self.assertEqual(first.reason, "no_need")
        self.assertEqual(second.reason, "hud_wait")
        self.assertEqual(Capture.hud_grabs, 1)
        self.assertEqual(input_handler.keys, [])

    def test_detect_only_reports_needs_without_opening_inventory(self) -> None:
        config = deepcopy(load_config())
        alert = cv2.imread(str(ASSETS / "validation_hud_clear.png"))
        cv2.rectangle(alert, (5, 40), (42, 77), (0, 160, 245), -1)
        cv2.line(alert, (15, 49), (15, 68), (255, 255, 255), 2)
        cv2.line(alert, (27, 49), (27, 68), (255, 255, 255), 2)

        class Capture:
            @staticmethod
            def grab_region(_region):
                return alert.copy()

            @staticmethod
            def grab():
                raise AssertionError("detect-only check must not capture inventory")

        class Input:
            @staticmethod
            def press_key(_key):
                raise AssertionError("detect-only check must not press a key")

        service = SustenanceService(config, Capture(), Input())
        result = service.check_needs(force=True)

        self.assertTrue(result.checked)
        self.assertFalse(result.water_needed)
        self.assertTrue(result.food_needed)
        self.assertEqual(result.reason, "need")

    def test_hunger_warning_does_not_open_inventory_until_food_is_configured(
        self,
    ) -> None:
        config = deepcopy(load_config())
        config["sustenance"]["food_enabled"] = False
        hud = cv2.imread(str(ASSETS / "validation_hud_clear.png"))
        cv2.rectangle(hud, (5, 40), (42, 77), (0, 160, 245), -1)
        cv2.line(hud, (15, 49), (15, 68), (255, 255, 255), 2)
        cv2.line(hud, (27, 49), (27, 68), (255, 255, 255), 2)

        class Capture:
            @staticmethod
            def grab_region(_region):
                return hud.copy()

            @staticmethod
            def grab():
                raise AssertionError("food is disabled; inventory must stay closed")

        class Input:
            @staticmethod
            def press_key(_key):
                raise AssertionError("I must not be pressed")

        service = SustenanceService(config, Capture(), Input())
        result = service.tick()
        self.assertTrue(result.hunger_detected)
        self.assertEqual(result.reason, "food_not_configured")

    def test_hunger_warning_double_clicks_food(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "inventory_open_wait_sec": 0,
                "after_item_wait_sec": 0,
                "inventory_close_wait_sec": 0,
            }
        )
        hud = cv2.imread(str(ASSETS / "validation_hud_clear.png"))
        cv2.rectangle(hud, (5, 40), (42, 77), (0, 160, 245), -1)
        cv2.line(hud, (15, 49), (15, 68), (255, 255, 255), 2)
        cv2.line(hud, (27, 49), (27, 68), (255, 255, 255), 2)
        full = self._inventory_frame()

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)

            @staticmethod
            def grab_region(_region):
                return hud.copy()

            @staticmethod
            def grab():
                return full.copy()

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.double_clicks = []

            @staticmethod
            def press_key(_key):
                pass

            def double_click_at(self, x, y, interval):
                self.double_clicks.append((x, y, interval))

        input_handler = Input()
        service = SustenanceService(config, Capture(), input_handler)
        result = service.tick()
        self.assertTrue(result.ate)
        self.assertFalse(result.drank)
        self.assertEqual(len(input_handler.double_clicks), 1)
        x, y, _interval = input_handler.double_clicks[0]
        self.assertTrue(1370 <= x <= 1490)
        self.assertTrue(420 <= y <= 530)

    def test_combined_hunger_and_thirst_consumes_both_items(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "inventory_open_wait_sec": 0,
                "after_item_wait_sec": 0,
                "inventory_close_wait_sec": 0,
            }
        )
        thirst = cv2.imread(str(ASSETS / "validation_hud_alert.png"))
        hud = np.zeros((180, 190, 3), dtype=np.uint8)
        thirst_height, thirst_width = thirst.shape[:2]
        hud[:thirst_height, :thirst_width] = thirst
        cv2.rectangle(hud, (120, 40), (157, 77), (0, 160, 245), -1)
        cv2.line(hud, (130, 49), (130, 68), (255, 255, 255), 2)
        cv2.line(hud, (142, 49), (142, 68), (255, 255, 255), 2)
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        inventory = self._inventory_frame()
        frames = [blank, inventory, inventory, inventory]

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)

            @staticmethod
            def grab_region(_region):
                return hud.copy()

            @staticmethod
            def grab():
                return frames.pop(0).copy()

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.keys = []
                self.double_clicks = []

            def press_key(self, key):
                self.keys.append(key)

            def double_click_at(self, x, y, interval):
                self.double_clicks.append((x, y, interval))

        input_handler = Input()
        service = SustenanceService(config, Capture(), input_handler)
        result = service.tick()

        self.assertTrue(result.drank)
        self.assertTrue(result.ate)
        self.assertTrue(result.water_found)
        self.assertTrue(result.food_found)
        self.assertEqual(["i", "i"], input_handler.keys)
        self.assertEqual(2, len(input_handler.double_clicks))
        self.assertEqual(1, service.stats["drinks"])
        self.assertEqual(1, service.stats["meals"])

    def test_inventory_open_is_retried_until_header_is_confirmed(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "inventory_open_wait_sec": 0,
                "inventory_open_attempts": 3,
                "inventory_open_retry_sec": 0,
                "after_item_wait_sec": 0,
                "inventory_close_wait_sec": 0,
            }
        )
        alert = cv2.imread(str(ASSETS / "validation_hud_alert.png"))
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        inventory = self._inventory_frame()
        frames = [blank, blank, blank, blank, blank, inventory, inventory]

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)

            @staticmethod
            def grab_region(_region):
                return alert.copy()

            @staticmethod
            def grab():
                return frames.pop(0).copy()

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.keys = []
                self.double_clicks = []

            def press_key(self, key):
                self.keys.append(key)

            def double_click_at(self, x, y, interval):
                self.double_clicks.append((x, y, interval))

        inputs = Input()
        service = SustenanceService(config, Capture(), inputs)
        result = service.tick()

        self.assertTrue(result.drank)
        self.assertEqual(3, result.inventory_attempts)
        self.assertEqual(["i", "i", "i", "i"], inputs.keys)
        self.assertEqual(1, len(inputs.double_clicks))

    def test_confirmed_inventory_frame_is_reused_for_item_search(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "after_item_wait_sec": 0,
                "inventory_close_wait_sec": 0,
            }
        )
        inventory = self._inventory_frame()
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)
            grabs = 0

            @classmethod
            def grab(cls):
                cls.grabs += 1
                return inventory.copy() if cls.grabs == 1 else blank.copy()

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.double_clicks = []

            @staticmethod
            def press_key(_key):
                pass

            def double_click_at(self, x, y, interval):
                self.double_clicks.append((x, y, interval))

        inputs = Input()
        service = SustenanceService(config, Capture(), inputs)
        result = service.consume_detected_needs(
            SustenanceResult(
                checked=True,
                water_detected=True,
                water_needed=True,
                reason="need",
            )
        )

        self.assertTrue(result.drank)
        self.assertEqual(1, Capture.grabs)
        self.assertEqual(1, len(inputs.double_clicks))
        self.assertGreaterEqual(
            service.last_item_scores["water"],
            service._detector.item_threshold,
        )

    def test_inventory_failure_has_no_blind_final_toggle(self) -> None:
        config = deepcopy(load_config())
        config["sustenance"].update(
            {
                "inventory_open_wait_sec": 0,
                "inventory_open_attempts": 3,
                "inventory_open_retry_sec": 0,
            }
        )
        alert = cv2.imread(str(ASSETS / "validation_hud_alert.png"))
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class Capture:
            rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)

            @staticmethod
            def grab_region(_region):
                return alert.copy()

            @staticmethod
            def grab():
                return blank.copy()

            @staticmethod
            def scaled_region(region):
                return list(region)

        class Input:
            def __init__(self):
                self.keys = []

            def press_key(self, key):
                self.keys.append(key)

            @staticmethod
            def double_click_at(_x, _y, _interval):
                raise AssertionError("nothing should be clicked")

        inputs = Input()
        service = SustenanceService(config, Capture(), inputs)
        result = service.tick()

        self.assertEqual("inventory_not_open", result.reason)
        self.assertEqual(3, result.inventory_attempts)
        self.assertEqual(["i", "i", "i"], inputs.keys)


if __name__ == "__main__":
    unittest.main()
