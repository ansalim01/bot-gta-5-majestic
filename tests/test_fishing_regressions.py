from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main_fishing
from bot.config_loader import FISHING_CONFIG_PATH, TEMPLATES_DIR, load_config
from bot.detectors.fish_fight import FishFightDetector
from bot.detectors.fish_phase import FishPhase, FishPhaseDetector
from bot.detectors.fish_prompt import FishPromptDetector
from bot.input_handler import InputHandler
from bot.jobs.fisher import FisherJob
from bot.services.sustenance import SustenanceResult
from bot.state_machine import BotState, StateMachine
from bot.telemetry.fishing_session import FishingSessionTelemetry


class FishingDetectorRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(FISHING_CONFIG_PATH)

    def test_persistent_hook_control_is_waiting_not_bite(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        template = cv2.imread(str(TEMPLATES_DIR / "fishing" / "wait_instruction_current.png"))
        self.assertIsNotNone(template)
        assert template is not None
        th, tw = template.shape[:2]
        frame[1022 : 1022 + th, 720 : 720 + tw] = template

        # The current HUD shows a bright bottom-right fishing control during
        # the whole wait. It must not turn the phase into BITE.
        frame[1030:1065, 1740:1905] = 255
        phase = FishPhaseDetector(self.config).detect(frame)
        self.assertEqual(FishPhase.WAITING, phase)

    def test_horizontal_bar_recovers_reeling_without_icon(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        x0 = int(frame.shape[1] * 0.356)
        y0 = int(frame.shape[0] * 0.82)
        frame[y0 : y0 + 10, x0 + 2 : x0 + 302] = (0, 255, 255)

        phase = FishPhaseDetector(self.config).detect(frame)
        self.assertEqual(FishPhase.REELING, phase)

    def test_short_scene_colored_blob_is_not_a_fight_bar(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        x0 = int(frame.shape[1] * 0.356)
        y0 = int(frame.shape[0] * 0.79)
        # Models a yellow/green rock crossing the old narrow stub detector.
        frame[y0 : y0 + 35, x0 : x0 + 120] = (0, 255, 255)

        fight = FishFightDetector(self.config)
        progress, yellow = fight._fill_state(frame)
        self.assertIsNone(progress)
        self.assertFalse(yellow)
        self.assertEqual(FishPhase.UNKNOWN, FishPhaseDetector(self.config).detect(frame))

    def test_bite_threshold_rejects_recorded_false_positive_band(self) -> None:
        prompt = self.config["prompt"]
        self.assertGreater(float(prompt["bite_text_threshold"]), 0.59)
        self.assertGreater(float(prompt["bite_absolute_threshold"]), 0.59)
        self.assertEqual(0.60, float(prompt["bite_text_threshold"]))
        self.assertEqual(1.0, float(self.config["fishing"]["min_bite_delay_sec"]))
        self.assertEqual(2, int(self.config["fishing"]["bite_confirm_frames"]))


class FishingControlRegressionTests(unittest.TestCase):
    def test_missing_timing_bar_stops_space_after_three_attempts(self) -> None:
        state = StateMachine()
        state.toggle()

        class MissingTiming:
            _appear_timeout = 0.0
            poll_sec = 0.0

            @staticmethod
            def reset():
                return None

            @staticmethod
            def is_active(_frame):
                return False

        class FakeInput:
            def __init__(self):
                self.keys = []
                self.release_count = 0

            def press_key_named(self, key):
                self.keys.append(key)

            def release_all_keys(self):
                self.release_count += 1

        class FakeTelemetry:
            def __init__(self):
                self.failures = []

            @staticmethod
            def event(_name, **_data):
                return None

            def failure(self, reason, **_data):
                self.failures.append(reason)

        inputs = FakeInput()
        telemetry = FakeTelemetry()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._input = inputs
        job._telemetry = telemetry
        job._timing = MissingTiming()
        job._grab_timing = lambda: np.zeros((75, 600, 3), dtype=np.uint8)
        job._sustenance_enabled = False
        job._sustenance_checked_for_cast = False
        job._timing_session = False
        job._timing_no_slider_streak = 0
        job._auto_cast = True
        job._cast_key = "space"
        job._cast_pressed = False
        job._cast_start_attempts = 0
        job._cast_start_max_attempts = 3
        job._cast_start_retry_delay = 0.0
        job._setup_enabled = True
        job._setup_complete = True
        job._save_screenshots = True
        job._save_debug_screenshot = MagicMock(return_value=True)
        job._debug = False
        job._stats = {"timing_hits": 0, "casts": 0}

        with patch("bot.jobs.fisher.time.sleep"):
            for _attempt in range(3):
                state.state = BotState.CAST_TIMING
                job._timing_session = False
                job._tick_cast_timing()

        self.assertEqual(["space", "space", "space"], inputs.keys)
        self.assertEqual(1, inputs.release_count)
        self.assertFalse(job._setup_complete)
        self.assertTrue(job._setup_recovery_passive)
        self.assertEqual(BotState.IDLE, state.state)
        self.assertFalse(state.running)
        job._save_debug_screenshot.assert_called_once_with("cast_start_retry_limit")
        self.assertIn("cast_start_retry_limit_reached", telemetry.failures)

    def test_missing_catch_modal_is_unidentified_catch_and_continues(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.REELING

        class FakeCapture:
            @staticmethod
            def grab_region(_region):
                return np.zeros((10, 10, 3), dtype=np.uint8)

            @staticmethod
            def grab():
                return np.zeros((1080, 1920, 3), dtype=np.uint8)

            @staticmethod
            def crop_from_frame(frame, _region):
                return frame

        class FakeModal:
            @staticmethod
            def is_visible(_frame):
                return False

        class FakeFight:
            last_solve_reason = "ui_gone"
            timeout_sec = 120.0

            @staticmethod
            def solve(*_args, **_kwargs):
                return True

        class FakeInput:
            def __init__(self):
                self.release_count = 0

            @staticmethod
            def set_reel_direction(_direction):
                return None

            def release_all_keys(self):
                self.release_count += 1

        class FakeTelemetry:
            def __init__(self):
                self.failures = []
                self.events = []

            def event(self, name, **data):
                self.events.append((name, data))

            def failure(self, reason, **_data):
                self.failures.append(reason)

        inputs = FakeInput()
        telemetry = FakeTelemetry()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._modal = FakeModal()
        job._fight = FakeFight()
        job._input = inputs
        job._telemetry = telemetry
        job._modal_region = [0, 0, 10, 10]
        job._fight_bar_region = [0, 0, 10, 10]
        job._prompt_region = [0, 0, 10, 10]
        job._reel_modal_wait = 0.0
        job._debug = False
        job._save_screenshots = True
        job._save_debug_screenshot = MagicMock(return_value=True)
        job._cast_pressed = True
        job._timing_session = False
        job._sustenance_checked_for_cast = True
        job._normal_cycle_ready_at = 0.0
        job._setup_enabled = True
        job._setup_complete = True
        job._timing_active_in = lambda _frame: False
        job._stats = {"unidentified_catches": 0}
        job._auto_cast = True
        job._mode = "cast_and_reel"
        job._cast_start_attempts = 2

        job._tick_reeling()

        self.assertEqual(BotState.CASTING, state.state)
        self.assertTrue(state.running)
        self.assertEqual(0, inputs.release_count)
        self.assertEqual([], telemetry.failures)
        self.assertEqual(1, job._stats["unidentified_catches"])
        self.assertIn(
            "catch_without_modal",
            [name for name, _data in telemetry.events],
        )
        job._save_debug_screenshot.assert_called_once()
        self.assertEqual(
            "catch_without_modal",
            job._save_debug_screenshot.call_args.args[0],
        )

    def test_missing_catch_modal_with_next_cast_bar_continues(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.REELING

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class FakeCapture:
            @staticmethod
            def grab_region(_region):
                return np.zeros((10, 10, 3), dtype=np.uint8)

            @staticmethod
            def grab():
                return frame

            @staticmethod
            def crop_from_frame(source, _region):
                return source

        class FakeModal:
            @staticmethod
            def is_visible(_frame):
                return False

        class FakeFight:
            last_solve_reason = "ui_gone"
            timeout_sec = 120.0

            @staticmethod
            def solve(*_args, **_kwargs):
                return True

        class FakeInput:
            def __init__(self):
                self.release_count = 0

            @staticmethod
            def set_reel_direction(_direction):
                return None

            def release_all_keys(self):
                self.release_count += 1

        class FakeTelemetry:
            def __init__(self):
                self.failures = []
                self.events = []

            def event(self, name, **data):
                self.events.append((name, data))

            def failure(self, reason, **_data):
                self.failures.append(reason)

        inputs = FakeInput()
        telemetry = FakeTelemetry()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._modal = FakeModal()
        job._fight = FakeFight()
        job._input = inputs
        job._telemetry = telemetry
        job._modal_region = [0, 0, 10, 10]
        job._fight_bar_region = [0, 0, 10, 10]
        job._prompt_region = [0, 0, 10, 10]
        job._reel_modal_wait = 1.0
        job._debug = False
        job._save_screenshots = True
        job._save_debug_screenshot = MagicMock(return_value=True)
        job._cast_pressed = True
        job._cast_start_attempts = 2
        job._timing_session = False
        job._timing_ignore_until = 999.0
        job._sustenance_checked_for_cast = True
        job._normal_cycle_ready_at = 0.0
        job._setup_enabled = True
        job._setup_complete = True
        job._timing_active_in = lambda _frame: True

        job._tick_reeling()

        self.assertEqual(BotState.CAST_TIMING, state.state)
        self.assertTrue(state.running)
        self.assertEqual(0, inputs.release_count)
        self.assertFalse(job._cast_pressed)
        self.assertEqual(0, job._cast_start_attempts)
        self.assertFalse(job._sustenance_checked_for_cast)
        self.assertEqual(0.0, job._timing_ignore_until)
        self.assertIn("fish_escaped", telemetry.failures)
        self.assertNotIn("reel_modal_missing", telemetry.failures)
        self.assertEqual(
            "fish_escaped",
            job._save_debug_screenshot.call_args.args[0],
        )

    def test_pending_sustenance_survives_false_no_need_recheck(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.FISHING_SUSTENANCE
        pending = SustenanceResult(
            checked=True,
            water_detected=True,
            water_needed=True,
            reason="need",
        )

        class FakeService:
            last_thirst_score = 0.4
            last_hunger_score = 0.0

            def __init__(self):
                self.consumed = []

            @staticmethod
            def action_block_reason():
                return None

            @staticmethod
            def check_needs(*, force=False):
                return SustenanceResult(checked=True, reason="no_need")

            def consume_detected_needs(self, needs, *, allow_action=True):
                self.consumed.append(needs)
                return SustenanceResult(
                    checked=True,
                    drank=True,
                    water_detected=True,
                    water_needed=True,
                    reason="consumed",
                )

        service = FakeService()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._sustenance = service
        job._sustenance_pending_needs = pending
        job._sustenance_retry_at = 0.0
        job._sustenance_last_deferred_reason = ""
        job._sustenance_failure_count = 0
        job._sustenance_max_failures = 3
        job._sustenance_after_action = 0.0
        job._stats = {"drinks": 0, "meals": 0}
        job._telemetry = None
        job._exit_fishing_for_sustenance = MagicMock(return_value=True)
        job._finish_sustenance_cycle = MagicMock()

        job._tick_sustenance_pending()

        self.assertEqual(1, len(service.consumed))
        self.assertTrue(service.consumed[0].water_needed)
        self.assertEqual(1, job._stats["drinks"])
        job._finish_sustenance_cycle.assert_called_once()

    def test_sustenance_failure_limit_requests_safe_pause(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.FISHING_SUSTENANCE
        pending = SustenanceResult(
            checked=True,
            water_detected=True,
            water_needed=True,
            reason="need",
        )

        class FakeService:
            @staticmethod
            def consume_detected_needs(_needs, *, allow_action=True):
                return SustenanceResult(
                    checked=False,
                    water_detected=True,
                    water_needed=True,
                    water_found=False,
                    food_found=False,
                    inventory_attempts=3,
                    reason="inventory_not_open",
                )

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._sustenance = FakeService()
        job._sustenance_pending_needs = pending
        job._sustenance_failure_count = 2
        job._sustenance_max_failures = 3
        job._stats = {"drinks": 0, "meals": 0}
        job._telemetry = None
        job._exit_fishing_for_sustenance = MagicMock(return_value=True)
        job._pause_for_manual_recovery = MagicMock()

        job._attempt_pending_sustenance(pending)

        job._pause_for_manual_recovery.assert_called_once_with("sustenance_retry_limit_reached")
        self.assertTrue(job._sustenance_pending_needs.water_needed)

    def test_waiting_bite_ignores_completed_timing_bar_during_grace(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.WAITING_BITE
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        class FakeCapture:
            @staticmethod
            def grab_union(_regions):
                return frame

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._bite_poll_regions = []
        job._telemetry = None
        job._debug = False
        job._bite_timeout = 0.0
        job._bite_wait_start = time.perf_counter()
        job._caught_modal_visible_in = lambda _frame: False
        timing_probe = MagicMock(return_value=True)
        job._timing_active_in = timing_probe
        job._should_hook = lambda _frame: False
        job._timing_ignore_until = time.perf_counter() + 10.0

        job._tick_waiting_bite()

        self.assertEqual(BotState.WAITING_BITE, state.state)
        timing_probe.assert_not_called()

        job._timing_ignore_until = 0.0
        job._tick_waiting_bite()

        self.assertEqual(BotState.CAST_TIMING, state.state)
        timing_probe.assert_called_once_with(frame)

    def test_daylight_bite_triggers_after_two_confirmed_frames(self) -> None:
        class Scores:
            bite = 0.66
            keyword = 0.66
            wait = 0.35
            waiting = False

        class FakePrompt:
            @staticmethod
            def text_scores(_frame):
                return Scores()

        job = FisherJob.__new__(FisherJob)
        job._bite_wait_start = time.perf_counter() - 20.0
        job._min_bite_delay = 5.0
        job._baseline_ready = True
        job._baseline_text = 0.42
        job._text_delta_min = 0.10
        job._bite_text_threshold = 0.60
        job._bite_absolute_threshold = 0.72
        job._bite_keyword_threshold = 0.72
        job._bite_confirm_frames = 2
        job._bite_streak = 0
        job._hook_trigger = "bite_text"
        job._debug = False
        job._prompt = FakePrompt()
        job._bite_text_region = [0, 0, 10, 10]
        job._crop = lambda frame, _region: frame
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        self.assertFalse(job._should_hook(frame))
        self.assertTrue(job._should_hook(frame))

    def test_early_daylight_like_prompt_residue_is_ignored(self) -> None:
        job = FisherJob.__new__(FisherJob)
        job._bite_wait_start = time.perf_counter() - 0.5
        job._min_bite_delay = 1.0
        job._last_bite_block_reason = ""

        self.assertFalse(job._should_hook(np.zeros((10, 10, 3), dtype=np.uint8)))
        self.assertEqual("min_delay", job._last_bite_block_reason)

    def test_show_overlay_false_skips_overlay_capture_in_hot_states(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.WAITING_BITE

        class FakeInput:
            @staticmethod
            def is_game_focused():
                return True

            @staticmethod
            def release_all_keys():
                return None

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._input = FakeInput()
        job._telemetry = None
        job._paused_cleanup_done = False
        job._input_focus_blocked = False
        job._sustenance_pending_needs = None
        job._setup_enabled = False
        job._setup_complete = True
        job._debug = True
        job._show_overlay = False
        job._update_fps = lambda: None
        job._tick_waiting_bite = lambda: None
        job._show_debug_overlay = MagicMock()

        job.tick()

        job._show_debug_overlay.assert_not_called()

    def test_debug_screenshots_are_cropped_compressed_and_limited(self) -> None:
        class FakeTelemetry:
            session_id = "test_session"

            @staticmethod
            def event(_name, **_data):
                return None

        job = FisherJob.__new__(FisherJob)
        job._config = {"screen": {"resolution": [1920, 1080]}}
        job._telemetry = FakeTelemetry()
        job._bite_text_region = [645, 980, 650, 95]
        job._bobber_region = [1280, 745, 380, 300]
        job._hook_prompt_region = [1730, 1025, 190, 55]
        job._fight_bar_region = [660, 860, 600, 75]
        job._fish_icon_region = [860, 360, 200, 180]
        job._modal_region = [700, 290, 520, 420]
        job._timing_region = [660, 860, 600, 75]
        job._screenshot_format = "jpg"
        job._screenshot_quality = 82
        job._screenshot_max_per_session = 3
        job._screenshot_max_per_tag = 1
        job._screenshot_count = 0
        job._screenshot_tag_counts = {}
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch("bot.jobs.fisher.cv2.imwrite", return_value=True) as write:
            self.assertTrue(job._save_debug_screenshot("bite_detected", frame))
            self.assertFalse(job._save_debug_screenshot("bite_detected", frame))

        write.assert_called_once()
        path, written_frame, options = write.call_args.args
        self.assertTrue(path.endswith(".jpg"))
        self.assertIn("fishing_screenshots", path)
        self.assertLess(written_frame.shape[0] * written_frame.shape[1], 1080 * 1920)
        self.assertEqual([cv2.IMWRITE_JPEG_QUALITY, 82], options)

    def test_input_is_blocked_when_foreground_window_is_not_gta(self) -> None:
        handler = InputHandler(
            {
                "screen": {
                    "require_foreground_for_input": True,
                    "window_titles": ["Grand Theft Auto V", "Majestic"],
                    "process_names": ["GTA5", "RAGEMP"],
                }
            }
        )
        with (
            patch("bot.input_handler.sys.platform", "win32"),
            patch.object(
                handler,
                "_foreground_window_identity",
                return_value=("notepad", "notepad"),
            ),
            patch("bot.input_handler.pydirectinput.keyDown") as key_down,
            patch("bot.input_handler.pydirectinput.keyUp") as key_up,
        ):
            self.assertFalse(handler.press_key("space", duration=0))
        key_down.assert_not_called()
        key_up.assert_not_called()

    def test_matching_title_does_not_allow_a_non_game_process(self) -> None:
        handler = InputHandler(
            {
                "screen": {
                    "require_foreground_for_input": True,
                    "window_titles": ["Majestic"],
                    "process_names": ["GTA5", "RAGEMP"],
                }
            }
        )
        with (
            patch("bot.input_handler.sys.platform", "win32"),
            patch.object(
                handler,
                "_foreground_window_identity",
                return_value=("majestic launcher", "majesticlauncher"),
            ),
            patch("bot.input_handler.pydirectinput.keyDown") as key_down,
        ):
            self.assertFalse(handler.press_key("space", duration=0))
        key_down.assert_not_called()

    def test_input_is_allowed_for_titleless_gta_process(self) -> None:
        handler = InputHandler(
            {
                "screen": {
                    "require_foreground_for_input": True,
                    "window_titles": ["Grand Theft Auto V"],
                    "process_names": ["GTA5"],
                }
            }
        )
        with (
            patch("bot.input_handler.sys.platform", "win32"),
            patch.object(
                handler,
                "_foreground_window_identity",
                return_value=("", "gta5"),
            ),
            patch("bot.input_handler.pydirectinput.keyDown") as key_down,
            patch("bot.input_handler.pydirectinput.keyUp") as key_up,
        ):
            self.assertTrue(handler.press_key("space", duration=0))
        key_down.assert_called_once_with("space")
        key_up.assert_called_once_with("space")

    def test_input_can_restore_game_focus_before_sending_key(self) -> None:
        handler = InputHandler(
            {
                "screen": {
                    "require_foreground_for_input": True,
                    "auto_focus_game_for_input": True,
                    "window_titles": ["Grand Theft Auto V"],
                    "process_names": ["GTA5"],
                }
            }
        )
        with (
            patch("bot.input_handler.sys.platform", "win32"),
            patch.object(handler, "is_game_focused", return_value=False),
            patch.object(handler, "_focus_game_window", return_value=True) as focus,
            patch("bot.input_handler.pydirectinput.keyDown") as key_down,
            patch("bot.input_handler.pydirectinput.keyUp") as key_up,
        ):
            self.assertTrue(handler.press_key("space", duration=0))

        focus.assert_called_once_with()
        key_down.assert_called_once_with("space")
        key_up.assert_called_once_with("space")

    def test_short_failed_session_with_no_cast_is_not_reported_as_ok(self) -> None:
        telemetry = FishingSessionTelemetry({"telemetry": {"enabled": False}})
        telemetry.failure("setup_start_prompt_timeout")

        conclusions = telemetry._build_conclusions(
            {"casts": 0, "hooks": 0, "fish_caught": 0},
            duration=2.0,
            performance={},
        )

        titles = {item["title"] for item in conclusions}
        self.assertIn("За сессию не выполнено ни одного заброса", titles)
        self.assertNotIn("Критических ошибок не обнаружено", titles)

    def test_f9_waiting_state_advances_to_automatic_cast(self) -> None:
        config = load_config(FISHING_CONFIG_PATH)
        self.assertTrue(config["fishing"]["auto_cast"])

        state = StateMachine()
        state.toggle()
        self.assertEqual(BotState.WAITING, state.state)

        class FakeCapture:
            @staticmethod
            def grab():
                return np.zeros((1080, 1920, 3), dtype=np.uint8)

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._auto_cast = True
        job._mode = "cast_and_reel"
        job._cast_pressed = False
        job._timing_active_in = lambda _frame: False
        job._detect_phase = lambda _frame: FishPhase.UNKNOWN

        job._tick_waiting()
        self.assertEqual(BotState.CASTING, state.state)

    def test_timeout_does_not_abandon_a_visible_waiting_session(self) -> None:
        config = load_config(FISHING_CONFIG_PATH)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        template = cv2.imread(str(TEMPLATES_DIR / "fishing" / "wait_instruction_current.png"))
        assert template is not None
        th, tw = template.shape[:2]
        frame[1022 : 1022 + th, 720 : 720 + tw] = template
        frame[1030:1065, 1740:1905] = 255

        state = StateMachine()
        state.toggle()
        state.state = BotState.WAITING_BITE

        class FakeCapture:
            @staticmethod
            def grab_union(_regions):
                return frame

            @staticmethod
            def crop_from_frame(source, region):
                x, y, w, h = region
                return source[y : y + h, x : x + w]

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._bite_poll_regions = []
        job._bite_text_region = config["regions"]["bite_text"]
        job._hook_prompt_region = config["regions"]["hook_prompt"]
        job._prompt = FishPromptDetector(config)
        job._debug = False
        job._save_screenshots = False
        job._bite_timeout = 35.0
        job._bite_wait_start = time.perf_counter() - 36.0
        job._caught_modal_visible_in = lambda _frame: False
        job._timing_active_in = lambda _frame: False
        job._should_hook = lambda _frame: False

        old_start = job._bite_wait_start
        job._tick_waiting_bite()
        self.assertEqual(BotState.WAITING_BITE, state.state)
        self.assertGreater(job._bite_wait_start, old_start)

    def test_release_all_keys_includes_fishing_inputs_and_mouse(self) -> None:
        handler = InputHandler({})
        with (
            patch("bot.input_handler.pydirectinput.keyUp") as key_up,
            patch("bot.input_handler.pydirectinput.mouseUp") as mouse_up,
        ):
            handler.release_all_keys()

        released = {call.args[0] for call in key_up.call_args_list}
        self.assertIn("space", released)
        self.assertIn("enter", released)
        self.assertIn("a", released)
        self.assertIn("d", released)
        mouse_up.assert_called_once()

    def test_admin_process_does_not_relaunch(self) -> None:
        with (
            patch("main_fishing._is_admin", return_value=True),
            patch("main_fishing._request_admin_relaunch") as relaunch,
        ):
            self.assertTrue(main_fishing._ensure_input_privileges())
        relaunch.assert_not_called()

    def test_non_admin_process_relaunches_and_exits(self) -> None:
        with (
            patch("main_fishing._is_admin", return_value=False),
            patch("main_fishing._request_admin_relaunch", return_value=True) as relaunch,
        ):
            self.assertFalse(main_fishing._ensure_input_privileges())
        relaunch.assert_called_once()

    def test_f10_state_is_not_overwritten_after_reel_solver_returns(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.REELING

        class FakeCapture:
            def grab_region(self, _region):
                return np.zeros((10, 10, 3), dtype=np.uint8)

        class FakeModal:
            @staticmethod
            def is_visible(_frame):
                return False

        class FakeInput:
            @staticmethod
            def set_reel_direction(_direction):
                return None

        class FakeFight:
            last_solve_reason = "aborted"
            timeout_sec = 120.0

            @staticmethod
            def solve(*_args, **_kwargs):
                state.stop()
                return False

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._modal = FakeModal()
        job._input = FakeInput()
        job._fight = FakeFight()
        job._modal_region = [0, 0, 10, 10]
        job._debug = False
        job._save_screenshots = False

        job._tick_reeling()
        self.assertEqual(BotState.STOPPED, state.state)
        self.assertFalse(state.running)

    def test_reel_timeout_rechecks_phase_before_another_cast(self) -> None:
        state = StateMachine()
        state.toggle()
        state.state = BotState.REELING

        class FakeCapture:
            @staticmethod
            def grab_region(_region):
                return np.zeros((10, 10, 3), dtype=np.uint8)

            @staticmethod
            def grab():
                return np.zeros((1080, 1920, 3), dtype=np.uint8)

            @staticmethod
            def crop_from_frame(frame, _region):
                return frame

        class FakeModal:
            @staticmethod
            def is_visible(_frame):
                return False

        class FakeInput:
            @staticmethod
            def set_reel_direction(_direction):
                return None

        class FakeFight:
            last_solve_reason = "timeout"
            timeout_sec = 120.0

            @staticmethod
            def solve(*_args, **_kwargs):
                return False

        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._modal = FakeModal()
        job._input = FakeInput()
        job._fight = FakeFight()
        job._modal_region = [0, 0, 10, 10]
        job._debug = False
        job._save_screenshots = False
        job._cast_pressed = True
        job._sustenance_checked_for_cast = True

        job._tick_reeling()

        self.assertEqual(BotState.WAITING, state.state)
        self.assertFalse(job._cast_pressed)
        self.assertFalse(job._sustenance_checked_for_cast)


if __name__ == "__main__":
    unittest.main()
