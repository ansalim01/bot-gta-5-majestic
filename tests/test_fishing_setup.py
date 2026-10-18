from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from bot.config_loader import FISHING_CONFIG_PATH, TEMPLATES_DIR, load_config
from bot.detectors.fish_phase import FishPhase
from bot.detectors.fishing_setup import FishingSetupDetector
from bot.jobs.fisher import FisherJob
from bot.services.sustenance import SustenanceResult
from bot.state_machine import BotState, StateMachine
from bot.telemetry.fishing_session import FishingSessionTelemetry

ROOT = Path(__file__).resolve().parent.parent


class FishingSetupDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(FISHING_CONFIG_PATH)
        cls.detector = FishingSetupDetector(cls.config)
        cls.start_template = cv2.imread(str(TEMPLATES_DIR / "fishing" / "start_fishing_prompt.png"))
        cls.equipment_template = cv2.imread(
            str(TEMPLATES_DIR / "fishing" / "equipment_ready_prompt.png")
        )
        assert cls.start_template is not None
        assert cls.equipment_template is not None

    @staticmethod
    def blank_frame() -> np.ndarray:
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def start_frame(self) -> np.ndarray:
        frame = self.blank_frame()
        height, width = self.start_template.shape[:2]
        frame[16 : 16 + height, 12 : 12 + width] = self.start_template
        return frame

    def equipment_frame(self) -> np.ndarray:
        frame = self.blank_frame()
        height, width = self.equipment_template.shape[:2]
        frame[1015 : 1015 + height, 1515 : 1515 + width] = self.equipment_template
        return frame

    def test_start_prompt_is_distinct_from_equipment_screen(self) -> None:
        scores = self.detector.scores(self.start_frame())
        self.assertGreaterEqual(scores.start, self.detector.start_threshold)
        self.assertLess(scores.equipment, self.detector.equipment_threshold)

    def test_start_prompt_ignores_translucent_background_change(self) -> None:
        frame = self.start_frame()
        height, width = self.start_template.shape[:2]
        prompt = frame[16 : 16 + height, 12 : 12 + width]
        gray = cv2.cvtColor(prompt, cv2.COLOR_BGR2GRAY)
        background = gray < 215
        prompt[background] = (125, 125, 125)

        scores = self.detector.scores(frame)

        self.assertGreaterEqual(scores.start, self.detector.start_threshold)

    def test_equipment_screen_is_distinct_from_start_prompt(self) -> None:
        scores = self.detector.scores(self.equipment_frame())
        self.assertGreaterEqual(
            scores.equipment,
            self.detector.equipment_threshold,
        )
        self.assertLess(scores.start, self.detector.start_threshold)

    def test_setup_presses_e_twice_then_enters_existing_cycle(self) -> None:
        frames = [self.start_frame(), self.equipment_frame()]

        class FakeCapture:
            @staticmethod
            def grab():
                return frames.pop(0)

        class FakeInput:
            def __init__(self) -> None:
                self.keys: list[str] = []

            def press_key_named(self, key: str) -> None:
                self.keys.append(key)

        state = StateMachine()
        state.toggle()
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_key = "e"
        job._setup_after_start = 0.0
        job._setup_after_equipment = 0.0
        job._setup_start_timeout = 30.0
        job._setup_equipment_timeout = 20.0
        job._setup_stage_started = 0.0
        job._setup_settle_until = 0.0
        job._normal_cycle_ready_at = 0.0
        job._detect_phase = lambda _frame: FishPhase.UNKNOWN

        job._tick_setup()
        self.assertEqual(["e"], inputs.keys)
        self.assertEqual(BotState.FISHING_EQUIPMENT, state.state)
        self.assertFalse(job._setup_complete)

        job._tick_setup()
        self.assertEqual(["e", "e"], inputs.keys)
        self.assertEqual(BotState.WAITING, state.state)
        self.assertTrue(job._setup_complete)

    def test_unrelated_phase_detection_does_not_skip_start_prompt(self) -> None:
        class FakeCapture:
            @staticmethod
            def grab():
                return np.zeros((1080, 1920, 3), dtype=np.uint8)

        class FakeInput:
            keys: list[str] = []

            def press_key_named(self, key: str) -> None:
                self.keys.append(key)

        state = StateMachine()
        state.toggle()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = FakeInput()
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_key = "e"
        job._setup_after_start = 0.0
        job._setup_after_equipment = 0.0
        job._setup_start_timeout = 30.0
        job._setup_equipment_timeout = 20.0
        job._setup_stage_started = time.perf_counter()
        job._setup_settle_until = 0.0
        job._normal_cycle_ready_at = 0.0
        job._detect_phase = lambda _frame: FishPhase.CAST_TIMING

        job._tick_setup()

        self.assertEqual([], job._input.keys)
        self.assertFalse(job._setup_complete)
        self.assertEqual(BotState.FISHING_START, state.state)

    def test_start_prompt_is_retried_until_equipment_screen_appears(self) -> None:
        frames = [self.start_frame(), self.start_frame(), self.equipment_frame()]

        class FakeCapture:
            @staticmethod
            def grab():
                return frames.pop(0)

        class FakeInput:
            def __init__(self) -> None:
                self.keys: list[str] = []

            def press_key_named(self, key: str) -> None:
                self.keys.append(key)

        state = StateMachine()
        state.toggle()
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_key = "e"
        job._setup_after_start = 0.0
        job._setup_after_equipment = 0.0
        job._setup_start_timeout = 30.0
        job._setup_equipment_timeout = 20.0
        job._setup_stage_started = 0.0
        job._setup_settle_until = 0.0
        job._normal_cycle_ready_at = 0.0

        job._tick_setup()
        job._tick_setup()
        self.assertEqual(["e", "e"], inputs.keys)
        self.assertFalse(job._setup_complete)
        self.assertEqual(BotState.FISHING_EQUIPMENT, state.state)

        job._tick_setup()
        self.assertEqual(["e", "e", "e"], inputs.keys)
        self.assertTrue(job._setup_complete)
        self.assertEqual(BotState.WAITING, state.state)

    def test_missing_start_prompt_steps_forward_then_back_before_e(self) -> None:
        frames = [
            self.blank_frame(),
            self.blank_frame(),
            self.blank_frame(),
            self.start_frame(),
        ]

        class FakeCapture:
            @staticmethod
            def grab():
                return frames.pop(0)

        class FakeInput:
            def __init__(self) -> None:
                self.actions: list[tuple[str, str, int]] = []

            def tap_key(self, key: str, duration_ms: int) -> None:
                self.actions.append(("step", key, duration_ms))

            def press_key_named(self, key: str) -> None:
                self.actions.append(("press", key, 0))

        state = StateMachine()
        state.toggle()
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_key = "e"
        job._setup_after_start = 2.0
        job._setup_after_equipment = 2.0
        job._setup_start_timeout = 30.0
        job._setup_equipment_timeout = 20.0
        job._setup_prompt_search_delay = 1.0
        job._setup_prompt_first_step = "s"
        job._setup_prompt_step_hold_ms = 500
        job._setup_prompt_step_wait = 1.0
        job._setup_prompt_cycle_pause = 2.0
        job._setup_stage_started = time.perf_counter()
        job._setup_settle_until = 0.0
        job._setup_nudge_direction = "s"

        job._tick_setup()
        job._setup_nudge_next_at = 0.0
        job._tick_setup()
        job._setup_nudge_next_at = 0.0
        job._tick_setup()
        job._tick_setup()

        self.assertEqual(
            [
                ("step", "s", 500),
                ("step", "w", 500),
                ("press", "e", 0),
            ],
            inputs.actions,
        )
        self.assertEqual(BotState.FISHING_EQUIPMENT, state.state)
        self.assertGreater(job._setup_settle_until, time.perf_counter())

    def test_missing_prompt_never_moves_more_than_one_pair(self) -> None:
        frame = self.blank_frame()

        class FakeCapture:
            @staticmethod
            def grab():
                return frame.copy()

        class FakeInput:
            def __init__(self) -> None:
                self.steps: list[str] = []

            def tap_key(self, key: str, _duration_ms: int) -> None:
                self.steps.append(key)

        state = StateMachine()
        state.toggle()
        state.state = BotState.FISHING_START
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._telemetry = None
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_start_timeout = 30.0
        job._setup_stage_started = time.perf_counter()
        job._setup_settle_until = 0.0
        job._setup_prompt_first_step = "s"
        job._setup_prompt_step_hold_ms = 500
        job._setup_prompt_step_wait = 1.0
        job._setup_prompt_cycle_pause = 2.0
        job._setup_prompt_max_steps = 2
        job._setup_nudge_steps = 0
        job._setup_nudge_direction = "s"
        job._setup_recovery_passive = False

        for _ in range(8):
            job._setup_nudge_next_at = 0.0
            job._tick_setup()

        self.assertEqual(["s", "w"], inputs.steps)
        self.assertTrue(state.running)
        self.assertEqual(BotState.FISHING_START, state.state)

    def test_passive_recovery_requires_start_prompt_and_never_moves(self) -> None:
        frames = [self.equipment_frame(), self.start_frame()]

        class FakeCapture:
            @staticmethod
            def grab():
                return frames.pop(0)

        class FakeInput:
            def __init__(self) -> None:
                self.actions: list[tuple[str, str]] = []

            def tap_key(self, key: str, _duration_ms: int) -> None:
                self.actions.append(("step", key))

            def press_key_named(self, key: str) -> None:
                self.actions.append(("press", key))

        state = StateMachine()
        state.toggle()
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._telemetry = None
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_key = "e"
        job._setup_after_start = 2.0
        job._setup_start_timeout = 30.0
        job._setup_stage_started = 0.0
        job._setup_settle_until = 0.0
        job._setup_prompt_search_delay = 1.0
        job._setup_prompt_first_step = "s"
        job._setup_prompt_step_hold_ms = 500
        job._setup_prompt_step_wait = 1.0
        job._setup_prompt_cycle_pause = 2.0
        job._setup_prompt_max_steps = 2
        job._setup_recovery_passive = True

        # Equipment alone must not resume an uncertain recovery.
        job._tick_setup()
        self.assertEqual([], inputs.actions)
        self.assertEqual(BotState.FISHING_START, state.state)

        # A real start prompt clears passive mode and allows E.
        job._tick_setup()
        self.assertEqual([("press", "e")], inputs.actions)
        self.assertFalse(job._setup_recovery_passive)
        self.assertEqual(BotState.FISHING_EQUIPMENT, state.state)

    def test_setup_timeout_pauses_instead_of_repeating_steps(self) -> None:
        frame = self.blank_frame()

        class FakeCapture:
            @staticmethod
            def grab():
                return frame.copy()

        class FakeInput:
            def __init__(self) -> None:
                self.release_count = 0

            @staticmethod
            def tap_key(_key: str, _duration_ms: int) -> None:
                raise AssertionError("step limit is already exhausted")

            def release_all_keys(self) -> None:
                self.release_count += 1

        state = StateMachine()
        state.toggle()
        state.state = BotState.FISHING_START
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._telemetry = None
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = False
        job._setup_start_timeout = 30.0
        job._setup_stage_started = time.perf_counter() - 31.0
        job._setup_settle_until = 0.0
        job._setup_prompt_max_steps = 2
        job._setup_nudge_steps = 2
        job._setup_recovery_passive = False
        job._cast_pressed = False
        job._timing_session = False
        job._sustenance_checked_for_cast = False
        job._normal_cycle_ready_at = 0.0

        job._tick_setup()

        self.assertEqual(BotState.IDLE, state.state)
        self.assertFalse(state.running)
        self.assertTrue(job._setup_recovery_passive)
        self.assertEqual(1, inputs.release_count)

    def test_runtime_equipment_screen_is_confirmed_before_another_cast(
        self,
    ) -> None:
        frame = self.equipment_frame()

        class FakeCapture:
            @staticmethod
            def grab():
                return frame.copy()

        class FakeInput:
            def __init__(self) -> None:
                self.keys: list[str] = []

            def press_key_named(self, key: str) -> None:
                self.keys.append(key)

        state = StateMachine()
        state.toggle()
        state.state = BotState.CASTING
        inputs = FakeInput()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._capture = FakeCapture()
        job._input = inputs
        job._setup_detector = self.detector
        job._setup_enabled = True
        job._setup_complete = True
        job._setup_key = "e"
        job._setup_after_start = 0.0
        job._setup_after_equipment = 0.0
        job._setup_start_timeout = 30.0
        job._setup_equipment_timeout = 20.0
        job._setup_stage_started = 0.0
        job._setup_settle_until = 0.0
        job._normal_cycle_ready_at = 0.0
        job._cast_pressed = True
        job._timing_session = False
        job._sustenance_checked_for_cast = True

        job._tick_casting()

        self.assertEqual([], inputs.keys)
        self.assertFalse(job._setup_complete)
        self.assertEqual(BotState.FISHING_EQUIPMENT, state.state)

        job._tick_setup()

        self.assertEqual(["e"], inputs.keys)
        self.assertTrue(job._setup_complete)
        self.assertEqual(BotState.WAITING, state.state)


class FishingSustenanceIntegrationTests(unittest.TestCase):
    @staticmethod
    def make_job(needs: SustenanceResult, consumed: SustenanceResult):
        class Service:
            last_thirst_score = 0.91 if needs.water_needed else 0.05
            last_hunger_score = 0.88 if needs.food_needed else 0.04

            def __init__(self) -> None:
                self.check_calls = 0
                self.consume_calls = 0

            def check_needs(self, *, force=False):
                assert force
                self.check_calls += 1
                return needs

            @staticmethod
            def action_block_reason():
                return None

            def consume_detected_needs(self, detected, *, allow_action=True):
                self.consume_calls += 1
                assert detected is needs
                assert allow_action
                return consumed

        class Input:
            def __init__(self) -> None:
                self.keys: list[str] = []
                self.release_count = 0

            def press_key_named(self, key: str) -> None:
                self.keys.append(key)

            def release_all_keys(self) -> None:
                self.release_count += 1

        state = StateMachine()
        state.toggle()
        state.state = BotState.CASTING
        service = Service()
        inputs = Input()
        job = FisherJob.__new__(FisherJob)
        job._sm = state
        job._input = inputs
        job._sustenance = service
        job._sustenance_enabled = True
        job._sustenance_checked_for_cast = False
        job._sustenance_pending_needs = None
        job._sustenance_retry_at = 0.0
        job._sustenance_last_deferred_reason = ""
        job._sustenance_exit_key = "esc"
        job._sustenance_exit_settle = 0.0
        job._sustenance_exit_attempts = 1
        job._sustenance_failure_retry = 0.25
        job._sustenance_after_action = 0.0
        job._timing_session = False
        job._cast_pressed = False
        job._setup_enabled = True
        job._setup_complete = True
        job._normal_cycle_ready_at = 0.0
        job._setup_stage_started = 0.0
        job._setup_settle_until = 0.0
        job._stats = {
            "sustenance_pauses": 0,
            "drinks": 0,
            "meals": 0,
        }

        class Capture:
            @staticmethod
            def grab():
                return np.zeros((1080, 1920, 3), dtype=np.uint8)

        class SetupDetector:
            start_threshold = 0.76
            equipment_threshold = 0.76

            @staticmethod
            def scores(_frame):
                return SimpleNamespace(start=0.0, equipment=0.0)

        job._capture = Capture()
        job._setup_detector = SetupDetector()
        return job, state, service, inputs

    def test_no_need_continues_cast_without_pressing_escape(self) -> None:
        needs = SustenanceResult(checked=True, reason="no_need")
        job, state, service, inputs = self.make_job(needs, needs)

        first = job._check_sustenance_before_cast()
        second = job._check_sustenance_before_cast()

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(BotState.CASTING, state.state)
        self.assertEqual([], inputs.keys)
        self.assertEqual(1, service.check_calls)
        self.assertEqual(0, service.consume_calls)

    def test_need_exits_consumes_and_restarts_fishing_setup(self) -> None:
        needs = SustenanceResult(
            checked=True,
            water_detected=True,
            hunger_detected=True,
            water_needed=True,
            food_needed=True,
            reason="need",
        )
        consumed = SustenanceResult(
            checked=True,
            drank=True,
            ate=True,
            water_detected=True,
            hunger_detected=True,
            water_needed=True,
            food_needed=True,
            reason="consumed",
        )
        job, state, service, inputs = self.make_job(needs, consumed)

        interrupted = job._check_sustenance_before_cast()

        self.assertTrue(interrupted)
        self.assertEqual(["esc"], inputs.keys)
        self.assertEqual(1, inputs.release_count)
        self.assertEqual(1, service.consume_calls)
        self.assertEqual(1, job._stats["sustenance_pauses"])
        self.assertEqual(1, job._stats["drinks"])
        self.assertEqual(1, job._stats["meals"])
        self.assertFalse(job._setup_complete)
        self.assertEqual(BotState.FISHING_START, state.state)

    def test_exit_retries_esc_until_equipment_prompt_disappears(self) -> None:
        needs = SustenanceResult(
            checked=True,
            water_detected=True,
            water_needed=True,
            reason="need",
        )
        consumed = SustenanceResult(
            checked=True,
            drank=True,
            water_detected=True,
            water_needed=True,
            reason="consumed",
        )
        job, state, service, inputs = self.make_job(needs, consumed)
        job._sustenance_exit_attempts = 3
        scores = [
            SimpleNamespace(start=0.0, equipment=0.99),
            SimpleNamespace(start=0.0, equipment=0.99),
            SimpleNamespace(start=0.99, equipment=0.0),
        ]
        job._setup_detector.scores = lambda _frame: scores.pop(0)

        self.assertTrue(job._check_sustenance_before_cast())

        self.assertEqual(["esc", "esc"], inputs.keys)
        self.assertEqual(1, service.consume_calls)
        self.assertEqual(BotState.FISHING_START, state.state)

    def test_inventory_failure_blocks_cast_and_keeps_need_pending(self) -> None:
        needs = SustenanceResult(
            checked=True,
            water_detected=True,
            water_needed=True,
            reason="need",
        )
        failed = SustenanceResult(
            checked=False,
            water_detected=True,
            water_needed=True,
            water_found=False,
            inventory_attempts=3,
            reason="inventory_not_open",
        )
        job, state, service, inputs = self.make_job(needs, failed)

        self.assertTrue(job._check_sustenance_before_cast())

        self.assertEqual(["esc"], inputs.keys)
        self.assertEqual(1, service.consume_calls)
        self.assertEqual(BotState.FISHING_SUSTENANCE, state.state)
        self.assertIs(job._sustenance_pending_needs, needs)
        self.assertTrue(job._setup_complete)
        self.assertEqual(0, job._stats["drinks"])
        self.assertTrue(job._exit_fishing_for_sustenance())
        self.assertEqual(
            ["esc"],
            inputs.keys,
            "a confirmed fishing exit must not press ESC again on retry",
        )

    def test_retry_cooldown_blocks_cast_instead_of_deferring_to_fishing(
        self,
    ) -> None:
        needs = SustenanceResult(
            checked=True,
            water_detected=True,
            water_needed=True,
            reason="need",
        )
        job, state, service, inputs = self.make_job(needs, needs)
        service.action_block_reason = lambda: "retry_wait"

        self.assertTrue(job._check_sustenance_before_cast())

        self.assertEqual(BotState.FISHING_SUSTENANCE, state.state)
        self.assertIs(job._sustenance_pending_needs, needs)
        self.assertEqual([], inputs.keys)
        self.assertEqual(0, service.consume_calls)

    def test_need_consumes_then_restarts_complete_e_e_sequence(self) -> None:
        needs = SustenanceResult(
            checked=True,
            water_detected=True,
            hunger_detected=True,
            water_needed=True,
            food_needed=True,
            reason="need",
        )
        consumed = SustenanceResult(
            checked=True,
            drank=True,
            ate=True,
            water_detected=True,
            hunger_detected=True,
            water_needed=True,
            food_needed=True,
            reason="consumed",
        )
        job, state, service, inputs = self.make_job(needs, consumed)

        self.assertTrue(job._check_sustenance_before_cast())
        self.assertEqual(BotState.FISHING_START, state.state)

        config = load_config(FISHING_CONFIG_PATH)
        detector = FishingSetupDetector(config)
        start_template = cv2.imread(str(TEMPLATES_DIR / "fishing" / "start_fishing_prompt.png"))
        equipment_template = cv2.imread(
            str(TEMPLATES_DIR / "fishing" / "equipment_ready_prompt.png")
        )
        assert start_template is not None
        assert equipment_template is not None

        start = np.zeros((1080, 1920, 3), dtype=np.uint8)
        height, width = start_template.shape[:2]
        start[16 : 16 + height, 12 : 12 + width] = start_template
        equipment = np.zeros((1080, 1920, 3), dtype=np.uint8)
        height, width = equipment_template.shape[:2]
        equipment[1015 : 1015 + height, 1515 : 1515 + width] = equipment_template
        frames = [start, equipment]

        class Capture:
            @staticmethod
            def grab():
                return frames.pop(0)

        job._capture = Capture()
        job._setup_detector = detector
        job._setup_key = "e"
        job._setup_after_start = 0.0
        job._setup_after_equipment = 0.0
        job._setup_start_timeout = 30.0
        job._setup_equipment_timeout = 20.0
        job._setup_settle_until = 0.0

        job._tick_setup()
        job._tick_setup()

        self.assertEqual(["esc", "e", "e"], inputs.keys)
        self.assertEqual(1, service.check_calls)
        self.assertEqual(1, service.consume_calls)
        self.assertTrue(job._setup_complete)
        self.assertEqual(BotState.WAITING, state.state)

    def test_fish_target_stops_bot_only_after_limit_is_reached(self) -> None:
        no_need = SustenanceResult(checked=True, reason="no_need")
        job, state, _service, inputs = self.make_job(no_need, no_need)
        job._target_fish_count = 2
        job._stats["fish_caught"] = 1

        self.assertFalse(job._stop_if_fish_target_reached())
        self.assertTrue(state.running)
        self.assertEqual(0, inputs.release_count)

        job._stats["fish_caught"] = 2
        self.assertTrue(job._stop_if_fish_target_reached())
        self.assertEqual(BotState.STOPPED, state.state)
        self.assertEqual(1, inputs.release_count)


class FishingTelemetryTests(unittest.TestCase):
    def test_summary_explains_detected_failures(self) -> None:
        directory = ROOT / "logs" / "test_fishing_telemetry"
        directory.mkdir(parents=True, exist_ok=True)
        telemetry = FishingSessionTelemetry(
            {
                "mode": "cast_and_reel",
                "fishing": {"auto_cast": True},
                "telemetry": {
                    "enabled": True,
                    "directory": str(directory),
                },
            }
        )
        telemetry.observe_state("CAST_TIMING")
        telemetry.failure("timing_timeout")
        telemetry.observe_state("REELING")
        telemetry.failure("reel_timeout", duration_sec=120)
        telemetry.performance(350, 900, 2.8)

        summary = telemetry.finish(
            {
                "fish_caught": 0,
                "casts": 3,
                "hooks": 3,
                "timing_hits": 2,
                "timeouts": 0,
            }
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        titles = {item["title"] for item in summary["conclusions"]}
        self.assertTrue(any("тайминга" in title for title in titles))
        self.assertTrue(any("сорвалась" in title for title in titles))
        self.assertTrue(any("Низкая успешность" in title for title in titles))
        self.assertTrue(telemetry.events_path.exists())
        self.assertTrue(telemetry.summary_path.exists())
        latest = directory / "latest_summary.json"
        parsed = json.loads(latest.read_text(encoding="utf-8"))
        self.assertEqual(3, parsed["stats"]["casts"])


if __name__ == "__main__":
    unittest.main()
