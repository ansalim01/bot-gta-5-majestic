"""A rejected input must not be counted as a completed game action."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from bot.config_loader import GYM_CONFIG_PATH, SUSTENANCE_CONFIG_PATH, TEMPLATES_DIR, load_config
from bot.input_handler import InputHandler
from bot.jobs.gym import GymJob
from bot.services.sustenance import SustenanceResult, SustenanceService
from bot.state_machine import StateMachine


def make_sustenance():
    config = load_config(SUSTENANCE_CONFIG_PATH, include_user_overrides=False)
    config["sustenance"].update(after_item_wait_sec=0, inventory_close_wait_sec=0)
    capture = MagicMock()
    capture.rect = SimpleNamespace(left=0, top=0, width=1920, height=1080)
    capture.grab.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    inputs = MagicMock()
    inputs.double_click_at.return_value = False
    service = SustenanceService(config, capture, inputs)
    service._inventory_offset = lambda _frame: (0, 0)
    service._shifted_region = lambda _name, _offset: [0, 0, 100, 100]
    service._detector.find_item = lambda _crop, _kind: SimpleNamespace(score=1.0, center=(20, 20))
    return service, inputs


def test_blocked_water_click_does_not_complete_need_or_increment_drinks():
    service, inputs = make_sustenance()
    result = service.consume_detected_needs(SustenanceResult(checked=True, water_needed=True))
    inputs.double_click_at.assert_called_once()
    assert result.drank is False
    assert result.water_found is True
    assert result.reason == "input_blocked"
    assert service.stats["drinks"] == 0
    assert service.stats["failures"] == 1


def test_blocked_inventory_open_is_not_followed_by_a_blind_close():
    service, inputs = make_sustenance()
    service._inventory_offset = lambda _frame: None
    inputs.press_key.return_value = False
    result = service.consume_detected_needs(SustenanceResult(checked=True, water_needed=True))
    assert result.reason == "input_blocked"
    assert inputs.press_key.call_count == 1
    inputs.double_click_at.assert_not_called()


def make_gym():
    config = load_config(GYM_CONFIG_PATH, include_user_overrides=False)
    config["gym"].update(start_settle_sec=0, confirm_frames=1, reaction_delay_ms=[0, 0])
    capture = MagicMock()
    capture.grab_region.return_value = cv2.imread(str(TEMPLATES_DIR / "gym" / "prompt_w.png"))
    inputs = MagicMock()
    state = StateMachine()
    state.toggle()
    return GymJob(config, capture, inputs, state), inputs


def test_blocked_gym_start_does_not_increment_cycles():
    job, inputs = make_gym()
    inputs.press_key.return_value = False
    with patch("bot.jobs.gym.time.perf_counter", return_value=100.0):
        job.tick()
    assert job.stats["cycles"] == 0
    inputs.press_key.return_value = True
    with patch("bot.jobs.gym.time.perf_counter", return_value=101.0):
        job.tick()
    assert job.stats["cycles"] == 1


def test_blocked_gym_letter_remains_ready_for_retry():
    job, inputs = make_gym()
    inputs.press_key.return_value = True
    job.tick()
    inputs.press_key.return_value = False
    with patch("bot.jobs.gym.time.sleep"):
        job.tick()
    assert job.stats["prompts"] == 0
    inputs.press_key.return_value = True
    with patch("bot.jobs.gym.time.sleep"):
        job.tick()
    assert job.stats["prompts"] == 1


def test_emergency_release_includes_split_and_inventory_keys():
    handler = InputHandler({})
    with (
        patch("bot.input_handler.pydirectinput.keyUp") as release,
        patch("bot.input_handler.pydirectinput.mouseUp"),
    ):
        handler.release_all_keys()
    keys = {call.args[0] for call in release.call_args_list}
    assert {"f", "i"} <= keys
