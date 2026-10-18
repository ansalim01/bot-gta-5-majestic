"""Exercise entire blackjack ticks using synthetic observations and no live input."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bot.config_loader import BLACKJACK_CONFIG_PATH, load_config
from bot.detectors.blackjack_score import BlackjackScore
from bot.jobs.blackjack import BlackjackJob


@pytest.fixture
def table():
    config = load_config(BLACKJACK_CONFIG_PATH, include_user_overrides=False)
    config["debug"] = {"save_live_crop": False, "save_event_screenshots": False}
    config["blackjack"].update(confirm_frames=1, action_cooldown_sec=0)
    config["regions"] = {
        name: [index, 0, 20, 20]
        for index, name in enumerate(
            ["score", "action_prompt", "bet_panel", "bet_value", "result_notice"]
        )
    }
    capture = MagicMock()
    capture.scale_region.side_effect = lambda region: region
    observation = {"score": BlackjackScore(player=16, dealer=10, confidence=0.99), "actions": True}

    def crop(_frame, region):
        brightness = (
            255 if region == config["regions"]["action_prompt"] and observation["actions"] else 0
        )
        return np.full((20, 20, 3), brightness, dtype=np.uint8)

    capture.crop_from_frame.side_effect = crop
    detector = MagicMock()
    detector.detect.side_effect = lambda _crop: observation["score"]
    detector.detect_number.return_value = (10, 0.99)
    telemetry = MagicMock()
    telemetry.session_id = "synthetic-test"
    inputs = MagicMock()
    inputs.press_key_named.return_value = True
    with (
        patch("bot.jobs.blackjack.BlackjackScoreDetector", return_value=detector),
        patch("bot.jobs.blackjack.BlackjackSessionTelemetry", return_value=telemetry),
    ):
        job = BlackjackJob(config, capture, inputs)
    return job, observation, inputs


def test_confirmed_turn_sends_one_action_and_suppresses_immediate_duplicate(table):
    job, observation, inputs = table
    assert job.tick() == observation["score"]
    assert job.tick() == observation["score"]
    inputs.press_key_named.assert_called_once_with("enter")
    assert job.stats["decisions"] == 1


def test_observe_only_tick_does_not_send_a_turn_action(table):
    job, _observation, inputs = table
    job.tick(allow_action=False)
    inputs.press_key_named.assert_not_called()


def test_dealer_animation_never_creates_a_player_action(table):
    job, observation, inputs = table
    observation["actions"] = False
    observation["score"] = BlackjackScore(player=16, dealer=20, confidence=0.99)
    job.tick()
    inputs.press_key_named.assert_not_called()


def test_capture_failure_is_counted_without_sending_input(table):
    job, _observation, inputs = table
    job._capture.grab_union.side_effect = OSError("capture unavailable")
    assert job.tick() is None
    assert job.stats["capture_errors"] == 1
    inputs.press_key_named.assert_not_called()


def test_ocr_miss_does_not_send_input(table):
    job, observation, inputs = table
    observation["score"] = None
    assert job.tick() is None
    assert job.stats["ocr_misses"] == 1
    inputs.press_key_named.assert_not_called()


@pytest.mark.parametrize("auto_action, allow_action", [(True, False), (False, True)])
def test_pending_action_is_not_retried_when_input_is_disabled(table, auto_action, allow_action):
    job, observation, inputs = table
    job.tick()
    inputs.press_key_named.reset_mock()
    observation["actions"] = False
    job._last_action_at -= 10
    job._auto_action = auto_action
    job.tick(allow_action=allow_action)
    inputs.press_key_named.assert_not_called()


def test_unacknowledged_terminal_hit_can_still_retry_when_input_is_enabled(table):
    job, observation, inputs = table
    job.tick()
    observation["actions"] = False
    job._last_action_at -= 10
    job.tick()
    assert inputs.press_key_named.call_count == 2
    assert job.stats["decisions"] == 1
    assert job.stats["action_retries"] == 1
