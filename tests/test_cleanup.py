"""Cleanup must reach input release and capture close even after another failure."""

from unittest.mock import MagicMock

import pytest

from bot.jobs.blackjack import BlackjackJob
from main import MinerBot
from main_blackjack import BlackjackBot
from main_fishing import FisherBot
from main_gym import GymBot
from main_sustenance import SustenanceBot


def test_blackjack_telemetry_failure_still_releases_keys():
    job = BlackjackJob.__new__(BlackjackJob)
    job._round = None
    job._stats = {}
    job._telemetry = MagicMock()
    job._telemetry.finish.side_effect = OSError("report failed")
    job._input = MagicMock()
    with pytest.raises(OSError, match="report failed"):
        job.cleanup()
    job._input.release_all_keys.assert_called_once()


@pytest.mark.parametrize("bot_type", [MinerBot, FisherBot, BlackjackBot, GymBot, SustenanceBot])
def test_startup_capture_error_closes_capture_and_stops_listener(bot_type):
    bot = bot_type.__new__(bot_type)
    bot._config = {}
    bot._toggle_key = "f9"
    bot._stop_key = "f10"
    bot._toggle_name = "f9"
    bot._stop_name = "f10"
    bot._check_name = "f8"
    bot._job = MagicMock()
    bot._input = MagicMock()
    bot._capture = MagicMock()
    bot._capture.find_window.side_effect = OSError("capture failed")
    bot._listener = MagicMock()
    with pytest.raises(OSError, match="capture failed"):
        bot.run()
    bot._capture.close.assert_called_once()
    bot._listener.stop.assert_called_once()


@pytest.mark.parametrize("bot_type", [MinerBot, FisherBot])
def test_job_cleanup_failure_still_closes_other_resources(bot_type):
    bot = bot_type.__new__(bot_type)
    bot._job = MagicMock()
    bot._job.cleanup.side_effect = OSError("cleanup failed")
    bot._capture = MagicMock()
    bot._listener = MagicMock()
    with pytest.raises(OSError, match="cleanup failed"):
        bot._cleanup()
    bot._capture.close.assert_called_once()
    bot._listener.stop.assert_called_once()
