"""Telemetry must release its file after I/O failures as well as normal exit."""

from unittest.mock import MagicMock, patch

import pytest

from bot.telemetry.blackjack_session import BlackjackSessionTelemetry
from bot.telemetry.fishing_session import FishingSessionTelemetry


@pytest.fixture(params=[BlackjackSessionTelemetry, FishingSessionTelemetry])
def telemetry(request, tmp_path):
    instance = request.param({"telemetry": {"directory": str(tmp_path)}})
    instance._stream.close()
    instance._stream = MagicMock()
    return instance


def test_write_failure_closes_stream_without_waiting_for_process_exit(telemetry):
    stream = telemetry._stream
    stream.write.side_effect = OSError("disk full")
    telemetry.event("sample")
    assert telemetry.enabled is False
    assert telemetry._stream is None
    stream.close.assert_called_once()


def test_disabled_telemetry_still_closes_an_existing_stream(telemetry):
    stream = telemetry._stream
    telemetry.enabled = False
    telemetry.finish({})
    assert telemetry._stream is None
    stream.close.assert_called_once()


def test_summary_failure_still_closes_stream(telemetry):
    stream = telemetry._stream
    method = (
        "_write_summary"
        if isinstance(telemetry, BlackjackSessionTelemetry)
        else "_performance_summary"
    )
    with patch.object(telemetry, method, side_effect=RuntimeError("summary failed")):
        with pytest.raises(RuntimeError, match="summary failed"):
            telemetry.finish({})
    assert telemetry._stream is None
    stream.close.assert_called_once()


def test_close_failure_does_not_crash_shutdown(telemetry):
    telemetry._stream.close.side_effect = OSError("flush failed")
    telemetry.finish({})
    assert telemetry._stream is None
