"""The panel must decode a byte stream independently of process chunk sizes."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QProcess

from bot.control_protocol import EVENT_PREFIX, encode_event
from gamebot_ui.process_manager import BotProcessManager


def make_manager():
    manager = BotProcessManager()
    manager._active_id = "fishing"
    manager._process = MagicMock()
    manager._process.readAllStandardOutput.return_value = b""
    manager._process.state.return_value = QProcess.Running
    manager._process.write.side_effect = len
    return manager


def test_utf8_characters_and_events_survive_every_byte_boundary():
    payload = ("Поймана рыба\r\n" + encode_event("stats", values={"улов": 2}) + "\n").encode(
        "utf-8"
    )
    for boundary in range(len(payload) + 1):
        manager = make_manager()
        lines = []
        manager.log_line.connect(lambda bot_id, line: lines.append((bot_id, line)))
        for chunk in (payload[:boundary], payload[boundary:]):
            manager._process.readAllStandardOutput.return_value = chunk
            manager._read_output()
        assert lines == [("fishing", "Поймана рыба")]
        assert manager.stats("fishing") == {"улов": 2}


@pytest.mark.parametrize("raw_json", ["null", "[]", "42", '"text"', "{invalid"])
def test_invalid_event_payload_remains_visible_in_log(raw_json):
    manager = make_manager()
    lines = []
    manager.log_line.connect(lambda _bot_id, line: lines.append(line))
    manager._handle_event(raw_json)
    assert lines == [raw_json]


def test_final_event_without_newline_is_processed_before_worker_cleanup():
    manager = make_manager()
    process = manager._process
    process.readAllStandardOutput.return_value = encode_event("stats", values={"fish": 3}).encode()
    manager._on_finished(0, QProcess.NormalExit)
    assert manager.stats("fishing") == {"fish": 3}
    assert manager.active_id is None
    assert manager.status("fishing") == "idle"
    process.deleteLater.assert_called_once()


def test_failure_to_start_releases_worker_and_reports_original_error():
    manager = make_manager()
    process = manager._process
    process.errorString.return_value = "Python executable missing"
    errors = []
    manager.error.connect(lambda bot_id, message: errors.append((bot_id, message)))
    manager._on_process_error(QProcess.FailedToStart)
    assert manager.active_id is None
    assert manager._process is None
    assert manager.status("fishing") == "error"
    assert errors == [("fishing", "Python executable missing")]
    process.deleteLater.assert_called_once()


def test_worker_and_panel_use_the_same_event_prefix():
    # A protocol event must remain a single line even with multiline messages.
    encoded = encode_event("error", message="Первая строка\nВторая строка")
    assert encoded.startswith(EVENT_PREFIX)
    assert "\n" not in encoded


@pytest.mark.parametrize("action", ["stop", "shutdown"])
def test_delayed_switch_is_cancelled_by_explicit_stop(action):
    manager = make_manager()
    scheduled = []
    with patch(
        "gamebot_ui.process_manager.QTimer.singleShot",
        side_effect=lambda _ms, fn: scheduled.append(fn),
    ):
        manager.switch_to("gym")
        manager._on_finished(0, QProcess.NormalExit)
        getattr(manager, action)()
        with patch.object(manager, "start") as start:
            for callback in scheduled:
                callback()
            start.assert_not_called()


def test_ready_and_state_events_do_not_undo_a_stop_request():
    manager = make_manager()
    with patch("gamebot_ui.process_manager.QTimer.singleShot"):
        manager.stop()
    manager._handle_event('{"type":"ready"}')
    manager._handle_event('{"type":"state","state":"WATCHING [ON]","running":true}')
    assert manager.status("fishing") == "stopping"


@pytest.mark.parametrize("command", ["pause", "resume"])
def test_pause_or_resume_cannot_undo_pending_stop(command):
    manager = make_manager()
    with patch("gamebot_ui.process_manager.QTimer.singleShot"):
        manager.stop()
    manager._process.write.reset_mock()
    getattr(manager, command)("fishing")
    assert manager.status("fishing") == "stopping"
    manager._process.write.assert_not_called()


def test_switch_to_unavailable_bot_does_not_stop_current_worker():
    manager = make_manager()
    with patch.object(manager, "stop") as stop:
        manager.switch_to("miner")
    stop.assert_not_called()
    manager._process.write.assert_not_called()


def test_shutdown_tolerates_process_released_inside_wait():
    manager = make_manager()
    process = manager._process

    def wait(_timeout):
        manager._process = None
        return False

    process.waitForFinished.side_effect = wait
    manager.shutdown()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_obsolete_worker_callbacks_cannot_change_the_new_worker():
    manager = make_manager()
    current = manager._process
    previous = MagicMock()
    manager._read_output(previous)
    manager._on_finished(1, QProcess.CrashExit, previous)
    manager._on_process_error(QProcess.FailedToStart, previous)
    assert manager._process is current
    assert manager.active_id == "fishing"
    current.deleteLater.assert_not_called()
    current.readAllStandardOutput.assert_not_called()


def test_delayed_switch_starts_requested_bot_once():
    manager = make_manager()
    scheduled = []
    with patch(
        "gamebot_ui.process_manager.QTimer.singleShot",
        side_effect=lambda _ms, fn: scheduled.append(fn),
    ):
        manager.switch_to("gym")
        manager._on_finished(0, QProcess.NormalExit)
        with patch.object(manager, "_start_process") as start:
            for callback in scheduled:
                callback()
            for callback in scheduled:
                callback()
            start.assert_called_once()
            assert start.call_args.args[0].bot_id == "gym"


def test_explicit_new_start_supersedes_a_queued_switch():
    manager = make_manager()
    scheduled = []
    with patch(
        "gamebot_ui.process_manager.QTimer.singleShot",
        side_effect=lambda _ms, fn: scheduled.append(fn),
    ):
        manager.switch_to("gym")
        manager._on_finished(0, QProcess.NormalExit)
        with patch.object(manager, "_start_process") as start:
            manager.start("blackjack")
            for callback in scheduled:
                callback()
            start.assert_called_once()
            assert start.call_args.args[0].bot_id == "blackjack"
