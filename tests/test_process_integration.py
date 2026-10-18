"""Run a synthetic worker through real Qt process pipes, without game input."""

import sys
import time
from unittest.mock import patch

from PySide6.QtCore import QProcess
from PySide6.QtTest import QTest

from gamebot_ui.process_manager import BotProcessManager

WORKER = r"""
import json
import sys
prefix = "@@GAMEBOT_EVENT@@"
def event(**data):
    print(prefix + json.dumps(data, ensure_ascii=False), flush=True)
event(type="ready")
for line in sys.stdin:
    command = line.strip()
    if command == "pause":
        event(type="state", state="IDLE [OFF]", running=False)
    elif command == "resume":
        event(type="state", state="WATCHING [ON]", running=True)
    elif command == "stop":
        # Verify the manager drains a final event without a newline.
        sys.stdout.write(prefix + json.dumps({"type":"stats","values":{"fish":3}}))
        sys.stdout.flush()
        break
"""


def wait_until(condition):
    deadline = time.perf_counter() + 5
    while not condition() and time.perf_counter() < deadline:
        QTest.qWait(10)
    assert condition(), "Timed out waiting for the synthetic Qt worker"


def test_real_worker_can_start_pause_resume_stop_and_restart(qt_app):
    class SyntheticProcess(QProcess):
        def start(self, _program, _arguments):
            super().start(sys.executable, ["-u", "-c", WORKER])

    manager = BotProcessManager()
    try:
        with patch("gamebot_ui.process_manager.QProcess", SyntheticProcess):
            for _ in range(2):
                assert manager.start("fishing") is True
                wait_until(lambda: manager.status("fishing") == "running")
                manager.pause("fishing")
                wait_until(lambda: manager.status("fishing") == "paused")
                manager.resume("fishing")
                wait_until(lambda: manager.status("fishing") == "running")
                manager.stop()
                wait_until(lambda: manager.active_id is None)
                assert manager.status("fishing") == "idle"
                assert manager.stats("fishing") == {"fish": 3}
    finally:
        manager.shutdown()


def test_real_failed_start_releases_process(qt_app):
    class MissingProcess(QProcess):
        def start(self, _program, _arguments):
            super().start("C:/gamebot-nonexistent-test-folder/python.exe", [])

    manager = BotProcessManager()
    errors = []
    manager.error.connect(lambda bot_id, message: errors.append((bot_id, message)))
    try:
        with patch("gamebot_ui.process_manager.QProcess", MissingProcess):
            assert manager.start("fishing") is True
            wait_until(lambda: manager.active_id is None)
            assert manager.status("fishing") == "error"
            assert len(errors) == 1
    finally:
        manager.shutdown()
