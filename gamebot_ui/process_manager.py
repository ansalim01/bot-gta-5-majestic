"""Qt process manager for isolated bot workers."""

from __future__ import annotations

import codecs
import json
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from bot.control_protocol import EVENT_PREFIX

from .models import BOT_BY_ID, ROOT, BotSpec


class BotProcessManager(QObject):
    status_changed = Signal(str, str)
    stats_changed = Signal(str, object)
    log_line = Signal(str, str)
    error = Signal(str, str)
    active_changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._active_id: str | None = None
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._status: dict[str, str] = {bot_id: "idle" for bot_id in BOT_BY_ID}
        self._stats: dict[str, dict[str, Any]] = {}
        self._pending_start: str | None = None
        self._start_generation = 0
        self._shutting_down = False

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def status(self, bot_id: str) -> str:
        return self._status.get(bot_id, "idle")

    def stats(self, bot_id: str) -> dict[str, Any]:
        return dict(self._stats.get(bot_id, {}))

    def start(self, bot_id: str) -> bool:
        if self._shutting_down:
            return False
        spec = BOT_BY_ID.get(bot_id)
        if spec is None or not spec.available:
            self.error.emit(bot_id, "Этот бот пока находится в разработке.")
            return False
        self._cancel_pending_start()
        if self._process is not None:
            if self._active_id == bot_id:
                if self.status(bot_id) == "paused":
                    self.resume(bot_id)
                    return True
                return False
            return False

        self._start_process(spec)
        return True

    def switch_to(self, bot_id: str) -> None:
        spec = BOT_BY_ID.get(bot_id)
        if self._shutting_down:
            return
        if spec is None or not spec.available:
            self.error.emit(bot_id, "Этот бот пока находится в разработке.")
            return
        self._cancel_pending_start()
        if self._active_id and self._active_id != bot_id:
            self._pending_start = bot_id
            self._request_stop()
        else:
            self.start(bot_id)

    def _start_process(self, spec: BotSpec) -> None:
        process = QProcess(self)
        process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(lambda: self._read_output(process))
        process.finished.connect(lambda code, status: self._on_finished(code, status, process))
        process.errorOccurred.connect(lambda error: self._on_process_error(error, process))

        self._process = process
        self._active_id = spec.bot_id
        self._buffer = ""
        self._decoder.reset()
        self._set_status(spec.bot_id, "starting")
        self.active_changed.emit(spec.bot_id)

        process.start(
            sys.executable,
            ["-u", str(Path(ROOT) / "bot_runner.py"), "--bot", spec.bot_id],
        )

    def pause(self, bot_id: str) -> None:
        if self.status(bot_id) not in {"starting", "running"}:
            return
        if self._active_id == bot_id and self._write_command("pause"):
            self._set_status(bot_id, "paused")

    def resume(self, bot_id: str) -> None:
        if self.status(bot_id) != "paused":
            return
        if self._active_id == bot_id and self._write_command("resume"):
            self._set_status(bot_id, "running")

    def check_now(self, bot_id: str) -> None:
        if self._active_id == bot_id:
            self._write_command("check_now")

    def stop(self, bot_id: str | None = None) -> None:
        if bot_id is not None and bot_id != self._active_id:
            if bot_id == self._pending_start:
                self._cancel_pending_start()
            return
        self._cancel_pending_start()
        self._request_stop()

    def _request_stop(self) -> None:
        if self._process is None:
            return
        active_id = self._active_id
        if active_id and self.status(active_id) == "stopping":
            return
        if active_id:
            self._set_status(active_id, "stopping")
        self._write_command("stop")
        process = self._process
        QTimer.singleShot(
            4500,
            lambda process=process: self._force_terminate_if_needed(process),
        )

    def shutdown(self) -> None:
        self._shutting_down = True
        self._cancel_pending_start()
        process = self._process
        if process is None:
            return
        self._write_command("stop")
        if not process.waitForFinished(3500):
            process.terminate()
            if not process.waitForFinished(1200):
                process.kill()
                process.waitForFinished(800)

    def _cancel_pending_start(self) -> None:
        self._pending_start = None
        self._start_generation += 1

    def _start_pending(self, bot_id: str, generation: int) -> None:
        if (
            not self._shutting_down
            and generation == self._start_generation
            and self._pending_start == bot_id
        ):
            self.start(bot_id)

    def _write_command(self, command: str) -> bool:
        if self._process is None or self._process.state() == QProcess.NotRunning:
            return False
        return self._process.write(f"{command}\n".encode("utf-8")) >= 0

    def _force_terminate_if_needed(self, process: QProcess) -> None:
        # A new worker may already be running by the time this old stop timer
        # fires. Never apply an old worker's escalation to self._process as it
        # may now point at that new session.
        if self._process is not process:
            return
        try:
            if process.state() == QProcess.NotRunning:
                return
        except RuntimeError:
            # deleteLater() may already have destroyed the Qt object.
            return
        process.terminate()
        QTimer.singleShot(
            1200,
            lambda process=process: self._force_kill_if_needed(process),
        )

    def _force_kill_if_needed(self, process: QProcess) -> None:
        if self._process is not process:
            return
        try:
            if process.state() != QProcess.NotRunning:
                process.kill()
        except RuntimeError:
            return

    def _read_output(self, process: QProcess | None = None) -> None:
        if process is not None and self._process is not process:
            return
        if self._process is None or self._active_id is None:
            return
        chunk = self._decoder.decode(bytes(self._process.readAllStandardOutput()))
        self._buffer += chunk.replace("\r", "\n")
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._handle_output_line(line)

    def _handle_output_line(self, line: str) -> None:
        clean = line.strip()
        if not clean or self._active_id is None:
            return
        if clean.startswith(EVENT_PREFIX):
            self._handle_event(clean[len(EVENT_PREFIX) :])
        else:
            self.log_line.emit(self._active_id, clean)

    def _handle_event(self, raw_json: str) -> None:
        if self._active_id is None:
            return
        try:
            event = json.loads(raw_json)
        except json.JSONDecodeError:
            self.log_line.emit(self._active_id, raw_json)
            return
        if not isinstance(event, dict):
            self.log_line.emit(self._active_id, raw_json)
            return
        event_type = event.get("type")
        if event_type == "ready":
            if self.status(self._active_id) != "stopping":
                self._set_status(self._active_id, "running")
        elif event_type == "state":
            running = bool(event.get("running"))
            state_text = str(event.get("state", ""))
            status = "running" if running else "paused"
            if "STOPPED" in state_text:
                status = "stopping"
            if self.status(self._active_id) != "stopping":
                self._set_status(self._active_id, status)
            self.log_line.emit(self._active_id, f"Состояние: {state_text}")
        elif event_type == "stats":
            values = event.get("values", {})
            if isinstance(values, dict):
                self._stats[self._active_id] = values
                self.stats_changed.emit(self._active_id, values)
        elif event_type == "error":
            self._set_status(self._active_id, "error")
            self.error.emit(
                self._active_id,
                str(event.get("message", "Неизвестная ошибка")),
            )

    def _on_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
        process: QProcess | None = None,
    ) -> None:
        if process is not None and self._process is not process:
            return
        active_id = self._active_id
        self._read_output()
        self._buffer += self._decoder.decode(b"", final=True)
        self._handle_output_line(self._buffer)
        self._buffer = ""
        if active_id:
            self._set_status(active_id, "idle" if exit_code == 0 else "error")
            if exit_code != 0:
                self.error.emit(
                    active_id,
                    f"Процесс завершился с кодом {exit_code}.",
                )
        self._release_process()

    def _release_process(self) -> None:
        """Release a finished worker, including one that failed to start."""
        process = self._process
        self._process = None
        self._active_id = None
        self._buffer = ""
        self._decoder.reset()
        self.active_changed.emit(None)
        if process is not None:
            process.deleteLater()

        pending = self._pending_start
        generation = self._start_generation
        if pending:
            QTimer.singleShot(150, lambda: self._start_pending(pending, generation))

    def _on_process_error(
        self, error: QProcess.ProcessError, process: QProcess | None = None
    ) -> None:
        if process is not None and self._process is not process:
            return
        if self._active_id is None:
            return
        if error == QProcess.Crashed:
            return
        message = self._process.errorString() if self._process else str(error)
        self._set_status(self._active_id, "error")
        self.error.emit(self._active_id, message)
        if error == QProcess.FailedToStart:
            self._release_process()

    def _set_status(self, bot_id: str, status: str) -> None:
        if self._status.get(bot_id) == status:
            return
        self._status[bot_id] = status
        self.status_changed.emit(bot_id, status)
