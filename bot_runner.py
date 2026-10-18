"""Run one existing bot behind a tiny line-based control protocol.

The desktop app writes commands to stdin and receives structured events on
stdout. Human-readable output from the existing bots remains untouched.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from typing import Any

from bot.control_protocol import EVENT_PREFIX as EVENT_PREFIX
from bot.control_protocol import encode_event
from bot.state_machine import BotState

_EVENT_LOCK = threading.Lock()


def emit(event_type: str, **payload: Any) -> None:
    # Bot status lines may end with end="". Start a separate line and write
    # the whole event at once so command/monitor threads cannot interleave it.
    with _EVENT_LOCK:
        sys.stdout.write("\n" + encode_event(event_type, **payload) + "\n")
        sys.stdout.flush()


def create_bot(bot_id: str):
    if bot_id == "miner":
        from main import MinerBot

        return MinerBot()
    if bot_id == "fishing":
        from main_fishing import FisherBot

        return FisherBot()
    if bot_id == "gym":
        from main_gym import GymBot

        return GymBot()
    if bot_id == "sustenance":
        from main_sustenance import SustenanceBot

        return SustenanceBot()
    if bot_id == "blackjack":
        from main_blackjack import BlackjackBot

        return BlackjackBot()
    raise ValueError(f"Unknown bot: {bot_id}")


def _stats(bot: Any) -> dict[str, Any]:
    job = getattr(bot, "_job", None)
    source = job if job is not None else getattr(bot, "_service", None)
    if source is None:
        return {}
    stats = getattr(source, "stats", {})
    if callable(stats):
        stats = stats()
    return dict(stats) if isinstance(stats, dict) else {}


def _stop_bot(bot: Any, stop_event: threading.Event) -> None:
    state_machine = getattr(bot, "_sm", None)
    if state_machine is not None:
        state_machine.stop()
    stop_event.set()
    input_handler = getattr(bot, "_input", None)
    if input_handler is not None:
        input_handler.release_all_keys()


def command_loop(bot: Any, stop_event: threading.Event) -> None:
    try:
        _read_commands(bot, stop_event)
    except (OSError, ValueError) as exc:
        emit("error", message=f"Control channel failed: {exc}")
    finally:
        # EOF means the controlling panel disappeared. Do not leave an
        # unattended worker running after losing its command pipe.
        if not stop_event.is_set():
            try:
                _stop_bot(bot, stop_event)
            except Exception as exc:
                emit("error", message=f"Failed to release input: {exc}")


def _read_commands(bot: Any, stop_event: threading.Event) -> None:
    for raw_line in sys.stdin:
        if stop_event.is_set():
            return
        command = raw_line.strip().lower()
        if not command:
            continue
        state_machine = getattr(bot, "_sm", None)
        try:
            if command == "pause":
                if state_machine is not None and state_machine.running:
                    state_machine.toggle()
                    input_handler = getattr(bot, "_input", None)
                    if input_handler is not None:
                        input_handler.release_all_keys()
                    emit("command", command="pause", accepted=True)
            elif command in {"start", "resume"}:
                if (
                    state_machine is not None
                    and state_machine.state != BotState.STOPPED
                    and not state_machine.running
                ):
                    state_machine.toggle()
                    emit("command", command="resume", accepted=True)
            elif command == "check_now":
                service = getattr(bot, "_service", None)
                if service is not None:
                    service.request_check()
                    emit("command", command="check_now", accepted=True)
            elif command == "stop":
                _stop_bot(bot, stop_event)
                emit("command", command="stop", accepted=True)
                return
            else:
                emit(
                    "command",
                    command=command,
                    accepted=False,
                    reason="unknown_command",
                )
        except Exception as exc:
            emit("error", message=f"Command {command!r} failed: {exc}")


def monitor_loop(bot: Any, stop_event: threading.Event) -> None:
    last_state = ""
    last_stats: dict[str, Any] | None = None
    while not stop_event.wait(0.35):
        state_machine = getattr(bot, "_sm", None)
        if state_machine is None:
            continue
        state = state_machine.state_label()
        stats = _stats(bot)
        if state != last_state:
            emit("state", state=state, running=bool(state_machine.running))
            last_state = state
        if stats != last_stats:
            emit("stats", values=stats)
            last_stats = stats


def run_bot(bot_id: str) -> int:
    try:
        bot = create_bot(bot_id)
    except Exception as exc:
        emit("error", message=f"Не удалось подготовить бот: {exc}")
        traceback.print_exc()
        return 1

    stop_event = threading.Event()
    commands = threading.Thread(
        target=command_loop,
        args=(bot, stop_event),
        name="gamebot-command-loop",
        daemon=True,
    )
    monitor = threading.Thread(
        target=monitor_loop,
        args=(bot, stop_event),
        name="gamebot-monitor",
        daemon=True,
    )
    try:
        # Initialize before the command thread can receive pause/stop.
        state_machine = getattr(bot, "_sm")
        state_machine.toggle()
        commands.start()
        monitor.start()
        if not stop_event.is_set():
            emit("ready", bot_id=bot_id)
        bot.run()
        return 0
    except KeyboardInterrupt:
        emit("state", state="STOPPED", running=False)
        return 0
    except Exception as exc:
        emit("error", message=str(exc))
        traceback.print_exc()
        return 1
    finally:
        stop_event.set()
        input_handler = getattr(bot, "_input", None)
        if input_handler is not None:
            try:
                input_handler.release_all_keys()
            except Exception:
                pass
        emit("finished", bot_id=bot_id)
        time.sleep(0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bot",
        required=True,
        choices=("miner", "fishing", "gym", "sustenance", "blackjack"),
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    return run_bot(parse_args().bot)


if __name__ == "__main__":
    raise SystemExit(main())
