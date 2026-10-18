"""Worker control must remain safe during startup and controller disconnects."""

import io
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import bot_runner
from bot.state_machine import BotState, StateMachine


def test_stop_received_immediately_during_startup_is_not_overwritten():
    bot = SimpleNamespace(_sm=StateMachine(), _input=MagicMock())
    states = []
    bot.run = lambda: states.append(bot._sm.state)

    class ImmediateCommands:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            if self.target is bot_runner.command_loop:
                self.target(*self.args)

    with (
        patch.object(bot_runner, "create_bot", return_value=bot),
        patch.object(bot_runner.threading, "Thread", ImmediateCommands),
        patch.object(bot_runner.sys, "stdin", io.StringIO("stop\n")),
    ):
        assert bot_runner.run_bot("fishing") == 0
    assert states == [BotState.STOPPED]


def test_controller_disconnect_stops_bot_and_releases_input():
    bot = SimpleNamespace(_sm=StateMachine(), _input=MagicMock())
    bot._sm.toggle()
    stopped = threading.Event()
    with patch.object(bot_runner.sys, "stdin", io.StringIO("")):
        bot_runner.command_loop(bot, stopped)
    assert bot._sm.state == BotState.STOPPED
    assert stopped.is_set()
    bot._input.release_all_keys.assert_called_once()


def test_resume_cannot_revive_a_stopped_worker():
    bot = SimpleNamespace(_sm=StateMachine(), _input=MagicMock())
    bot._sm.stop()
    states = []
    with (
        patch.object(bot_runner.sys, "stdin", io.StringIO("resume\n")),
        patch.object(
            bot_runner, "emit", side_effect=lambda *_args, **_kwargs: states.append(bot._sm.state)
        ),
    ):
        bot_runner.command_loop(bot, threading.Event())
    assert bot._sm.state == BotState.STOPPED
    assert all(state == BotState.STOPPED for state in states)


def test_event_is_separated_from_unterminated_human_status_line():
    stream = io.StringIO()
    stream.write("FPS: 60")
    with patch.object(bot_runner.sys, "stdout", stream):
        bot_runner.emit("ready", bot_id="fishing")
    assert stream.getvalue().splitlines()[1].startswith(bot_runner.EVENT_PREFIX)
