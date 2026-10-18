from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QScrollArea

from bot.config_loader import _deep_merge, user_config_path
from gamebot_ui.app import MainWindow
from gamebot_ui.config_store import ConfigStore
from gamebot_ui.models import AVAILABLE_BOTS, BOT_BY_ID, BOT_SPECS, PLANNED_BOTS
from gamebot_ui.pages import BotPage
from gamebot_ui.process_manager import BotProcessManager

ROOT = Path(__file__).resolve().parent.parent


class ConfigOverrideTests(unittest.TestCase):
    def test_deep_merge_only_replaces_selected_leaves(self) -> None:
        base = {
            "fishing": {"auto_cast": True, "bite_timeout_sec": 35},
            "hotkeys": {"toggle": "f9", "stop": "f10"},
        }
        overrides = {
            "fishing": {"bite_timeout_sec": 50},
            "hotkeys": {"toggle": "f8"},
        }

        merged = _deep_merge(base, overrides)

        self.assertTrue(merged["fishing"]["auto_cast"])
        self.assertEqual(50, merged["fishing"]["bite_timeout_sec"])
        self.assertEqual("f8", merged["hotkeys"]["toggle"])
        self.assertEqual("f10", merged["hotkeys"]["stop"])
        self.assertEqual(35, base["fishing"]["bite_timeout_sec"])

    def test_builtin_config_maps_to_local_user_override(self) -> None:
        result = user_config_path(ROOT / "config" / "fishing.yaml")
        self.assertEqual(ROOT / "config" / "user" / "fishing.yaml", result)

    def test_arbitrary_config_does_not_gain_an_override(self) -> None:
        self.assertIsNone(user_config_path(ROOT / "tests" / "fixture.yaml"))


class ControlPanelMetadataTests(unittest.TestCase):
    def test_real_bot_entrypoints_exist(self) -> None:
        self.assertEqual(
            {"blackjack", "fishing", "gym", "sustenance"},
            {spec.bot_id for spec in AVAILABLE_BOTS},
        )
        for spec in AVAILABLE_BOTS:
            self.assertTrue(spec.entrypoint)
            self.assertTrue((ROOT / str(spec.entrypoint)).exists())
            self.assertTrue(spec.config_path and spec.config_path.exists())

    def test_reference_professions_have_placeholders(self) -> None:
        self.assertEqual(10, len(PLANNED_BOTS))
        expected = {
            "miner",
            "woodcutter",
            "courier",
            "farmer",
            "mushroom",
            "seamstress",
            "taxi",
            "carjacker",
            "anti_afk",
            "demorgan",
        }
        self.assertEqual(expected, {spec.bot_id for spec in PLANNED_BOTS})

    def test_ids_are_unique(self) -> None:
        self.assertEqual(len(BOT_SPECS), len(BOT_BY_ID))


class BotPageLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_bot_pages_fit_default_window_without_scrollbars(self) -> None:
        # MainWindow 1120x720 minus the 222px sidebar and 54px title bar.
        for spec in AVAILABLE_BOTS:
            with self.subTest(bot_id=spec.bot_id):
                page = BotPage(spec, ConfigStore())
                page.resize(898, 666)
                page.show()
                self.app.processEvents()

                scroll = page.findChild(QScrollArea)
                self.assertIsNotNone(scroll)
                self.assertEqual(0, scroll.horizontalScrollBar().maximum())
                self.assertEqual(0, scroll.verticalScrollBar().maximum())

                page.close()
                page.deleteLater()

    def test_blackjack_editor_values_are_written_to_user_override(self) -> None:
        captured = {}

        class RecordingStore(ConfigStore):
            def save_values(self, _spec, values):
                captured.update(values)
                return ROOT / "config" / "user" / "blackjack.yaml"

        page = BotPage(BOT_BY_ID["blackjack"], RecordingStore())
        page.setting_rows["blackjack.bet_amount"].editor.setValue(25)
        page.setting_rows["blackjack.risk.max_session_loss"].editor.setValue(80)

        self.assertTrue(page.save_settings(notify=False))

        self.assertEqual(25, captured["blackjack.bet_amount"])
        self.assertEqual(80, captured["blackjack.risk.max_session_loss"])
        page.close()
        page.deleteLater()

    def test_start_automatically_saves_visible_settings_before_worker(self) -> None:
        order = []

        class RecordingStore(ConfigStore):
            def save_values(self, spec, values):
                order.append(("save", values["blackjack.bet_amount"]))
                return ROOT / "config" / "user" / f"{spec.config_name}.yaml"

        class Manager:
            active_id = None

            @staticmethod
            def start(bot_id):
                order.append(("start", bot_id))
                return True

        page = BotPage(BOT_BY_ID["blackjack"], RecordingStore())
        page.setting_rows["blackjack.bet_amount"].editor.setValue(35)
        window = MagicMock()
        window.pages = {"blackjack": page}
        window.manager = Manager()

        with patch("gamebot_ui.app.QTimer.singleShot"):
            MainWindow._start_bot(window, "blackjack")

        self.assertEqual(
            [("save", 35), ("start", "blackjack")],
            order,
        )
        page.close()
        page.deleteLater()


class ProcessManagerTimerTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self.writes: list[bytes] = []

        @staticmethod
        def state():
            return QProcess.Running

        def write(self, payload: bytes) -> int:
            self.writes.append(payload)
            return len(payload)

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

    @staticmethod
    def manager_with(process):
        manager = BotProcessManager()
        manager._process = process
        manager._active_id = "fishing"
        return manager

    def test_old_stop_timer_never_terminates_new_worker(self) -> None:
        old = self.FakeProcess()
        new = self.FakeProcess()
        manager = self.manager_with(old)
        scheduled = []

        with patch(
            "gamebot_ui.process_manager.QTimer.singleShot",
            side_effect=lambda delay, callback: scheduled.append((delay, callback)),
        ):
            manager.stop("fishing")
            self.assertEqual(1, len(scheduled))
            manager._process = new
            scheduled[0][1]()

        self.assertEqual(0, old.terminate_calls)
        self.assertEqual(0, new.terminate_calls)
        self.assertEqual(0, new.kill_calls)

    def test_stop_timer_escalates_only_its_own_worker(self) -> None:
        process = self.FakeProcess()
        manager = self.manager_with(process)
        scheduled = []

        with patch(
            "gamebot_ui.process_manager.QTimer.singleShot",
            side_effect=lambda delay, callback: scheduled.append((delay, callback)),
        ):
            manager.stop("fishing")
            self.assertEqual(4500, scheduled[0][0])
            scheduled[0][1]()
            self.assertEqual(1, process.terminate_calls)
            self.assertEqual(1200, scheduled[1][0])
            scheduled[1][1]()

        self.assertEqual(1, process.kill_calls)


if __name__ == "__main__":
    unittest.main()
