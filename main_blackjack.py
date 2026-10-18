"""Majestic RP blackjack assistant entry point."""

from __future__ import annotations

import time
from pathlib import Path

from pynput import keyboard

from bot.capture import GameCapture
from bot.config_loader import BLACKJACK_CONFIG_PATH, load_config
from bot.input_handler import InputHandler
from bot.jobs.blackjack import BlackjackJob
from bot.runtime import close_bot_resources
from bot.state_machine import BotState, StateMachine
from bot.windows import is_admin as _is_admin
from bot.windows import relaunch_as_admin


def _ensure_capture_privileges() -> bool:
    """GTA may deny GDI capture/input to a lower-privilege process."""
    if _is_admin():
        return True
    try:
        launched = relaunch_as_admin(Path(__file__))
    except (AttributeError, OSError) as exc:
        print(f"[21] Не удалось запросить права администратора: {exc}", flush=True)
        return False
    if launched:
        return False  # Elevated child now owns the session.
    print("[21] Запуск отменён: без прав администратора GTA блокирует захват экрана.", flush=True)
    return False


class BlackjackBot:
    def __init__(self) -> None:
        self._config = load_config(BLACKJACK_CONFIG_PATH)
        self._capture = GameCapture(self._config)
        self._input = InputHandler(self._config)
        self._sm = StateMachine()
        self._job = BlackjackJob(self._config, self._capture, self._input)
        self._listener: keyboard.Listener | None = None

    @staticmethod
    def _hotkey(name: str):
        return getattr(keyboard.Key, name.lower(), keyboard.Key.f9)

    def _on_press(self, key) -> None:
        hotkeys = self._config.get("hotkeys", {})
        if key == self._hotkey(str(hotkeys.get("toggle", "f9"))):
            self._sm.toggle()
            print(f"[21] {self._sm.state_label()}", flush=True)
        elif key == self._hotkey(str(hotkeys.get("stop", "f10"))):
            self._sm.stop()
            self._input.release_all_keys()

    def run(self) -> None:
        print("Majestic Blackjack: F9 — старт/пауза, F10 — остановка", flush=True)
        section = self._config.get("blackjack", {})
        risk = section.get("risk", {})
        mode = "автонажатия" if section.get("auto_action") else "советчик"
        print(f"Режим: {mode}.", flush=True)
        print(
            "[21] Стратегия: effective-48 max-EV v11. Применены настройки: "
            f"ставка={section.get('bet_amount', 10)}, "
            f"дабл={section.get('double_policy', 'conservative')}, "
            f"лимит убытка={risk.get('max_session_loss', 0)}, "
            f"просадка={risk.get('max_drawdown_from_peak', 0)}, "
            f"раздач={risk.get('max_hands_per_session', 0)}.",
            flush=True,
        )
        try:
            self._capture.find_window()
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
            while self._sm.state != BotState.STOPPED:
                if self._sm.running:
                    self._job.tick(allow_action=True)
                time.sleep(0.06)
        finally:
            close_bot_resources(self._job.cleanup, self._capture, self._listener)


def main() -> None:
    if not _ensure_capture_privileges():
        return
    BlackjackBot().run()


if __name__ == "__main__":
    main()
