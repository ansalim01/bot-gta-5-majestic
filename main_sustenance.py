"""Standalone runner for the reusable food/water service."""

from __future__ import annotations

import time
from pathlib import Path

from pynput import keyboard

from bot.capture import GameCapture
from bot.config_loader import SUSTENANCE_CONFIG_PATH, load_config
from bot.hotkeys import parse_hotkey
from bot.input_handler import InputHandler
from bot.runtime import close_bot_resources
from bot.services.sustenance import SustenanceService
from bot.state_machine import BotState, StateMachine
from bot.windows import is_admin as _is_admin
from bot.windows import relaunch_as_admin


def _request_admin_relaunch() -> bool:
    return relaunch_as_admin(Path(__file__))


def _ensure_input_privileges() -> bool:
    if _is_admin():
        return True
    print("[BOT] Для управления GTA нужны права администратора.")
    print("[BOT] Подтвердите запрос Windows UAC.")
    try:
        launched = _request_admin_relaunch()
    except (AttributeError, OSError) as exc:
        print(f"[BOT] Не удалось запросить права: {exc}")
        launched = False
    if launched:
        print("[BOT] Открыта новая консоль с правами администратора.")
    else:
        print("[BOT] Запуск отменён.")
    return False


class SustenanceBot:
    def __init__(self) -> None:
        self._config = load_config(SUSTENANCE_CONFIG_PATH)
        self._capture = GameCapture(self._config)
        self._input = InputHandler(self._config)
        self._service = SustenanceService(self._config, self._capture, self._input)
        self._sm = StateMachine()
        self._listener: keyboard.Listener | None = None
        hotkeys = self._config.get("hotkeys", {})
        self._toggle_name = str(hotkeys.get("toggle", "f9"))
        self._check_name = str(hotkeys.get("check_now", "f8"))
        self._stop_name = str(hotkeys.get("stop", "f10"))
        self._last_status = 0.0

    @staticmethod
    def _hotkey(name: str):
        return parse_hotkey(name, extended=False)

    def _on_press(self, key) -> None:
        if key == self._hotkey(self._toggle_name):
            self._sm.toggle()
            print(f"\n[BOT] {self._sm.state_label()}", flush=True)
        elif key == self._hotkey(self._check_name):
            self._service.request_check()
            print("\n[BOT] Запрошена проверка еды и воды.", flush=True)
        elif key == self._hotkey(self._stop_name):
            self._sm.stop()
            self._input.release_all_keys()
            print("\n[BOT] STOPPED", flush=True)

    def run(self) -> None:
        print("=" * 58)
        print("  Majestic RP — отдельный бот еды и воды")
        print("=" * 58)
        print(f"  {self._toggle_name.upper()}: старт/пауза")
        print(f"  {self._check_name.upper()}: проверить прямо сейчас")
        print(f"  {self._stop_name.upper()}: остановка")
        print("  При значке стакана бот откроет I и дважды нажмёт воду.")
        print("  При значке вилки с ножом — найдёт и съест батончик.")
        print("  Основной HUD проверяется раз в минуту.")
        print("  Без значка потребности инвентарь не открывается.")
        print("=" * 58)

        try:
            rect = self._capture.find_window()
            if rect:
                print(f"[BOT] Захват: {rect.width}x{rect.height} ({self._capture.window_identity})")
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
            while self._sm.state != BotState.STOPPED:
                if self._sm.running:
                    self._service.tick()
                now = time.perf_counter()
                if now - self._last_status >= 2.0:
                    stats = self._service.stats
                    print(
                        f"\r{self._sm.state_label()} | "
                        f"выпито={stats['drinks']} съедено={stats['meals']} "
                        f"вода={self._service.last_thirst_score:.2f} "
                        f"голод={self._service.last_hunger_score:.2f}".ljust(115),
                        end="",
                        flush=True,
                    )
                    self._last_status = now
                time.sleep(0.10)
        except KeyboardInterrupt:
            print("\n[BOT] Остановлено с клавиатуры.")
        finally:
            close_bot_resources(self._input.release_all_keys, self._capture, self._listener)
            print("\n[BOT] Завершено.")


def main() -> None:
    if not _ensure_input_privileges():
        return
    SustenanceBot().run()


if __name__ == "__main__":
    main()
