"""Majestic RP outdoor gym bot entry point."""

from __future__ import annotations

import time
from pathlib import Path

from pynput import keyboard

from bot.capture import GameCapture
from bot.config_loader import (
    GYM_CONFIG_PATH,
    SUSTENANCE_CONFIG_PATH,
    load_config,
)
from bot.hotkeys import parse_hotkey
from bot.input_handler import InputHandler
from bot.jobs.gym import GymJob
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
    print("[BOT] Для ввода в GTA нужны права администратора.")
    print("[BOT] Подтвердите запрос Windows UAC.")
    try:
        launched = _request_admin_relaunch()
    except (AttributeError, OSError) as exc:
        print(f"[BOT] Не удалось запросить права администратора: {exc}")
        launched = False
    if launched:
        print("[BOT] Бот открыт в новой консоли с правами администратора.")
    else:
        print("[BOT] Запуск отменён: без этих прав GTA может не принять ввод.")
    # The elevated child owns the bot. Always stop this non-admin parent.
    return False


class GymBot:
    def __init__(self) -> None:
        self._config = load_config(GYM_CONFIG_PATH)
        self._capture = GameCapture(self._config)
        self._input = InputHandler(self._config)
        self._sm = StateMachine()
        self._sustenance = SustenanceService(
            load_config(SUSTENANCE_CONFIG_PATH),
            self._capture,
            self._input,
        )
        self._job = GymJob(
            self._config,
            self._capture,
            self._input,
            self._sm,
            before_cycle=self._check_sustenance_before_cycle,
        )
        self._listener: keyboard.Listener | None = None
        hotkeys = self._config.get("hotkeys", {})
        self._toggle_name = str(hotkeys.get("toggle", "f9"))
        self._stop_name = str(hotkeys.get("stop", "f10"))
        self._last_status = 0.0

    def _check_sustenance_before_cycle(self) -> None:
        """Check HUD once immediately before GymJob presses E."""
        if not self._sm.running:
            return
        print("\n[BOT] Перед подходом проверяю голод и жажду...", flush=True)
        self._sustenance.request_check()
        result = self._sustenance.tick(allow_action=self._sm.running)
        if result.drank or result.ate:
            actions = []
            if result.ate:
                actions.append("поел")
            if result.drank:
                actions.append("выпил воду")
            print(
                f"[BOT] {' и '.join(actions)}; продолжаю качалку.",
                flush=True,
            )
        elif result.reason in ("no_need", "hud_wait"):
            print("[BOT] Еда и вода не нужны; продолжаю.", flush=True)

    @staticmethod
    def _hotkey(name: str):
        return parse_hotkey(name, extended=False)

    def _on_press(self, key) -> None:
        if key == self._hotkey(self._toggle_name):
            self._sm.toggle()
            print(f"\n[BOT] {self._sm.state_label()}", flush=True)
        elif key == self._hotkey(self._stop_name):
            self._sm.stop()
            self._input.release_all_keys()
            print("\n[BOT] STOPPED", flush=True)

    def run(self) -> None:
        print("=" * 54)
        print("  Majestic RP — бот для уличной качалки")
        print("=" * 54)
        print(f"  {self._toggle_name.upper()}: старт/пауза")
        print(f"  {self._stop_name.upper()}: полная остановка")
        print("  Подойдите к снаряду и нажмите F9.")
        print("  Перед каждым E бот проверит на HUD голод и жажду.")
        print("  Затем нажмёт E и распознает W/D/S/A.")
        print("=" * 54)

        try:
            rect = self._capture.find_window()
            if rect:
                print(f"[BOT] Окно: {rect.width}x{rect.height} ({self._capture.window_identity})")
            else:
                print("[BOT] Окно GTA не найдено, используется основной монитор.")

            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
            while self._sm.state != BotState.STOPPED:
                self._job.tick()
                now = time.perf_counter()
                if now - self._last_status >= 2.0:
                    stats = self._job.stats
                    sustain = self._sustenance.stats
                    print(
                        f"\r{self._sm.state_label()} | циклы: {stats['cycles']} | "
                        f"нажатия: {stats['prompts']} "
                        f"(W:{stats['w']} D:{stats['d']} "
                        f"S:{stats['s']} A:{stats['a']}) | "
                        f"еда:{sustain['meals']} вода:{sustain['drinks']}".ljust(125),
                        end="",
                        flush=True,
                    )
                    self._last_status = now
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n[BOT] Остановлено с клавиатуры.")
        finally:
            close_bot_resources(self._job.cleanup, self._capture, self._listener)
            print("\n[BOT] Завершено.")


def main() -> None:
    if not _ensure_input_privileges():
        return
    GymBot().run()


if __name__ == "__main__":
    main()
