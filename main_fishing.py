"""Majestic RP Fisher Bot — entry point."""

from __future__ import annotations

import time
from pathlib import Path

from pynput import keyboard

from bot.capture import GameCapture
from bot.config_loader import (
    FISHING_CONFIG_PATH,
    LOGS_DIR,
    SUSTENANCE_CONFIG_PATH,
    load_config,
)
from bot.hotkeys import parse_hotkey
from bot.input_handler import InputHandler
from bot.jobs.fisher import FisherJob
from bot.runtime import close_bot_resources
from bot.services.sustenance import SustenanceService
from bot.state_machine import BotState, StateMachine
from bot.windows import is_admin as _is_admin
from bot.windows import relaunch_as_admin


def _request_admin_relaunch() -> bool:
    return relaunch_as_admin(Path(__file__))


def _ensure_input_privileges() -> bool:
    """Elevate once on Windows so GTA accepts synthetic keyboard input."""
    if _is_admin():
        return True

    print("[BOT] Для управления GTA нужны права администратора.")
    print("[BOT] Запрашиваю перезапуск через UAC — подтвердите окно Windows.")
    try:
        launched = _request_admin_relaunch()
    except (AttributeError, OSError) as exc:
        print(f"[BOT] Не удалось открыть окно UAC: {exc}")
        launched = False

    if launched:
        print("[BOT] Открыта новая консоль с правами администратора.")
    else:
        print("[BOT] Запуск отменён. Без этих прав игра не принимает Space/A/D.")
    return False


class FisherBot:
    def __init__(self) -> None:
        self._config = load_config(FISHING_CONFIG_PATH)
        self._capture = GameCapture(self._config)
        self._input = InputHandler(self._config)
        self._sm = StateMachine()
        self._sustenance = SustenanceService(
            load_config(SUSTENANCE_CONFIG_PATH),
            self._capture,
            self._input,
        )
        self._job = FisherJob(
            self._config,
            self._capture,
            self._input,
            self._sm,
            sustenance_service=self._sustenance,
        )
        self._listener: keyboard.Listener | None = None
        self._last_status_time = 0.0
        self._tick_ms_sum = 0.0
        self._tick_ms_max = 0.0
        self._tick_ms_count = 0

        hotkeys = self._config.get("hotkeys", {})
        self._toggle_key = hotkeys.get("toggle", "f9")
        self._stop_key = hotkeys.get("stop", "f10")

    def _parse_hotkey(self, name: str):
        return parse_hotkey(name, extended=True)

    def _hotkey_label(self, name: str) -> str:
        return name.replace("_", " ").upper()

    def _on_press(self, key) -> None:
        toggle = self._parse_hotkey(self._toggle_key)
        stop = self._parse_hotkey(self._stop_key)

        if key == toggle:
            self._sm.toggle()
            print(f"[BOT] {self._sm.state_label()}")
        elif key == stop:
            self._sm.stop()
            self._input.release_all_keys()
            print("[BOT] STOPPED")

    def start_hotkeys(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def run(self) -> None:
        mode = self._config.get("mode", "cast_and_reel")
        print("=" * 50)
        print("  Majestic RP Fisher Bot")
        print("=" * 50)
        print("  Privilege: Administrator")
        print(f"  Mode: {mode}")
        print(
            f"  Toggle: {self._hotkey_label(self._toggle_key)}  |  Stop: {self._hotkey_label(self._stop_key)}"
        )
        if mode == "minigame_only":
            print("  Вы сами забрасываете — бот подсекает и вываживает (A/D)")
        elif mode == "cast_and_reel":
            auto = self._config.get("fishing", {}).get("auto_cast", False)
            if auto:
                print("  Бот забрасывает, ждёт поклёвку, подсекает и вываживает")
            else:
                print("  Вы забрасываете сами — бот ловит тайминг, клёв, вываживание")
        elif mode == "full":
            print("  Полный цикл: маршруты + рыбалка + сдача улова")
        print("  Калибровка: python tools/calibrate.py screenshot.png --fishing")
        print("  Тест:       python tools/test_fishing.py screenshot.png")
        if self._config.get("fishing", {}).get("auto_start_activity", True):
            print("  Старт:      E на приглашении -> E после выбора снаряжения")
        if self._config.get("fishing", {}).get("sustenance_enabled", True):
            print("  Еда/вода:   проверка перед забросом; ESC -> предметы -> повторный старт")
        target = max(
            0,
            int(self._config.get("fishing", {}).get("target_fish_count", 0)),
        )
        if target:
            print(f"  Цель:       поймать {target} рыб и завершить работу")
        if self._config.get("telemetry", {}).get("enabled", True):
            print("  Аналитика:  logs/fishing_sessions/latest_summary.txt")
        print("=" * 50)

        try:
            rect = self._capture.find_window()
            cfg_res = self._config.get("screen", {}).get("resolution", [1920, 1080])
            if rect:
                sx, sy = self._capture.scale_factors
                print(f"Game window: {rect.width}x{rect.height} at ({rect.left}, {rect.top})")
                print(f"Capture target: {self._capture.window_identity}")
                print(f"Config base: {cfg_res[0]}x{cfg_res[1]}  scale: {sx:.3f}x / {sy:.3f}y")
            else:
                print("WARNING: Game window not found, using primary monitor fallback")

            dbg = self._config.get("debug", {})
            if dbg.get("enabled"):
                print(f"  Debug карта:  {LOGS_DIR / 'live_debug_fishing.png'}")
                print(f"  Этап 1 crop:  {LOGS_DIR / 'timing_bar_live.png'}")
                print(f"  Этап 2 crop:  {LOGS_DIR / 'bite_text_live.png'}")
                if dbg.get("live_window"):
                    print("  Live окно:   cv2.imshow (live_window: true)")

            self.start_hotkeys()

            while self._sm.state != BotState.STOPPED:
                tick_start = time.perf_counter()
                self._job.tick()
                tick_ms = (time.perf_counter() - tick_start) * 1000.0
                self._tick_ms_sum += tick_ms
                self._tick_ms_max = max(self._tick_ms_max, tick_ms)
                self._tick_ms_count += 1

                now = time.perf_counter()
                if now - self._last_status_time >= 2.0:
                    self._last_status_time = now
                    stats = self._job.stats
                    avg_ms = self._tick_ms_sum / self._tick_ms_count if self._tick_ms_count else 0.0
                    perf = (
                        f" | tick avg/max: {avg_ms:.0f}/{self._tick_ms_max:.0f}ms"
                        if self._config.get("debug", {}).get("enabled")
                        else ""
                    )
                    status = (
                        f"{self._sm.state_label()} | "
                        f"FPS: {self._job.fps:.0f} | "
                        f"fish: {stats['fish_caught']}"
                        f"{'/' + str(target) if target else ''} | "
                        f"casts: {stats['casts']} | "
                        f"food/water: {stats['meals']}/{stats['drinks']}{perf}"
                    )
                    # Pad the \r status line so shorter reprints don't leave
                    # tails of the previous line in the console.
                    print("\r" + status.ljust(70), end="", flush=True)
                    self._job.record_performance(avg_ms, self._tick_ms_max)
                    self._tick_ms_sum = 0.0
                    self._tick_ms_max = 0.0
                    self._tick_ms_count = 0

                time.sleep(0.016)
        except KeyboardInterrupt:
            print("\n[BOT] Interrupted")
        except Exception as exc:
            self._job.record_unhandled_error(exc)
            raise
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        close_bot_resources(self._job.cleanup, self._capture, self._listener)
        print("\n[BOT] Shutdown complete")


def main() -> int:
    if not _ensure_input_privileges():
        return 0
    bot = FisherBot()
    bot.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
