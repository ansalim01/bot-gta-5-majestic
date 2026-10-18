"""Majestic RP Miner Bot — entry point."""

from __future__ import annotations

import time

from pynput import keyboard

from bot.capture import GameCapture
from bot.config_loader import load_config
from bot.hotkeys import parse_hotkey
from bot.input_handler import InputHandler
from bot.jobs.miner import MinerJob
from bot.runtime import close_bot_resources
from bot.state_machine import BotState, StateMachine


class MinerBot:
    def __init__(self) -> None:
        self._config = load_config()
        self._capture = GameCapture(self._config)
        self._input = InputHandler(self._config)
        self._sm = StateMachine()
        self._job = MinerJob(self._config, self._capture, self._input, self._sm)
        self._listener: keyboard.Listener | None = None
        self._last_status_time = 0.0

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
        mode = self._config.get("mode", "minigame_only")
        print("=" * 50)
        print("  Majestic RP Miner Bot")
        print("=" * 50)
        print(f"  Mode: {mode}")
        print(
            f"  Toggle: {self._hotkey_label(self._toggle_key)}  |  Stop: {self._hotkey_label(self._stop_key)}"
        )
        if mode == "minigame_only":
            print("  Вы сами бегаете и жмёте ЛКМ — бот кликает жилы в мини-игре")
            print("  1) F6 в игре — убрать HUD  2) F9 — включить бота")
        elif mode == "walk_minigame":
            print("  Бот идёт по записанному маршруту + кликает жилы")
            print("  Запишите маршруты: python tools/record_route.py fork_tunnel2")
        elif mode == "full":
            nav = self._config.get("navigation", {})
            nav_mode = nav.get("mode", "routes")
            print(f"  Навигация: {nav_mode} | F9 старт | F10 стоп")
            if nav_mode == "routes":
                r = nav.get("routes", {})
                print(f"  Маршруты: {r.get('to_rock')} → камень, {r.get('to_drop')} → сдача")
        else:
            print("  Устройтесь шахтёром, подбегите к камню, нажмите F9")
        print("=" * 50)

        try:
            rect = self._capture.find_window()
            cfg_res = self._config.get("screen", {}).get("resolution", [1920, 1080])
            if rect:
                sx, sy = self._capture.scale_factors
                print(f"Game window: {rect.width}x{rect.height} at ({rect.left}, {rect.top})")
                print(f"Config base: {cfg_res[0]}x{cfg_res[1]}  scale: {sx:.3f}x / {sy:.3f}y")
                if abs(sx - sy) > 0.02:
                    print("NOTE: non-uniform scale (fullscreen borderless) — ROI auto-adjusted")
            else:
                print("WARNING: Game window not found, using primary monitor fallback")

            self.start_hotkeys()

            while self._sm.state != BotState.STOPPED:
                self._job.tick()

                now = time.perf_counter()
                if now - self._last_status_time >= 2.0:
                    self._last_status_time = now
                    stats = self._job.stats
                    print(
                        f"\r{self._sm.state_label()} | "
                        f"FPS: {self._job.fps:.0f} | "
                        f"minigames: {stats['minigames']} | "
                        f"veins: {stats['veins_clicked']}",
                        end="",
                        flush=True,
                    )

                time.sleep(0.016)
        except KeyboardInterrupt:
            print("\n[BOT] Interrupted")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        close_bot_resources(self._job.cleanup, self._capture, self._listener)
        print("\n[BOT] Shutdown complete")


def main():
    bot = MinerBot()
    bot.run()


if __name__ == "__main__":
    main()
