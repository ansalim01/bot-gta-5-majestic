"""Record movement routes: WASD + mouse (как в игре)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pynput import keyboard, mouse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config_loader import ROUTES_DIR, load_config
from bot.navigation.route_player import save_route

MOUSE_SAMPLE_MS = 33
STOP_KEYS = {keyboard.Key.esc}

_KEY_MAP = {
    "f1": keyboard.Key.f1,
    "f2": keyboard.Key.f2,
    "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4,
    "f5": keyboard.Key.f5,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
    "insert": keyboard.Key.insert,
    "ins": keyboard.Key.insert,
    "home": keyboard.Key.home,
    "pause": keyboard.Key.pause,
}


def parse_record_key(name: str) -> keyboard.Key:
    key = _KEY_MAP.get(name.lower().replace(" ", ""))
    if key is None:
        raise ValueError(f"Unknown key: {name}")
    return key


def record_key_label(key: keyboard.Key) -> str:
    for label, mapped in _KEY_MAP.items():
        if mapped == key and len(label) <= 4:
            return label.upper()
    return str(key)


class RouteRecorder:
    def __init__(self, route_name: str, record_key: keyboard.Key) -> None:
        self._route_name = route_name
        self._record_key = record_key
        self._events: list[dict] = []
        self._start_time = 0.0
        self._recording = False
        self._pressed: set[str] = set()
        self._last_mouse_pos: tuple[int, int] | None = None
        self._last_mouse_sample = 0.0
        self._key_map = {
            "w": "w",
            "a": "a",
            "s": "s",
            "d": "d",
            "shift": "shift",
            "shift_l": "shift",
            "shift_r": "shift",
        }

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start_time) * 1000

    def _append(self, event: dict) -> None:
        self._events.append(event)

    def _normalize_key(self, key) -> str | None:
        if key == self._record_key:
            return None
        if hasattr(key, "char") and key.char:
            return self._key_map.get(key.char.lower())
        if hasattr(key, "name"):
            return self._key_map.get(key.name.lower())
        return None

    def _toggle_recording(self) -> None:
        if not self._recording:
            self._events.clear()
            self._pressed.clear()
            self._last_mouse_pos = None
            self._start_time = time.perf_counter()
            self._recording = True
            print("\n>>> Recording STARTED — переключитесь в игру (Alt+Tab)")
        else:
            self._recording = False
            for key in list(self._pressed):
                self._append(
                    {
                        "time_ms": round(self._elapsed_ms(), 1),
                        "action": "keyup",
                        "key": key,
                    }
                )
            self._pressed.clear()
            path = save_route(self._route_name, self._events, ROUTES_DIR)
            mouse_events = sum(1 for e in self._events if e.get("action") == "mouserel")
            key_events = len(self._events) - mouse_events
            print(f"\n>>> Recording STOPPED — {key_events} keys, {mouse_events} mouse")
            print(f">>> Saved: {path}")

    def on_move(self, x: int, y: int) -> None:
        if not self._recording:
            return
        now = time.perf_counter()
        if now - self._last_mouse_sample < MOUSE_SAMPLE_MS / 1000.0:
            return
        self._last_mouse_sample = now

        if self._last_mouse_pos is None:
            self._last_mouse_pos = (x, y)
            return

        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)
        if dx == 0 and dy == 0:
            return

        self._append(
            {
                "time_ms": round(self._elapsed_ms(), 1),
                "action": "mouserel",
                "dx": int(dx),
                "dy": int(dy),
            }
        )

    def on_press(self, key) -> None:
        if key == self._record_key:
            self._toggle_recording()
            return

        if key in STOP_KEYS:
            return False

        if not self._recording:
            return

        normalized = self._normalize_key(key)
        if normalized and normalized not in self._pressed:
            self._pressed.add(normalized)
            self._append(
                {
                    "time_ms": round(self._elapsed_ms(), 1),
                    "action": "keydown",
                    "key": normalized,
                }
            )

    def on_release(self, key) -> None:
        if key in STOP_KEYS:
            if self._recording:
                self._toggle_recording()
            return False

        if not self._recording:
            return

        normalized = self._normalize_key(key)
        if normalized and normalized in self._pressed:
            self._pressed.discard(normalized)
            self._append(
                {
                    "time_ms": round(self._elapsed_ms(), 1),
                    "action": "keyup",
                    "key": normalized,
                }
            )


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/record_route.py <route_name> [toggle_key]")
        print("  По умолчанию toggle из config/miner.yaml -> hotkeys.record_toggle (F11)")
        print("  ESC = остановить и сохранить")
        sys.exit(1)

    route_name = sys.argv[1]
    config = load_config()
    key_name = (
        sys.argv[2] if len(sys.argv) > 2 else config.get("hotkeys", {}).get("record_toggle", "f11")
    )
    record_key = parse_record_key(key_name)
    label = record_key_label(record_key)

    recorder = RouteRecorder(route_name, record_key)

    print(f"Route recorder: {route_name}")
    print(f"{label} = start/stop записи (глобально, из игры тоже)")
    print("ESC = stop + save | Запись: W + мышь")
    print("1) Встаньте в начало пути  2) Нажмите", label, " 3) Идите в игре  4) Снова", label)

    mouse_listener = mouse.Listener(on_move=recorder.on_move)
    mouse_listener.start()

    with keyboard.Listener(
        on_press=recorder.on_press,
        on_release=recorder.on_release,
    ) as keyboard_listener:
        keyboard_listener.join()

    mouse_listener.stop()


if __name__ == "__main__":
    main()
