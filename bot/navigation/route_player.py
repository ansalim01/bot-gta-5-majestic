"""Route recording and playback for navigation."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import yaml

from bot.config_loader import read_config_mapping
from bot.input_handler import InputHandler


class RoutePlayer:
    def __init__(self, input_handler: InputHandler) -> None:
        self._input = input_handler
        self._route: list[dict[str, Any]] = []
        self._index = 0
        self._start_time = 0.0
        self._playing = False
        self._route_name = ""
        self._mouse_jitter = 0
        self._auto_walk = False

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def route_name(self) -> str:
        return self._route_name

    def load(self, path: Path) -> bool:
        if self._playing:
            self.stop()
        self._route = []
        self._route_name = ""
        self._index = 0
        self._auto_walk = False
        if not path.exists():
            return False
        data = read_config_mapping(path)
        events = data.get("events", [])
        self._validate_events(events)
        name = data.get("name", path.stem)
        if not isinstance(name, str):
            raise ValueError(f"Route name must be a string: {path}")
        self._route = events
        self._route_name = name
        self._auto_walk = not any(
            e.get("action") == "keydown" and e.get("key") == "w" for e in self._route
        )
        return len(self._route) > 0

    @staticmethod
    def _validate_events(events: Any) -> None:
        if not isinstance(events, list):
            raise ValueError("Route events must be a list")
        previous_time = 0.0
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"Route event {index} must be a mapping")
            timestamp = event.get("time_ms")
            if (
                type(timestamp) not in (int, float)
                or (isinstance(timestamp, float) and not math.isfinite(timestamp))
                or timestamp < previous_time
            ):
                raise ValueError(f"Route event {index} has an invalid or decreasing time_ms")
            previous_time = timestamp
            action = event.get("action")
            if action in ("keydown", "keyup", "press"):
                if not isinstance(event.get("key"), str) or not event["key"]:
                    raise ValueError(f"Route event {index} requires a key")
            elif action == "mouserel":
                if any(type(event.get(axis, 0)) is not int for axis in ("dx", "dy")):
                    raise ValueError(f"Route event {index} requires integer mouse offsets")
            else:
                raise ValueError(f"Route event {index} has an unknown action: {action!r}")

    def start(self) -> None:
        if not self._route:
            return
        self._index = 0
        self._start_time = time.perf_counter()
        self._playing = True
        if self._auto_walk:
            self._input.key_down("w")
        self._apply_event(self._route[0])

    def stop(self) -> None:
        self._playing = False
        if self._auto_walk:
            self._input.key_up("w")
        self._input.release_all_keys()

    def tick(self) -> bool:
        """Advance route playback. Returns True when route finished."""
        if not self._playing or not self._route:
            return True

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        while self._index < len(self._route) - 1:
            next_event = self._route[self._index + 1]
            jitter = 0 if next_event.get("action") == "mouserel" else random.uniform(-15, 15)
            if elapsed_ms >= next_event["time_ms"] + jitter:
                self._index += 1
                self._apply_event(self._route[self._index])
            else:
                break

        last_time = self._route[-1]["time_ms"] if self._route else 0
        if elapsed_ms >= last_time + 200:
            self.stop()
            return True

        return False

    def _apply_event(self, event: dict[str, Any]) -> None:
        action = event.get("action")
        key = event.get("key", "")

        if action == "keydown":
            self._input.key_down(key)
        elif action == "keyup":
            self._input.key_up(key)
        elif action == "press":
            self._input.press_key(key)
        elif action == "mouserel":
            dx = int(event.get("dx", 0))
            dy = int(event.get("dy", 0))
            if dx or dy:
                self._input.turn_camera(dx, dy)


def save_route(name: str, events: list[dict[str, Any]], routes_dir: Path) -> Path:
    routes_dir.mkdir(parents=True, exist_ok=True)
    path = routes_dir / f"{name}.yaml"
    data = {"name": name, "events": events}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return path
