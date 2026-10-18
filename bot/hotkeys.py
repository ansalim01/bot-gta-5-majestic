"""Shared special-key parsing for the standalone bot runners."""

from __future__ import annotations

from collections.abc import Mapping

from pynput import keyboard

_COMMON_KEYS = {
    **{f"f{number}": getattr(keyboard.Key, f"f{number}") for number in range(1, 13)},
    "insert": keyboard.Key.insert,
    "delete": keyboard.Key.delete,
    "home": keyboard.Key.home,
    "end": keyboard.Key.end,
    "pause": keyboard.Key.pause,
}
_EXTENDED_KEYS = {
    **_COMMON_KEYS,
    "ins": keyboard.Key.insert,
    "del": keyboard.Key.delete,
    "page_up": keyboard.Key.page_up,
    "page_down": keyboard.Key.page_down,
    "scroll_lock": keyboard.Key.scroll_lock,
}


def parse_hotkey(name: str, *, extended: bool = False) -> keyboard.Key:
    """Preserve the original F9 fallback and each runner's accepted key set."""
    mapping = _EXTENDED_KEYS if extended else _COMMON_KEYS
    return mapping.get(name.lower(), keyboard.Key.f9)


def validate_hotkeys(
    hotkeys: Mapping[str, str], *, extended: bool = False, all_special_keys: bool = False
) -> None:
    """Reject unsupported or colliding controls before the panel saves them."""
    mapping = (
        keyboard.Key.__members__
        if all_special_keys
        else (_EXTENDED_KEYS if extended else _COMMON_KEYS)
    )
    assigned: dict[keyboard.Key, str] = {}
    for control, name in hotkeys.items():
        key = mapping.get(name.lower()) if isinstance(name, str) else None
        if key is None:
            raise ValueError(f"hotkeys.{control}: неподдерживаемая клавиша {name!r}")
        if key in assigned:
            raise ValueError(f"hotkeys.{control} и hotkeys.{assigned[key]} используют одну клавишу")
        assigned[key] = control
