import pytest
from pynput import keyboard

from bot.hotkeys import parse_hotkey


@pytest.mark.parametrize("number", range(1, 13))
def test_function_keys_are_case_insensitive(number):
    expected = getattr(keyboard.Key, f"f{number}")
    assert parse_hotkey(f"F{number}") == expected
    assert parse_hotkey(f"f{number}", extended=True) == expected


@pytest.mark.parametrize("name", ["insert", "delete", "home", "end", "pause"])
def test_common_special_keys(name):
    assert parse_hotkey(name) == getattr(keyboard.Key, name)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ins", keyboard.Key.insert),
        ("del", keyboard.Key.delete),
        ("page_up", keyboard.Key.page_up),
        ("page_down", keyboard.Key.page_down),
        ("scroll_lock", keyboard.Key.scroll_lock),
    ],
)
def test_extended_keys_preserve_runner_specific_support(name, expected):
    assert parse_hotkey(name) == keyboard.Key.f9
    assert parse_hotkey(name, extended=True) == expected


def test_unknown_key_keeps_the_existing_f9_fallback():
    assert parse_hotkey("unknown") == keyboard.Key.f9
