"""Keyboard and mouse input for the game."""

from __future__ import annotations

import ctypes
import ntpath
import random
import sys
import time
from ctypes import wintypes
from typing import Any

import pydirectinput

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False

# The game samples input once per rendered frame; with PAUSE=0 a bare
# pydirectinput.press() emits down+up in well under a millisecond and the
# game can miss the tap entirely (observed as "Space pressed but no
# hook/cast happened"). Hold every tap for ~3 frames at 60fps.
KEY_HOLD_SEC = 0.045
CLICK_HOLD_SEC = 0.03


class InputHandler:
    def __init__(self, config: dict[str, Any]) -> None:
        mining = config.get("mining", {})
        fishing = config.get("fishing", {})
        self._lmb_interval = mining.get(
            "lmb_interval_ms", fishing.get("cast_interval_ms", [35, 70])
        )
        self._lmb_random = mining.get("lmb_random_ms", fishing.get("cast_random_ms", [5, 25]))
        self._last_lmb_time = 0.0
        self._reel_key: str | None = None
        screen = config.get("screen", {})
        self._require_game_focus = bool(screen.get("require_foreground_for_input", False))
        self._auto_focus_game = bool(screen.get("auto_focus_game_for_input", False))
        self._focus_retry_sec = max(0.2, float(screen.get("focus_retry_sec", 0.75)))
        self._last_focus_attempt_at = 0.0
        self._window_titles = tuple(
            str(title).strip().lower()
            for title in screen.get("window_titles", [])
            if str(title).strip()
        )
        self._process_names = {
            str(name).strip().lower().removesuffix(".exe")
            for name in screen.get("process_names", [])
            if str(name).strip()
        }
        self._last_foreground_identity = ""

    @staticmethod
    def _delay_ms(range_ms: list[int]) -> float:
        return random.uniform(range_ms[0], range_ms[1]) / 1000.0

    @property
    def last_foreground_identity(self) -> str:
        return self._last_foreground_identity

    def is_game_focused(self) -> bool:
        """Return whether input may safely be sent to the foreground window."""
        if not self._require_game_focus or sys.platform != "win32":
            return True
        title, process = self._foreground_window_identity()
        self._last_foreground_identity = f"{title or '<no title>'} [{process or 'unknown process'}]"
        title_match = any(token in title for token in self._window_titles)
        process_match = process in self._process_names
        # When Windows exposes the executable name, treat it as authoritative:
        # a launcher, editor or screenshot viewer may contain "Majestic"/GTA
        # in its title but must never receive the bot's global input.
        return process_match if process else title_match

    def _ensure_game_focused(self) -> bool:
        """Focus the configured game window, then verify it before input."""
        if self.is_game_focused():
            return True
        if not self._auto_focus_game or sys.platform != "win32":
            return False
        now = time.perf_counter()
        if now - self._last_focus_attempt_at < self._focus_retry_sec:
            return False
        self._last_focus_attempt_at = now
        return self._focus_game_window()

    def _focus_game_window(self) -> bool:
        try:
            user32 = ctypes.windll.user32
            matches: list[int] = []
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def callback(hwnd, _lparam):
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = max(0, int(user32.GetWindowTextLengthW(hwnd)))
                title_buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title_buffer, length + 1)
                title = title_buffer.value.strip().lower()
                process_id = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
                process = self._process_name(process_id.value)
                title_match = any(token in title for token in self._window_titles)
                process_match = process in self._process_names
                matches_game = process_match if process else title_match
                if matches_game:
                    matches.append(int(hwnd))
                    return False
                return True

            user32.EnumWindows(callback_type(callback), 0)
            if not matches:
                return False
            hwnd = wintypes.HWND(matches[0])
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.08)
            return self.is_game_focused()
        except (AttributeError, OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _foreground_window_identity() -> tuple[str, str]:
        try:
            user32 = ctypes.windll.user32
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
            user32.GetWindowTextLengthW.restype = ctypes.c_int
            user32.GetWindowTextW.argtypes = [
                wintypes.HWND,
                wintypes.LPWSTR,
                ctypes.c_int,
            ]
            user32.GetWindowThreadProcessId.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.DWORD),
            ]

            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "", ""
            length = max(0, int(user32.GetWindowTextLengthW(hwnd)))
            title_buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title_buffer, length + 1)
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            process = InputHandler._process_name(process_id.value)
            return title_buffer.value.strip().lower(), process
        except (AttributeError, OSError, ValueError):
            return "", ""

    @staticmethod
    def _process_name(process_id: int) -> str:
        if process_id <= 0:
            return ""
        handle = None
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.QueryFullProcessImageNameW.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

            handle = kernel32.OpenProcess(0x1000, False, process_id)
            if not handle:
                return ""
            size = wintypes.DWORD(32768)
            path_buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(
                handle,
                0,
                path_buffer,
                ctypes.byref(size),
            ):
                return ""
            return ntpath.basename(path_buffer.value).lower().removesuffix(".exe")
        except (AttributeError, OSError, ValueError):
            return ""
        finally:
            if handle:
                try:
                    ctypes.windll.kernel32.CloseHandle(handle)
                except (AttributeError, OSError, ValueError):
                    pass

    def click_left(self) -> bool:
        if not self._ensure_game_focused():
            return False
        pydirectinput.mouseDown()
        try:
            time.sleep(CLICK_HOLD_SEC)
        finally:
            pydirectinput.mouseUp()
        return True

    def click_at(self, x: int, y: int) -> bool:
        if not self._ensure_game_focused():
            return False
        pydirectinput.moveTo(x, y)
        time.sleep(0.025)
        pydirectinput.mouseDown(x, y)
        try:
            time.sleep(CLICK_HOLD_SEC)
        finally:
            pydirectinput.mouseUp(x, y)
        return True

    def double_click_at(self, x: int, y: int, interval_sec: float = 0.12) -> bool:
        """Send a real two-click sequence at one inventory slot."""
        if not self._ensure_game_focused():
            return False
        pydirectinput.moveTo(x, y)
        time.sleep(0.025)
        for click_index in range(2):
            pydirectinput.mouseDown(x, y)
            try:
                time.sleep(CLICK_HOLD_SEC)
            finally:
                pydirectinput.mouseUp(x, y)
            if click_index == 0:
                time.sleep(max(interval_sec, 0.0))
        return True

    def press_e(self) -> None:
        self.press_key("e")

    def press_enter(self) -> None:
        self.press_key("enter")

    def press_space(self) -> None:
        self.press_key("space")

    def press_key_named(self, key: str) -> bool:
        key = key.lower()
        if key in ("lmb", "mouse1", "left", "mouse_left"):
            return self.click_left()
        return self.press_key(key)

    def key_down(self, key: str) -> bool:
        if not self._ensure_game_focused():
            return False
        pydirectinput.keyDown(key)
        return True

    def key_up(self, key: str) -> None:
        pydirectinput.keyUp(key)

    def press_key(self, key: str, duration: float = KEY_HOLD_SEC) -> bool:
        if not self._ensure_game_focused():
            return False
        pydirectinput.keyDown(key)
        try:
            time.sleep(duration)
        finally:
            pydirectinput.keyUp(key)
        return True

    def turn_camera(self, dx: int, dy: int = 0) -> bool:
        if dx == 0 and dy == 0:
            return False
        if not self._ensure_game_focused():
            return False
        pydirectinput.moveRel(dx, dy, relative=True)
        return True

    def tap_key(self, key: str, duration_ms: int) -> bool:
        if not self._ensure_game_focused():
            return False
        pydirectinput.keyDown(key)
        try:
            time.sleep(max(duration_ms, 0) / 1000.0)
        finally:
            pydirectinput.keyUp(key)
        return True

    def spam_lmb_if_ready(self) -> bool:
        """Click LMB if enough time passed. Returns True if clicked."""
        now = time.perf_counter()
        interval = self._delay_ms(self._lmb_interval) + self._delay_ms(self._lmb_random)
        if now - self._last_lmb_time >= interval:
            if self.click_left():
                self._last_lmb_time = now
                return True
        return False

    def reset_lmb_timer(self) -> None:
        self._last_lmb_time = 0.0

    def set_reel_direction(self, direction: str | None) -> bool:
        """Hold A or D for reel minigame; None releases both."""
        if direction is not None and not self.is_game_focused():
            direction = None
        if direction == self._reel_key:
            return direction is not None
        for key in ("a", "d"):
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
        self._reel_key = None
        if direction == "left":
            pydirectinput.keyDown("a")
            self._reel_key = "left"
            return True
        elif direction == "right":
            pydirectinput.keyDown("d")
            self._reel_key = "right"
            return True
        return False

    def release_all_keys(self) -> None:
        self._reel_key = None
        # Include every key used by both jobs. In particular, Space/Enter used
        # to be omitted, so an exception or F10 racing a timed press could
        # leave the fishing action held in the game.
        for key in ("w", "a", "s", "d", "shift", "e", "f", "i", "space", "enter", "tab"):
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
        try:
            pydirectinput.mouseUp()
        except Exception:
            pass
