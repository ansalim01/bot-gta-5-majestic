"""Screen capture from the GTA game window."""

from __future__ import annotations

import ctypes
import ntpath
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

import mss
import numpy as np

try:
    import pygetwindow as gw
except ImportError:
    gw = None


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int

    @property
    def monitor_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def _ensure_dpi_aware() -> None:
    """Without DPI awareness Windows virtualizes GetWindowRect/capture
    coordinates whenever display scaling isn't 100%, silently shifting every
    region. Idempotent; fails harmlessly if awareness is already set."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class GameCapture:
    def __init__(self, config: dict[str, Any]) -> None:
        _ensure_dpi_aware()
        self._config = config
        self._titles: list[str] = config.get("screen", {}).get(
            "window_titles",
            ["Grand Theft Auto V", "GTA5", "Majestic"],
        )
        self._process_names = {
            str(name).lower().removesuffix(".exe")
            for name in config.get("screen", {}).get(
                "process_names",
                ["GTA5", "PlayGTAV", "RAGEMP"],
            )
        }
        self._resolution = config.get("screen", {}).get("resolution", [1920, 1080])
        self._force_monitor = bool(config.get("screen", {}).get("force_monitor", False))
        self._monitor_index = int(config.get("screen", {}).get("monitor_index", 1))
        self._rect: WindowRect | None = None
        self._window_identity = "not found"
        self._sct = mss.mss()
        # find_window() walks every top-level window (EnumWindows + GetWindowTextW
        # per window) — calling it on every single grab() was a real per-frame cost.
        # The window almost never moves mid-session, so re-check at most once a
        # second instead of every frame.
        self._last_refresh = 0.0
        self._refresh_interval = config.get("screen", {}).get("refresh_interval_sec", 1.0)

    def find_window(self) -> WindowRect | None:
        if self._force_monitor:
            rect = self._configured_monitor_rect()
            if rect is not None:
                self._rect = rect
                self._window_identity = (
                    f"forced monitor[{self._monitor_index}] {rect.width}x{rect.height}"
                )
                return rect
        for finder in (self._find_window_win32, self._find_window_pygetwindow):
            rect = finder()
            if rect is not None:
                self._rect = self._clamp_to_monitor(rect)
                return self._rect
        return None

    def _configured_monitor_rect(self) -> WindowRect | None:
        try:
            monitors = self._sct.monitors
        except Exception:
            return None
        if len(monitors) <= 1:
            return None
        index = self._monitor_index
        if index < 1 or index >= len(monitors):
            index = 1
        monitor = monitors[index]
        return WindowRect(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )

    @property
    def window_identity(self) -> str:
        return self._window_identity

    def _clamp_to_monitor(self, rect: WindowRect) -> WindowRect:
        """GetWindowRect() on a borderless/maximized GTA window commonly reports
        a few px of invisible resize border past the real screen edge (e.g.
        left=-8, top=-8, width=screen_w+16). Capturing that exact rect makes
        mss grab a region that straddles outside the monitor, which is both
        wasteful (extra pixels copied every frame for nothing) and was
        suspected of pushing mss onto a slower capture path on some setups.
        Clip the rect to whichever monitor it overlaps most before grabbing."""
        try:
            monitors = self._sct.monitors[1:] or self._sct.monitors
        except Exception:
            return rect
        if not monitors:
            return rect

        def overlap_area(mon: dict[str, int]) -> int:
            ox1 = max(rect.left, mon["left"])
            oy1 = max(rect.top, mon["top"])
            ox2 = min(rect.left + rect.width, mon["left"] + mon["width"])
            oy2 = min(rect.top + rect.height, mon["top"] + mon["height"])
            return max(0, ox2 - ox1) * max(0, oy2 - oy1)

        best = max(monitors, key=overlap_area)
        x1 = max(rect.left, best["left"])
        y1 = max(rect.top, best["top"])
        x2 = min(rect.left + rect.width, best["left"] + best["width"])
        y2 = min(rect.top + rect.height, best["top"] + best["height"])
        if x2 <= x1 or y2 <= y1:
            return rect
        return WindowRect(left=x1, top=y1, width=x2 - x1, height=y2 - y1)

    def _find_window_win32(self) -> WindowRect | None:
        if sys.platform != "win32":
            return None

        user32 = ctypes.windll.user32
        matches: list[tuple[int, WindowRect, str]] = []

        def process_name(hwnd: int) -> str:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if not pid.value:
                return ""

            kernel32 = ctypes.windll.kernel32
            open_process = kernel32.OpenProcess
            open_process.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            open_process.restype = wintypes.HANDLE
            handle = open_process(0x1000, False, pid.value)  # limited query
            if not handle:
                return ""
            query_name = kernel32.QueryFullProcessImageNameW
            query_name.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            query_name.restype = wintypes.BOOL
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            try:
                size = wintypes.DWORD(32768)
                path = ctypes.create_unicode_buffer(size.value)
                if not query_name(
                    handle,
                    0,
                    path,
                    ctypes.byref(size),
                ):
                    return ""
                filename = ntpath.basename(path.value)
                return ntpath.splitext(filename)[0].lower()
            finally:
                close_handle(handle)

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value.lower()

            proc_name = process_name(hwnd)
            title_match = any(t.lower() in title for t in self._titles)
            process_match = proc_name in self._process_names
            if not title_match and not process_match:
                return True
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width > 100 and height > 100:
                # An exact game process beats a broad title such as "Majestic",
                # which also matches the launcher/webengine windows.
                priority = 2 if process_match else 1
                matches.append(
                    (
                        priority,
                        WindowRect(rect.left, rect.top, width, height),
                        f"process={proc_name or '?'} title={title or '<empty>'}",
                    )
                )
            return True

        try:
            user32.EnumWindows(enum_proc, 0)
        except Exception:
            return None

        if not matches:
            return None
        selected = max(
            matches,
            key=lambda item: (item[0], item[1].width * item[1].height),
        )
        self._window_identity = selected[2]
        return selected[1]

    def _find_window_pygetwindow(self) -> WindowRect | None:
        if gw is None:
            return None

        for title in self._titles:
            windows = gw.getWindowsWithTitle(title)
            for win in windows:
                if win.width > 100 and win.height > 100:
                    self._window_identity = f"title={win.title or '<empty>'} (pygetwindow)"
                    return WindowRect(
                        left=win.left,
                        top=win.top,
                        width=win.width,
                        height=win.height,
                    )

        for win in gw.getAllWindows():
            title_lower = (win.title or "").lower()
            if any(t.lower() in title_lower for t in self._titles):
                if win.width > 100 and win.height > 100:
                    self._window_identity = f"title={win.title or '<empty>'} (pygetwindow)"
                    return WindowRect(
                        left=win.left,
                        top=win.top,
                        width=win.width,
                        height=win.height,
                    )
        return None

    def _fallback_fullscreen(self) -> WindowRect:
        configured = self._configured_monitor_rect()
        if configured is not None:
            self._rect = configured
            return self._rect

        w, h = self._resolution
        self._rect = WindowRect(left=0, top=0, width=w, height=h)
        return self._rect

    def refresh_window(self) -> WindowRect | None:
        now = time.perf_counter()
        if self._rect is not None and now - self._last_refresh < self._refresh_interval:
            return self._rect
        self._last_refresh = now
        old_rect = self._rect
        rect = self.find_window()
        if rect is None:
            return self._rect
        if old_rect is not None and rect != old_rect:
            print(
                "[BOT] game window changed: "
                f"{old_rect.width}x{old_rect.height}@({old_rect.left},{old_rect.top}) "
                f"-> {rect.width}x{rect.height}@({rect.left},{rect.top}); "
                f"{self._window_identity}",
                flush=True,
            )
        return rect

    @property
    def rect(self) -> WindowRect:
        if self._rect is None:
            found = self.find_window()
            if found is None:
                return self._fallback_fullscreen()
        assert self._rect is not None
        return self._rect

    @property
    def scale_factors(self) -> tuple[float, float]:
        rect = self.rect
        cfg_w, cfg_h = self._resolution[0], self._resolution[1]
        return rect.width / cfg_w, rect.height / cfg_h

    def grab(self) -> np.ndarray:
        rect = self.refresh_window() or self.rect
        screenshot = self._sct.grab(rect.monitor_dict)
        frame = np.array(screenshot, dtype=np.uint8)
        return np.ascontiguousarray(frame[:, :, :3])

    def grab_region(self, region: list[int]) -> np.ndarray:
        """Grab a region relative to the game window [x, y, w, h]."""
        from bot.config_loader import scale_region

        rect = self.rect
        scaled = scale_region(region, self._resolution, rect.width, rect.height)
        x, y, w, h = scaled
        monitor = {
            "left": rect.left + x,
            "top": rect.top + y,
            "width": max(w, 1),
            "height": max(h, 1),
        }
        screenshot = self._sct.grab(monitor)
        frame = np.array(screenshot, dtype=np.uint8)
        return np.ascontiguousarray(frame[:, :, :3])

    def grab_union(self, regions: list[list[int]]) -> np.ndarray:
        """Capture only the bounding box covering several config-space
        [x, y, w, h] regions in ONE mss call, placed into a full-window-sized
        zero canvas so existing crop_from_frame() calls keep working unchanged
        (it derives scale from frame.shape, which must stay window-sized).
        On setups where capture cost scales with pixel area (some GDI/BitBlt
        paths do, especially with a fullscreen-exclusive game), grabbing a
        small bbox instead of the whole 1920x1080 window is dramatically
        cheaper when only a handful of small regions are actually needed."""
        from bot.config_loader import scale_region

        rect = self.rect
        scaled = [scale_region(r, self._resolution, rect.width, rect.height) for r in regions]
        x1 = max(0, min((r[0] for r in scaled), default=0))
        y1 = max(0, min((r[1] for r in scaled), default=0))
        x2 = min(rect.width, max((r[0] + r[2] for r in scaled), default=0))
        y2 = min(rect.height, max((r[1] + r[3] for r in scaled), default=0))

        canvas = np.zeros((rect.height, rect.width, 3), dtype=np.uint8)
        if x2 <= x1 or y2 <= y1:
            return canvas

        monitor = {
            "left": rect.left + x1,
            "top": rect.top + y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
        screenshot = self._sct.grab(monitor)
        sub = np.array(screenshot, dtype=np.uint8)[:, :, :3]
        canvas[y1:y2, x1:x2] = sub
        return canvas

    def crop_from_frame(self, frame: np.ndarray, region: list[int]) -> np.ndarray:
        """Slice a region out of an already-captured full-window frame — no new
        mss/GDI call. Each separate mss.grab() costs ~10ms+ regardless of region
        size, so batching many small region reads into one full-frame grab plus
        cheap numpy slicing is far faster than grabbing each region individually."""
        from bot.config_loader import scale_region

        h, w = frame.shape[:2]
        x, y, rw, rh = scale_region(region, self._resolution, w, h)
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        rw = max(0, min(rw, w - x))
        rh = max(0, min(rh, h - y))
        return frame[y : y + rh, x : x + rw]

    def scaled_region(self, region: list[int]) -> list[int]:
        from bot.config_loader import scale_region

        rect = self.rect
        return scale_region(region, self._resolution, rect.width, rect.height)

    def to_screen_coords(self, x: int, y: int, region: list[int] | None = None) -> tuple[int, int]:
        """Convert relative coords to absolute screen coords."""
        rect = self.rect
        if region:
            from bot.config_loader import scale_region

            region = scale_region(region, self._resolution, rect.width, rect.height)
        offset_x = region[0] if region else 0
        offset_y = region[1] if region else 0
        return rect.left + offset_x + x, rect.top + offset_y + y

    def close(self) -> None:
        self._sct.close()
