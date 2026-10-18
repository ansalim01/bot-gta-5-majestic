"""Cast timing bar — white slider must hit the green zone (Space)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


SLIDER_MISSING = -1


@dataclass
class TimingAnalysis:
    active: bool
    in_zone: bool
    slider_x: int
    green_left: int
    green_right: int
    green_width: int
    green_center: int = 0
    # Raw per-frame zone detection. With a locked zone `active` stays True
    # even after the bar left the screen — this is the only signal that says
    # whether the green zone was REALLY seen on this frame.
    zone_live: bool = True

    @property
    def slider_found(self) -> bool:
        return self.slider_x >= 0


class CastTimingDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("cast_timing", {})

        locate = cfg.get("locate_green", {})
        self._locate_low = np.array(locate.get("low", [30, 40, 40]))
        self._locate_high = np.array(locate.get("high", [100, 255, 255]))
        zone = cfg.get("zone_green", {})
        self._zone_low = np.array(zone.get("low", [40, 100, 100]))
        self._zone_high = np.array(zone.get("high", [85, 255, 255]))

        self._slider_gray_min = cfg.get("slider_gray_min", 215)
        # Margin shrinks the hit zone inward from each edge (safety buffer
        # against edge mis-detection). 0 = press anywhere inside the visible
        # green zone — matches the real minigame (no need to hit dead center).
        self._zone_edge_margin_px = int(cfg.get("zone_edge_margin_px", 0))
        # Capture→keypress latency compensation: by the time Space reaches the
        # game the slider has moved on. Real misses in the logs were presses
        # 0-4px from a zone edge (slider=365 zone=242-367, slider=337
        # zone=337-459) followed by a dead 40s bite-timeout. We aim at the
        # PREDICTED slider position (current + velocity * lead) and only press
        # when that lands in the central band of the zone.
        self._press_lead_ms = cfg.get("press_lead_ms", 80)
        # Fraction of the zone width trimmed from EACH edge for the press
        # window (0.2 -> press only in the central 60%).
        self._zone_press_band = cfg.get("zone_press_band_frac", 0.20)
        self._timeout = cfg.get("timeout_sec", 6)
        self._appear_timeout = cfg.get("appear_timeout_sec", 2.0)
        self._poll_ms = cfg.get("poll_ms", 4)
        self._tick_ms = cfg.get("tick_ms", 2)
        self._press_keys = cfg.get("press_keys", ["space"])
        self._prev_slider_x: int | None = None
        self._prev_sample_t: float | None = None
        self._prev_aim_x: float | None = None
        self._inactive_streak = 0
        self._locked_zone: tuple[int, int] | None = None

    @property
    def tick_sec(self) -> float:
        return self._tick_ms / 1000.0

    @property
    def poll_sec(self) -> float:
        return self._poll_ms / 1000.0

    def reset(self) -> None:
        self._prev_slider_x = None
        self._prev_sample_t = None
        self._prev_aim_x = None
        self._inactive_streak = 0
        self._locked_zone = None

    def _scaled_min_green_width(self, frame_width: int) -> int:
        return max(10, int(frame_width * 0.03))

    @staticmethod
    def _segments(indices: np.ndarray, max_gap: int = 4) -> list[tuple[int, int]]:
        if indices.size == 0:
            return []
        out: list[tuple[int, int]] = []
        start = int(indices[0])
        prev = int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if value - prev > max_gap:
                out.append((start, prev))
                start = value
            prev = value
        out.append((start, prev))
        return out

    @staticmethod
    def locate_bar_rect(frame: np.ndarray, green_low: np.ndarray, green_high: np.ndarray) -> list[int] | None:
        if frame.size == 0:
            return None
        h, w = frame.shape[:2]
        y0 = int(h * 0.62)
        search = frame[y0:, :]
        if search.size == 0:
            return None

        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, green_low, green_high)
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)

        row_green = np.sum(green > 0, axis=1)
        row_bright = np.sum(gray > 175, axis=1)
        row_signal = np.maximum(row_green, row_bright)
        if row_signal.max() < max(8, w * 0.015):
            return None

        active_rows = np.where(row_signal >= max(5, w * 0.012))[0]
        if active_rows.size == 0:
            return None

        ry1, ry2 = int(active_rows[0]), int(active_rows[-1])
        band_gray = gray[ry1 : ry2 + 1, :]
        band_green = green[ry1 : ry2 + 1, :]
        band_signal = np.maximum(band_gray > 120, band_green > 0)
        cols = np.where(np.any(band_signal, axis=0))[0]
        if cols.size < int(w * 0.22):
            return None

        pad_x = max(4, int(w * 0.008))
        pad_y = max(2, int(h * 0.006))
        x1 = max(0, int(cols[0]) - pad_x)
        x2 = min(w, int(cols[-1]) + pad_x)
        y1 = max(0, y0 + ry1 - pad_y)
        y2 = min(h, y0 + ry2 + pad_y + 1)
        return [x1, y1, x2 - x1, y2 - y1]

    def crop_bar(self, window_frame: np.ndarray) -> np.ndarray:
        rect = self.locate_bar_rect(window_frame, self._locate_low, self._locate_high)
        if not rect:
            return np.array([])
        x, y, w, h = rect
        crop = window_frame[y : y + h, x : x + w]
        bar_h = max(12, int(h * 0.42))
        return np.ascontiguousarray(crop[:bar_h, :])

    def _find_green_zone(self, zone_green: np.ndarray, frame_width: int) -> tuple[int, int] | None:
        """Largest green blob on the bar (connected component)."""
        h = zone_green.shape[0]
        row_start = int(h * 0.15)
        row_end = max(row_start + 1, int(h * 0.85))
        mask = zone_green[row_start:row_end, :]
        min_w = max(8, int(frame_width * 0.03))
        max_w = int(frame_width * 0.22)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best: tuple[int, int] | None = None
        best_area = 0
        for i in range(1, n):
            x0, _y0, ww, hh, area = stats[i]
            if area < 20 or ww < min_w or ww > max_w or hh < 2:
                continue
            if area > best_area:
                best_area = area
                best = (x0, x0 + ww - 1)

        return best

    def _find_slider_x(self, gray: np.ndarray) -> int | None:
        """Tall bright vertical line — not the small ruler tick marks.

        Vectorized (no per-column Python loop) — this used to be the main
        bottleneck dragging capture rate down to ~6-7 FPS, which made the
        slider jump 100+ px between samples and skip clean over the green
        zone. Fast enough now to run at full capture rate.
        """
        if gray.size == 0:
            return None

        h, w = gray.shape[:2]
        row_a = int(h * 0.18)
        row_b = max(row_a + 1, int(h * 0.82))
        band = gray[row_a:row_b, :]
        min_height = max(4, int(band.shape[0] * 0.45))

        bright = band >= self._slider_gray_min
        height = bright.sum(axis=0).astype(np.float32)
        col_max = band.max(axis=0).astype(np.float32)
        scores = np.where(height >= min_height, height * col_max, 0.0)

        if scores.max() <= 0:
            return None

        peak = int(np.argmax(scores))
        x0 = max(0, peak - 3)
        x1 = min(w, peak + 4)
        return int(min(w - 1, max(0, x0 + np.argmax(scores[x0:x1]))))

    def _in_zone_bounds(self, slider_x: int, green_left: int, green_right: int) -> bool:
        """Anywhere inside the visible green zone counts — not just its center."""
        margin = self._zone_edge_margin_px
        return (green_left + margin) <= slider_x <= (green_right - margin)

    def analyze(self, frame: np.ndarray, *, use_lock: bool = True) -> TimingAnalysis:
        """Full analysis of the timing bar crop.

        use_lock=True (default): zone is locked on first detection and kept
        for the rest of the tracking session for center stability — intended
        ONLY for the active `_tick_cast_timing()` solve loop, which calls
        `reset()` at the start of every new session.

        use_lock=False: stateless "is the bar visible right now" probe, no
        side effects. Use this for one-off phase/state checks (anything
        outside the dedicated solve session) — otherwise a lock acquired
        during one cast session would make `is_active()` report True forever
        afterwards, even after the bar is long gone.
        """
        inactive = TimingAnalysis(False, False, 0, 0, 0, 0, 0)
        if frame.size == 0:
            return inactive

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zone_green = cv2.inRange(hsv, self._zone_low, self._zone_high)

        zone = self._find_green_zone(zone_green, frame.shape[1])
        zone_live = zone is not None
        if use_lock:
            if zone is None and self._locked_zone is None:
                return inactive
            if zone is not None:
                if self._locked_zone is None:
                    self._locked_zone = zone
                else:
                    # Re-validate a lock acquired mid fade-in: if the live
                    # zone overlaps the locked one, widen the lock to their
                    # union. Widen-only, and only for overlapping detections —
                    # keeps the center stable when the slider occludes/splits
                    # the green blob, and ignores unrelated green blobs.
                    lock_l, lock_r = self._locked_zone
                    live_l, live_r = zone
                    if live_l <= lock_r and live_r >= lock_l:
                        self._locked_zone = (min(lock_l, live_l), max(lock_r, live_r))
            green_left, green_right = self._locked_zone
        else:
            if zone is None:
                return inactive
            green_left, green_right = zone
        green_width = green_right - green_left
        green_center = (green_left + green_right) // 2
        slider_x = self._find_slider_x(gray)
        if slider_x is None:
            return TimingAnalysis(
                True, False, SLIDER_MISSING, green_left, green_right, green_width,
                green_center, zone_live=zone_live,
            )

        in_zone = self._in_zone_bounds(slider_x, green_left, green_right)
        return TimingAnalysis(
            True, in_zone, slider_x, green_left, green_right, green_width,
            green_center, zone_live=zone_live,
        )

    def analyze_fresh(self, frame: np.ndarray) -> TimingAnalysis:
        return self.analyze(frame, use_lock=False)

    def is_active(self, frame: np.ndarray) -> bool:
        return self.analyze(frame).active

    def is_active_fresh(self, frame: np.ndarray) -> bool:
        analysis = self.analyze_fresh(frame)
        return analysis.active and analysis.slider_found

    def wait_for_bar(
        self,
        get_frame: Callable[[], np.ndarray],
        should_abort: Callable[[], bool] | None = None,
    ) -> bool:
        deadline = time.perf_counter() + self._appear_timeout
        while time.perf_counter() < deadline:
            if should_abort and should_abort():
                return False
            if self.is_active(get_frame()):
                return True
            time.sleep(self.poll_sec)
        return self.is_active(get_frame())

    @staticmethod
    def _segment_covers_zone(x0: int, x1: int, green_left: int, green_right: int) -> bool:
        a, b = min(x0, x1), max(x0, x1)
        return a <= green_right and b >= green_left

    def evaluate(self, analysis: TimingAnalysis, frame_width: int) -> tuple[bool, str | None]:
        """Decide hit/miss from an already-computed analysis. Keeps prev_slider_x between calls.

        No frame re-analysis here — analyze() is the only expensive call and the
        caller (fisher.py) must compute it exactly once per tick.
        """
        # `active=False` only ever happens before a zone lock exists; once
        # locked, "the bar is gone" shows up as a run of zone_live misses
        # (the lock keeps active=True forever) — both feed the same streak.
        if not analysis.active or not analysis.zone_live:
            self._inactive_streak += 1
            if self._inactive_streak >= 8:
                return False, "lost"
            if not analysis.active:
                return False, None
        else:
            self._inactive_streak = 0

        if not analysis.slider_found:
            return False, None

        green_left, green_right = analysis.green_left, analysis.green_right
        slider_x = analysis.slider_x
        prev = self._prev_slider_x
        now = time.perf_counter()
        dt = None
        if prev is not None and self._prev_sample_t is not None:
            dt = now - self._prev_sample_t
        # Velocity is only trustworthy across a fresh pair of samples — the
        # state-entry ticks run at ~1 FPS and a second-old prev says nothing.
        fresh = dt is not None and 0.0 < dt <= 0.4

        # Detection-glitch filter, time-aware: a jump is bogus only if it
        # implies an impossible slider speed (~3 bar widths per second). The
        # old fixed 150px cutoff also rejected perfectly real movement on the
        # slow first ticks of the state, where 300+ px steps are legitimate.
        if dt is not None and dt > 0.0 and abs(slider_x - prev) / dt > frame_width * 3.0:
            self._commit_sample(slider_x, now, None)
            return False, None

        velocity = (slider_x - prev) / dt if fresh else 0.0

        # Aim at where the slider will be when the game registers Space, and
        # only press inside the central band — an edge press means the slider
        # exits the zone during input latency (the observed misses).
        aim_x = slider_x + velocity * (self._press_lead_ms / 1000.0)
        band = max(self._zone_edge_margin_px, int(analysis.green_width * self._zone_press_band))
        band_left, band_right = green_left + band, green_right - band
        if band_left > band_right:
            band_left = band_right = analysis.green_center
        prev_aim = self._prev_aim_x

        if band_left <= aim_x <= band_right:
            self._commit_sample(slider_x, now, aim_x)
            return True, "in_zone"

        # Fast slider: the aim point can hop clean over the band between two
        # samples — press on the crossing. Fresh pairs only: with a second-old
        # prev the slider may have bounced several times in between, and a
        # press on such a segment is a coin flip that costs a full
        # bite-timeout when it loses.
        if fresh and prev_aim is not None and self._segment_covers_zone(
            int(prev_aim), int(aim_x), band_left, band_right
        ):
            self._commit_sample(slider_x, now, aim_x)
            return True, "crossed_zone"

        # Raw-position backstop for the first fresh samples of a session,
        # when velocity (and therefore aim) is still zero.
        if fresh and self._segment_covers_zone(prev, slider_x, green_left, green_right):
            self._commit_sample(slider_x, now, aim_x)
            return True, "crossed_zone"

        self._commit_sample(slider_x, now, aim_x)
        return False, None

    def _commit_sample(self, slider_x: int, t: float, aim_x: float | None) -> None:
        self._prev_slider_x = slider_x
        self._prev_sample_t = t
        self._prev_aim_x = aim_x

    def solve_step(self, frame: np.ndarray) -> tuple[bool, str | None]:
        """Convenience wrapper: analyze() + evaluate() in one call (used by solve())."""
        analysis = self.analyze(frame)
        return self.evaluate(analysis, frame.shape[1])

    def solve(
        self,
        get_frame: Callable[[], np.ndarray],
        press_key: Callable[[str], None],
        should_abort: Callable[[], bool] | None = None,
        wait_for_appear: bool = True,
    ) -> tuple[bool, str | None]:
        self.reset()
        if wait_for_appear:
            if not self.wait_for_bar(get_frame, should_abort):
                return False, None

        start = time.perf_counter()
        while time.perf_counter() - start < self._timeout:
            if should_abort and should_abort():
                return False, None
            hit, reason = self.solve_step(get_frame())
            if hit:
                for key in self._press_keys:
                    press_key(key)
                time.sleep(0.03)
                return True, reason
            if reason == "lost":
                return False, None
            time.sleep(self.tick_sec)
        return False, None

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        debug = np.ascontiguousarray(frame.copy())
        # Stateless probe: rendering a debug image must never acquire/extend
        # the session's zone lock (this is called from debug paths mid-session).
        analysis = self.analyze(frame, use_lock=False)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        zone = cv2.inRange(hsv, self._zone_low, self._zone_high)
        overlay = debug.copy()
        overlay[zone > 0] = (0, 200, 0)
        cv2.addWeighted(overlay, 0.45, debug, 0.55, 0, debug)

        if analysis.active:
            cx = analysis.green_center
            dist = abs(analysis.slider_x - cx) if analysis.slider_found else -1
            zone_color = (0, 255, 255) if analysis.in_zone else (0, 255, 0)
            cv2.rectangle(
                debug,
                (analysis.green_left, 0),
                (analysis.green_right, debug.shape[0]),
                zone_color,
                2,
            )
            if analysis.slider_found:
                cv2.line(
                    debug,
                    (analysis.slider_x, 0),
                    (analysis.slider_x, debug.shape[0]),
                    (255, 255, 255),
                    2,
                )
            if analysis.in_zone:
                label = "HIT!"
            elif analysis.slider_found:
                label = (
                    f"slider {analysis.slider_x} dist={dist} "
                    f"-> zone {analysis.green_left}-{analysis.green_right}"
                )
            else:
                label = f"no slider -> zone {analysis.green_left}-{analysis.green_right}"
            cv2.putText(
                debug,
                label,
                (5, max(14, debug.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
            )
        else:
            cv2.putText(
                debug,
                "no zone",
                (5, max(14, debug.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
        return debug
