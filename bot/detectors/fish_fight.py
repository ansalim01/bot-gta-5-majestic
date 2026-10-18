"""Fish fight — track center fish marker movement, steer A/D opposite."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import cv2
import numpy as np

# The fish marker is a small square UI box floating over the water — the only
# such box in the marker ROI. It has TWO states (verified on real recordings):
# pale GREEN (H 57-80, high V) during normal tug-of-war, and WHITE/GRAY while
# the fish surges (progress drains or stalls no matter what is held). Searching
# for it by color directly is far more reliable than template matching: on a
# night recording TM_CCOEFF scored rock texture (~0.53) nearly the same as the
# real icon (~0.58) and sometimes ABOVE it, while the color-box search returns
# exactly one hit on the icon during a fight and zero hits before/after
# (including the cast-timing bar screen, splashes and the waiting bobber).
ICON_GREEN_LOW = np.array([35, 25, 70])
ICON_GREEN_HIGH = np.array([95, 255, 255])
ICON_WHITE_LOW = np.array([0, 0, 185])
ICON_WHITE_HIGH = np.array([180, 50, 255])


class SteerDirection(Enum):
    LEFT = auto()  # hold A
    RIGHT = auto()  # hold D
    CENTER = auto()


@dataclass
class FightAnalysis:
    active: bool
    direction: SteerDirection
    marker_x: int
    delta_x: int
    source: str
    # Reel bar fill 0..1, or None when the bar row wasn't located this frame.
    progress: float | None = None


@dataclass
class MarkerHit:
    x: int
    y: int
    area: float
    # True when the icon matched the WHITE (surge) state, False for green.
    white: bool = False


class FishFightDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        reel_cfg = config.get("reel", {})
        self._yellow_low = np.array(reel_cfg.get("yellow_low", [10, 80, 100]))
        self._yellow_high = np.array(reel_cfg.get("yellow_high", [40, 255, 255]))
        self._yellow_ratio_min = reel_cfg.get("yellow_ratio_min", 0.006)
        self._fish_gray_min = reel_cfg.get("fish_icon_gray_min", 185)
        self._inactive_frames = reel_cfg.get("inactive_frames_to_finish", 12)
        self._timeout = reel_cfg.get("timeout_sec", 55)
        self._tick_ms = reel_cfg.get("tick_ms", 8)
        self._prompt_white_min = reel_cfg.get("reel_prompt_white_min", 0.028)

        # Tall ROI: during the final reeling phase the icon rises high above
        # the water (seen at y≈0.10 of frame right before a catch) — a short
        # ROI lost it there and the bot abandoned a winning fight.
        marker = reel_cfg.get("marker_roi_pct", [0.30, 0.06, 0.40, 0.56])
        self._marker_x_pct, self._marker_y_pct, self._marker_w_pct, self._marker_h_pct = marker

        self._consecutive_inactive = 0
        self._seen_active = False
        self._appear_timeout = reel_cfg.get("appear_timeout_sec", 5.0)
        self._last_steer: SteerDirection = SteerDirection.CENTER
        self._last_move_time = 0.0

        # --- Camera-pan steering ---
        # The on-screen fish icon barely moves and never changes orientation —
        # what actually conveys "which way the fish is pulling" is the game
        # CAMERA panning left/right (the background rotates while the 2D HUD
        # stays put, like turning the mouse). Confirmed by measuring background
        # vs. HUD shift on a real recording: HUD shift ~0.00px, background shift
        # ~20px/0.5s, flipping sign when the fish changes direction. We track
        # that background shift via phase correlation instead of any icon.
        pan = reel_cfg.get("pan_roi_pct", [0.31, 0.05, 0.42, 0.23])
        self._pan_x_pct, self._pan_y_pct, self._pan_w_pct, self._pan_h_pct = pan
        self._pan_min_dt = reel_cfg.get("pan_min_dt_sec", 0.15)
        self._pan_min_response = reel_cfg.get("pan_min_response", 0.20)
        self._pan_velocity_min = reel_cfg.get("pan_velocity_min_px_sec", 30.0)
        self._pan_invert = reel_cfg.get("invert_pan_steer", False)
        self._pan_ref: tuple[np.ndarray, float] | None = None
        self._last_pan_shift = 0.0

        # --- Progress-bar feedback (safety net on top of camera-pan) ---
        # --- Progress bar ---
        # The bar's on-screen ROW shifts with window size (1920x1040 windowed vs
        # full-screen put it ~60px apart — a fixed pct ROI cut the bar off and
        # made the bot abandon a live fight at 0.73 fill). Instead of a fixed
        # ROI we LOCATE the fill row each frame: the fill always starts at the
        # bar's left edge (x≈0.356 of frame width, stable across sessions), so
        # a narrow column band there lights up on the fill rows at any progress
        # ≥~0.02. The cast-timing bar in the same strip has a GREEN zone
        # (hue>40) — no pure yellow — so "yellow present on the fill row" also
        # cleanly separates "reel fight on screen" from "cast bar on screen".
        self._prog_hue_low = reel_cfg.get("progress_hue_low", 10)
        self._prog_hue_high = reel_cfg.get("progress_hue_high", 90)
        # x-extent of the bar fill area and of the left "stub" band used to
        # locate the row (fractions of frame width; bar spans 683-1237 @1920,
        # net icon at the right end excluded).
        self._fill_x0 = reel_cfg.get("fill_x0_pct", 0.356)
        self._fill_x1 = reel_cfg.get("fill_x1_pct", 0.630)
        self._fill_y0 = reel_cfg.get("fill_y0_pct", 0.76)
        self._fill_y1 = reel_cfg.get("fill_y1_pct", 0.88)
        # Minimum contiguous horizontal fill needed to trust the bar without
        # the floating fish icon. At 1920px this is 144px (~27% of the bar).
        # Lower progress is still tracked through the icon.
        self._fill_min_run_pct = reel_cfg.get("fill_min_run_pct", 0.075)

        # --- Quiet flip: the fish switches sides WITHOUT a camera pan ---
        # Seen on real lost fights: pan goes silent, the icon stays green
        # (tug-of-war, not a surge) and progress steadily drains 0.5→0.2 over
        # ~15s while we hold the stale key. The fish flipped sides silently.
        # When pan is quiet AND the icon is green AND progress dropped by
        # quiet_flip_drop within quiet_flip_window_sec, flip the held key once
        # and keep holding. Pan stays the primary signal — any real pan reading
        # resets this. NOT the old ping-pong override: that one fired from
        # ±0.01 noise every 0.8s and fought the pan; this needs a drop far
        # above noise, only acts in pan silence, and only outside surges.
        self._quiet_flip_window = reel_cfg.get("quiet_flip_window_sec", 2.0)
        self._quiet_flip_drop = reel_cfg.get("quiet_flip_drop", 0.05)
        self._quiet_prog_history: list[tuple[float, float]] = []

    def reset(self) -> None:
        self._consecutive_inactive = 0
        self._seen_active = False
        self._last_steer = SteerDirection.CENTER
        self._last_move_time = 0.0
        self._pan_ref = None
        self._last_pan_shift = 0.0
        self._quiet_prog_history = []
        self._last_solve_reason = ""

    @property
    def last_solve_reason(self) -> str:
        return self._last_solve_reason

    @property
    def timeout_sec(self) -> float:
        return self._timeout

    @property
    def tick_sec(self) -> float:
        return self._tick_ms / 1000.0

    def _marker_roi(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = frame.shape[:2]
        x = int(w * self._marker_x_pct)
        y = int(h * self._marker_y_pct)
        rw = max(40, int(w * self._marker_w_pct))
        rh = max(40, int(h * self._marker_h_pct))
        x = min(x, w - rw)
        y = min(y, h - rh)
        return frame[y : y + rh, x : x + rw], x, y

    def _pan_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x = int(w * self._pan_x_pct)
        y = int(h * self._pan_y_pct)
        rw = max(40, int(w * self._pan_w_pct))
        rh = max(40, int(h * self._pan_h_pct))
        x = min(x, max(0, w - rw))
        y = min(y, max(0, h - rh))
        return frame[y : y + rh, x : x + rw]

    def _measure_pan_shift(self, full_frame: np.ndarray) -> tuple[float, float] | None:
        """Phase-correlate the background ROI against the last reference frame.
        Returns (shift_x_px, dt_sec) once enough time has passed for a reliable
        reading, else None (caller should hold the previous steer direction)."""
        roi = self._pan_roi(full_frame)
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        now = time.perf_counter()

        if self._pan_ref is None:
            self._pan_ref = (gray, now)
            return None

        ref_gray, ref_t = self._pan_ref
        dt = now - ref_t
        if dt < self._pan_min_dt:
            return None

        self._pan_ref = (gray, now)
        if ref_gray.shape != gray.shape:
            return None
        shift, response = cv2.phaseCorrelate(ref_gray, gray)
        if response < self._pan_min_response:
            return None
        return float(shift[0]), dt

    def _steer_from_pan(self, full_frame: np.ndarray) -> tuple[SteerDirection, int]:
        measured = self._measure_pan_shift(full_frame)
        if measured is None:
            return SteerDirection.CENTER, int(self._last_pan_shift)
        shift_x, dt = measured
        self._last_pan_shift = shift_x
        velocity = shift_x / dt if dt > 0 else 0.0
        threshold = self._pan_velocity_min * (full_frame.shape[1] / 1920)

        # Background moving RIGHT -> hold D (RIGHT). Background moving LEFT ->
        # hold A (LEFT). Flip with reel.invert_pan_steer if it feels backwards
        # in practice.
        pulled_left = velocity >= threshold
        pulled_right = velocity <= -threshold
        if self._pan_invert:
            pulled_left, pulled_right = pulled_right, pulled_left

        if pulled_left:
            return SteerDirection.RIGHT, int(shift_x)
        if pulled_right:
            return SteerDirection.LEFT, int(shift_x)
        return SteerDirection.CENTER, int(shift_x)

    def _fill_state(self, full_frame: np.ndarray) -> tuple[float | None, bool]:
        """Locate the reel bar's fill row and measure it.

        Returns (progress 0..1 or None if the bar isn't on screen,
        yellow_present) where yellow_present means the located row carries
        actual YELLOW fill (hue 10-40) — true only for the reel fight bar,
        never for the cast-timing bar whose zone is pure green (hue ~55).

        The old locator looked only at a narrow vertical stub. Yellow/green
        scenery crossing that x coordinate formed solid vertical blobs, so a
        finished fight could stay "active" for the full timeout. A real bar is
        a thin, multi-row HORIZONTAL run starting at its fixed left edge.
        Require that geometry and use the run length as progress."""
        h, w = full_frame.shape[:2]
        y0, y1 = int(h * self._fill_y0), int(h * self._fill_y1)
        fx0, fx1 = int(w * self._fill_x0), int(w * self._fill_x1)
        roi = full_frame[y0:y1, fx0:fx1]
        if roi.size == 0:
            return None, False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lit_lo = np.array([self._prog_hue_low, 60, 110])
        lit_hi = np.array([self._prog_hue_high, 255, 255])
        mask = cv2.inRange(hsv, lit_lo, lit_hi)

        start_tolerance = max(6, int(w * 0.006))
        min_run_width = max(20, int(w * self._fill_min_run_pct))
        min_run = max(4, int(h * 0.006))

        def left_run_length(row: np.ndarray) -> int:
            padded = np.pad((row > 0).astype(np.int8), (1, 1))
            edges = np.flatnonzero(np.diff(padded))
            best = 0
            for start, end in zip(edges[::2], edges[1::2]):
                if start <= start_tolerance:
                    best = max(best, int(end - start))
            return best

        lengths = np.array([left_run_length(row) for row in mask], dtype=np.int32)
        strong = lengths >= min_run_width
        candidates: list[tuple[int, int, float]] = []
        i = 0
        while i < len(strong):
            if not strong[i]:
                i += 1
                continue
            j = i
            while j < len(strong) and strong[j]:
                j += 1
            if j - i >= min_run:
                candidates.append((i, j, float(np.median(lengths[i:j]))))
            i = j
        if not candidates:
            return None, False

        row0, row1, fill_width = max(candidates, key=lambda item: item[2])
        prog = min(1.0, fill_width / max(mask.shape[1], 1))
        yellow = cv2.inRange(
            hsv[row0:row1],
            np.array([self._prog_hue_low, 60, 110]),
            np.array([40, 255, 255]),
        )
        yellow_present = int(np.count_nonzero(yellow.any(axis=0))) >= max(6, int(w * 0.004))
        return prog, yellow_present

    def _measure_progress(self, full_frame: np.ndarray) -> float | None:
        return self._fill_state(full_frame)[0]

    def _find_icon_box(
        self, hsv: np.ndarray, low: np.ndarray, high: np.ndarray, frame_width: int
    ) -> MarkerHit | None:
        mask = cv2.inRange(hsv, low, high)
        # Bridge the dark fish silhouette cut through the icon's middle so the
        # box comes back as one solid blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Icon measured ~1000-2300 px² on a 1920-wide frame — scale bounds by
        # frame area, with generous margins.
        scale_sq = (frame_width / 1920) ** 2
        min_area = 250 * scale_sq
        max_area = 6000 * scale_sq

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: MarkerHit | None = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue
            # Solid box, not scattered foliage/water glints/splash.
            if area / max(bw * bh, 1) < 0.5:
                continue
            if best is None or area > best.area:
                best = MarkerHit(x + bw // 2, y + bh // 2, area)
        return best

    def find_center_marker(self, full_frame: np.ndarray) -> MarkerHit | None:
        """Fish icon over the water — green (normal fight) or white (fish
        surging). Green is checked first: the green state also has enough
        bright pixels to weakly match the white mask, never vice versa."""
        roi, ox, oy = self._marker_roi(full_frame)
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hit = self._find_icon_box(hsv, ICON_GREEN_LOW, ICON_GREEN_HIGH, full_frame.shape[1])
        white = False
        if hit is None:
            hit = self._find_icon_box(hsv, ICON_WHITE_LOW, ICON_WHITE_HIGH, full_frame.shape[1])
            white = True
        if hit is None:
            return None
        return MarkerHit(hit.x + ox, hit.y + oy, hit.area, white)

    def find_bar_fish(self, bar_frame: np.ndarray) -> MarkerHit | None:
        """White fish silhouette on the bottom tension bar."""
        if bar_frame.size == 0:
            return None

        gray = cv2.cvtColor(bar_frame, cv2.COLOR_BGR2GRAY)
        bright = gray >= self._fish_gray_min
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(bright.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: MarkerHit | None = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 18 or area > bar_frame.shape[0] * bar_frame.shape[1] * 0.25:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 3 or bh < 3 or bh > bar_frame.shape[0] * 0.85:
                continue
            cx, cy = x + bw // 2, y + bh // 2
            score = area + cx * 0.05
            if score > best_score:
                best_score = score
                best = MarkerHit(cx, cy, area)

        return best

    def is_reel_prompt(self, prompt_frame: np.ndarray) -> bool:
        if prompt_frame.size == 0:
            return False
        gray = cv2.cvtColor(prompt_frame, cv2.COLOR_BGR2GRAY)
        return float(np.count_nonzero(gray > 180)) / gray.size >= self._prompt_white_min

    def bar_progress_ratio(self, bar_frame: np.ndarray) -> float:
        if bar_frame.size == 0:
            return 0.0
        hsv = cv2.cvtColor(bar_frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, self._yellow_low, self._yellow_high)
        return float(np.count_nonzero(yellow)) / yellow.size

    def _bar_has_yellow(self, bar_frame: np.ndarray) -> bool:
        return self.bar_progress_ratio(bar_frame) >= self._yellow_ratio_min

    @staticmethod
    def is_sane_bar_crop(crop: np.ndarray, frame_width: int) -> bool:
        if crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 8 or w < int(frame_width * 0.15):
            return False
        if w > int(frame_width * 0.85):
            return False
        return h <= w

    def crop_bar(self, window_frame: np.ndarray) -> np.ndarray:
        from bot.detectors.cast_timing import CastTimingDetector

        rect = CastTimingDetector.locate_bar_rect(
            window_frame, np.array([30, 40, 40]), np.array([100, 255, 255])
        )
        if not rect:
            return np.array([])
        x, y, w, h = rect
        crop = window_frame[y : y + h, x : x + w]
        bar_h = max(14, int(h * 0.5))
        crop = np.ascontiguousarray(crop[:bar_h, :])
        if not self.is_sane_bar_crop(crop, window_frame.shape[1]):
            return np.array([])
        return crop

    def analyze(
        self,
        full_frame: np.ndarray,
        bar_frame: np.ndarray,
        prompt_frame: np.ndarray | None = None,
    ) -> FightAnalysis:
        inactive = FightAnalysis(False, SteerDirection.CENTER, 0, 0, "none")

        # Strict content-located yellow fill (window-size independent). The
        # loose fixed-region yellow ratio is not an activity source: scenery
        # in that crop used to keep ended fights alive until the hard timeout.
        progress, yellow_present = self._fill_state(full_frame)
        reel_active = yellow_present

        # The fish icon (green or white) confirms the reel UI is on screen —
        # it's NOT the steering signal (see _steer_from_pan). "Fight is on"
        # must come from the icon or the yellow fill ONLY: the white-blob
        # find_bar_fish fallback used to keep the fight "active" forever after
        # a fish escaped, because it matched the cast-timing bar's white
        # slider that appears in the same screen region.
        icon = self.find_center_marker(full_frame)
        if icon is None and not reel_active:
            return inactive
        # Cosmetic only — gives marker_x something to report while the icon
        # is briefly lost.
        marker = icon if icon is not None else self.find_bar_fish(bar_frame)

        direction, shift_x = self._steer_from_pan(full_frame)
        source = "camera_pan"

        if direction != SteerDirection.CENTER:
            self._last_steer = direction
            self._last_move_time = time.perf_counter()
            self._quiet_prog_history = []
        elif self._last_steer != SteerDirection.CENTER:
            # The game wants A or D held continuously the whole fight — never
            # let go just because camera-pan went quiet for a moment (resting
            # fish, or a pull too subtle for phase-correlation to measure).
            # Releasing here is exactly the multi-second "does nothing" gap
            # that let progress drain back down before a fish was lost.
            direction = self._last_steer
            source = "hold"
            flipped = self._quiet_flip(icon, progress)
            if flipped is not None:
                direction = flipped
                source = "quiet_flip"

        marker_x = marker.x if marker is not None else 0
        return FightAnalysis(True, direction, marker_x, shift_x, source, progress)

    def _quiet_flip(self, icon: MarkerHit | None, prog: float | None) -> SteerDirection | None:
        """While pan is silent: if the green icon shows a normal tug-of-war but
        progress steadily drains, the fish flipped sides without panning the
        camera — flip the held key once. Returns None to keep holding."""
        if icon is None or icon.white:
            # Surge (white) or icon lost — progress drains regardless of keys
            # there, a flip would just leave us on a random side. Hold.
            self._quiet_prog_history = []
            return None
        if prog is None or prog <= 0.0:
            return None
        now = time.perf_counter()
        self._quiet_prog_history.append((now, prog))
        min_keep = now - self._quiet_flip_window
        while len(self._quiet_prog_history) > 2 and self._quiet_prog_history[0][0] < min_keep:
            self._quiet_prog_history.pop(0)
        oldest_t, oldest_p = self._quiet_prog_history[0]
        if now - oldest_t < self._quiet_flip_window * 0.7:
            return None
        if oldest_p - prog < self._quiet_flip_drop:
            return None
        flipped = (
            SteerDirection.LEFT
            if self._last_steer == SteerDirection.RIGHT
            else SteerDirection.RIGHT
        )
        self._last_steer = flipped
        self._quiet_prog_history = []
        return flipped

    def bar_fight_active(
        self,
        full_frame: np.ndarray,
        bar_frame: np.ndarray,
        *,
        timing_active: bool = False,
    ) -> bool:
        """Reeling minigame, confirmed by the yellow progress fill.

        The floating icon is often covered by the player model. Requiring both
        signals prevented state recovery even though analyze() already accepts
        either the icon or the fill as proof that the fight is active.
        """
        if timing_active:
            return False
        _, yellow_present = self._fill_state(full_frame)
        if yellow_present:
            return True
        if self.find_center_marker(full_frame) is None:
            return False
        if not self._bar_has_yellow(bar_frame):
            return False
        return self.bar_progress_ratio(bar_frame) >= self._yellow_ratio_min + 0.004

    def is_active(
        self,
        full_frame: np.ndarray,
        bar_frame: np.ndarray,
        prompt_frame: np.ndarray | None = None,
        prompt_detector: Any | None = None,
    ) -> bool:
        if (
            prompt_detector
            and prompt_frame is not None
            and prompt_detector.is_reeling(prompt_frame)
        ):
            marker = self.find_center_marker(full_frame)
            if marker is not None or self.bar_progress_ratio(bar_frame) >= 0.01:
                return True
        return self.bar_fight_active(full_frame, bar_frame)

    def track_completion(
        self,
        full_frame: np.ndarray,
        bar_frame: np.ndarray,
        prompt_frame: np.ndarray | None = None,
    ) -> bool:
        if self.is_active(full_frame, bar_frame, prompt_frame):
            self._consecutive_inactive = 0
            return False
        self._consecutive_inactive += 1
        return self._consecutive_inactive >= self._inactive_frames

    def solve(
        self,
        get_full_frame: Callable[[], np.ndarray],
        get_bar_frame: Callable[[], np.ndarray],
        get_prompt_frame: Callable[[], np.ndarray],
        set_steer: Callable[[str | None], None],
        is_caught: Callable[[np.ndarray], bool] | None = None,
        should_abort: Callable[[], bool] | None = None,
        on_steer: Callable[[FightAnalysis], None] | None = None,
    ) -> bool:
        self.reset()
        self._last_solve_reason = ""
        start = time.perf_counter()
        while time.perf_counter() - start < self._timeout:
            if should_abort and should_abort():
                set_steer(None)
                self._last_solve_reason = "aborted"
                return False

            # One screen capture per iteration — get_full_frame() first, then
            # get_bar_frame()/get_prompt_frame() are expected to slice that same
            # frame (no extra mss calls) and is_caught() reuses it too.
            full = get_full_frame()

            if is_caught and is_caught(full):
                set_steer(None)
                self._last_solve_reason = "caught_modal"
                return True

            bar = get_bar_frame()
            prompt = get_prompt_frame()

            analysis = self.analyze(full, bar, prompt)

            if not analysis.active:
                if self._seen_active:
                    # Was actively reeling, now the bar/marker disappeared.
                    # KEEP HOLDING the last key through the miss streak — a
                    # 1-2 frame detection flicker (splash over the icon, fill
                    # row missed) used to release A/D instantly, the exact
                    # mid-fight let-go the hold logic in analyze() prevents.
                    # Release only once the streak says the fight really ended.
                    self._consecutive_inactive += 1
                    if self._consecutive_inactive >= self._inactive_frames:
                        set_steer(None)
                        self._last_solve_reason = "ui_gone"
                        return True
                else:
                    set_steer(None)
                    if time.perf_counter() - start >= self._appear_timeout:
                        # The reel UI never showed up at all within
                        # appear_timeout — the hook press likely didn't start
                        # a fight. Bail out as a failure instead of wrongly
                        # reporting "caught".
                        self._last_solve_reason = "ui_never_appeared"
                        return False
                if on_steer:
                    on_steer(analysis)
            else:
                self._seen_active = True
                self._consecutive_inactive = 0
                if analysis.direction == SteerDirection.LEFT:
                    set_steer("left")
                elif analysis.direction == SteerDirection.RIGHT:
                    set_steer("right")
                else:
                    set_steer(None)
                if on_steer:
                    on_steer(analysis)

            time.sleep(self.tick_sec)

        set_steer(None)
        self._last_solve_reason = "timeout"
        return False

    def draw_debug(
        self,
        full_frame: np.ndarray,
        bar_frame: np.ndarray,
        analysis: FightAnalysis | None = None,
    ) -> np.ndarray:
        """Render a debug frame. Pass the analysis already computed this tick:
        analyze() is NOT read-only (it consumes a camera-pan measurement and
        can trigger a quiet-flip), so calling it from the debug path perturbs
        live steering. The self-computed fallback is for offline tools only."""
        debug = np.ascontiguousarray(full_frame.copy())
        if analysis is None:
            analysis = self.analyze(full_frame, bar_frame)

        pan_roi = self._pan_roi(full_frame)
        ph, pw = pan_roi.shape[:2]
        px = int(full_frame.shape[1] * self._pan_x_pct)
        py = int(full_frame.shape[0] * self._pan_y_pct)
        cv2.rectangle(debug, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
        cv2.putText(
            debug,
            "camera-pan ROI",
            (px, max(py - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        if analysis.active:
            key = (
                "A"
                if analysis.direction == SteerDirection.LEFT
                else ("D" if analysis.direction == SteerDirection.RIGHT else "-")
            )
            cv2.putText(
                debug,
                f"HOLD {key}  pan_shift={analysis.delta_x:+d}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        return debug
