"""Ore vein minigame detection and clicking."""

from __future__ import annotations

import math
import random
import time
from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR, scaled_rocks_relative
from bot.detectors.progress_bar import ProgressBarDetector


class MinigameDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        minigame_cfg = config.get("minigame", {})
        regions = config.get("regions", {})

        self._use_templates = minigame_cfg.get("use_templates", False)
        self._threshold = minigame_cfg.get("match_threshold", 0.88)
        self._color_detection = minigame_cfg.get("color_detection", True)
        self._min_area = minigame_cfg.get("min_vein_area", 8)
        self._max_area = minigame_cfg.get("max_vein_area", 600)
        self._click_delay = minigame_cfg.get("click_delay_ms", [100, 180])
        self._post_click_wait = minigame_cfg.get("post_click_wait_ms", 150) / 1000.0
        self._timeout = minigame_cfg.get("timeout_sec", 15)
        self._rock_dark_threshold = minigame_cfg.get("rock_dark_threshold", 85)
        self._dedupe_distance = minigame_cfg.get("dedupe_distance", 20)
        self._empty_frames_to_finish = minigame_cfg.get("empty_frames_to_finish", 12)
        self._final_sweep_attempts = minigame_cfg.get("final_sweep_attempts", 3)
        self._use_rocks_mask = minigame_cfg.get("use_rocks_mask", True)
        self._use_auto_rock_mask = minigame_cfg.get("use_auto_rock_mask", True)
        self._click_cooldown_radius = minigame_cfg.get("click_cooldown_radius", 24)
        self._max_clicks_per_minigame = minigame_cfg.get("max_clicks_per_minigame", 8)
        self._min_progress_to_finish = minigame_cfg.get("min_progress_to_finish", 0.92)

        vein_hsv = minigame_cfg.get("vein_hsv", {})
        self._yellow_low = np.array(vein_hsv.get("yellow_low", [12, 45, 125]))
        self._yellow_high = np.array(vein_hsv.get("yellow_high", [42, 255, 255]))
        self._white_low = np.array(vein_hsv.get("white_low", [0, 0, 200]))
        self._white_high = np.array(vein_hsv.get("white_high", [180, 70, 255]))

        relaxed_hsv = minigame_cfg.get("vein_hsv_relaxed", {})
        self._relaxed_yellow_low = np.array(relaxed_hsv.get("yellow_low", [10, 30, 110]))
        self._relaxed_yellow_high = np.array(relaxed_hsv.get("yellow_high", [45, 255, 255]))
        self._relaxed_white_low = np.array(relaxed_hsv.get("white_low", [0, 0, 175]))
        self._relaxed_white_high = np.array(relaxed_hsv.get("white_high", [180, 90, 255]))

        self._config_regions = regions
        self._config_resolution = config.get("screen", {}).get("resolution", [1920, 1080])
        self._window_size: tuple[int, int] | None = None

        self._minigame_region = regions.get("minigame", [520, 340, 880, 420])
        self._rocks_mask_relative = self._to_relative_mask(
            regions.get("rocks_mask"),
            self._minigame_region,
        )

        self._progress_bar = ProgressBarDetector(config)
        self._templates = self._load_templates() if self._use_templates else []
        self._empty_frame_count = 0
        self._clicked_positions: list[tuple[int, int]] = []

    @staticmethod
    def _to_relative_mask(
        absolute_mask: list[int] | None,
        minigame_region: list[int],
    ) -> list[int] | None:
        if not absolute_mask or len(absolute_mask) != 4:
            return None
        mg_x, mg_y = minigame_region[0], minigame_region[1]
        return [
            absolute_mask[0] - mg_x,
            absolute_mask[1] - mg_y,
            absolute_mask[2],
            absolute_mask[3],
        ]

    def _load_templates(self) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        if not TEMPLATES_DIR.exists():
            return templates

        for path in sorted(TEMPLATES_DIR.glob("vein_*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                img = cv2.imread(str(path))
                if img is not None:
                    templates.append(img)
        return templates

    def reload_templates(self) -> None:
        self._templates = self._load_templates() if self._use_templates else []

    def is_active(
        self,
        frame: np.ndarray,
        progress_frame: np.ndarray | None = None,
    ) -> bool:
        return self.is_minigame_open(frame, progress_frame)

    def is_minigame_open(
        self,
        frame: np.ndarray,
        progress_frame: np.ndarray | None = None,
    ) -> bool:
        if progress_frame is not None and self._progress_bar.is_visible(progress_frame):
            return True
        return len(self.find_veins(frame)) > 0

    def is_complete(
        self,
        frame: np.ndarray,
        progress_frame: np.ndarray | None = None,
    ) -> bool:
        if self._progress_complete(progress_frame):
            return True
        if not self.is_minigame_open(frame, progress_frame):
            return True
        return False

    def find_veins(self, frame: np.ndarray) -> list[tuple[int, int]]:
        return self._find_veins_in_frame(frame, relaxed=False)

    def find_veins_relaxed(self, frame: np.ndarray) -> list[tuple[int, int]]:
        return self._find_veins_in_frame(frame, relaxed=True)

    def find_crystals(self, frame: np.ndarray) -> list[tuple[int, int]]:
        return self.find_veins(frame)

    def _rocks_mask_active(self, frame: np.ndarray) -> bool:
        if not self._use_rocks_mask or not self._rocks_mask_relative:
            return False
        rx, ry, rw, rh = self._rocks_mask_relative
        fh, fw = frame.shape[:2]
        return rx >= 0 and ry >= 0 and rw > 0 and rh > 0 and rx + rw <= fw and ry + rh <= fh

    def _search_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        if self._use_auto_rock_mask:
            bounds = self._auto_rock_bounds(frame)
            if bounds is not None:
                x1, y1, x2, y2 = bounds
                return frame[y1:y2, x1:x2], x1, y1

        if self._rocks_mask_active(frame):
            rx, ry, rw, rh = self._rocks_mask_relative
            return frame[ry : ry + rh, rx : rx + rw], rx, ry

        return frame, 0, 0

    def _auto_rock_bounds(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        mask = self._rock_surface_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = frame.shape[0] * frame.shape[1] * 0.012
        boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 28 and h >= 22:
                boxes.append((x, y, x + w, y + h))

        if not boxes:
            return None

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        pad = 8
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        if x2 - x1 < 40 or y2 - y1 < 30:
            return None
        return x1, y1, x2, y2

    def _rock_surface_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rock = cv2.inRange(hsv, np.array([75, 10, 15]), np.array([140, 220, 200]))
        rug = cv2.inRange(hsv, np.array([128, 28, 55]), np.array([180, 255, 255]))
        mask = cv2.bitwise_and(rock, cv2.bitwise_not(rug))
        mask = cv2.bitwise_and(mask, cv2.inRange(gray, 18, 200))
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def _is_rug_pixel(
        self, hsv: np.ndarray, x: int, y: int, gray: np.ndarray | None = None
    ) -> bool:
        hue, sat, val = [int(v) for v in hsv[y, x]]
        if gray is not None and int(gray[y, x]) < 95:
            return False
        if 128 <= hue <= 180 and sat > 28 and val > 65:
            return True
        if hue <= 16 and sat > 75 and val > 95:
            return True
        return False

    def _point_on_rock(self, frame: np.ndarray, cx: int, cy: int, radius: int = 12) -> bool:
        h, w = frame.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rock_mask = self._rock_surface_mask(frame)
        rock = rug = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 5 or dist > radius:
                    continue
                x, y = cx + dx, cy + dy
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if self._is_rug_pixel(hsv, x, y, gray):
                    rug += 1
                elif rock_mask[y, x] > 0 or int(gray[y, x]) < 115:
                    rock += 1
        return rock >= 4 and (rug <= 6 or rug / max(rock, 1) < 0.18)

    def _find_veins_in_frame(self, frame: np.ndarray, relaxed: bool) -> list[tuple[int, int]]:
        search_frame, offset_x, offset_y = self._search_frame(frame)
        points: list[tuple[int, int, float]] = []
        skip_paper_filter = self._use_rocks_mask and self._rocks_mask_active(frame)
        paper_cutoff_y = int(frame.shape[0] * 0.72) if skip_paper_filter else None

        if self._templates and not relaxed:
            points.extend(self._find_by_template(search_frame, offset_x, offset_y))

        if self._color_detection:
            points.extend(
                self._find_by_color(
                    search_frame,
                    offset_x,
                    offset_y,
                    relaxed=relaxed,
                    paper_cutoff_y=paper_cutoff_y,
                    full_frame=frame,
                )
            )

        min_dist = self._dedupe_distance - 2 if relaxed else self._dedupe_distance
        return self._deduplicate_points(points, max(min_dist, 12))

    def set_regions(
        self,
        minigame_region: list[int],
        rocks_mask_relative: list[int] | None = None,
    ) -> None:
        self._minigame_region = minigame_region
        if rocks_mask_relative:
            self._rocks_mask_relative = rocks_mask_relative

    def sync_window_scale(self, window_width: int, window_height: int) -> None:
        """Recompute rocks mask for the actual game window (fullscreen / borderless)."""
        size = (window_width, window_height)
        if self._window_size == size:
            return
        self._window_size = size
        relative = scaled_rocks_relative(
            self._config_regions,
            self._config_resolution,
            window_width,
            window_height,
        )
        if relative:
            self._rocks_mask_relative = relative

    def _pixel_is_rock(self, gray: np.ndarray, hsv: np.ndarray, x: int, y: int) -> bool:
        g = int(gray[y, x])
        if g < 115:
            return True
        hue, sat, val = [int(v) for v in hsv[y, x]]
        if sat > 55 and hue < 20:
            return False
        if 80 <= hue <= 140 and sat >= 12 and val <= 145:
            return True
        return False

    def _count_rock_neighbors(
        self, gray: np.ndarray, hsv: np.ndarray, cx: int, cy: int, radius: int = 14
    ) -> int:
        h, w = gray.shape[:2]
        count = 0
        for i in range(8):
            angle = i * math.pi / 4
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            if 0 <= x < w and 0 <= y < h and self._pixel_is_rock(gray, hsv, x, y):
                count += 1
        return count

    def _has_adjacent_rock(self, gray: np.ndarray, cx: int, cy: int) -> bool:
        h, w = gray.shape[:2]
        for radius in (8, 12, 16, 20):
            for i in range(8):
                angle = i * math.pi / 4
                x = int(cx + radius * math.cos(angle))
                y = int(cy + radius * math.sin(angle))
                if 0 <= x < w and 0 <= y < h and int(gray[y, x]) < 120:
                    return True
        return False

    def _is_gold_vein_pixel(self, hsv: np.ndarray, cx: int, cy: int, relaxed: bool = False) -> bool:
        hue, sat, val = [int(v) for v in hsv[cy, cx]]
        min_sat = 38 if relaxed else 40
        if 12 <= hue <= 40 and sat >= min_sat and val >= 120:
            return True
        # Светлая жила на синем камне (блик фонаря даёт hue ~165–180)
        if 132 <= hue <= 185 and sat >= min_sat and val >= 130:
            return True
        return False

    def _is_vein_color(self, hsv: np.ndarray, cx: int, cy: int, relaxed: bool = False) -> bool:
        if self._is_gold_vein_pixel(hsv, cx, cy, relaxed=relaxed):
            return True
        if relaxed:
            hue, sat, val = [int(v) for v in hsv[cy, cx]]
            if sat < 65 and val >= 165:
                return True
        return False

    def _has_adjacent_rock_on_full(self, frame: np.ndarray, cx: int, cy: int) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        for radius in (6, 10, 14, 18, 22):
            for i in range(8):
                angle = i * math.pi / 4
                x = int(cx + radius * math.cos(angle))
                y = int(cy + radius * math.sin(angle))
                if 0 <= x < w and 0 <= y < h and int(gray[y, x]) < 115:
                    return True
        return False

    def _is_lantern_or_rug_glow(
        self,
        hsv: np.ndarray,
        cx: int,
        cy: int,
        gray: np.ndarray | None = None,
        full_frame: np.ndarray | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> bool:
        hue, sat, val = [int(v) for v in hsv[cy, cx]]
        on_rock = full_frame is not None and self._point_on_rock(
            full_frame, cx + offset_x, cy + offset_y
        )
        near_rock = full_frame is not None and self._has_adjacent_rock_on_full(
            full_frame, cx + offset_x, cy + offset_y
        )
        if 5 <= hue <= 22 and sat > 65 and val > 140:
            if on_rock or near_rock or (gray is not None and self._has_adjacent_rock(gray, cx, cy)):
                return False
            return True
        if sat > 50 and hue > 150:
            if on_rock or near_rock:
                return False
            return True
        return False

    def _validate_vein_point(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        cx: int,
        cy: int,
        relaxed: bool = False,
        full_frame: np.ndarray | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> bool:
        h, w = gray.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return False

        full_cx, full_cy = cx + offset_x, cy + offset_y
        on_rock = False
        near_rock = False
        if full_frame is not None:
            on_rock = self._point_on_rock(full_frame, full_cx, full_cy)
            near_rock = self._has_adjacent_rock_on_full(full_frame, full_cx, full_cy)
            fh = full_frame.shape[0]
            if full_cy > fh * 0.72:
                return False
            if not relaxed and not on_rock:
                if not (
                    near_rock
                    and self._is_gold_vein_pixel(hsv, cx, cy, relaxed=False)
                    and self._count_rock_neighbors(gray, hsv, cx, cy) >= 2
                ):
                    return False
            elif relaxed and not on_rock and not near_rock:
                return False

        if self._is_lantern_or_rug_glow(
            hsv, cx, cy, gray, full_frame=full_frame, offset_x=offset_x, offset_y=offset_y
        ):
            return False

        if not self._is_vein_color(hsv, cx, cy, relaxed=relaxed):
            return False

        center = float(gray[cy, cx])
        min_center = 118 if (relaxed or near_rock) else 120
        if center < min_center:
            return False

        if not self._has_adjacent_rock(gray, cx, cy):
            return False

        if relaxed and near_rock and not on_rock:
            if self._is_gold_vein_pixel(hsv, cx, cy, relaxed=True):
                return True

        rock_neighbors = self._count_rock_neighbors(gray, hsv, cx, cy)
        if rock_neighbors >= 2:
            return True

        outer_r = 20
        inner_r = 5
        y1, y2 = max(0, cy - outer_r), min(h, cy + outer_r)
        x1, x2 = max(0, cx - outer_r), min(w, cx + outer_r)

        rock_count = 0
        total = 0
        for y in range(y1, y2):
            for x in range(x1, x2):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if inner_r < dist <= outer_r:
                    total += 1
                    if self._pixel_is_rock(gray, hsv, x, y):
                        rock_count += 1

        if total == 0:
            return center >= 140

        min_ratio = 0.18 if (relaxed or near_rock) else 0.24
        return rock_count / total >= min_ratio

    def _find_by_template(
        self, frame: np.ndarray, offset_x: int = 0, offset_y: int = 0
    ) -> list[tuple[int, int, float]]:
        found: list[tuple[int, int, float]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for template in self._templates:
            t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            th, tw = t_gray.shape[:2]
            if th >= frame.shape[0] or tw >= frame.shape[1]:
                continue

            result = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self._threshold)

            for y, x in zip(*locations):
                cx = x + tw // 2
                cy = y + th // 2
                if not self._validate_vein_point(gray, hsv, cx, cy):
                    continue
                score = float(result[y, x])
                found.append((cx + offset_x, cy + offset_y, score))

        return found

    def _extract_centers_from_blob(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        min_area: int,
    ) -> list[tuple[int, int]]:
        x, y, w, h = cv2.boundingRect(contour)
        roi = mask[y : y + h, x : x + w]
        if roi.size == 0:
            return []

        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        peak_min = max(2.5, float(dist.max()) * 0.35)
        _, peak_mask = cv2.threshold(dist, peak_min, 255, cv2.THRESH_BINARY)
        peak_mask = peak_mask.astype(np.uint8)
        peak_mask = cv2.erode(peak_mask, np.ones((3, 3), np.uint8))

        sub_contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: list[tuple[int, int]] = []
        for sub in sub_contours:
            if cv2.contourArea(sub) < 1:
                continue
            moments = cv2.moments(sub)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"]) + x
            cy = int(moments["m01"] / moments["m00"]) + y
            centers.append((cx, cy))
        return centers

    def _append_vein_center(
        self,
        found: list[tuple[int, int, float]],
        cx: int,
        cy: int,
        offset_x: int,
        offset_y: int,
        paper_cutoff_y: int | None,
        frame_h: int,
        gray: np.ndarray,
        hsv: np.ndarray,
        full_frame: np.ndarray | None,
        relaxed: bool,
    ) -> bool:
        full_cy = cy + offset_y
        if paper_cutoff_y is not None:
            if full_cy > paper_cutoff_y:
                return False
        elif cy > frame_h * 0.72:
            return False

        if not self._validate_vein_point(
            gray,
            hsv,
            cx,
            cy,
            relaxed=relaxed,
            full_frame=full_frame,
            offset_x=offset_x,
            offset_y=offset_y,
        ):
            return False

        hue, sat, val = [int(v) for v in hsv[cy, cx]]
        score = sat / 255.0 + val / 512.0
        if 12 <= hue <= 40:
            score += 1.0
        if full_frame is not None and self._point_on_rock(full_frame, cx + offset_x, cy + offset_y):
            score += 2.0
        found.append((cx + offset_x, cy + offset_y, score))
        return True

    def _find_by_color(
        self,
        frame: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        relaxed: bool = False,
        paper_cutoff_y: int | None = None,
        full_frame: np.ndarray | None = None,
    ) -> list[tuple[int, int, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if relaxed:
            yellow_low, yellow_high = self._relaxed_yellow_low, self._relaxed_yellow_high
            white_low, white_high = self._relaxed_white_low, self._relaxed_white_high
            min_area = max(8, self._min_area - 2)
            vein_mask = cv2.bitwise_or(
                cv2.inRange(hsv, yellow_low, yellow_high),
                cv2.inRange(hsv, white_low, white_high),
            )
            blue_vein = cv2.inRange(hsv, np.array([140, 45, 130]), np.array([182, 255, 255]))
            vein_mask = cv2.bitwise_or(vein_mask, blue_vein)
        else:
            min_area = self._min_area
            vein_mask = cv2.inRange(hsv, self._yellow_low, self._yellow_high)
        mask = vein_mask

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if full_frame is not None:
            rock_mask = self._rock_surface_mask(full_frame)
            ry1, rx1 = offset_y, offset_x
            ry2 = min(rock_mask.shape[0], ry1 + frame.shape[0])
            rx2 = min(rock_mask.shape[1], rx1 + frame.shape[1])
            rock_crop = rock_mask[ry1:ry2, rx1:rx2]
            if rock_crop.shape[:2] == frame.shape[:2]:
                rock_crop = cv2.dilate(rock_crop, np.ones((13, 13), np.uint8))
                mask = cv2.bitwise_and(mask, rock_crop)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found: list[tuple[int, int, float]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            if area > 80:
                added = False
                for cx, cy in self._extract_centers_from_blob(contour, mask, min_area):
                    if self._append_vein_center(
                        found,
                        cx,
                        cy,
                        offset_x,
                        offset_y,
                        paper_cutoff_y,
                        frame.shape[0],
                        gray,
                        hsv,
                        full_frame,
                        relaxed,
                    ):
                        added = True
                if added:
                    continue

            if area > self._max_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            self._append_vein_center(
                found,
                cx,
                cy,
                offset_x,
                offset_y,
                paper_cutoff_y,
                frame.shape[0],
                gray,
                hsv,
                full_frame,
                relaxed,
            )

        return found

    def _deduplicate_points(
        self, points: list[tuple[int, int, float]], min_dist: int
    ) -> list[tuple[int, int]]:
        if not points:
            return []

        points.sort(key=lambda p: p[2], reverse=True)
        unique: list[tuple[int, int]] = []

        for x, y, _ in points:
            if all((x - ux) ** 2 + (y - uy) ** 2 >= min_dist**2 for ux, uy in unique):
                unique.append((x, y))

        unique.sort(key=lambda p: (p[1], p[0]))
        return unique

    def _filter_clicked(self, veins: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not self._clicked_positions:
            return veins
        r = self._click_cooldown_radius
        return [
            (x, y)
            for x, y in veins
            if all((x - px) ** 2 + (y - py) ** 2 >= r**2 for px, py in self._clicked_positions)
        ]

    def _find_any_veins(
        self,
        frame: np.ndarray,
        allow_relaxed: bool = False,
    ) -> list[tuple[int, int]]:
        veins = self._filter_clicked(self.find_veins(frame))
        if veins:
            return veins
        if not allow_relaxed:
            return []
        return self._filter_clicked(self.find_veins_relaxed(frame))

    def _progress_complete(self, progress_frame: np.ndarray | None) -> bool:
        return progress_frame is not None and self._progress_bar.is_complete(progress_frame)

    def _can_finish_minigame(
        self,
        frame: np.ndarray,
        progress_frame: np.ndarray | None,
    ) -> bool:
        if self._find_any_veins(frame, allow_relaxed=True):
            return False
        if progress_frame is not None and self._progress_bar.is_visible(progress_frame):
            return self.get_progress_fill(progress_frame) >= self._min_progress_to_finish
        return bool(self._clicked_positions)

    def _final_sweep(
        self,
        get_frame,
        to_screen_coords,
        click_fn,
        get_progress_frame=None,
    ) -> bool:
        for _ in range(self._final_sweep_attempts):
            time.sleep(0.12)
            frame = get_frame()
            progress_frame = get_progress_frame() if get_progress_frame else None

            if self._progress_complete(progress_frame):
                return True

            veins = self._find_any_veins(frame, allow_relaxed=True)
            if not veins:
                continue

            x, y = veins[0]
            sx, sy = to_screen_coords(x, y)
            click_fn(sx, sy)
            self._clicked_positions.append((x, y))
            time.sleep(self._post_click_wait)
            return False

        frame = get_frame()
        progress_frame = get_progress_frame() if get_progress_frame else None
        return self._can_finish_minigame(frame, progress_frame)

    def solve(
        self,
        get_frame,
        to_screen_coords,
        click_fn,
        get_progress_frame=None,
        should_abort=None,
    ) -> bool:
        start = time.perf_counter()
        self._empty_frame_count = 0
        self._strict_empty_count = 0
        self._clicked_positions = []

        while time.perf_counter() - start < self._timeout:
            if should_abort and should_abort():
                return False

            frame = get_frame()
            progress_frame = get_progress_frame() if get_progress_frame else None

            if self._progress_complete(progress_frame):
                return True

            if len(self._clicked_positions) >= self._max_clicks_per_minigame:
                if self._can_finish_minigame(frame, progress_frame):
                    return True

            if self._clicked_positions and not self.is_minigame_open(frame, progress_frame):
                return True

            allow_relaxed = self._strict_empty_count >= 3
            veins = self._find_any_veins(frame, allow_relaxed=allow_relaxed)

            if veins:
                self._empty_frame_count = 0
                self._strict_empty_count = 0
                x, y = veins[0]
                sx, sy = to_screen_coords(x, y)
                click_fn(sx, sy)
                self._clicked_positions.append((x, y))
                time.sleep(
                    self._post_click_wait
                    + random.uniform(self._click_delay[0], self._click_delay[1]) / 1000.0
                )
                continue

            if not allow_relaxed:
                self._strict_empty_count += 1

            self._empty_frame_count += 1
            if self._empty_frame_count >= self._empty_frames_to_finish:
                frame = get_frame()
                progress_frame = get_progress_frame() if get_progress_frame else None
                if self._final_sweep(get_frame, to_screen_coords, click_fn, get_progress_frame):
                    if self._can_finish_minigame(frame, progress_frame):
                        return True
                self._empty_frame_count = 0
                continue

            time.sleep(0.06)

        return False

    def get_progress_fill(self, progress_frame: np.ndarray) -> float:
        return self._progress_bar.fill_ratio(progress_frame)

    def draw_debug(
        self,
        frame: np.ndarray,
        progress_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        debug = np.ascontiguousarray(frame.copy())

        if self._rocks_mask_relative and self._use_rocks_mask:
            rx, ry, rw, rh = self._rocks_mask_relative
            cv2.rectangle(debug, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)

        for x, y in self.find_veins(frame):
            cv2.circle(debug, (x, y), 8, (0, 0, 255), 2)
        for x, y in self.find_veins_relaxed(frame):
            cv2.circle(debug, (x, y), 5, (0, 165, 255), 1)

        if progress_frame is not None:
            fill = self.get_progress_fill(progress_frame)
            cv2.putText(
                debug,
                f"fill: {fill:.0%}",
                (5, debug.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return debug
