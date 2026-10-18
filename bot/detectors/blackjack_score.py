"""Read blackjack totals shown by the Majestic casino UI.

The detector deliberately reads only the digits between square brackets in
``У вас [..], у дилера [..]``.  Restricting OCR to ten possible glyphs makes it
small, local and much less likely to turn unrelated HUD text into an action.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class BlackjackScore:
    player: int
    dealer: int
    confidence: float
    # After a split Majestic renders ``You [left] and [right], dealer [up]``.
    # Keeping the second hand optional preserves the ordinary two-total API.
    player_second: int | None = None


def _normalise(mask: np.ndarray, size: tuple[int, int] = (20, 32)) -> np.ndarray:
    if mask.size and int(mask.max()) <= 1:
        mask = (mask * 255).astype(np.uint8)
    points = cv2.findNonZero(mask)
    if points is None:
        return np.zeros((size[1], size[0]), np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    glyph = mask[y : y + h, x : x + w]
    target_w, target_h = size
    scale = min((target_w - 2) / max(w, 1), (target_h - 2) / max(h, 1))
    resized = cv2.resize(
        glyph,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((target_h, target_w), np.uint8)
    yy = (target_h - resized.shape[0]) // 2
    xx = (target_w - resized.shape[1]) // 2
    canvas[yy : yy + resized.shape[0], xx : xx + resized.shape[1]] = resized
    return (canvas > 80).astype(np.uint8)


class BlackjackScoreDetector:
    def __init__(self, min_confidence: float = 0.64) -> None:
        self.min_confidence = float(min_confidence)
        self._templates = self._make_templates()

    @staticmethod
    def _font_paths() -> list[Path]:
        root = Path("C:/Windows/Fonts")
        candidates = (
            "segoeui.ttf",
            "segoeuib.ttf",
            "arial.ttf",
            "arialbd.ttf",
            "ARIALN.TTF",
            "ARIALNB.TTF",
        )
        paths = [root / name for name in candidates if (root / name).exists()]
        return paths

    @staticmethod
    @lru_cache(maxsize=1)
    def _majestic_font_blobs() -> tuple[bytes, ...]:
        """Load the exact Proxima Nova faces embedded in Majestic's client.

        The UI bundle keeps its WOFF2 fonts as base64 data URLs. Pillow's
        FreeType build can render those bytes directly, so nothing is written
        outside the project and the detector follows client updates.
        """
        configured = os.environ.get("MAJESTIC_CLIENT_DLL")
        dll_path = (
            Path(configured)
            if configured
            else (Path.home() / "AppData/Roaming/majestic-launcher/Multiplayer/majestic-client.dll")
        )
        try:
            bundle = dll_path.read_bytes()
        except OSError:
            return ()

        blobs: list[bytes] = []
        # cz = Regular 400, Fz = Semibold 600 in the Majestic UI bundle.
        for variable in (b"cz", b"Fz"):
            match = re.search(
                rb"(?:const |,)" + variable + rb"=\"data:[^;\"]+;base64,([A-Za-z0-9+/=]+)\"",
                bundle,
            )
            if match is not None:
                try:
                    blobs.append(base64.b64decode(match.group(1), validate=True))
                except (ValueError, base64.binascii.Error):
                    continue
        return tuple(blobs)

    def _make_templates(self) -> dict[str, list[np.ndarray]]:
        result: dict[str, list[np.ndarray]] = {str(i): [] for i in range(10)}
        font_blobs = self._majestic_font_blobs()
        font_sources: list[str | bytes] = list(font_blobs) or [
            str(path) for path in self._font_paths()
        ]
        for digit in result:
            for source in font_sources:
                for font_size in (18, 20, 22, 24):
                    font_ref = io.BytesIO(source) if isinstance(source, bytes) else source
                    font = ImageFont.truetype(font_ref, font_size)
                    image = Image.new("L", (40, 45), 0)
                    draw = ImageDraw.Draw(image)
                    draw.text((4, 0), digit, fill=255, font=font, stroke_width=0)
                    rendered = cv2.morphologyEx(
                        cv2.inRange(np.asarray(image), 150, 255),
                        cv2.MORPH_CLOSE,
                        np.ones((2, 1), np.uint8),
                    )
                    result[digit].append(_normalise(rendered))
        if not font_sources:  # Useful for tests and non-Windows development.
            for digit in result:
                image = np.zeros((40, 30), np.uint8)
                cv2.putText(image, digit, (2, 31), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                result[digit].append(_normalise(image))
        return result

    @staticmethod
    def _binary(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # The label is light grey; a small close reconnects anti-aliased strokes.
        mask = cv2.inRange(gray, 150, 255)
        # Reconnect vertical anti-aliased strokes without joining a bracket to
        # a neighbouring 7 (they are only one horizontal pixel apart).
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 1), np.uint8))

    @staticmethod
    def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for idx in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[idx])
            if 7 <= h <= 34 and 1 <= w <= 24 and area >= 5:
                components.append(
                    (x, y, w, h, (labels[y : y + h, x : x + w] == idx).astype(np.uint8))
                )
        return sorted(components, key=lambda item: item[0])

    @staticmethod
    def _bracket_orientation(item: tuple[int, int, int, int, np.ndarray]) -> str | None:
        _, _, w, h, glyph = item
        # In the game font brackets reach above/below the digits. Requiring
        # that extra height prevents a narrow ``1`` from becoming ``]``.
        if h < 15 or w > max(9, int(h * 0.65)):
            return None
        top = float(glyph[: max(2, h // 5)].mean())
        bottom = float(glyph[-max(2, h // 5) :].mean())
        edge = max(1, w // 3)
        left = float(glyph[:, :edge].mean())
        right = float(glyph[:, -edge:].mean())
        if top <= 0.12 or bottom <= 0.12:
            return None
        if left >= 0.55 and right <= 0.35:
            return "left"
        if right >= 0.55 and left <= 0.35:
            return "right"
        return None

    def _classify(self, glyph: np.ndarray) -> tuple[str, float]:
        sample = _normalise(glyph)
        sample_f = sample.astype(np.float32)
        sample_norm = float(np.linalg.norm(sample_f)) or 1.0
        scores: dict[str, float] = {}
        for digit, templates in self._templates.items():
            scores[digit] = max(
                float(
                    np.sum(sample_f * template) / (sample_norm * (np.linalg.norm(template) or 1.0))
                )
                for template in templates
            )
        ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        digit, score = ordered[0]
        margin = score - ordered[1][1]
        confidence = min(1.0, max(0.0, score * 0.72 + margin * 1.8))
        return digit, confidence

    def detect_number(
        self,
        frame: np.ndarray,
        *,
        maximum: int = 1_000_000,
        min_confidence: float | None = None,
    ) -> tuple[int, float] | None:
        """Read an isolated numeric value, such as ``ЭТА СТАВКА``.

        Callers must provide a tight crop containing only the value. This is
        intentionally separate from score parsing, whose brackets provide
        stronger structural validation.
        """
        glyphs = [
            item
            for item in self._components(self._binary(frame))
            if 8 <= item[3] <= 30 and item[2] >= 2
        ]
        if not glyphs or len(glyphs) > 7:
            return None
        recognised = [self._classify(item[4]) for item in glyphs]
        confidence = min(score for _, score in recognised)
        threshold = self.min_confidence if min_confidence is None else float(min_confidence)
        if confidence < threshold:
            return None
        value = int("".join(digit for digit, _ in recognised))
        if not 0 <= value <= maximum:
            return None
        return value, confidence

    def detect(self, frame: np.ndarray) -> BlackjackScore | None:
        components = self._components(self._binary(frame))
        if not components:
            return None
        # Brackets extend several pixels above and below all digits. A narrow
        # digit 1 has a similar edge profile to `]`, but never that full height.
        bracket_height = max(item[3] for item in components)
        left_brackets = [
            item
            for item in components
            if item[3] >= bracket_height - 1 and self._bracket_orientation(item) == "left"
        ]
        right_brackets = [
            item
            for item in components
            if item[3] >= bracket_height - 1 and self._bracket_orientation(item) == "right"
        ]
        if len(left_brackets) < 2 or len(right_brackets) < 2:
            return None

        # Try consecutive bracket pairs. Ordinary hands contain two pairs;
        # split hands contain three (left player hand, right hand, dealer).
        values: list[tuple[int, float, int]] = []
        for left in left_brackets:
            candidates = [right for right in right_brackets if right[0] > left[0]]
            if not candidates:
                continue
            right = min(candidates, key=lambda item: item[0])
            lx, _, lw, _, _ = left
            rx, _, _, _, _ = right
            if not 3 <= rx - (lx + lw) <= 40:
                continue
            glyphs = [item for item in components if item[0] > lx + lw and item[0] + item[2] < rx]
            if not 1 <= len(glyphs) <= 2:
                continue
            recognised = [self._classify(item[4]) for item in glyphs]
            value = int("".join(digit for digit, _ in recognised))
            confidence = min(score for _, score in recognised)
            if 0 <= value <= 31:
                values.append((value, confidence, lx))

        if len(values) < 2:
            return None
        values.sort(key=lambda item: item[2])
        player = values[0]
        if len(values) >= 3:
            player_second, dealer = values[1], values[2]
        else:
            player_second, dealer = None, values[1]
        confidence = min(
            player[1],
            dealer[1],
            player_second[1] if player_second is not None else 1.0,
        )
        # Final dealer totals may exceed 21 (the captured sessions contain
        # ordinary busts up to 26). Player-turn validation is stricter and is
        # handled by BlackjackJob, where the dealer must be an up-card 1..11.
        if confidence < self.min_confidence or not (1 <= dealer[0] <= 31):
            return None
        return BlackjackScore(
            player=player[0],
            dealer=dealer[0],
            confidence=confidence,
            player_second=(player_second[0] if player_second is not None else None),
        )
