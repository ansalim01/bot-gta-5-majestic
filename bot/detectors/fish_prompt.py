"""Fishing prompt detection — bite when «Клюет!» text or bobber icon appears."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from bot.config_loader import TEMPLATES_DIR, fishing_region

FISHING_TEMPLATES = TEMPLATES_DIR / "fishing"

BITE_TEXT_NAMES = (
    "bite_klyuet_full.png",
    "bite_klyuet.png",
    "bite_keyword.png",
    "bite_prompt_v3.png",
)
# Strict subset of BITE_TEXT_NAMES — kept as a name set (not a separate template
# load) so keyword scoring reuses the matchTemplate results computed for
# bite_text_score() instead of re-matching the same images a second time.
KEYWORD_BITE_NAMES = {"bite_klyuet_full.png", "bite_klyuet.png", "bite_keyword.png"}
BOBBER_BITE_NAMES = ("bobber_icon.png", "bobber_bite.png")
HOOK_TEMPLATE_NAMES = ("hook_lmb.png", "hook_prompt.png", "hook_enter_new.png")
WAIT_TEMPLATE_NAMES = (
    "wait_instruction_current.png",
    "wait_prompt_v3.png",
    "wait_prompt_v2.png",
    "wait_prompt.png",
)
REEL_TEMPLATE_NAMES = ("reel_prompt.png",)


@dataclass
class TextScores:
    bite: float
    keyword: float
    wait: float
    waiting: bool


class FishPromptDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("prompt", {})
        regions = config.get("regions", {})
        bite_text_region = fishing_region(regions, "bite_text")
        bobber_region = fishing_region(regions, "bobber_widget")
        self._bite_text_ref_w = int(bite_text_region[2])
        self._bite_text_ref_h = int(bite_text_region[3])
        self._bobber_ref_w = int(bobber_region[2])
        self._bobber_ref_h = int(bobber_region[3])
        self._match_threshold = cfg.get("match_threshold", 0.85)
        self._bite_margin = cfg.get("bite_margin", 0.003)
        # 0.45 matches the previously hardcoded cap (the old code silently
        # clamped any configured value down to 0.45).
        self._wait_floor = cfg.get("wait_match_floor", 0.45)
        self._bite_text_threshold = cfg.get("bite_text_threshold", 0.60)
        self._bobber_threshold = cfg.get("bobber_bite_threshold", 0.70)
        self._hook_match_threshold = cfg.get("hook_match_threshold", 0.80)
        self._hook_white_min = cfg.get("hook_prompt_white_min", 0.035)
        # On the current HUD the bottom-right "hook fish [Space]" control is
        # visible for the whole waiting phase, not only when a fish bites.
        # It is useful as a "fishing session still alive" signal, but must not
        # itself trigger a hook when bite_text is the configured source.
        hook_trigger = config.get("fishing", {}).get("hook_trigger", "bite_text")
        self._hook_is_bite_signal = hook_trigger != "bite_text"
        self._bite_text_named = self._load_named_with_names(BITE_TEXT_NAMES)
        self._bite_text_templates = [img for _, img in self._bite_text_named]
        self._keyword_templates = [
            img for name, img in self._bite_text_named if name in KEYWORD_BITE_NAMES
        ]
        self._bobber_templates = self._load_named(BOBBER_BITE_NAMES)
        self._hook_templates = self._load_named(HOOK_TEMPLATE_NAMES)
        self._wait_templates = self._load_named(WAIT_TEMPLATE_NAMES)
        self._reel_templates = self._load_named(REEL_TEMPLATE_NAMES)
        hook_region = fishing_region(regions, "hook_prompt")
        self._hook_ref_w = int(hook_region[2])
        self._hook_ref_h = int(hook_region[3])

        # A missing template silently zeroes its score — the bot then just
        # "never sees" bites/waits with no hint why. Say it out loud once.
        for label, templates in (
            ("bite_text", self._bite_text_templates),
            ("bobber", self._bobber_templates),
            ("hook", self._hook_templates),
            ("wait", self._wait_templates),
        ):
            if not templates:
                print(
                    f"[BOT] WARNING: no '{label}' templates found in "
                    f"{FISHING_TEMPLATES} — this detector is disabled",
                    flush=True,
                )

    def _load_named_with_names(self, names: tuple[str, ...]) -> list[tuple[str, np.ndarray]]:
        pairs: list[tuple[str, np.ndarray]] = []
        for name in names:
            path = FISHING_TEMPLATES / name
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            if img is not None:
                pairs.append((name, img))
        return pairs

    def _load_named(self, names: tuple[str, ...]) -> list[np.ndarray]:
        return [img for _, img in self._load_named_with_names(names)]

    def _resize_to(self, frame: np.ndarray, ref_w: int, ref_h: int) -> np.ndarray:
        if frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        if w == ref_w and h == ref_h:
            return frame
        return cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC)

    def _best_match_on_frame(self, frame: np.ndarray, templates: list[np.ndarray]) -> float:
        if frame.size == 0 or not templates:
            return 0.0
        return max(self._match_template_score(frame, tpl) for tpl in templates)

    def text_scores(self, bite_text_frame: np.ndarray) -> TextScores:
        """Bite/keyword/wait scores + the `is_waiting` verdict in a single pass
        over the bite_text crop. The naive equivalent (bite_text_score() +
        keyword_bite_score() + wait_score() + is_waiting()) ends up running
        matchTemplate ~17 times on the same image because keyword templates are
        a subset of the bite templates and is_waiting() recomputes wait/bite
        scores internally; this does it in 7 (one per distinct template)."""
        if bite_text_frame.size == 0:
            return TextScores(0.0, 0.0, 0.0, False)
        frame = self._resize_to(bite_text_frame, self._bite_text_ref_w, self._bite_text_ref_h)

        bite_best = 0.0
        keyword_best = 0.0
        for name, tpl in self._bite_text_named:
            score = self._match_template_score(frame, tpl)
            bite_best = max(bite_best, score)
            if name in KEYWORD_BITE_NAMES:
                keyword_best = max(keyword_best, score)

        wait_best = self._best_match_on_frame(frame, self._wait_templates)
        waiting = wait_best >= self._wait_floor and wait_best > bite_best + self._bite_margin
        return TextScores(bite=bite_best, keyword=keyword_best, wait=wait_best, waiting=waiting)

    def _match_template_score(self, frame: np.ndarray, tpl: np.ndarray) -> float:
        use_tpl = tpl
        if tpl.shape[0] > frame.shape[0] or tpl.shape[1] > frame.shape[1]:
            scale = min(frame.shape[0] / tpl.shape[0], frame.shape[1] / tpl.shape[1]) * 0.95
            use_tpl = cv2.resize(tpl, (int(tpl.shape[1] * scale), int(tpl.shape[0] * scale)))
        res = cv2.matchTemplate(frame, use_tpl, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def bite_text_score(self, bite_text_frame: np.ndarray) -> float:
        frame = self._resize_to(bite_text_frame, self._bite_text_ref_w, self._bite_text_ref_h)
        return self._best_match_on_frame(frame, self._bite_text_templates)

    def bobber_bite_score(self, bobber_frame: np.ndarray) -> float:
        if bobber_frame.size == 0:
            return 0.0
        frame = bobber_frame
        if frame.shape[0] < 40 or frame.shape[1] < 40:
            frame = self._resize_to(frame, self._bobber_ref_w, self._bobber_ref_h)
        return self._best_match_on_frame(frame, self._bobber_templates)

    def hook_match_score(self, hook_prompt_frame: np.ndarray) -> float:
        if hook_prompt_frame.size == 0 or not self._hook_templates:
            return 0.0
        frame = self._resize_to(hook_prompt_frame, self._hook_ref_w, self._hook_ref_h)
        return self._best_match_on_frame(frame, self._hook_templates)

    def hook_white_ratio(self, hook_prompt_frame: np.ndarray) -> float:
        if hook_prompt_frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(hook_prompt_frame, cv2.COLOR_BGR2GRAY)
        return float(np.count_nonzero(gray > 175)) / gray.size

    def is_hook_visible(self, hook_prompt_frame: np.ndarray) -> bool:
        if hook_prompt_frame.size == 0:
            return False
        if self.hook_match_score(hook_prompt_frame) >= self._hook_match_threshold:
            return True
        return self.hook_white_ratio(hook_prompt_frame) >= self._hook_white_min

    def is_fishing_control_visible(self, hook_prompt_frame: np.ndarray) -> bool:
        """The persistent bottom-right fishing control is still on screen."""
        return self.is_hook_visible(hook_prompt_frame)

    @property
    def bite_text_threshold(self) -> float:
        return self._bite_text_threshold

    def is_bite_text_visible(self, bite_text_frame: np.ndarray) -> bool:
        return self.bite_text_score(bite_text_frame) >= self._bite_text_threshold

    def is_bobber_biting(self, bobber_frame: np.ndarray) -> bool:
        return self.bobber_bite_score(bobber_frame) >= self._bobber_threshold

    def is_biting(
        self,
        bite_text_frame: np.ndarray,
        hook_prompt_frame: np.ndarray,
        bobber_frame: np.ndarray | None = None,
    ) -> bool:
        """True when «Клюет!» text, bobber splash icon, or «Подсечь рыбу» prompt is visible."""
        if self._hook_is_bite_signal and self.is_hook_visible(hook_prompt_frame):
            return True
        if self.is_bite_text_visible(bite_text_frame):
            return True
        if bobber_frame is not None and self.is_bobber_biting(bobber_frame):
            return True
        return False

    def bite_scores(self, prompt_frame: np.ndarray) -> tuple[float, float]:
        bite = self.bite_text_score(prompt_frame)
        wait = self._best_match_on_frame(
            self._resize_to(prompt_frame, self._bite_text_ref_w, self._bite_text_ref_h),
            self._wait_templates,
        )
        return bite, wait

    def wait_score(self, bite_text_frame: np.ndarray) -> float:
        if bite_text_frame.size == 0:
            return 0.0
        frame = self._resize_to(bite_text_frame, self._bite_text_ref_w, self._bite_text_ref_h)
        return self._best_match_on_frame(frame, self._wait_templates)

    def keyword_bite_score(self, prompt_frame: np.ndarray) -> float:
        if prompt_frame.size == 0:
            return 0.0
        frame = self._resize_to(prompt_frame, self._bite_text_ref_w, self._bite_text_ref_h)
        return self._best_match_on_frame(frame, self._keyword_templates)

    def is_waiting(
        self,
        bite_text_frame: np.ndarray,
        hook_prompt_frame: np.ndarray | None = None,
    ) -> bool:
        if (
            self._hook_is_bite_signal
            and hook_prompt_frame is not None
            and self.is_hook_visible(hook_prompt_frame)
        ):
            return False
        frame = self._resize_to(bite_text_frame, self._bite_text_ref_w, self._bite_text_ref_h)
        wait_score = self._best_match_on_frame(frame, self._wait_templates)
        bite_score = self.bite_text_score(bite_text_frame)
        return wait_score >= self._wait_floor and wait_score > bite_score + self._bite_margin

    def is_reeling(self, prompt_frame: np.ndarray) -> bool:
        if not self._reel_templates:
            return False
        frame = self._resize_to(prompt_frame, self._bite_text_ref_w, self._bite_text_ref_h)
        reel_score = self._best_match_on_frame(frame, self._reel_templates)
        return reel_score >= self._match_threshold

    def draw_debug(
        self,
        bite_text_frame: np.ndarray,
        hook_frame: np.ndarray,
        bobber_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        debug = np.ascontiguousarray(
            bite_text_frame.copy() if bite_text_frame.size else hook_frame.copy()
        )
        txt_sc = self.bite_text_score(bite_text_frame)
        hook_sc = self.hook_match_score(hook_frame)
        bob_sc = self.bobber_bite_score(bobber_frame) if bobber_frame is not None else 0.0
        biting = self.is_biting(bite_text_frame, hook_frame, bobber_frame)
        label = "BITE!" if biting else "waiting"
        cv2.putText(
            debug,
            f"{label} txt={txt_sc:.2f} hook={hook_sc:.2f} bob={bob_sc:.2f}",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0) if biting else (200, 200, 200),
            1,
        )
        return debug
