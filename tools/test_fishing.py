"""Offline test for fishing detectors on reference screenshots."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config_loader import FISHING_CONFIG_PATH, LOGS_DIR, load_config, scale_region
from bot.detectors.cast_timing import CastTimingDetector
from bot.detectors.catch_modal import CatchModalDetector
from bot.detectors.fish_fight import FishFightDetector
from bot.detectors.fish_phase import FishPhaseDetector
from bot.detectors.fish_prompt import FishPromptDetector
from bot.detectors.fishing_setup import FishingSetupDetector


def crop(frame: np.ndarray, region: list[int]) -> np.ndarray:
    x, y, w, h = region
    return frame[y : y + h, x : x + w]


def grab_timing_crop(
    frame: np.ndarray, timing: CastTimingDetector, timing_region: list[int]
) -> np.ndarray:
    """Same priority as FisherJob._grab_timing: fixed ROI first, then auto-locate."""
    fixed = crop(frame, timing_region)
    if fixed.size > 0 and timing.is_active(fixed):
        return fixed
    auto = timing.crop_bar(frame)
    if auto.size > 0 and timing.is_active(auto):
        return auto
    return fixed if fixed.size > 0 else auto


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tools/test_fishing.py <screenshot.png> [--no-gui]")
        print("  Tests all 6 fishing stages on a 1920x1080 screenshot (or scaled).")
        sys.exit(1)

    path = Path(sys.argv[1])
    no_gui = "--no-gui" in sys.argv
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    config = load_config(FISHING_CONFIG_PATH)
    regions = config.get("regions", {})
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read: {path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")
    cfg_res = config.get("screen", {}).get("resolution", [1920, 1080])
    scaled_regions = {name: scale_region(rect, cfg_res, w, h) for name, rect in regions.items()}

    timing = CastTimingDetector(config)
    fight = FishFightDetector(config)
    prompt = FishPromptDetector(config)
    modal = CatchModalDetector(config)
    phase_det = FishPhaseDetector(config)
    setup = FishingSetupDetector(config)

    def grab(name: str):
        return crop(frame, scaled_regions[name])

    phase = phase_det.detect(frame)
    timing_crop = grab_timing_crop(frame, timing, scaled_regions["timing_bar"])
    fight_crop = fight.crop_bar(frame)
    timing_a = timing.analyze(timing_crop)
    timing_fresh = timing.is_active_fresh(timing_crop)
    fight_a = fight.analyze(
        frame,
        fight_crop if fight_crop.size else grab("fight_bar"),
        grab("fishing_prompt"),
    )
    bite_text_crop = grab("bite_text")
    bobber_crop = grab("bobber_widget")
    hook_crop = grab("hook_prompt")
    txt_sc = prompt.bite_text_score(bite_text_crop)
    bob_sc = prompt.bobber_bite_score(bobber_crop)
    hook_sc = prompt.hook_match_score(hook_crop)
    biting = prompt.is_biting(bite_text_crop, hook_crop, bobber_crop)
    waiting = prompt.is_waiting(bite_text_crop, hook_crop)
    caught = modal.is_visible(grab("catch_modal"))
    setup_scores = setup.scores(frame)

    slider_label = timing_a.slider_x if timing_a.slider_found else -1
    print(f"Phase: {phase.name}")
    print(f"Cast timing fresh (no ghost): {timing_fresh}")
    print(
        f"Cast timing: active={timing_a.active} slider={slider_label} "
        f"zone={timing_a.green_left}-{timing_a.green_right} center={timing_a.green_center} "
        f"slider_found={timing_a.slider_found} in_zone={timing_a.in_zone} green_w={timing_a.green_width}"
    )
    print(f"Fight: active={fight_a.active} steer={fight_a.direction.name} src={fight_a.source}")
    print(
        f"Bite: text={txt_sc:.3f} bobber={bob_sc:.3f} hook={hook_sc:.3f} "
        f"biting={biting} waiting={waiting}"
    )
    print(f"Caught modal: {caught}")
    print(
        f"Setup: start={setup_scores.start:.3f} "
        f"equipment={setup_scores.equipment:.3f} "
        f"start_visible={setup_scores.start >= setup.start_threshold} "
        f"equipment_visible={setup_scores.equipment >= setup.equipment_threshold}"
    )

    result = frame.copy()
    for name, key, color in [
        ("timing", "timing_bar", (0, 255, 0)),
        ("fish", "fish_icon", (0, 255, 255)),
        ("bite_text", "bite_text", (255, 0, 255)),
        ("bobber", "bobber_widget", (0, 200, 255)),
        ("hook", "hook_prompt", (255, 165, 0)),
        ("modal", "catch_modal", (255, 128, 0)),
        ("start", "fishing_start_prompt", (180, 255, 80)),
        ("equipment", "fishing_equipment_prompt", (80, 180, 255)),
    ]:
        x, y, rw, rh = scaled_regions[key]
        cv2.rectangle(result, (x, y), (x + rw, y + rh), color, 2)
        cv2.putText(result, name, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    dbg = timing.draw_debug(timing_crop if timing_crop.size else grab("timing_bar"))
    x, y, rw, rh = scaled_regions["timing_bar"]
    if rw > 0 and rh > 0:
        result[y : y + rh, x : x + rw] = cv2.resize(dbg, (rw, rh))

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_dbg = prompt.draw_debug(bite_text_crop, hook_crop, bobber_crop)
    cv2.imwrite(str(LOGS_DIR / "prompt_debug.png"), prompt_dbg)
    timing_dbg = timing.draw_debug(timing_crop if timing_crop.size else grab("timing_bar"))
    timing_out = LOGS_DIR / "timing_debug.png"
    cv2.imwrite(str(timing_out), timing_dbg)
    out = LOGS_DIR / "fishing_test_result.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")
    print(f"Saved: {timing_out}")

    if not no_gui:
        cv2.imshow("Fishing test", cv2.resize(result, (1280, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
