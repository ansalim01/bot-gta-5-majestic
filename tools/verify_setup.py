"""Verify capture, scaling and detection for fullscreen / borderless GTA."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.capture import GameCapture
from bot.config_loader import (
    LOGS_DIR,
    load_config,
    scale_regions,
    scaled_rocks_relative,
)
from bot.detectors.minigame import MinigameDetector


def simulate_fullscreen(image_path: Path, window_w: int, window_h: int) -> None:
    """Test detection as if screenshot were captured at given window size."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Cannot read: {image_path}")
        return

    config = load_config()
    cfg_res = config.get("screen", {}).get("resolution", [1920, 1080])
    img_h, img_w = image.shape[:2]

    if img_w != window_w or img_h != window_h:
        print(f"  Image is {img_w}x{img_h}, simulating window {window_w}x{window_h}")
        image = cv2.resize(image, (window_w, window_h), interpolation=cv2.INTER_LINEAR)

    regions = scale_regions(config.get("regions", {}), cfg_res, window_w, window_h)
    minigame = regions["minigame"]
    progress = regions["minigame_progress"]
    rocks = regions["rocks_mask"]

    x, y, w, h = minigame
    minigame_frame = image[y : y + h, x : x + w].copy()
    px, py, pw, ph = progress
    progress_frame = image[py : py + ph, px : px + pw].copy()

    detector = MinigameDetector(config)
    detector.sync_window_scale(window_w, window_h)

    veins = detector.find_veins(minigame_frame)
    active = detector.is_active(minigame_frame, progress_frame)
    fill = detector.get_progress_fill(progress_frame)

    print(f"  Window {window_w}x{window_h}: veins={len(veins)} active={active} progress={fill:.0%}")
    print(f"  Scaled minigame ROI: {minigame}")
    print(f"  Scaled rocks mask: {rocks}")
    print(f"  Rocks relative in crop: {detector._rocks_mask_relative}")


def main() -> None:
    config = load_config()
    cfg_res = config.get("screen", {}).get("resolution", [1920, 1080])
    capture = GameCapture(config)

    print("=" * 55)
    print("  Miner Bot — fullscreen / capture verification")
    print("=" * 55)

    rect = capture.find_window()
    if rect:
        sx, sy = capture.scale_factors
        print(f"Game window: {rect.width}x{rect.height} at ({rect.left}, {rect.top})")
        print(f"Config base: {cfg_res[0]}x{cfg_res[1]}  scale: {sx:.4f}x / {sy:.4f}y")
        regions = scale_regions(config.get("regions", {}), cfg_res, rect.width, rect.height)
        for name, r in regions.items():
            print(f"  {name}: {r}")
        rocks_rel = scaled_rocks_relative(
            config.get("regions", {}), cfg_res, rect.width, rect.height
        )
        print(f"  rocks_relative (in minigame crop): {rocks_rel}")

        try:
            frame = capture.grab_region(config["regions"]["minigame"])
            progress = capture.grab_region(config["regions"]["minigame_progress"])
            detector = MinigameDetector(config)
            detector.sync_window_scale(rect.width, rect.height)
            veins = detector.find_veins(frame)
            active = detector.is_active(frame, progress)
            print(
                f"Live grab: frame {frame.shape[1]}x{frame.shape[0]} veins={len(veins)} active={active}"
            )

            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            out = LOGS_DIR / "verify_live_minigame.png"
            cv2.imwrite(str(out), detector.draw_debug(frame, progress))
            print(f"Debug saved: {out}")
        except Exception as exc:
            print(f"Live grab failed: {exc}")
    else:
        mon = capture.rect
        print(f"No GTA window — using monitor: {mon.width}x{mon.height} at ({mon.left}, {mon.top})")

    templates = ROOT / "config" / "templates"
    for name in ("reference_2veins.png", "reference_minigame.png"):
        path = templates / name
        if not path.exists():
            continue
        print(f"\nOffline test: {name}")
        simulate_fullscreen(path, 1920, 1080)
        simulate_fullscreen(path, 1936, 1048)

    capture.close()
    print("\nDone. Start bot with F9 when minigame is open on screen.")


if __name__ == "__main__":
    main()
