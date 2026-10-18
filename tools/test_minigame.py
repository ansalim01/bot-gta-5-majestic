"""Test minigame vein detection on a static screenshot."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config_loader import LOGS_DIR, load_config, scale_regions, scaled_rocks_relative
from bot.detectors.minigame import MinigameDetector


def crop_region(image: np.ndarray, region: list[int]) -> np.ndarray:
    x, y, w, h = region
    return image[y : y + h, x : x + w].copy()


def auto_extract_vein_templates(
    image: np.ndarray,
    rocks_region: list[int],
    templates_dir: Path,
    config: dict,
) -> int:
    """Auto-crop vein templates from rocks ROI using color detection."""
    rocks = crop_region(image, rocks_region)
    detector = MinigameDetector(config)
    detector._rocks_mask_offset = None

    hsv = cv2.cvtColor(rocks, cv2.COLOR_BGR2HSV)
    vein_hsv = config.get("minigame", {}).get("vein_hsv", {})
    yellow_low = np.array(vein_hsv.get("yellow_low", [12, 60, 140]))
    yellow_high = np.array(vein_hsv.get("yellow_high", [40, 255, 255]))
    white_low = np.array(vein_hsv.get("white_low", [0, 0, 200]))
    white_high = np.array(vein_hsv.get("white_high", [180, 50, 255]))

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, yellow_low, yellow_high),
        cv2.inRange(hsv, white_low, white_high),
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = config.get("minigame", {}).get("min_vein_area", 15)
    max_area = config.get("minigame", {}).get("max_vein_area", 800)

    templates_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if saved >= 5:
            break

        x, y, w, h = cv2.boundingRect(contour)
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(rocks.shape[1], x + w + pad)
        y2 = min(rocks.shape[0], y + h + pad)
        crop = rocks[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        path = templates_dir / f"vein_{saved + 1:02d}.png"
        cv2.imwrite(str(path), crop)
        saved += 1
        print(f"Auto-extracted template: {path}")

    return saved


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/test_minigame.py <screenshot.png> [--extract]")
        sys.exit(1)

    screenshot_path = Path(sys.argv[1])
    do_extract = "--extract" in sys.argv

    if not screenshot_path.exists():
        print(f"File not found: {screenshot_path}")
        sys.exit(1)

    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"Cannot read image: {screenshot_path}")
        sys.exit(1)

    config = load_config()
    cfg_res = config.get("screen", {}).get("resolution", [1920, 1080])
    img_h, img_w = image.shape[:2]

    regions = scale_regions(config.get("regions", {}), cfg_res, img_w, img_h)
    minigame_region = regions.get("minigame", [277, 173, 469, 213])
    progress_region = regions.get("minigame_progress", [373, 346, 277, 30])
    rocks_region = regions.get("rocks_mask", [309, 193, 405, 142])

    templates_dir = ROOT / "config" / "templates"

    if do_extract:
        count = auto_extract_vein_templates(image, rocks_region, templates_dir, config)
        print(f"Extracted {count} vein templates")

    minigame_frame = crop_region(image, minigame_region)
    progress_frame = crop_region(image, progress_region)

    mg_x, mg_y = minigame_region[0], minigame_region[1]
    rocks_relative = scaled_rocks_relative(config.get("regions", {}), cfg_res, img_w, img_h)

    detector = MinigameDetector(config)
    detector.set_regions(minigame_region, rocks_relative)

    veins = detector.find_veins(minigame_frame)
    active = detector.is_active(minigame_frame, progress_frame)
    complete = detector.is_complete(minigame_frame, progress_frame)
    fill = detector.get_progress_fill(progress_frame)

    print(f"Minigame active: {active}")
    print(f"Minigame complete: {complete}")
    print(f"Progress fill: {fill:.0%}")
    print(f"Veins found: {len(veins)}")
    for i, (vx, vy) in enumerate(veins):
        abs_x = mg_x + vx
        abs_y = mg_y + vy
        print(f"  vein {i + 1}: ({vx}, {vy}) abs=({abs_x}, {abs_y})")

    result = image.copy()

    for name, region, color in [
        ("minigame", minigame_region, (0, 255, 0)),
        ("rocks_mask", rocks_region, (255, 255, 0)),
        ("progress", progress_region, (255, 0, 255)),
    ]:
        x, y, w, h = region
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    debug_mini = detector.draw_debug(minigame_frame, progress_frame)
    mg_x, mg_y, mg_w, mg_h = minigame_region
    result[mg_y : mg_y + mg_h, mg_x : mg_x + mg_w] = debug_mini

    pg_x, pg_y, pg_w, pg_h = progress_region
    pg_debug = detector._progress_bar.draw_debug(progress_frame)
    result[pg_y : pg_y + pg_h, pg_x : pg_x + pg_w] = cv2.resize(
        pg_debug, (pg_w, pg_h), interpolation=cv2.INTER_LINEAR
    )

    for vx, vy in veins:
        abs_x = mg_x + vx
        abs_y = mg_y + vy
        cv2.circle(result, (abs_x, abs_y), 10, (0, 0, 255), 2)
        cv2.circle(result, (abs_x, abs_y), 3, (0, 0, 255), -1)

    cv2.putText(
        result,
        f"veins={len(veins)} progress={fill:.0%} active={active}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "minigame_test_result.png"
    cv2.imwrite(str(out_path), result)
    print(f"Result saved to {out_path}")

    if "--no-gui" not in sys.argv:
        cv2.imshow("Minigame Test", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
