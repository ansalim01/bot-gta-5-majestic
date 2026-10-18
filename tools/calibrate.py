"""ROI calibration tool — select regions on a screenshot and save to config YAML."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
MINER_CONFIG = ROOT / "config" / "miner.yaml"
FISHING_CONFIG = ROOT / "config" / "fishing.yaml"

MINER_REGION_NAMES = [
    "notification",
    "minigame",
    "minigame_progress",
    "rocks_mask",
    "carry_indicator",
    "minimap",
    "carry_banner",
    "mining_hud",
]

FISHING_REGION_NAMES = [
    "notification",
    "timing_bar",
    "fishing_prompt",
    "hook_prompt",
    "fish_icon",
    "fight_bar",
    "catch_modal",
    "keep_button",
    "minimap",
]
current_region_idx = 0
regions: dict[str, list[int]] = {}
CONFIG_PATH = MINER_CONFIG
REGION_NAMES = MINER_REGION_NAMES
drawing = False
start_point: tuple[int, int] | None = None
current_rect: tuple[int, int, int, int] | None = None
image = None
display = None


def load_existing_regions() -> dict[str, list[int]]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config.get("regions", {})
    return {}


def mouse_callback(event, x, y, _flags, _param):
    global drawing, start_point, current_rect, display, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing and start_point:
        display = image.copy()
        cv2.rectangle(display, start_point, (x, y), (0, 255, 0), 2)
        name = REGION_NAMES[current_region_idx]
        cv2.putText(
            display,
            f"Selecting: {name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    elif event == cv2.EVENT_LBUTTONUP and drawing and start_point:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)
        if rw > 5 and rh > 5:
            current_rect = (rx, ry, rw, rh)


def save_regions():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "regions" not in config:
        config["regions"] = {}
    config["regions"].update(regions)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"Saved regions to {CONFIG_PATH}")
    for name, rect in regions.items():
        print(f"  {name}: {rect}")


def draw_ui():
    global display
    display = image.copy()

    for name, rect in regions.items():
        x, y, w, h = rect
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(display, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    if current_rect:
        x, y, w, h = current_rect
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    name = REGION_NAMES[current_region_idx]
    cv2.putText(
        display,
        f"Region: {name} | N=next S=save Q=quit",
        (10, display.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main():
    global image, current_region_idx, current_rect, regions, CONFIG_PATH, REGION_NAMES

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]

    if "--fishing" in flags:
        CONFIG_PATH = FISHING_CONFIG
        REGION_NAMES = FISHING_REGION_NAMES

    if len(args) < 1:
        print("Usage: python tools/calibrate.py <screenshot.png> [--fishing]")
        print("  Select ROI with mouse drag. N=next region, S=save, Q=quit")
        sys.exit(1)

    screenshot_path = Path(args[0])
    if not screenshot_path.exists():
        print(f"File not found: {screenshot_path}")
        sys.exit(1)

    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"Cannot read image: {screenshot_path}")
        sys.exit(1)

    regions = load_existing_regions()
    draw_ui()

    window = "Calibrate ROI"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    print("Controls: drag=select, N=next region, S=save, Q=quit")
    print(f"Regions to calibrate: {REGION_NAMES}")

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            if current_rect:
                name = REGION_NAMES[current_region_idx]
                regions[name] = list(current_rect)
                current_rect = None
                print(f"Saved {name}: {regions[name]}")
            current_region_idx = (current_region_idx + 1) % len(REGION_NAMES)
            print(f"Now selecting: {REGION_NAMES[current_region_idx]}")
            draw_ui()
        elif key == ord("s"):
            if current_rect:
                name = REGION_NAMES[current_region_idx]
                regions[name] = list(current_rect)
                current_rect = None
            save_regions()

    if regions:
        save_regions()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
