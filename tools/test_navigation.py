"""Test minimap navigation and mine phase detection on screenshots."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config_loader import load_config, scale_regions
from bot.detectors.mine_phase import MinePhaseDetector
from bot.navigation.minimap_nav import MinimapNavigator


def analyze_image(path: Path, config: dict, width: int, height: int) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"SKIP {path.name}: cannot read")
        return

    img = cv2.resize(img, (width, height))
    regions = scale_regions(
        config.get("regions", {}), config["screen"]["resolution"], width, height
    )

    class FakeCapture:
        def grab_region(self, region):
            x, y, w, h = region
            return img[y : y + h, x : x + w]

    detector = MinePhaseDetector(config)
    navigator = MinimapNavigator(config)
    phase = detector.detect(img, FakeCapture().grab_region)

    minimap = FakeCapture().grab_region(regions["minimap"])
    blip = navigator.find_target_blip(minimap)
    blip_info = (
        f"({blip.x},{blip.y}) d={blip.distance:.0f} a={blip.angle_deg:.0f}" if blip else "none"
    )
    print(f"{path.name:40} phase={phase.name:10} blip={blip_info}")

    if "--save" in sys.argv:
        debug = navigator.draw_debug(minimap)
        out = ROOT / "logs" / "debug" / f"nav_{path.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), debug)


def main() -> None:
    config = load_config()
    width, height = 1920, 1080

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        nav_dir = ROOT / "config" / "templates" / "navigation"
        paths = sorted(nav_dir.glob("*.png"))

    if not paths:
        print("No images found. Usage: python tools/test_navigation.py [image.png ...]")
        sys.exit(1)

    print(f"Testing {len(paths)} images at {width}x{height}\n")
    for path in paths:
        analyze_image(path, config, width, height)


if __name__ == "__main__":
    main()
