"""Extract ore vein template crops from a screenshot."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "config" / "templates"

drawing = False
start_point: tuple[int, int] | None = None
current_rect: tuple[int, int, int, int] | None = None
image = None
display = None
vein_counter = 1


def next_vein_path() -> Path:
    global vein_counter
    while True:
        path = TEMPLATES_DIR / f"vein_{vein_counter:02d}.png"
        vein_counter += 1
        if not path.exists():
            return path
        vein_counter += 1


def mouse_callback(event, x, y, _flags, _param):
    global drawing, start_point, current_rect, display, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing and start_point:
        display = image.copy()
        cv2.rectangle(display, start_point, (x, y), (0, 255, 0), 2)
        cv2.putText(
            display,
            "Drag to select vein | S=save crop | Q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
        if rw > 3 and rh > 3:
            current_rect = (rx, ry, rw, rh)


def save_crop():
    global current_rect, display

    if current_rect is None:
        print("No selection to save")
        return

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    x, y, w, h = current_rect
    crop = image[y : y + h, x : x + w]
    path = next_vein_path()
    cv2.imwrite(str(path), crop)
    print(f"Saved {path} ({w}x{h})")
    current_rect = None
    display = image.copy()


def main():
    global image, display

    if len(sys.argv) < 2:
        print("Usage: python tools/extract_vein.py <screenshot.png>")
        print("  Drag to select ore vein, S=save, Q=quit")
        sys.exit(1)

    screenshot_path = Path(sys.argv[1])
    if not screenshot_path.exists():
        print(f"File not found: {screenshot_path}")
        sys.exit(1)

    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"Cannot read image: {screenshot_path}")
        sys.exit(1)

    display = image.copy()
    window = "Extract Vein Template"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    print("Drag to select vein area. S=save crop, Q=quit")

    while True:
        if current_rect:
            disp = display.copy()
            x, y, w, h = current_rect
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(window, disp)
        else:
            cv2.imshow(window, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_crop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
