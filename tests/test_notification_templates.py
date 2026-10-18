from pathlib import Path

import cv2
import numpy as np
import pytest

from bot.config_loader import load_config
from bot.detectors.notification import NotificationDetector

FIXTURES = Path(__file__).parent / "fixtures" / "notifications"


@pytest.mark.parametrize("action", ["drop", "mine"])
@pytest.mark.parametrize("background", [0, 35, 90])
def test_action_is_distinguished_without_chat(action, background):
    """The shared 'press E' text must not trigger the other action."""
    detector = NotificationDetector(load_config())
    prompt = cv2.imread(str(FIXTURES / f"{action}.png"))
    assert prompt is not None
    frame = np.full((120, 400, 3), background, dtype=np.uint8)
    height, width = prompt.shape[:2]
    frame[8 : 8 + height, 7 : 7 + width] = prompt

    assert detector.is_drop_visible(frame) == (action == "drop")
    assert detector.is_mine_visible(frame) == (action == "mine")


def test_plain_background_is_not_an_action():
    detector = NotificationDetector(load_config())
    frame = np.zeros((120, 400, 3), dtype=np.uint8)

    assert not detector.is_drop_visible(frame)
    assert not detector.is_mine_visible(frame)
