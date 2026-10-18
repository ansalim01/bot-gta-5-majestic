from __future__ import annotations

import time
import unittest
from pathlib import Path

import cv2
import yaml

from bot.detectors.gym_prompt import GymPromptDetector
from bot.jobs.gym import GymJob
from bot.state_machine import StateMachine

ROOT = Path(__file__).resolve().parents[1]


class GymPromptDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(ROOT / "config" / "gym.yaml", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        cls.detector = GymPromptDetector(config)

    @staticmethod
    def _image(key: str):
        return cv2.imread(str(ROOT / "config" / "templates" / "gym" / f"prompt_{key}.png"))

    def test_recognizes_every_captcha_variant(self) -> None:
        templates = ROOT / "config" / "templates" / "gym"
        for key in ("w", "d", "s", "a"):
            paths = sorted(templates.glob(f"prompt_{key}*.png"))
            self.assertGreater(len(paths), 1)
            for path in paths:
                with self.subTest(key=key, template=path.name):
                    result = self.detector.detect(cv2.imread(str(path)))
                    self.assertEqual(result.key, key)
                    self.assertGreaterEqual(result.score, 0.95)

    def test_recognizes_small_rotations(self) -> None:
        for angle in (-4.0, 4.0):
            for key in ("w", "d", "s", "a"):
                with self.subTest(key=key, angle=angle):
                    matrix = cv2.getRotationMatrix2D((57, 57), angle, 1.0)
                    rotated = cv2.warpAffine(self._image(key), matrix, (114, 114))
                    self.assertEqual(self.detector.detect(rotated).key, key)

    def test_rejects_missing_prompt(self) -> None:
        no_prompt = cv2.imread(str(ROOT / "config" / "templates" / "gym" / "no_prompt.png"))
        result = self.detector.detect(no_prompt)
        self.assertIsNone(result.key)
        self.assertFalse(result.circle_visible)

    def test_shape_fallback_handles_all_unseen_letters(self) -> None:
        templates = ROOT / "config" / "templates" / "gym"
        samples = {
            "a": ["validation_a_new.png", "validation_a_narrow.png"],
            "w": ["validation_w_new.png"],
            "d": ["validation_d_new.png"],
            "s": ["validation_s_new.png"],
        }
        for key, filenames in samples.items():
            for filename in filenames:
                with self.subTest(key=key, filename=filename):
                    image = cv2.imread(str(templates / filename))
                    result = self.detector.detect(image)
                    self.assertEqual(result.key, key)
                    self.assertEqual(result.source, "shape")


class GymJobTests(unittest.TestCase):
    @staticmethod
    def _config():
        with open(ROOT / "config" / "gym.yaml", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["gym"]["start_settle_sec"] = 0
        config["gym"]["reaction_delay_ms"] = [0, 0]
        return config

    @staticmethod
    def _prompt(name: str):
        return cv2.imread(str(ROOT / "config" / "templates" / "gym" / name))

    def test_start_e_is_followed_by_recognized_w_press(self) -> None:
        config = self._config()
        prompt = self._prompt("prompt_w.png")

        class Capture:
            @staticmethod
            def grab_region(_region):
                return prompt.copy()

        class Input:
            def __init__(self):
                self.pressed = []

            def press_key(self, key, duration):
                self.pressed.append((key, duration))

            @staticmethod
            def release_all_keys():
                pass

        state = StateMachine()
        input_handler = Input()
        job = GymJob(config, Capture(), input_handler, state)
        state.toggle()

        job.tick()  # starts the exercise with E
        job.tick()  # first W confirmation frame
        job.tick()  # second W confirmation frame -> press W

        self.assertEqual(
            [key for key, _duration in input_handler.pressed],
            ["e", "w"],
        )

    def test_sustenance_check_runs_before_cycle_e(self) -> None:
        config = self._config()
        events = []

        class Capture:
            @staticmethod
            def grab_region(_region):
                return GymJobTests._prompt("no_prompt.png")

        class Input:
            @staticmethod
            def press_key(key, _duration):
                events.append(key)

            @staticmethod
            def release_all_keys():
                pass

        state = StateMachine()
        job = GymJob(
            config,
            Capture(),
            Input(),
            state,
            before_cycle=lambda: events.append("sustenance"),
        )
        state.toggle()
        job.tick()
        self.assertEqual(events, ["sustenance", "e"])

    def test_pause_between_four_prompts_does_not_press_e_early(self) -> None:
        config = self._config()

        class Capture:
            frame = GymJobTests._prompt("prompt_w.png")

            @classmethod
            def grab_region(cls, _region):
                return cls.frame.copy()

        class Input:
            def __init__(self):
                self.pressed = []

            def press_key(self, key, duration):
                self.pressed.append((key, duration))

            @staticmethod
            def release_all_keys():
                pass

        state = StateMachine()
        input_handler = Input()
        job = GymJob(config, Capture(), input_handler, state)
        state.toggle()
        job.tick()
        job.tick()
        job.tick()
        self.assertEqual(
            [key for key, _duration in input_handler.pressed],
            ["e", "w"],
        )

        Capture.frame = self._prompt("no_prompt.png")
        job.tick()
        job.tick()
        job._absent_since = time.perf_counter() - config["gym"]["restart_after_absent_sec"] + 1.0
        job.tick()
        self.assertEqual(
            [key for key, _duration in input_handler.pressed],
            ["e", "w"],
        )

        job._absent_since = time.perf_counter() - config["gym"]["restart_after_absent_sec"] - 0.1
        job.tick()
        self.assertEqual(
            [key for key, _duration in input_handler.pressed],
            ["e", "w", "e"],
        )


if __name__ == "__main__":
    unittest.main()
