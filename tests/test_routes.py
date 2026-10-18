"""Loading a missing or broken route must never replay an earlier route."""

from unittest.mock import MagicMock

import pytest
import yaml

from bot.config_loader import ROUTES_DIR
from bot.jobs.miner import MinerJob
from bot.navigation.route_player import RoutePlayer


def write_route(path, events):
    path.write_text(yaml.safe_dump({"name": "route", "events": events}), encoding="utf-8")


def test_missing_load_discards_previously_loaded_route(tmp_path):
    path = tmp_path / "route.yaml"
    write_route(path, [{"action": "keydown", "key": "w", "time_ms": 0}])
    inputs = MagicMock()
    player = RoutePlayer(inputs)
    assert player.load(path) is True
    assert player.load(tmp_path / "missing.yaml") is False
    player.start()
    assert player.playing is False
    inputs.key_down.assert_not_called()


def test_load_during_playback_releases_the_old_route(tmp_path):
    path = tmp_path / "route.yaml"
    write_route(path, [{"action": "keydown", "key": "w", "time_ms": 0}])
    inputs = MagicMock()
    player = RoutePlayer(inputs)
    player.load(path)
    player.start()
    player.load(tmp_path / "missing.yaml")
    assert player.playing is False
    inputs.release_all_keys.assert_called_once()


@pytest.mark.parametrize(
    "events",
    [
        "invalid",
        [None],
        [{"action": "keydown", "time_ms": 0}],
        [{"action": "unknown", "time_ms": 0}],
        [{"action": "keydown", "key": "w", "time_ms": -1}],
        [{"action": "keydown", "key": "w", "time_ms": float("nan")}],
        [
            {"action": "keydown", "key": "w", "time_ms": 20},
            {"action": "keyup", "key": "w", "time_ms": 10},
        ],
    ],
)
def test_invalid_route_is_rejected_before_any_input(tmp_path, events):
    path = tmp_path / "route.yaml"
    write_route(path, events)
    inputs = MagicMock()
    player = RoutePlayer(inputs)
    with pytest.raises(ValueError):
        player.load(path)
    player.start()
    assert player.playing is False
    inputs.key_down.assert_not_called()


def test_miner_does_not_announce_or_start_an_empty_route(tmp_path):
    path = tmp_path / "empty.yaml"
    write_route(path, [])
    job = MinerJob.__new__(MinerJob)
    job._debug = False
    job._route_player = MagicMock()
    job._route_player.load.return_value = False
    assert job._start_route_file(path) is False
    job._route_player.start.assert_not_called()


def test_all_bundled_routes_still_load_without_sending_input():
    for path in ROUTES_DIR.glob("*.yaml"):
        inputs = MagicMock()
        assert RoutePlayer(inputs).load(path) is True
        inputs.key_down.assert_not_called()
