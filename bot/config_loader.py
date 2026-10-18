"""Load and access bot configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "miner.yaml"
FISHING_CONFIG_PATH = ROOT_DIR / "config" / "fishing.yaml"
GYM_CONFIG_PATH = ROOT_DIR / "config" / "gym.yaml"
SUSTENANCE_CONFIG_PATH = ROOT_DIR / "config" / "sustenance.yaml"
BLACKJACK_CONFIG_PATH = ROOT_DIR / "config" / "blackjack.yaml"
TEMPLATES_DIR = ROOT_DIR / "config" / "templates"
ROUTES_DIR = ROOT_DIR / "config" / "routes"
LOGS_DIR = ROOT_DIR / "logs" / "debug"
USER_CONFIG_DIR = ROOT_DIR / "config" / "user"

# Single source of truth for fishing region fallbacks, matched to
# config/fishing.yaml. fisher.py, fish_phase.py and fish_prompt.py used to
# each carry their own drifted copies of these literals — a missing yaml key
# would silently pick a stale region depending on which module read it.
FISHING_DEFAULT_REGIONS: dict[str, list[int]] = {
    "fishing_start_prompt": [0, 0, 400, 90],
    "fishing_equipment_prompt": [1500, 990, 420, 90],
    "timing_bar": [660, 860, 600, 75],
    "fight_bar": [660, 860, 600, 75],
    "bite_text": [645, 980, 650, 95],
    "bobber_widget": [1280, 745, 380, 300],
    "fishing_prompt": [306, 871, 1308, 52],
    "hook_prompt": [1730, 1025, 190, 55],
    "fish_icon": [860, 360, 200, 180],
    "catch_modal": [700, 290, 520, 420],
    "keep_button": [776, 703, 178, 45],
}


def fishing_region(regions: dict[str, list[int]] | None, name: str) -> list[int]:
    return (regions or {}).get(name, FISHING_DEFAULT_REGIONS[name])


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge user-facing overrides without rewriting documented base YAML."""
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def user_config_path(path: Path) -> Path | None:
    """Return the local override path for one of the built-in bot configs."""
    resolved = path.resolve()
    if resolved.parent != CONFIG_PATH.parent.resolve():
        return None
    return USER_CONFIG_DIR / resolved.name


def load_config(
    path: Path | None = None,
    *,
    include_user_overrides: bool = True,
) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    config = read_config_mapping(config_path)

    override_path = user_config_path(config_path)
    if include_user_overrides and override_path and override_path.exists():
        overrides = read_config_mapping(override_path)
        config = _deep_merge(config, overrides)
    return config


def read_config_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, rejecting invalid roots before callers modify it."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def save_config(config: dict[str, Any], path: Path | None = None) -> None:
    """Replace a config atomically so failed writes preserve the previous file."""
    config_path = path or CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(
                config, stream, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def scale_region(
    region: list[int],
    config_resolution: list[int],
    actual_width: int,
    actual_height: int,
) -> list[int]:
    """Scale a [x, y, w, h] region from config resolution to actual screen size."""
    cfg_w, cfg_h = config_resolution[0], config_resolution[1]
    if cfg_w == actual_width and cfg_h == actual_height:
        return list(region)
    sx = actual_width / cfg_w
    sy = actual_height / cfg_h
    x, y, w, h = region
    return [int(x * sx), int(y * sy), int(w * sx), int(h * sy)]


def scale_regions(
    regions: dict[str, list[int]],
    config_resolution: list[int],
    actual_width: int,
    actual_height: int,
) -> dict[str, list[int]]:
    return {
        name: scale_region(rect, config_resolution, actual_width, actual_height)
        for name, rect in regions.items()
    }


def scaled_rocks_relative(
    regions: dict[str, list[int]],
    config_resolution: list[int],
    actual_width: int,
    actual_height: int,
) -> list[int] | None:
    """Rocks mask offset inside a scaled minigame crop."""
    rocks = regions.get("rocks_mask")
    minigame = regions.get("minigame")
    if not rocks or not minigame:
        return None
    scaled_rocks = scale_region(rocks, config_resolution, actual_width, actual_height)
    scaled_minigame = scale_region(minigame, config_resolution, actual_width, actual_height)
    return [
        scaled_rocks[0] - scaled_minigame[0],
        scaled_rocks[1] - scaled_minigame[1],
        scaled_rocks[2],
        scaled_rocks[3],
    ]
