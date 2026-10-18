"""Read base bot configs and atomically save small UI override files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bot.config_loader import (
    _deep_merge,
    load_config,
    read_config_mapping,
    save_config,
    user_config_path,
)
from bot.config_values import nested_set

from .models import BotSpec
from .validation import validate_settings


class ConfigStore:
    def load(self, spec: BotSpec) -> dict[str, Any]:
        if spec.config_path is None:
            return {}
        config = load_config(spec.config_path)
        validate_settings(spec, config)
        return config

    def save_values(self, spec: BotSpec, values: dict[str, Any]) -> Path:
        if spec.config_path is None:
            raise ValueError(f"{spec.title} does not have a configuration")

        destination = user_config_path(spec.config_path)
        if destination is None:
            raise ValueError(f"Unsupported config path: {spec.config_path}")
        current = read_config_mapping(destination) if destination.exists() else {}

        for path, value in values.items():
            nested_set(current, path, value)

        base = load_config(spec.config_path, include_user_overrides=False)
        validate_settings(spec, _deep_merge(base, current))
        save_config(current, destination)
        return destination
