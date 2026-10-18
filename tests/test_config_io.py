"""Regression coverage for preserving users' configuration on failed writes."""

from unittest.mock import patch

import pytest
import yaml

from bot.config_loader import load_config, read_config_mapping, save_config
from gamebot_ui.config_store import ConfigStore
from gamebot_ui.models import BOT_BY_ID


@pytest.mark.parametrize("contents", ["[]", "false", "0", "hello", "- item"])
def test_invalid_config_root_is_rejected(tmp_path, contents):
    path = tmp_path / "settings.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config(path)


def test_empty_config_is_a_mapping(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("# no settings yet\n", encoding="utf-8")
    assert read_config_mapping(path) == {}


def test_atomic_save_round_trips_unicode_and_removes_temporary_file(tmp_path):
    path = tmp_path / "nested" / "settings.yaml"
    config = {"title": "Рыбалка", "timing": {"delay": 0}, "enabled": False}
    save_config(config, path)
    assert read_config_mapping(path) == config
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_failed_save_preserves_previous_contents_and_cleans_up(tmp_path, failure):
    path = tmp_path / "settings.yaml"
    original = "# keep my comments\ntiming: 25\n"
    path.write_text(original, encoding="utf-8")
    target = (
        "bot.config_loader.yaml.safe_dump" if failure == "serialize" else "pathlib.Path.replace"
    )
    with patch(target, side_effect=OSError("write failed")):
        with pytest.raises(OSError, match="write failed"):
            save_config({"timing": 50}, path)
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("contents", ["[]", "false", "hotkeys: ["])
def test_ui_save_never_replaces_a_malformed_override(tmp_path, contents):
    destination = tmp_path / "blackjack.yaml"
    destination.write_text(contents, encoding="utf-8")
    with patch("gamebot_ui.config_store.user_config_path", return_value=destination):
        with pytest.raises((ValueError, yaml.YAMLError)):
            ConfigStore().save_values(BOT_BY_ID["blackjack"], {"blackjack.bet_amount": 25})
    assert destination.read_text(encoding="utf-8") == contents


def test_ui_save_preserves_settings_outside_visible_fields(tmp_path):
    destination = tmp_path / "blackjack.yaml"
    destination.write_text("blackjack:\n  bet_amount: 10\n  confirm_frames: 5\n", encoding="utf-8")
    with patch("gamebot_ui.config_store.user_config_path", return_value=destination):
        saved = ConfigStore().save_values(BOT_BY_ID["blackjack"], {"blackjack.bet_amount": 25})
    assert saved == destination
    assert read_config_mapping(saved) == {"blackjack": {"bet_amount": 25, "confirm_frames": 5}}
