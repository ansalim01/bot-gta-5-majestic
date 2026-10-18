"""Invalid configuration must be rejected before editors can silently coerce it."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from gamebot_ui.config_store import ConfigStore
from gamebot_ui.models import BOT_BY_ID
from gamebot_ui.pages import BotPage


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_invalid_boolean_config_is_not_converted_to_an_enabled_switch(value):
    config = ConfigStore().load(BOT_BY_ID["blackjack"])
    config["blackjack"]["auto_action"] = value
    with patch("gamebot_ui.config_store.load_config", return_value=config):
        with pytest.raises(ValueError):
            ConfigStore().load(BOT_BY_ID["blackjack"])


def test_configuration_failure_blocks_saving_and_starting(qt_app):
    store = MagicMock()
    store.load.side_effect = ValueError("invalid settings")
    page = BotPage(BOT_BY_ID["blackjack"], store)
    try:
        with patch("gamebot_ui.pages.QMessageBox.critical"):
            assert page.save_settings(notify=False) is False
        store.save_values.assert_not_called()
        assert page.start_button.isEnabled() is False
    finally:
        page.deleteLater()


@pytest.mark.parametrize(
    "values",
    [
        {"hotkeys.stop": "wrong-key"},
        {"hotkeys.stop": "f9"},
        {"hotkeys.toggle": ""},
        {"blackjack.bet_amount": True},
        {"blackjack.bet_amount": -1},
        {"blackjack.double_policy": "unknown"},
        {"blackjack.min_ocr_confidence": float("nan")},
    ],
)
def test_invalid_editor_values_never_overwrite_settings(tmp_path, values):
    path = tmp_path / "blackjack.yaml"
    original = "# preserve this file\nblackjack:\n  bet_amount: 10\n"
    path.write_text(original, encoding="utf-8")
    with patch("gamebot_ui.config_store.user_config_path", return_value=path):
        with pytest.raises(ValueError):
            ConfigStore().save_values(BOT_BY_ID["blackjack"], values)
    assert path.read_text(encoding="utf-8") == original


def test_partial_editor_failure_cannot_save_the_fields_created_before_it(qt_app):
    config = deepcopy(ConfigStore().load(BOT_BY_ID["blackjack"]))
    config["blackjack"]["bet_amount"] = "invalid integer"
    store = MagicMock()
    store.load.return_value = config
    page = BotPage(BOT_BY_ID["blackjack"], store)
    try:
        with patch("gamebot_ui.pages.QMessageBox.critical"):
            assert page.save_settings(notify=False) is False
        store.save_values.assert_not_called()
    finally:
        page.deleteLater()


@pytest.mark.parametrize("bot_id", ["blackjack", "fishing", "gym", "sustenance"])
def test_all_documented_settings_can_still_be_saved(tmp_path, bot_id):
    from bot.config_values import nested_get

    spec = BOT_BY_ID[bot_id]
    config = ConfigStore().load(spec)
    values = {field.path: nested_get(config, field.path) for field in spec.settings}
    path = tmp_path / f"{bot_id}.yaml"
    with patch("gamebot_ui.config_store.user_config_path", return_value=path):
        assert ConfigStore().save_values(spec, values) == path
    assert path.exists()


def test_aliases_cannot_assign_the_same_key_to_two_controls(tmp_path):
    path = tmp_path / "fishing.yaml"
    with patch("gamebot_ui.config_store.user_config_path", return_value=path):
        with pytest.raises(ValueError, match="одну клавишу"):
            ConfigStore().save_values(
                BOT_BY_ID["fishing"], {"hotkeys.toggle": "insert", "hotkeys.stop": "ins"}
            )
    assert not path.exists()
