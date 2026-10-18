"""A broken local session summary must not prevent the panel from opening."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gamebot_ui.diagnostics import format_fishing_diagnostics
from gamebot_ui.pages import BotPage


@pytest.mark.parametrize(
    "summary",
    [
        None,
        [],
        {"stats": None},
        {"stats": {"casts": "invalid"}},
        {"conclusions": [None]},
        {"conclusions": None},
    ],
)
def test_invalid_summary_is_reported_without_crashing_page(tmp_path, summary):
    path = tmp_path / "logs" / "fishing_sessions" / "latest_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary), encoding="utf-8")
    page = SimpleNamespace(diagnostics_label=MagicMock())
    with patch("gamebot_ui.pages.ROOT", tmp_path):
        BotPage.refresh_diagnostics(page)
    assert "Не удалось прочитать" in page.diagnostics_label.setText.call_args.args[0]


def test_valid_summary_escapes_text_and_keeps_counts():
    rendered = format_fishing_diagnostics(
        {
            "session_id": "<session>",
            "stats": {"casts": 3, "fish_caught": 2},
            "conclusions": [{"level": "warning", "title": "<title>", "recommendation": "<text>"}],
        }
    )
    assert "забросов 3" in rendered
    assert "поймано 2" in rendered
    assert "&lt;session&gt;" in rendered
    assert "<title>" not in rendered
