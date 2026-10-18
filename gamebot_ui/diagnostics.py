"""Validate and render a fishing summary without depending on file or widget state."""

from __future__ import annotations

import html
from typing import Any

from .theme import COLORS


def format_fishing_diagnostics(summary: Any) -> str:
    if not isinstance(summary, dict):
        raise ValueError("Summary must be an object")
    stats = summary.get("stats", {})
    if not isinstance(stats, dict):
        raise ValueError("Summary stats must be an object")
    for key in ("casts", "fish_caught"):
        if type(stats.get(key, 0)) is not int or stats.get(key, 0) < 0:
            raise ValueError(f"Summary stats.{key} must be a nonnegative integer")
    conclusions = summary.get("conclusions", [])
    if not isinstance(conclusions, list) or any(not isinstance(item, dict) for item in conclusions):
        raise ValueError("Summary conclusions must be a list of objects")
    stats = summary.get("stats", {})
    parts = [
        f"<b>Сессия {html.escape(str(summary.get('session_id', '—')))}</b> · "
        f"{html.escape(str(summary.get('duration_sec', 0)))} сек · "
        f"забросов {int(stats.get('casts', 0) or 0)} · "
        f"поймано {int(stats.get('fish_caught', 0) or 0)}"
    ]
    level_colors = {
        "ok": COLORS["success"],
        "warning": COLORS["warning"],
        "error": COLORS["danger"],
    }
    for conclusion in summary.get("conclusions", []):
        color = level_colors.get(
            str(conclusion.get("level")),
            COLORS["muted"],
        )
        title = html.escape(str(conclusion.get("title", "")))
        recommendation = html.escape(str(conclusion.get("recommendation", "")))
        parts.append(
            f"<span style='color:{color}'><b>● {title}</b></span><br>"
            f"<span style='color:{COLORS['muted']}'>{recommendation}</span>"
        )
    return "<br><br>".join(parts)
