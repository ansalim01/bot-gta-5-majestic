"""Shared wire format for worker events mixed with human-readable stdout."""

from __future__ import annotations

import json
from typing import Any

EVENT_PREFIX = "@@GAMEBOT_EVENT@@"


def encode_event(event_type: str, **payload: Any) -> str:
    """Encode one event; the caller appends a newline and flushes stdout."""
    return EVENT_PREFIX + json.dumps({"type": event_type, **payload}, ensure_ascii=False)
