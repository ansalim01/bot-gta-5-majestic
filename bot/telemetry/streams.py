"""Best-effort closing for diagnostic files after a failed write or flush."""

from __future__ import annotations

from typing import TextIO


def close_event_stream(stream: TextIO | None) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except OSError as exc:
        print(f"[BOT] WARNING: telemetry close failed: {exc}", flush=True)
