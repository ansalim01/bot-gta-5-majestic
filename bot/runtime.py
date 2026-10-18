"""Release a bot's resources even when an earlier cleanup operation fails."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any


def close_bot_resources(cleanup_job: Callable[[], None], capture: Any, listener: Any) -> None:
    with ExitStack() as stack:
        if listener is not None:
            stack.callback(listener.stop)
        stack.callback(capture.close)
        stack.callback(cleanup_job)
