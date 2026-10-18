"""Windows privilege helpers shared by CLI entry points and the desktop app."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def is_admin() -> bool:
    """Return whether the current process already has elevated privileges."""
    if sys.platform != "win32":
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def relaunch_as_admin(script: Path, arguments: Sequence[str] | None = None) -> bool:
    """Request UAC for a script; a successful caller must exit the parent."""
    script = script.resolve()
    parameters = subprocess.list2cmdline(
        [str(script), *(sys.argv[1:] if arguments is None else arguments)]
    )
    shell_execute = ctypes.windll.shell32.ShellExecuteW
    shell_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_int,
    ]
    shell_execute.restype = ctypes.c_void_p
    result = shell_execute(None, "runas", sys.executable, parameters, str(script.parent), 1)
    return bool(result and result > 32)
