import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from bot.windows import is_admin, relaunch_as_admin


def test_privileges_are_not_requested_outside_windows():
    with patch("bot.windows.sys.platform", "linux"):
        assert is_admin() is True


@pytest.mark.parametrize(
    "result, expected", [(0, False), (5, False), (32, False), (33, True), (64, True)]
)
def test_uac_return_code_and_arguments(result, expected, tmp_path):
    script = tmp_path / "folder with spaces" / "launcher.py"
    arguments = ["--config", "path with spaces.yaml"]
    api = MagicMock()
    api.shell32.ShellExecuteW.return_value = result
    with patch("bot.windows.ctypes.windll", api, create=True):
        assert relaunch_as_admin(script, arguments) is expected
    api.shell32.ShellExecuteW.assert_called_once_with(
        None,
        "runas",
        sys.executable,
        subprocess.list2cmdline([str(script.resolve()), *arguments]),
        str(script.resolve().parent),
        1,
    )


def test_admin_probe_handles_unavailable_windows_api():
    api = MagicMock()
    api.shell32.IsUserAnAdmin.side_effect = OSError("unavailable")
    with (
        patch("bot.windows.sys.platform", "win32"),
        patch("bot.windows.ctypes.windll", api, create=True),
    ):
        assert is_admin() is False
