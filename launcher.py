"""Launch the visual GameBot control panel."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

from bot.windows import is_admin as _is_admin
from bot.windows import relaunch_as_admin

ROOT = Path(__file__).resolve().parent


def _relaunch_as_admin() -> bool:
    return relaunch_as_admin(Path(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GameBot visual control panel")
    parser.add_argument(
        "--no-elevate",
        action="store_true",
        help="Open the panel without a Windows UAC request (UI development only).",
    )
    return parser.parse_args()


def _show_dependency_error(message: str) -> None:
    text = (
        "Не удалось открыть GameBot Control.\n\n"
        f"{message}\n\n"
        "Установите зависимости командой:\n"
        "python -m pip install -r requirements.txt"
    )
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(None, text, "GameBot Control", 0x10)
            return
        except (AttributeError, OSError):
            pass
    print(text, file=sys.stderr)


def main() -> int:
    args = parse_args()
    if not args.no_elevate and not _is_admin():
        try:
            if _relaunch_as_admin():
                return 0
        except (AttributeError, OSError):
            pass
        message = (
            "GameBot не запущен. Разрешите запрос контроля учётных записей (UAC): "
            "без одинаковых прав с GTA захват экрана и нажатия блокируются."
        )
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "GameBot Control", 0x10)
        except (AttributeError, OSError):
            print(message, file=sys.stderr)
        return 1

    try:
        from gamebot_ui import run
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            _show_dependency_error("Не найден графический модуль PySide6.")
            return 1
        raise
    return run([sys.argv[0]])


if __name__ == "__main__":
    raise SystemExit(main())
