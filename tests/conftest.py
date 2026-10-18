"""Keep automated UI checks off the real desktop."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def qt_app():
    """Exercise layout with the production theme and usable fonts."""
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication

    from gamebot_ui.theme import STYLESHEET

    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "segoeuib.ttf", "seguisym.ttf", "seguiemj.ttf"):
            if (font_dir / name).exists():
                QFontDatabase.addApplicationFont(str(font_dir / name))
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    yield app
    app.processEvents()
