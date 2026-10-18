"""Colors and Qt stylesheet for the desktop panel."""

from __future__ import annotations

COLORS = {
    "window": "#0d0f12",
    "sidebar": "#111318",
    "surface": "#171a20",
    "surface_hover": "#1d2128",
    "surface_alt": "#13161b",
    "border": "#252a33",
    "text": "#f4f5f7",
    "muted": "#8d939f",
    "accent": "#ff2f78",
    "accent_hover": "#ff4a8b",
    "success": "#35d79a",
    "warning": "#f0b45a",
    "danger": "#ff5d6c",
}


STYLESHEET = f"""
* {{
    font-family: "Segoe UI";
    font-size: 13px;
    color: {COLORS["text"]};
}}
QMainWindow, QWidget#appRoot {{
    background: {COLORS["window"]};
}}
QWidget#titleBar {{
    background: {COLORS["window"]};
    border-bottom: 1px solid {COLORS["border"]};
}}
QWidget#sidebar {{
    background: {COLORS["sidebar"]};
    border-right: 1px solid {COLORS["border"]};
}}
QLabel#brandName {{
    font-size: 19px;
    font-weight: 700;
}}
QLabel#brandAccent {{
    color: {COLORS["accent"]};
    font-size: 19px;
    font-weight: 700;
}}
QLabel#pageTitle {{
    font-size: 25px;
    font-weight: 700;
}}
QLabel#pageSubtitle, QLabel[muted="true"] {{
    color: {COLORS["muted"]};
}}
QLabel#sectionTitle {{
    font-size: 16px;
    font-weight: 650;
}}
QLabel#cardTitle {{
    font-size: 16px;
    font-weight: 650;
}}
QLabel#botHeroIcon {{
    background: rgba(255, 47, 120, 0.12);
    color: {COLORS["accent"]};
    border: 1px solid rgba(255, 47, 120, 0.30);
    border-radius: 18px;
    font-size: 24px;
    font-weight: 700;
}}
QFrame#card, QFrame#settingsPanel, QFrame#heroPanel, QFrame#statCard {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
}}
QFrame#card:hover {{
    background: {COLORS["surface_hover"]};
    border-color: #343a46;
}}
QFrame#placeholderCard {{
    background: {COLORS["surface_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
}}
QPushButton {{
    min-height: 34px;
    padding: 0 15px;
    background: {COLORS["surface_hover"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: #252a33;
    border-color: #363d49;
}}
QPushButton:pressed {{
    background: #13161b;
}}
QPushButton:disabled {{
    color: #5c626d;
    background: #15171b;
    border-color: #20242b;
}}
QPushButton[role="primary"] {{
    color: white;
    background: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
}}
QPushButton[role="primary"]:hover {{
    background: {COLORS["accent_hover"]};
    border-color: {COLORS["accent_hover"]};
}}
QPushButton[role="danger"] {{
    color: {COLORS["danger"]};
    background: rgba(255, 93, 108, 0.08);
    border-color: rgba(255, 93, 108, 0.35);
}}
QPushButton[role="ghost"] {{
    background: transparent;
    border-color: transparent;
    color: {COLORS["muted"]};
}}
QPushButton[role="ghost"]:hover {{
    color: {COLORS["text"]};
    background: {COLORS["surface_hover"]};
}}
QPushButton#sidebarButton {{
    min-height: 40px;
    padding: 0 12px;
    background: transparent;
    border: 0;
    border-radius: 8px;
    color: #aeb3bd;
    text-align: left;
    font-weight: 500;
}}
QPushButton#sidebarButton:hover {{
    color: {COLORS["text"]};
    background: #191c22;
}}
QPushButton#sidebarButton:checked {{
    color: white;
    background: rgba(255, 47, 120, 0.13);
    border-left: 3px solid {COLORS["accent"]};
    font-weight: 650;
}}
QPushButton#windowButton {{
    min-width: 34px;
    min-height: 30px;
    padding: 0;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: {COLORS["muted"]};
}}
QPushButton#windowButton:hover {{
    background: {COLORS["surface_hover"]};
    color: white;
}}
QPushButton#closeButton {{
    min-width: 34px;
    min-height: 30px;
    padding: 0;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: {COLORS["muted"]};
}}
QPushButton#closeButton:hover {{
    background: {COLORS["danger"]};
    color: white;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    min-height: 34px;
    padding: 0 10px;
    background: {COLORS["surface_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 7px;
    selection-background-color: {COLORS["accent"]};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS["accent"]};
}}
QFrame[compact="true"] QLineEdit,
QFrame[compact="true"] QSpinBox,
QFrame[compact="true"] QDoubleSpinBox,
QFrame[compact="true"] QComboBox {{
    min-height: 28px;
}}
QComboBox::drop-down {{
    width: 28px;
    border: 0;
}}
QComboBox QAbstractItemView {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    selection-background-color: rgba(255, 47, 120, 0.20);
    outline: 0;
}}
QToolTip {{
    color: white;
    background: #252a33;
    border: 1px solid #3a414d;
    padding: 6px;
}}
QScrollArea {{
    border: 0;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 3px;
}}
QScrollBar::handle:vertical {{
    background: #343944;
    min-height: 30px;
    border-radius: 3px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QPlainTextEdit {{
    background: #0b0d10;
    border: 1px solid {COLORS["border"]};
    border-radius: 9px;
    padding: 10px;
    color: #c8ccd4;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 12px;
    selection-background-color: {COLORS["accent"]};
}}
QMessageBox {{
    background: {COLORS["surface"]};
}}
QMenu {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    padding: 5px;
}}
QMenu::item {{
    min-width: 170px;
    padding: 8px 18px;
    border-radius: 5px;
}}
QMenu::item:selected {{
    background: rgba(255, 47, 120, 0.18);
}}
"""
