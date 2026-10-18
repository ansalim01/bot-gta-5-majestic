"""Small reusable widgets used throughout the control panel."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import COLORS

STATUS_LABELS = {
    "idle": ("Остановлен", COLORS["muted"]),
    "starting": ("Запускается", COLORS["warning"]),
    "running": ("Работает", COLORS["success"]),
    "paused": ("Пауза", COLORS["warning"]),
    "stopping": ("Останавливается", COLORS["warning"]),
    "error": ("Ошибка", COLORS["danger"]),
    "planned": ("Скоро", COLORS["muted"]),
}


class Switch(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checked = checked
        self.setFixedSize(42, 23)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self._checked = not self._checked
            self.update()
            self.toggled.emit(self._checked)
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        track = QColor(COLORS["accent"] if self._checked else "#343944")
        painter.setPen(Qt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0, 0, 42, 23), 11.5, 11.5)
        painter.setBrush(QColor("#ffffff"))
        x = 21 if self._checked else 2
        painter.drawEllipse(QRectF(x, 2, 19, 19))
        if self.hasFocus():
            painter.setPen(QPen(QColor(COLORS["text"]), 1, Qt.DotLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(0.5, 0.5, 41, 22), 11, 11)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Space, Qt.Key_Return):
            self._checked = not self._checked
            self.update()
            self.toggled.emit(self._checked)
            event.accept()
            return
        super().keyPressEvent(event)


class StatusPill(QFrame):
    def __init__(self, status: str = "idle", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 4, 10, 4)
        layout.setSpacing(6)
        self.dot = QLabel("●")
        self.dot.setFixedWidth(10)
        self.text = QLabel()
        layout.addWidget(self.dot)
        layout.addWidget(self.text)
        self.set_status(status)

    def set_status(self, status: str) -> None:
        label, color = STATUS_LABELS.get(status, STATUS_LABELS["idle"])
        self.set_label(label, color)

    def set_label(self, label: str, color: str) -> None:
        """Apply the same badge appearance to bot and game-window statuses."""
        red, green, blue, _ = QColor(color).getRgb()
        self.dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.text.setText(label)
        self.text.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 650;")
        self.setStyleSheet(
            f"QFrame {{ background: rgba({red}, {green}, {blue}, 0.08);"
            f" border: 1px solid rgba({red}, {green}, {blue}, 0.27);"
            " border-radius: 10px; }}"
        )
        self.adjustSize()


class StatCard(QFrame):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statCard")
        self.setMinimumWidth(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 8, 11, 8)
        layout.setSpacing(2)
        self.value = QLabel("0")
        self.value.setStyleSheet("font-size: 22px; font-weight: 700;")
        title = QLabel(label)
        title.setProperty("muted", True)
        title.setWordWrap(True)
        title.setToolTip(label)
        layout.addWidget(self.value)
        layout.addWidget(title)

    def set_value(self, value: object) -> None:
        self.value.setText(str(value))


class TitleBar(QWidget):
    minimize_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, is_admin: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setFixedHeight(54)
        self._drag_position: QPoint | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 10, 0)
        layout.setSpacing(10)

        brand = QLabel("GAMEBOT")
        brand.setStyleSheet(
            f"font-size: 12px; font-weight: 800; letter-spacing: 1px; color: {COLORS['accent']};"
        )
        layout.addWidget(brand)
        layout.addStretch()

        self.game_badge = StatusPill("idle")
        self.game_badge.text.setText("GTA не найдена")
        layout.addWidget(self.game_badge)

        admin = QLabel("  Администратор  " if is_admin else "  Без прав администратора  ")
        admin_color = COLORS["success"] if is_admin else COLORS["warning"]
        admin_rgb = QColor(admin_color).getRgb()
        admin.setFixedHeight(24)
        admin.setStyleSheet(
            f"color: {admin_color};"
            f" background: rgba({admin_rgb[0]}, {admin_rgb[1]}, {admin_rgb[2]}, 0.08);"
            f" border: 1px solid rgba({admin_rgb[0]}, {admin_rgb[1]}, {admin_rgb[2]}, 0.25);"
            " border-radius: 10px;"
            " font-size: 11px; font-weight: 600; padding: 3px 5px;"
        )
        layout.addWidget(admin)

        minimize = QPushButton("—")
        minimize.setObjectName("windowButton")
        minimize.clicked.connect(self.minimize_clicked)
        close = QPushButton("×")
        close.setObjectName("closeButton")
        close.clicked.connect(self.close_clicked)
        layout.addWidget(minimize)
        layout.addWidget(close)

    def set_game_found(self, found: bool) -> None:
        color = COLORS["success"] if found else COLORS["muted"]
        self.game_badge.set_label("GTA подключена" if found else "GTA не найдена", color)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_position = (
                event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_position is not None and event.buttons() & Qt.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_position)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_position = None
        super().mouseReleaseEvent(event)


class SidebarButton(QPushButton):
    def __init__(
        self,
        text: str,
        icon_text: str = "",
        badge: str = "",
        parent: QWidget | None = None,
    ) -> None:
        shown = f"  {icon_text}    {text}" if icon_text else f"  {text}"
        if badge:
            shown += f"    · {badge}"
        super().__init__(shown, parent)
        self.setObjectName("sidebarButton")
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class EmptyState(QFrame):
    def __init__(
        self,
        icon: str,
        title: str,
        text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("heroPanel")
        self.setMinimumHeight(250)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(35, 42, 35, 42)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 42px; font-weight: 700;")
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        description = QLabel(text)
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        description.setMaximumWidth(460)
        description.setProperty("muted", True)
        layout.addWidget(icon_label)
        layout.addWidget(heading)
        layout.addWidget(description)
