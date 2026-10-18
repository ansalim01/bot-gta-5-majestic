"""Dashboard, bot settings, logs, and application information pages."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .config_store import ConfigStore
from .diagnostics import format_fishing_diagnostics
from .models import (
    AVAILABLE_BOTS,
    PLANNED_BOTS,
    ROOT,
    BotSpec,
    nested_get,
)
from .settings import SettingRow as SettingRow
from .theme import COLORS
from .widgets import EmptyState, StatCard, StatusPill, Switch


def page_heading(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    description = QLabel(subtitle)
    description.setObjectName("pageSubtitle")
    layout.addWidget(heading)
    layout.addWidget(description)
    return widget, layout


class BotCard(QFrame):
    open_requested = Signal(str)
    start_requested = Signal(str)

    def __init__(self, spec: BotSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("card")
        self.setMinimumHeight(190)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 17, 18, 17)
        layout.setSpacing(10)

        top = QHBoxLayout()
        icon = QLabel(spec.icon)
        icon.setObjectName("botHeroIcon")
        icon.setFixedSize(50, 50)
        icon.setAlignment(Qt.AlignCenter)
        names = QVBoxLayout()
        names.setSpacing(3)
        title = QLabel(spec.title)
        title.setObjectName("cardTitle")
        description = QLabel(spec.description)
        description.setProperty("muted", True)
        description.setWordWrap(True)
        names.addWidget(title)
        names.addWidget(description)
        self.status = StatusPill("idle")
        top.addWidget(icon)
        top.addLayout(names, 1)
        top.addWidget(self.status, 0, Qt.AlignTop)
        layout.addLayout(top)

        tags = QHBoxLayout()
        tags.setSpacing(6)
        for tag in spec.tags:
            label = QLabel(f" {tag} ")
            label.setStyleSheet(
                f"color: {COLORS['muted']}; background: {COLORS['surface_alt']};"
                f" border: 1px solid {COLORS['border']}; border-radius: 7px;"
                " font-size: 11px; padding: 2px 4px;"
            )
            tags.addWidget(label)
        tags.addStretch()
        layout.addLayout(tags)
        layout.addStretch()

        actions = QHBoxLayout()
        details = QPushButton("Настроить")
        details.clicked.connect(lambda: self.open_requested.emit(spec.bot_id))
        self.start_button = QPushButton("Запустить")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(lambda: self.start_requested.emit(spec.bot_id))
        actions.addWidget(details)
        actions.addStretch()
        actions.addWidget(self.start_button)
        layout.addLayout(actions)

    def set_status(self, status: str) -> None:
        self.status.set_status(status)
        if status == "running":
            self.start_button.setText("Открыть")
        elif status == "paused":
            self.start_button.setText("Продолжить")
        elif status in {"starting", "stopping"}:
            self.start_button.setText("Подождите…")
            self.start_button.setEnabled(False)
            return
        else:
            self.start_button.setText("Запустить")
        self.start_button.setEnabled(True)


class DashboardPage(QWidget):
    open_bot = Signal(str)
    start_bot = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 25, 30, 28)
        root.setSpacing(20)
        heading, _ = page_heading(
            "Центр управления",
            "Запускайте сценарии, следите за состоянием и меняйте настройки без консоли.",
        )
        root.addWidget(heading)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 10)
        layout.setSpacing(18)

        title = QLabel("Доступные боты")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(14)
        self.cards: dict[str, BotCard] = {}
        for index, spec in enumerate(AVAILABLE_BOTS):
            card = BotCard(spec)
            card.open_requested.connect(self.open_bot)
            card.start_requested.connect(self.start_bot)
            self.cards[spec.bot_id] = card
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)

        planned_title = QLabel("В разработке")
        planned_title.setObjectName("sectionTitle")
        layout.addWidget(planned_title)
        planned_description = QLabel(
            "Карточки уже добавлены в структуру — функциональность можно подключать по мере появления новых сценариев."
        )
        planned_description.setWordWrap(True)
        planned_description.setProperty("muted", True)
        layout.addWidget(planned_description)

        planned_grid = QGridLayout()
        planned_grid.setSpacing(9)
        for index, spec in enumerate(PLANNED_BOTS):
            card = QFrame()
            card.setObjectName("placeholderCard")
            card.setCursor(Qt.PointingHandCursor)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(13, 11, 13, 11)
            icon = QLabel(spec.icon)
            icon.setStyleSheet(f"color: {COLORS['accent']}; font-size: 17px; font-weight: 700;")
            name = QLabel(spec.title)
            name.setStyleSheet("font-weight: 600;")
            soon = QLabel("СКОРО")
            soon.setStyleSheet(f"color: {COLORS['muted']}; font-size: 9px; font-weight: 700;")
            card_layout.addWidget(icon)
            card_layout.addWidget(name)
            card_layout.addStretch()
            card_layout.addWidget(soon)
            planned_grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(planned_grid)
        layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def set_bot_status(self, bot_id: str, status: str) -> None:
        card = self.cards.get(bot_id)
        if card:
            card.set_status(status)


class BotPage(QWidget):
    start_requested = Signal(str)
    pause_requested = Signal(str)
    resume_requested = Signal(str)
    stop_requested = Signal(str)
    check_requested = Signal(str)
    saved = Signal(str, str)

    def __init__(
        self,
        spec: BotSpec,
        config_store: ConfigStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.config_store = config_store
        self.status_name = "idle" if spec.available else "planned"
        self.setting_rows: dict[str, SettingRow] = {}
        self.stat_cards: dict[str, StatCard] = {}
        self._settings_error: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 18)
        root.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(10)

        layout.addWidget(self._build_hero())

        if not spec.available:
            self.pause_button.hide()
            self.stop_button.hide()
            self.start_button.setText("В разработке")
            self.start_button.setEnabled(False)
            layout.addWidget(
                EmptyState(
                    spec.icon,
                    "Этот бот пока не подключён",
                    "Каркас страницы уже готов. Когда появится рабочий сценарий, сюда добавятся настройки, управление и статистика.",
                )
            )
            layout.addStretch()
            scroll.setWidget(content)
            root.addWidget(scroll)
            return

        layout.addWidget(self._build_stats())

        if spec.bot_id == "fishing":
            diagnostics = QFrame()
            diagnostics.setObjectName("settingsPanel")
            diagnostics_layout = QVBoxLayout(diagnostics)
            diagnostics_layout.setContentsMargins(18, 15, 18, 15)
            diagnostics_layout.setSpacing(8)
            diagnostics_header = QHBoxLayout()
            diagnostics_title = QLabel("Последний анализ рыбалки")
            diagnostics_title.setObjectName("sectionTitle")
            self.diagnostics_toggle = QPushButton("Показать")
            self.diagnostics_toggle.setProperty("role", "ghost")
            self.diagnostics_toggle.clicked.connect(self._toggle_diagnostics)
            refresh = QPushButton("Обновить")
            refresh.clicked.connect(self.refresh_diagnostics)
            diagnostics_header.addWidget(diagnostics_title)
            diagnostics_header.addStretch()
            diagnostics_header.addWidget(self.diagnostics_toggle)
            diagnostics_header.addWidget(refresh)
            diagnostics_layout.addLayout(diagnostics_header)
            self.diagnostics_label = QLabel()
            self.diagnostics_label.setWordWrap(True)
            self.diagnostics_label.setTextFormat(Qt.RichText)
            self.diagnostics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            diagnostics_layout.addWidget(self.diagnostics_label)
            self.diagnostics_label.hide()
            layout.addWidget(diagnostics)
            self.refresh_diagnostics()

        layout.addWidget(self._build_settings_panel())
        layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)
        self.set_status("idle")

    def _build_hero(self) -> QFrame:
        spec = self.spec
        hero = QFrame()
        hero.setObjectName("heroPanel")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(15, 11, 15, 11)
        hero_layout.setSpacing(12)
        icon = QLabel(spec.icon)
        icon.setObjectName("botHeroIcon")
        icon.setFixedSize(50, 50)
        icon.setAlignment(Qt.AlignCenter)
        names = QVBoxLayout()
        title_row = QHBoxLayout()
        title = QLabel(spec.title)
        title.setStyleSheet("font-size: 21px; font-weight: 700;")
        self.status = StatusPill(self.status_name)
        title_row.addWidget(title)
        title_row.addWidget(self.status)
        title_row.addStretch()
        description = QLabel(spec.description)
        description.setProperty("muted", True)
        description.setWordWrap(True)
        description.setMinimumWidth(0)
        description.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        names.addLayout(title_row)
        names.addWidget(description)
        hero_layout.addWidget(icon)
        hero_layout.addLayout(names, 1)

        self.pause_button = QPushButton("Пауза")
        self.pause_button.clicked.connect(self._pause_or_resume)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit(spec.bot_id))
        self.start_button = QPushButton("Запустить")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(lambda: self.start_requested.emit(spec.bot_id))
        hero_layout.addWidget(self.pause_button)
        hero_layout.addWidget(self.stop_button)
        hero_layout.addWidget(self.start_button)

        return hero

    def _build_stats(self) -> QWidget:
        spec = self.spec
        stats_frame = QWidget()
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setHorizontalSpacing(10)
        stats_layout.setVerticalSpacing(8)
        stat_columns = min(5, max(1, len(spec.stats)))
        for index, (key, label) in enumerate(spec.stats):
            card = StatCard(label)
            self.stat_cards[key] = card
            stats_layout.addWidget(
                card,
                index // stat_columns,
                index % stat_columns,
            )
        for column in range(stat_columns):
            stats_layout.setColumnStretch(column, 1)

        return stats_frame

    def _build_settings_panel(self) -> QFrame:
        spec = self.spec
        config_store = self.config_store
        panel = QFrame()
        panel.setObjectName("settingsPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 11, 14, 11)
        panel_layout.setSpacing(0)
        panel_header = QHBoxLayout()
        panel_title = QLabel("Основные настройки")
        panel_title.setObjectName("sectionTitle")
        hint = QLabel("Применяются при следующем запуске")
        hint.setProperty("muted", True)
        panel_header.addWidget(panel_title)
        panel_header.addStretch()
        panel_header.addWidget(hint)
        panel_layout.addLayout(panel_header)
        panel_layout.addSpacing(5)

        try:
            config = config_store.load(spec)
            settings_grid = QGridLayout()
            settings_grid.setContentsMargins(0, 0, 0, 0)
            settings_grid.setHorizontalSpacing(18)
            settings_grid.setVerticalSpacing(0)
            rows_per_column = (len(spec.settings) + 1) // 2
            for index, field in enumerate(spec.settings):
                row = SettingRow(
                    field,
                    nested_get(config, field.path),
                    compact=True,
                )
                self.setting_rows[field.path] = row
                grid_row = index % rows_per_column
                grid_column = index // rows_per_column
                settings_grid.addWidget(row, grid_row, grid_column)
            settings_grid.setColumnStretch(0, 1)
            settings_grid.setColumnStretch(1, 1)
            panel_layout.addLayout(settings_grid)
        except Exception as exc:
            self._settings_error = str(exc)
            error = QLabel(f"Не удалось прочитать конфигурацию: {exc}")
            error.setWordWrap(True)
            error.setStyleSheet(f"color: {COLORS['danger']}; padding: 15px;")
            panel_layout.addWidget(error)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 9, 0, 0)
        actions.addStretch()
        if spec.bot_id == "sustenance":
            check = QPushButton("Проверить сейчас")
            check.clicked.connect(lambda: self.check_requested.emit(spec.bot_id))
            actions.addWidget(check)
        save = QPushButton("Сохранить настройки")
        save.setEnabled(self._settings_error is None)
        save.setProperty("role", "primary")
        save.clicked.connect(self._save)
        actions.addWidget(save)
        panel_layout.addLayout(actions)
        return panel

    def _save(self) -> None:
        self.save_settings()

    def save_settings(self, *, notify: bool = True) -> bool:
        """Persist the values currently visible in this page."""
        if self._settings_error is not None:
            QMessageBox.critical(self, "Не удалось загрузить настройки", self._settings_error)
            return False
        values = {path: row.value() for path, row in self.setting_rows.items()}
        for path, value in values.items():
            if isinstance(value, list) and len(value) == 2 and value[0] > value[1]:
                QMessageBox.warning(
                    self,
                    "Проверьте диапазон",
                    f"В параметре «{self.setting_rows[path].field.label}» "
                    "минимум не может быть больше максимума.",
                )
                return False
        try:
            destination = self.config_store.save_values(self.spec, values)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Не удалось сохранить",
                str(exc),
            )
            return False
        if notify:
            self.saved.emit(self.spec.bot_id, str(destination))
        return True

    def _pause_or_resume(self) -> None:
        if self.status_name == "paused":
            self.resume_requested.emit(self.spec.bot_id)
        else:
            self.pause_requested.emit(self.spec.bot_id)

    def set_status(self, status: str) -> None:
        self.status_name = status
        self.status.set_status(status)
        running = status in {"running", "paused", "starting", "stopping"}
        self.pause_button.setVisible(running)
        self.stop_button.setVisible(running)
        self.stop_button.setEnabled(running and status != "stopping")
        self.pause_button.setEnabled(status in {"running", "paused"})
        self.pause_button.setText("Продолжить" if status == "paused" else "Пауза")
        self.start_button.setVisible(not running)
        self.start_button.setEnabled(self.spec.available and self._settings_error is None)
        if status == "error":
            self.start_button.setText("Запустить снова")
        else:
            self.start_button.setText("Запустить")

    def set_stats(self, values: dict[str, Any]) -> None:
        for key, card in self.stat_cards.items():
            card.set_value(values.get(key, 0))

    def _toggle_diagnostics(self) -> None:
        label = getattr(self, "diagnostics_label", None)
        toggle = getattr(self, "diagnostics_toggle", None)
        if label is None or toggle is None:
            return
        expanded = label.isHidden()
        label.setVisible(expanded)
        toggle.setText("Скрыть" if expanded else "Показать")
        if expanded:
            self.refresh_diagnostics()

    def refresh_diagnostics(self) -> None:
        label = getattr(self, "diagnostics_label", None)
        if label is None:
            return
        path = ROOT / "logs" / "fishing_sessions" / "latest_summary.json"
        if not path.exists():
            label.setText(
                f"<span style='color:{COLORS['muted']}'>Отчёт появится после "
                "остановки первой сессии. Подробные события записываются во "
                "время работы.</span>"
            )
            return
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            label.setText(format_fishing_diagnostics(summary))
        except (OSError, ValueError) as exc:
            label.setText(
                f"<span style='color:{COLORS['danger']}'>Не удалось прочитать "
                f"отчёт: {html.escape(str(exc))}</span>"
            )
            return


class PlannedBotPage(BotPage):
    pass


class LogsPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 28)
        layout.setSpacing(16)
        heading, _ = page_heading(
            "Журнал",
            "События запуска, состояния и сообщения рабочих процессов.",
        )
        layout.addWidget(heading)
        actions = QHBoxLayout()
        actions.addStretch()
        clear = QPushButton("Очистить")
        clear.clicked.connect(lambda: self.console.clear())
        actions.addWidget(clear)
        layout.addLayout(actions)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Здесь появятся сообщения ботов…")
        layout.addWidget(self.console, 1)

    def append(self, bot_title: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{timestamp}] [{bot_title}] {message}")
        bar = self.console.verticalScrollBar()
        bar.setValue(bar.maximum())


class AppSettingsPage(QWidget):
    def __init__(
        self,
        root_path: Path,
        is_admin: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 28)
        layout.setSpacing(16)
        heading, _ = page_heading(
            "Настройки приложения",
            "Системная информация и поведение панели.",
        )
        layout.addWidget(heading)

        panel = QFrame()
        panel.setObjectName("settingsPanel")
        form = QVBoxLayout(panel)
        form.setContentsMargins(19, 17, 19, 17)
        title = QLabel("Среда запуска")
        title.setObjectName("sectionTitle")
        form.addWidget(title)
        for label, value, color in (
            ("Папка проекта", str(root_path), COLORS["text"]),
            (
                "Права администратора",
                "Получены" if is_admin else "Не получены",
                COLORS["success"] if is_admin else COLORS["warning"],
            ),
            (
                "Пользовательские настройки",
                str(root_path / "config" / "user"),
                COLORS["text"],
            ),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 11, 0, 11)
            name = QLabel(label)
            name.setProperty("muted", True)
            current = QLabel(value)
            current.setTextInteractionFlags(Qt.TextSelectableByMouse)
            current.setStyleSheet(f"color: {color}; font-weight: 600;")
            row.addWidget(name)
            row.addStretch()
            row.addWidget(current)
            form.addLayout(row)
        layout.addWidget(panel)

        tray = QFrame()
        tray.setObjectName("settingsPanel")
        tray_layout = QHBoxLayout(tray)
        tray_layout.setContentsMargins(19, 16, 19, 16)
        text = QVBoxLayout()
        label = QLabel("Сворачивать в системный трей")
        label.setStyleSheet("font-weight: 600;")
        help_label = QLabel(
            "При закрытии окна бот продолжит работу, а панель останется рядом с часами."
        )
        help_label.setWordWrap(True)
        help_label.setProperty("muted", True)
        text.addWidget(label)
        text.addWidget(help_label)
        tray_layout.addLayout(text, 1)
        switch = Switch(True)
        switch.setEnabled(False)
        tray_layout.addWidget(switch)
        layout.addWidget(tray)
        layout.addStretch()
