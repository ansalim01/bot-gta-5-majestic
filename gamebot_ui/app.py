"""Main PySide6 desktop application."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from bot.windows import is_admin as is_admin

from .config_store import ConfigStore
from .models import BOT_BY_ID, BOT_SPECS, ROOT
from .pages import AppSettingsPage, BotPage, DashboardPage, LogsPage
from .process_manager import BotProcessManager
from .theme import COLORS, STYLESHEET
from .widgets import SidebarButton, TitleBar

ASSET_ICON = ROOT / "assets" / "gamebot.svg"
GAME_TITLES = ("grand theft auto v", "gta5", "majestic", "rage multiplayer")


def game_window_exists() -> bool:
    if sys.platform != "win32":
        return False
    found = False
    user32 = ctypes.windll.user32
    enum_proc_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    def callback(hwnd, _lparam):
        nonlocal found
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.lower()
        if any(candidate in title for candidate in GAME_TITLES):
            found = True
            return False
        return True

    try:
        user32.EnumWindows(enum_proc_type(callback), 0)
    except (AttributeError, OSError):
        return False
    return found


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GameBot Control")
        self.setWindowIcon(QIcon(str(ASSET_ICON)))
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.resize(1120, 720)
        self.setMinimumSize(980, 640)
        self._allow_close = False
        self._tray_notice_shown = False
        self._is_admin = is_admin()

        self.manager = BotProcessManager(self)
        self.config_store = ConfigStore()
        self.pages: dict[str, QWidget] = {}
        self.sidebar_buttons: dict[str, SidebarButton] = {}

        root = QWidget()
        root.setObjectName("appRoot")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.title_bar = TitleBar(self._is_admin)
        self.title_bar.minimize_clicked.connect(self.showMinimized)
        self.title_bar.close_clicked.connect(self.close)
        outer.addWidget(self.title_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.setCentralWidget(root)

        self._build_pages()
        self._connect_manager()
        self._build_tray()
        self.open_page("overview")

        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self._refresh_game_status)
        self.game_timer.start(2500)
        self._refresh_game_status()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(222)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(13, 17, 13, 15)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        logo = QLabel("GB")
        logo.setFixedSize(38, 38)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"background: {COLORS['accent']}; color: white;"
            " border-radius: 11px; font-size: 13px; font-weight: 850;"
        )
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        name = QLabel("GameBot")
        name.setObjectName("brandName")
        control = QLabel("CONTROL PANEL")
        control.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 9px; font-weight: 700; letter-spacing: 1px;"
        )
        brand_text.addWidget(name)
        brand_text.addWidget(control)
        brand.addWidget(logo)
        brand.addLayout(brand_text)
        brand.addStretch()
        layout.addLayout(brand)
        layout.addSpacing(10)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        overview = SidebarButton("Обзор", "⌂")
        overview.clicked.connect(lambda: self.open_page("overview"))
        self._add_sidebar_button("overview", overview)
        layout.addWidget(overview)

        section = QLabel("БОТЫ")
        section.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 9px; font-weight: 750;"
            " letter-spacing: 1px; padding: 7px 10px 2px 10px;"
        )
        layout.addWidget(section)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bot_list = QWidget()
        bot_layout = QVBoxLayout(bot_list)
        bot_layout.setContentsMargins(0, 0, 3, 0)
        bot_layout.setSpacing(3)
        for spec in BOT_SPECS:
            badge = "" if spec.available else "скоро"
            button = SidebarButton(spec.title, spec.icon, badge)
            button.setToolTip(spec.description)
            button.clicked.connect(
                lambda _checked=False, bot_id=spec.bot_id: self.open_page(bot_id)
            )
            self._add_sidebar_button(spec.bot_id, button)
            bot_layout.addWidget(button)
        bot_layout.addStretch()
        scroll.setWidget(bot_list)
        layout.addWidget(scroll, 1)

        separator = QFrame()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background: {COLORS['border']};")
        layout.addWidget(separator)

        logs = SidebarButton("Журнал", "≡")
        logs.clicked.connect(lambda: self.open_page("logs"))
        self._add_sidebar_button("logs", logs)
        settings = SidebarButton("Настройки", "⚙")
        settings.clicked.connect(lambda: self.open_page("settings"))
        self._add_sidebar_button("settings", settings)
        layout.addWidget(logs)
        layout.addWidget(settings)

        self.sidebar_game = QLabel("●  Проверяю окно GTA…")
        self.sidebar_game.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 11px; padding: 8px 10px 2px;"
        )
        layout.addWidget(self.sidebar_game)
        return sidebar

    def _add_sidebar_button(self, page_id: str, button: SidebarButton) -> None:
        self.sidebar_buttons[page_id] = button
        self.button_group.addButton(button)

    def _build_pages(self) -> None:
        dashboard = DashboardPage()
        dashboard.open_bot.connect(self.open_page)
        dashboard.start_bot.connect(self._start_bot)
        self.dashboard = dashboard
        self._register_page("overview", dashboard)

        for spec in BOT_SPECS:
            page = BotPage(spec, self.config_store)
            page.start_requested.connect(self._start_bot)
            page.pause_requested.connect(self.manager.pause)
            page.resume_requested.connect(self.manager.resume)
            page.stop_requested.connect(self.manager.stop)
            page.check_requested.connect(self.manager.check_now)
            page.saved.connect(self._settings_saved)
            self._register_page(spec.bot_id, page)

        self.logs_page = LogsPage()
        self._register_page("logs", self.logs_page)
        settings = AppSettingsPage(ROOT, self._is_admin)
        self._register_page("settings", settings)

    def _register_page(self, page_id: str, page: QWidget) -> None:
        self.pages[page_id] = page
        self.stack.addWidget(page)

    def _connect_manager(self) -> None:
        self.manager.status_changed.connect(self._status_changed)
        self.manager.stats_changed.connect(self._stats_changed)
        self.manager.log_line.connect(self._log_line)
        self.manager.error.connect(self._manager_error)
        self.manager.active_changed.connect(self._active_changed)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(QIcon(str(ASSET_ICON)), self)
        self.tray.setToolTip("GameBot Control")
        menu = QMenu()
        open_action = QAction("Открыть панель", self)
        open_action.triggered.connect(self.show_panel)
        self.tray_pause = QAction("Пауза", self)
        self.tray_pause.triggered.connect(self._tray_pause_resume)
        self.tray_stop = QAction("Экстренно остановить", self)
        self.tray_stop.triggered.connect(lambda: self.manager.stop())
        logs_action = QAction("Показать журнал", self)
        logs_action.triggered.connect(lambda: self._show_page_from_tray("logs"))
        exit_action = QAction("Завершить приложение", self)
        exit_action.triggered.connect(self.exit_application)
        menu.addAction(open_action)
        menu.addSeparator()
        menu.addAction(self.tray_pause)
        menu.addAction(self.tray_stop)
        menu.addAction(logs_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()
        self._active_changed(None)

    def open_page(self, page_id: str) -> None:
        page = self.pages.get(page_id)
        button = self.sidebar_buttons.get(page_id)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        if button:
            button.setChecked(True)

    def _start_bot(self, bot_id: str) -> None:
        spec = BOT_BY_ID[bot_id]
        page = self.pages.get(bot_id)
        if isinstance(page, BotPage) and not page.save_settings(notify=False):
            return
        self._log_line(
            bot_id,
            "Текущие значения интерфейса сохранены и передаются новому процессу.",
        )
        active = self.manager.active_id
        activated = False
        if active and active != bot_id:
            active_spec = BOT_BY_ID[active]
            answer = QMessageBox.question(
                self,
                "Переключить бот?",
                f"Сейчас запущен «{active_spec.title}». Остановить его и запустить «{spec.title}»?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return
            self.manager.switch_to(bot_id)
            activated = True
        elif active == bot_id and self.manager.status(bot_id) == "paused":
            self.manager.resume(bot_id)
            activated = True
        else:
            activated = self.manager.start(bot_id)
        self.open_page(bot_id)
        if spec.available and activated:
            # Every worker sends input only to the verified foreground game
            # window. Minimize the panel so it cannot steal the first action.
            QTimer.singleShot(100, self.showMinimized)

    def _status_changed(self, bot_id: str, status: str) -> None:
        self.dashboard.set_bot_status(bot_id, status)
        page = self.pages.get(bot_id)
        if isinstance(page, BotPage):
            page.set_status(status)
            if bot_id == "fishing" and status in {"idle", "error"}:
                page.refresh_diagnostics()
        self.tray_pause.setText("Продолжить" if status == "paused" else "Пауза")
        title = BOT_BY_ID[bot_id].title
        self.tray.setToolTip(f"GameBot Control — {title}: {status}")

    def _stats_changed(self, bot_id: str, values: object) -> None:
        page = self.pages.get(bot_id)
        if isinstance(page, BotPage) and isinstance(values, dict):
            page.set_stats(values)

    def _log_line(self, bot_id: str, message: str) -> None:
        title = BOT_BY_ID.get(bot_id).title if bot_id in BOT_BY_ID else bot_id
        self.logs_page.append(title, message)

    def _manager_error(self, bot_id: str, message: str) -> None:
        self._log_line(bot_id, f"ОШИБКА: {message}")
        if self.isVisible():
            QMessageBox.critical(
                self,
                f"Ошибка: {BOT_BY_ID[bot_id].title}",
                message,
            )
        else:
            self.tray.showMessage(
                "Ошибка GameBot",
                message,
                QSystemTrayIcon.Critical,
                5000,
            )

    def _active_changed(self, bot_id: object) -> None:
        active = str(bot_id) if bot_id else None
        enabled = active is not None
        self.tray_pause.setEnabled(enabled)
        self.tray_stop.setEnabled(enabled)
        if active:
            self.tray.setToolTip(f"GameBot Control — {BOT_BY_ID[active].title}")
        else:
            self.tray.setToolTip("GameBot Control — всё остановлено")

    def _settings_saved(self, bot_id: str, destination: str) -> None:
        running = self.manager.active_id == bot_id
        suffix = " Применятся после остановки и нового запуска бота." if running else ""
        self._log_line(bot_id, f"Настройки сохранены: {destination}.{suffix}")
        self.tray.showMessage(
            "Настройки сохранены",
            f"Профиль «{BOT_BY_ID[bot_id].title}» обновлён.{suffix}",
            QSystemTrayIcon.Information,
            2500,
        )

    def _refresh_game_status(self) -> None:
        found = game_window_exists()
        self.title_bar.set_game_found(found)
        if found:
            self.sidebar_game.setText("●  GTA подключена")
            self.sidebar_game.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 11px; padding: 8px 10px 2px;"
            )
        else:
            self.sidebar_game.setText("●  GTA не найдена")
            self.sidebar_game.setStyleSheet(
                f"color: {COLORS['muted']}; font-size: 11px; padding: 8px 10px 2px;"
            )

    def _tray_pause_resume(self) -> None:
        active = self.manager.active_id
        if not active:
            return
        if self.manager.status(active) == "paused":
            self.manager.resume(active)
        else:
            self.manager.pause(active)

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.Trigger,
            QSystemTrayIcon.DoubleClick,
        ):
            self.show_panel()

    def _show_page_from_tray(self, page_id: str) -> None:
        self.show_panel()
        self.open_page(page_id)

    def show_panel(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self._tray_notice_shown:
            self._tray_notice_shown = True
            self.tray.showMessage(
                "GameBot продолжает работу",
                "Панель свёрнута в системный трей.",
                QSystemTrayIcon.Information,
                3000,
            )

    def exit_application(self) -> None:
        if self.manager.active_id:
            answer = QMessageBox.question(
                self,
                "Завершить приложение?",
                "Активный бот будет остановлен, а все удерживаемые клавиши — отпущены.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.manager.shutdown()
        self._allow_close = True
        self.tray.hide()
        self.close()
        QApplication.instance().quit()


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("GameBot Control")
    app.setOrganizationName("GameBot")
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(QIcon(str(ASSET_ICON)))
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()
