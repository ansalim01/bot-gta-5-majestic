"""Static UI metadata for available and planned bots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bot.config_values import nested_get as nested_get
from bot.config_values import nested_set as nested_set

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SettingField:
    path: str
    label: str
    kind: str
    help_text: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BotSpec:
    bot_id: str
    title: str
    icon: str
    description: str
    available: bool = False
    entrypoint: str | None = None
    config_name: str | None = None
    settings: tuple[SettingField, ...] = ()
    stats: tuple[tuple[str, str], ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def config_path(self) -> Path | None:
        if not self.config_name:
            return None
        return ROOT / "config" / f"{self.config_name}.yaml"


BOT_SPECS: tuple[BotSpec, ...] = (
    BotSpec(
        bot_id="blackjack",
        title="Блэкджек 21",
        icon="21",
        description="Использует max-EV стратегию для наблюдаемой effective-48 смеси карт.",
        available=True,
        entrypoint="main_blackjack.py",
        config_name="blackjack",
        tags=("Автоигра", "Effective-48 EV"),
        settings=(
            SettingField(
                "blackjack.auto_action",
                "Автоматически нажимать",
                "bool",
                "Если выключено, бот только печатает рекомендуемый ход.",
            ),
            SettingField(
                "blackjack.auto_bet",
                "Автоматически ставить",
                "bool",
                "Подтверждать Enter уже выбранный на игровом столе размер ставки.",
            ),
            SettingField(
                "blackjack.bet_amount",
                "Размер ставки для отчёта",
                "int",
                "Укажите сумму, выбранную на столе. Она используется в аналитике прибыли и ROI.",
                minimum=1,
                maximum=1000000,
                step=1,
            ),
            SettingField(
                "blackjack.double_policy",
                "Политика дабла",
                "choice",
                "Полный режим использует рассчитанную effective-48 EV-таблицу даблов.",
                choices=(
                    ("Консервативная: 10–11", "conservative"),
                    ("Полная EV-таблица", "basic"),
                    ("Без даблов", "never"),
                ),
            ),
            SettingField(
                "blackjack.split_enabled",
                "Использовать сплит",
                "bool",
                "Разделять пары по EV-таблице; Majestic сначала разыгрывает вторую показанную сумму.",
            ),
            SettingField(
                "blackjack.confirm_frames",
                "Подтверждение, кадров",
                "int",
                "Сколько одинаковых распознаваний требуется перед решением.",
                minimum=1,
                maximum=10,
                step=1,
            ),
            SettingField(
                "blackjack.min_ocr_confidence",
                "Уверенность OCR",
                "float",
                "При меньшей уверенности бот не совершает действие.",
                minimum=0.3,
                maximum=1.0,
                step=0.01,
            ),
            SettingField(
                "blackjack.risk.max_session_loss",
                "Лимит убытка (0 = выкл.)",
                "int",
                "После достижения лимита бот прекращает новые ставки. 0 отключает ограничение.",
                minimum=0,
                maximum=1000000,
                step=10,
            ),
            SettingField(
                "blackjack.risk.max_drawdown_from_peak",
                "Лимит просадки (0 = выкл.)",
                "int",
                "Максимальная потеря от лучшего результата сессии. 0 отключает ограничение.",
                minimum=0,
                maximum=1000000,
                step=10,
            ),
            SettingField(
                "blackjack.risk.max_hands_per_session",
                "Раздач за сессию",
                "int",
                "После этого числа раздач новые ставки прекращаются. 0 отключает ограничение.",
                minimum=0,
                maximum=100000,
                step=10,
            ),
            SettingField("hotkeys.toggle", "Старт / пауза", "text"),
            SettingField("hotkeys.stop", "Экстренная остановка", "text"),
            SettingField(
                "telemetry.enabled",
                "Собирать аналитику",
                "bool",
                "Записывать каждую ставку, решение и результат раздачи.",
            ),
        ),
        stats=(
            ("bets", "Ставки"),
            ("wins", "Победы"),
            ("losses", "Поражения"),
            ("pushes", "Ничьи"),
            ("net", "Итог"),
            ("roi_pct", "ROI, %"),
            ("cards_inferred", "Карт восстановлено"),
            ("running_count", "Effective-48 счёт (диагн.)"),
            ("max_drawdown", "Просадка"),
            ("risk_stopped", "Лимит риска"),
        ),
    ),
    BotSpec(
        bot_id="miner",
        title="Шахтёр",
        icon="⛏",
        description="Добыча руды, мини-игра и маршруты сдачи.",
        available=False,
        entrypoint="main.py",
        config_name="miner",
        tags=("Маршруты", "Мини-игра"),
        settings=(
            SettingField(
                "mode",
                "Режим работы",
                "choice",
                "Определяет, какую часть цикла выполняет бот.",
                choices=(
                    ("Только мини-игра", "minigame_only"),
                    ("Маршрут и мини-игра", "walk_minigame"),
                    ("Полный цикл", "full"),
                ),
            ),
            SettingField(
                "navigation.enabled",
                "Автоматическая навигация",
                "bool",
                "Использовать записанные маршруты между точками.",
            ),
            SettingField(
                "mining.lmb_interval_ms",
                "Интервал кликов, мс",
                "range_int",
                "Случайная задержка между нажатиями ЛКМ.",
                minimum=10,
                maximum=500,
                step=5,
            ),
            SettingField(
                "minigame.click_delay_ms",
                "Задержка по жилам, мс",
                "range_int",
                "Пауза перед кликом по найденной жиле.",
                minimum=0,
                maximum=1000,
                step=5,
            ),
            SettingField(
                "hotkeys.toggle",
                "Старт / пауза",
                "text",
                "Глобальная горячая клавиша.",
            ),
            SettingField(
                "hotkeys.stop",
                "Экстренная остановка",
                "text",
                "Останавливает бот и отпускает все клавиши.",
            ),
        ),
        stats=(
            ("cycles", "Циклы"),
            ("minigames", "Мини-игры"),
            ("veins_clicked", "Жилы"),
            ("nav_steps", "Шаги маршрута"),
        ),
    ),
    BotSpec(
        bot_id="fishing",
        title="Рыбалка",
        icon="◒",
        description="Заброс, подсечка, вываживание и сбор улова.",
        available=True,
        entrypoint="main_fishing.py",
        config_name="fishing",
        tags=("Автозаброс", "Вываживание"),
        settings=(
            SettingField(
                "fishing.auto_start_activity",
                "Автоматически начать рыбалку",
                "bool",
                "Нажать E на приглашении, подтвердить снаряжение и перейти к забросу.",
            ),
            SettingField(
                "fishing.sustenance_enabled",
                "Еда и вода перед забросом",
                "bool",
                "При голоде или жажде выйти через Esc, поесть/попить и заново запустить рыбалку.",
            ),
            SettingField(
                "mode",
                "Режим работы",
                "choice",
                "Полный автоматический цикл или только помощь в мини-игре.",
                choices=(
                    ("Полный цикл", "cast_and_reel"),
                    ("Только вываживание", "minigame_only"),
                ),
            ),
            SettingField(
                "fishing.auto_cast",
                "Автоматический заброс",
                "bool",
                "Самостоятельно начинать следующий заброс.",
            ),
            SettingField(
                "fishing.target_fish_count",
                "Сколько рыб поймать",
                "int",
                "После этого количества бот завершит рыбалку. Значение 0 снимает ограничение.",
                minimum=0,
                maximum=1000,
                step=1,
            ),
            SettingField(
                "fishing.bite_timeout_sec",
                "Ожидание поклёвки, сек",
                "int",
                "После таймаута бот начнёт новый цикл.",
                minimum=5,
                maximum=180,
                step=1,
            ),
            SettingField(
                "fishing.keep_click_delay_ms",
                "Задержка перед сбором, мс",
                "int",
                "Даёт окну улова закончить анимацию.",
                minimum=0,
                maximum=3000,
                step=25,
            ),
            SettingField(
                "fishing.after_collect_ms",
                "Пауза после улова, мс",
                "int",
                "Пауза перед следующим забросом.",
                minimum=0,
                maximum=5000,
                step=50,
            ),
            SettingField(
                "reel.invert_pan_steer",
                "Инвертировать A / D",
                "bool",
                "Поменять направление управления при вываживании.",
            ),
            SettingField(
                "telemetry.enabled",
                "Собирать диагностику сессии",
                "bool",
                "Записывать тайминги, причины ошибок и итоговые рекомендации.",
            ),
            SettingField(
                "hotkeys.toggle",
                "Старт / пауза",
                "text",
                "Глобальная горячая клавиша.",
            ),
            SettingField(
                "hotkeys.stop",
                "Экстренная остановка",
                "text",
                "Останавливает бот и отпускает все клавиши.",
            ),
        ),
        stats=(
            ("fish_caught", "Поймано"),
            ("unidentified_catches", "Без окна / мусор"),
            ("casts", "Забросы"),
            ("hooks", "Подсечки"),
            ("timeouts", "Таймауты"),
            ("drinks", "Напитки"),
            ("meals", "Приёмы еды"),
        ),
    ),
    BotSpec(
        bot_id="gym",
        title="Качалка",
        icon="◆",
        description="Распознавание подсказок W, A, S, D на тренажёре.",
        available=True,
        entrypoint="main_gym.py",
        config_name="gym",
        tags=("WASD", "Еда и вода"),
        settings=(
            SettingField(
                "gym.start_key",
                "Клавиша начала",
                "text",
                "Клавиша взаимодействия с тренажёром.",
            ),
            SettingField(
                "gym.reaction_delay_ms",
                "Реакция, мс",
                "range_int",
                "Случайная задержка перед ответом на подсказку.",
                minimum=0,
                maximum=1000,
                step=5,
            ),
            SettingField(
                "gym.start_retry_sec",
                "Повтор запуска, сек",
                "float",
                "Как скоро повторно нажать клавишу начала.",
                minimum=0.5,
                maximum=30,
                step=0.1,
            ),
            SettingField(
                "gym.min_match_score",
                "Точность распознавания",
                "float",
                "Минимальная уверенность распознавания буквы.",
                minimum=0.1,
                maximum=1.0,
                step=0.01,
            ),
            SettingField(
                "hotkeys.toggle",
                "Старт / пауза",
                "text",
                "Глобальная горячая клавиша.",
            ),
            SettingField(
                "hotkeys.stop",
                "Экстренная остановка",
                "text",
                "Останавливает бот и отпускает все клавиши.",
            ),
        ),
        stats=(
            ("cycles", "Подходы"),
            ("prompts", "Команды"),
            ("w", "Нажато W"),
            ("d", "Нажато D"),
        ),
    ),
    BotSpec(
        bot_id="sustenance",
        title="Еда и вода",
        icon="●",
        description="Следит за HUD и использует предметы из инвентаря.",
        available=True,
        entrypoint="main_sustenance.py",
        config_name="sustenance",
        tags=("Поддержка", "Инвентарь"),
        settings=(
            SettingField(
                "sustenance.water_enabled",
                "Использовать воду",
                "bool",
                "Автоматически пить при предупреждении о жажде.",
            ),
            SettingField(
                "sustenance.food_enabled",
                "Использовать еду",
                "bool",
                "Автоматически есть при предупреждении о голоде.",
            ),
            SettingField(
                "sustenance.hud_check_interval_sec",
                "Интервал проверки, сек",
                "int",
                "Как часто анализировать основной HUD.",
                minimum=5,
                maximum=600,
                step=5,
            ),
            SettingField(
                "sustenance.action_cooldown_sec",
                "Пауза между действиями, сек",
                "int",
                "Защита от повторного использования предмета.",
                minimum=1,
                maximum=180,
                step=1,
            ),
            SettingField(
                "sustenance.inventory_key",
                "Клавиша инвентаря",
                "text",
                "Клавиша открытия инвентаря GTA.",
            ),
            SettingField(
                "hotkeys.check_now",
                "Проверить сейчас",
                "text",
                "Горячая клавиша немедленной проверки.",
            ),
        ),
        stats=(
            ("checks", "Проверки"),
            ("drinks", "Напитки"),
            ("meals", "Приёмы еды"),
            ("failures", "Ошибки"),
        ),
    ),
    BotSpec(
        "woodcutter",
        "Лесоруб",
        "⌁",
        "Автоматизация рубки и доставки древесины.",
        tags=("Скоро",),
    ),
    BotSpec(
        "courier",
        "Курьер",
        "▣",
        "Маршруты получения и доставки заказов.",
        tags=("Скоро",),
    ),
    BotSpec(
        "farmer",
        "Фермер",
        "♧",
        "Сбор урожая и повторение рабочего цикла.",
        tags=("Скоро",),
    ),
    BotSpec(
        "mushroom",
        "Грибник",
        "♠",
        "Поиск точек и автоматический сбор грибов.",
        tags=("Скоро",),
    ),
    BotSpec(
        "seamstress",
        "Швейка",
        "⌇",
        "Прохождение мини-игры швейного производства.",
        tags=("Скоро",),
    ),
    BotSpec(
        "taxi",
        "Таксист",
        "▰",
        "Работа с заказами и маршрутами такси.",
        tags=("Скоро",),
    ),
    BotSpec(
        "carjacker",
        "Угонщик",
        "◇",
        "Сценарии заданий по угону транспорта.",
        tags=("Скоро",),
    ),
    BotSpec(
        "anti_afk",
        "Анти-AFK",
        "◉",
        "Безопасное поддержание активности персонажа.",
        tags=("Скоро",),
    ),
    BotSpec(
        "demorgan",
        "Деморган",
        "▥",
        "Автоматизация повторяющихся тюремных заданий.",
        tags=("Скоро",),
    ),
)


BOT_BY_ID: dict[str, BotSpec] = {spec.bot_id: spec for spec in BOT_SPECS}
AVAILABLE_BOTS: tuple[BotSpec, ...] = tuple(spec for spec in BOT_SPECS if spec.available)
PLANNED_BOTS: tuple[BotSpec, ...] = tuple(spec for spec in BOT_SPECS if not spec.available)
