"""Validate public settings before Qt editors or persistence can coerce values."""

from __future__ import annotations

import math
from typing import Any

from bot.config_values import nested_get
from bot.hotkeys import validate_hotkeys

from .models import BotSpec, SettingField


def validate_setting(field: SettingField, value: Any) -> None:
    def reject(reason: str) -> None:
        raise ValueError(f"{field.path}: {reason}")

    if field.kind == "bool":
        if type(value) is not bool:
            reject("ожидается true или false без кавычек")
    elif field.kind == "int":
        if type(value) is not int:
            reject("ожидается целое число")
    elif field.kind == "float":
        if type(value) not in (int, float) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            reject("ожидается конечное число")
    elif field.kind == "choice":
        if value not in [stored for _label, stored in field.choices]:
            reject(f"неизвестный вариант {value!r}")
    elif field.kind == "text":
        if not isinstance(value, str):
            reject("ожидается строка")
    elif field.kind == "range_int":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            reject("ожидается диапазон из двух целых чисел")
        if any(type(item) is not int for item in value):
            reject("границы диапазона должны быть целыми числами")
        if value[0] > value[1]:
            reject("минимум не может быть больше максимума")
    else:
        reject(f"неизвестный тип редактора {field.kind!r}")

    if field.kind in {"int", "float", "range_int"}:
        values = value if field.kind == "range_int" else (value,)
        for number in values:
            if field.minimum is not None and number < field.minimum:
                reject(f"значение должно быть не меньше {field.minimum}")
            if field.maximum is not None and number > field.maximum:
                reject(f"значение должно быть не больше {field.maximum}")


def validate_settings(spec: BotSpec, config: dict[str, Any]) -> None:
    for field in spec.settings:
        validate_setting(field, nested_get(config, field.path))
    controls = {
        field.path.removeprefix("hotkeys."): nested_get(config, field.path)
        for field in spec.settings
        if field.path.startswith("hotkeys.")
    }
    validate_hotkeys(
        controls,
        extended=spec.bot_id in {"miner", "fishing"},
        all_special_keys=spec.bot_id == "blackjack",
    )
