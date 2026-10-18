"""Reusable editors for settings described by SettingField metadata."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .models import SettingField
from .theme import COLORS
from .validation import validate_setting
from .widgets import Switch


class SettingRow(QFrame):
    def __init__(
        self,
        field: SettingField,
        value: Any,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        validate_setting(field, value)
        self.field = field
        self.compact = compact
        self.setProperty("compact", compact)
        self.setMinimumWidth(0)
        self.setStyleSheet(f"QFrame {{ border-bottom: 1px solid {COLORS['border']}; }}")
        layout = QHBoxLayout(self)
        row_padding = 2 if compact else 11
        layout.setContentsMargins(2, row_padding, 2, row_padding)
        layout.setSpacing(10 if compact else 18)
        text = QVBoxLayout()
        text.setSpacing(2)
        label = QLabel(field.label)
        label.setStyleSheet("font-weight: 600; border: 0;")
        label.setMinimumWidth(0)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        text.addWidget(label)
        if field.help_text and compact:
            self.setToolTip(field.help_text)
            label.setToolTip(field.help_text)
        elif field.help_text:
            help_label = QLabel(field.help_text)
            help_label.setProperty("muted", True)
            help_label.setWordWrap(True)
            help_label.setMinimumWidth(0)
            help_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            help_label.setStyleSheet(f"color: {COLORS['muted']}; font-size: 11px; border: 0;")
            text.addWidget(help_label)
        layout.addLayout(text, 1)
        self.editor = self._create_editor(value)
        self.editor.setAccessibleName(field.label)
        if field.help_text and compact:
            self.editor.setToolTip(field.help_text)
        layout.addWidget(self.editor, 0, Qt.AlignRight | Qt.AlignVCenter)

    def _create_editor(self, value: Any) -> QWidget:
        field = self.field
        if field.kind == "bool":
            return Switch(bool(value))
        if field.kind == "choice":
            editor = QComboBox()
            editor.setMinimumWidth(155 if self.compact else 205)
            for label, stored_value in field.choices:
                editor.addItem(label, stored_value)
            index = editor.findData(value)
            editor.setCurrentIndex(max(index, 0))
            return editor
        if field.kind == "int":
            editor = QSpinBox()
            editor.setMinimumWidth(105 if self.compact else 125)
            editor.setRange(
                int(field.minimum if field.minimum is not None else -1_000_000),
                int(field.maximum if field.maximum is not None else 1_000_000),
            )
            editor.setSingleStep(int(field.step or 1))
            editor.setValue(int(value or 0))
            return editor
        if field.kind == "float":
            editor = QDoubleSpinBox()
            editor.setMinimumWidth(105 if self.compact else 125)
            editor.setDecimals(2)
            editor.setRange(
                float(field.minimum if field.minimum is not None else -1_000_000),
                float(field.maximum if field.maximum is not None else 1_000_000),
            )
            editor.setSingleStep(float(field.step or 0.1))
            editor.setValue(float(value or 0))
            return editor
        if field.kind == "range_int":
            container = QWidget()
            container.setStyleSheet("border: 0; background: transparent;")
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            values = value if isinstance(value, (list, tuple)) else [0, 0]
            minimum = int(field.minimum if field.minimum is not None else 0)
            maximum = int(field.maximum if field.maximum is not None else 1_000_000)
            for index in range(2):
                spin = QSpinBox()
                spin.setProperty("rangeIndex", index)
                spin.setRange(minimum, maximum)
                spin.setSingleStep(int(field.step or 1))
                spin.setValue(int(values[index] if index < len(values) else 0))
                spin.setFixedWidth(92)
                row.addWidget(spin)
                if index == 0:
                    dash = QLabel("—")
                    dash.setProperty("muted", True)
                    dash.setStyleSheet("border: 0;")
                    row.addWidget(dash)
            return container
        editor = QLineEdit(str(value if value is not None else ""))
        editor.setMinimumWidth(120 if self.compact else 150)
        editor.setMaxLength(32)
        return editor

    def value(self) -> Any:
        editor = self.editor
        if isinstance(editor, Switch):
            return editor.isChecked()
        if isinstance(editor, QComboBox):
            return editor.currentData()
        if isinstance(editor, QSpinBox):
            return editor.value()
        if isinstance(editor, QDoubleSpinBox):
            return editor.value()
        if isinstance(editor, QLineEdit):
            return editor.text().strip().lower()
        spins = editor.findChildren(QSpinBox)
        if self.field.kind == "range_int" and len(spins) == 2:
            spins.sort(key=lambda item: int(item.property("rangeIndex")))
            return [spin.value() for spin in spins]
        return None
