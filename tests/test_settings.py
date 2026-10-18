from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSpinBox

from gamebot_ui.models import SettingField
from gamebot_ui.settings import SettingRow
from gamebot_ui.widgets import Switch


def test_zero_is_a_valid_integer_bound():
    app = QApplication.instance() or QApplication([])
    for minimum, maximum in [(0, 100), (-10, 0), (0, 0)]:
        row = SettingRow(SettingField("value", "Value", "int", minimum=minimum, maximum=maximum), 0)
        assert row.editor.minimum() == minimum
        assert row.editor.maximum() == maximum
        row.editor.setValue(minimum - 1)
        assert row.value() == minimum
        row.editor.setValue(maximum + 1)
        assert row.value() == maximum
        row.close()
        row.deleteLater()
    app.processEvents()


def test_switch_can_be_operated_from_the_keyboard(qt_app):
    switch = Switch()
    changes = []
    switch.toggled.connect(changes.append)
    QTest.keyClick(switch, Qt.Key_Space)
    assert switch.isChecked() is True
    QTest.keyClick(switch, Qt.Key_Return)
    assert switch.isChecked() is False
    assert changes == [True, False]
    switch.deleteLater()


def test_range_editor_respects_a_zero_maximum(qt_app):
    row = SettingRow(SettingField("range", "Range", "range_int", minimum=0, maximum=0), [0, 0])
    for spin in row.editor.findChildren(QSpinBox):
        assert spin.minimum() == 0
        assert spin.maximum() == 0
    assert row.value() == [0, 0]
    row.deleteLater()
