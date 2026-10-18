"""Bot state machine."""

from __future__ import annotations

from enum import Enum, auto


class BotState(Enum):
    IDLE = auto()
    WAITING = auto()
    NAVIGATE = auto()
    PRESS_E = auto()
    MINING = auto()
    MINIGAME = auto()
    CARRY = auto()
    ROUTE_TO_ROCK = auto()
    ROUTE_TO_DROP = auto()
    ROUTE_BACK = auto()
    # Fishing
    FISHING_START = auto()
    FISHING_EQUIPMENT = auto()
    FISHING_SUSTENANCE = auto()
    CASTING = auto()
    CAST_TIMING = auto()
    WAITING_BITE = auto()
    HOOKING = auto()
    REELING = auto()
    COLLECT = auto()
    INVENTORY_FULL = auto()
    ROUTE_TO_SPOT = auto()
    ROUTE_TO_SELL = auto()
    STOPPED = auto()


class StateMachine:
    def __init__(self) -> None:
        self._state = BotState.IDLE
        self._running = False

    @property
    def state(self) -> BotState:
        return self._state

    @state.setter
    def state(self, new_state: BotState) -> None:
        self._state = new_state

    @property
    def running(self) -> bool:
        return self._running and self._state not in (BotState.IDLE, BotState.STOPPED)

    @property
    def active(self) -> bool:
        return self._state not in (BotState.STOPPED,)

    def toggle(self) -> None:
        if self._state == BotState.STOPPED:
            self._state = BotState.IDLE
            self._running = False
            return

        if self._running:
            self._running = False
            self._state = BotState.IDLE
        else:
            self._running = True
            if self._state == BotState.IDLE:
                self._state = BotState.WAITING

    def stop(self) -> None:
        self._running = False
        self._state = BotState.STOPPED

    def resume_from_idle(self) -> None:
        if self._running and self._state == BotState.IDLE:
            self._state = BotState.WAITING

    def state_label(self) -> str:
        labels = {
            BotState.IDLE: "IDLE (paused)",
            BotState.WAITING: "WATCHING",
            BotState.NAVIGATE: "NAVIGATE",
            BotState.PRESS_E: "PRESS_E",
            BotState.MINING: "MINING",
            BotState.MINIGAME: "MINIGAME",
            BotState.CARRY: "CARRY",
            BotState.ROUTE_TO_ROCK: "ROUTE_TO_ROCK",
            BotState.ROUTE_TO_DROP: "ROUTE_TO_DROP",
            BotState.ROUTE_BACK: "ROUTE_BACK",
            BotState.FISHING_START: "FISHING_START",
            BotState.FISHING_EQUIPMENT: "FISHING_EQUIPMENT",
            BotState.FISHING_SUSTENANCE: "FOOD_AND_WATER",
            BotState.CASTING: "CASTING",
            BotState.CAST_TIMING: "CAST_TIMING",
            BotState.WAITING_BITE: "WAITING_BITE",
            BotState.HOOKING: "HOOKING",
            BotState.REELING: "REELING",
            BotState.COLLECT: "COLLECT",
            BotState.INVENTORY_FULL: "INVENTORY_FULL",
            BotState.ROUTE_TO_SPOT: "ROUTE_TO_SPOT",
            BotState.ROUTE_TO_SELL: "ROUTE_TO_SELL",
            BotState.STOPPED: "STOPPED",
        }
        status = "ON" if self._running else "OFF"
        return f"{labels.get(self._state, '?')} [{status}]"
