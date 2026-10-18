"""Blackjack decision loop for the Majestic casino table."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot.blackjack.strategy import (
    CARD_VALUES as CARD_VALUES,
)
from bot.blackjack.strategy import (
    DOUBLE as DOUBLE,
)
from bot.blackjack.strategy import (
    HIT as HIT,
)
from bot.blackjack.strategy import (
    SPLIT as SPLIT,
)
from bot.blackjack.strategy import (
    STAND as STAND,
)
from bot.blackjack.strategy import (
    add_card_to_total as add_card_to_total,
)
from bot.blackjack.strategy import (
    basic_strategy as basic_strategy,
)
from bot.blackjack.strategy import (
    double_allowed_by_policy as double_allowed_by_policy,
)
from bot.blackjack.strategy import (
    hi_lo_value as hi_lo_value,
)
from bot.blackjack.strategy import (
    infer_card_transition as infer_card_transition,
)
from bot.blackjack.strategy import (
    infer_initial_soft as infer_initial_soft,
)
from bot.blackjack.strategy import (
    majestic_balanced_count_value as majestic_balanced_count_value,
)
from bot.blackjack.strategy import (
    should_split_pair as should_split_pair,
)
from bot.blackjack.strategy import (
    soft_after_hit as soft_after_hit,
)
from bot.config_loader import ROOT_DIR
from bot.detectors.blackjack_score import BlackjackScore, BlackjackScoreDetector
from bot.telemetry.blackjack_session import BlackjackSessionTelemetry


@dataclass
class _Seen:
    key: tuple[int, ...] | None = None
    count: int = 0


@dataclass(frozen=True)
class _TableObservation:
    frame: np.ndarray
    score: BlackjackScore | None
    action_available: bool
    bet_available: bool
    bet_value: tuple[int, float] | None


@dataclass(frozen=True)
class _ActionRecommendation:
    action: str
    soft: bool
    pair_rank: int | None
    policy_blocked: bool
    double_blocked: bool


class BlackjackJob:
    def __init__(self, config: dict[str, Any], capture, input_handler) -> None:
        self._config = config
        self._capture = capture
        self._input = input_handler
        section = config.get("blackjack", {})
        self._detector = BlackjackScoreDetector(section.get("min_ocr_confidence", 0.64))
        self._confirm_frames = max(1, int(section.get("confirm_frames", 3)))
        self._auto_action = bool(section.get("auto_action", False))
        self._auto_bet = bool(section.get("auto_bet", True))
        self._bet_amount = max(1, int(section.get("bet_amount", 10)))
        configured_double_policy = str(section.get("double_policy", "conservative"))
        self._double_policy = (
            configured_double_policy
            if configured_double_policy in {"basic", "conservative", "never"}
            else "conservative"
        )
        self._split_enabled = bool(section.get("split_enabled", True))
        self._bet_key = str(section.get("bet_key", "enter"))
        self._bet_confirm_frames = max(1, int(section.get("bet_confirm_frames", 3)))
        self._min_bet_bright_pixels = int(section.get("min_bet_bright_pixels", 90))
        self._max_bet_bright_pixels = int(section.get("max_bet_bright_pixels", 2500))
        self._bet_retry_sec = max(1.0, float(section.get("bet_retry_sec", 2.0)))
        self._max_bet_retries = max(0, int(section.get("max_bet_retries", 2)))
        self._cooldown = float(section.get("action_cooldown_sec", 1.0))
        self._action_retry_sec = max(0.8, float(section.get("action_retry_sec", 1.25)))
        self._max_action_retries = max(0, int(section.get("max_action_retries", 2)))
        self._round_reset_sec = float(section.get("round_reset_sec", 3.0))
        risk = section.get("risk", {})
        self._max_session_loss = max(0, int(risk.get("max_session_loss", 100)))
        self._max_drawdown = max(0, int(risk.get("max_drawdown_from_peak", 100)))
        self._max_hands = max(0, int(risk.get("max_hands_per_session", 500)))
        self._risk_stop_reason: str | None = None
        self._keys = {
            HIT: str(section.get("hit_key", "enter")),
            STAND: str(section.get("stand_key", "space")),
            DOUBLE: str(section.get("double_key", "tab")),
            SPLIT: str(section.get("split_key", "f")),
        }
        self._seen = _Seen()
        self._last_action_key: tuple[int, ...] | None = None
        self._last_action_at = 0.0
        self._last_pressed_action: str | None = None
        self._last_input_blocked_key: tuple[int | None, int, int, str] | None = None
        self._action_retry_count = 0
        self._action_retry_exhausted = False
        self._action_missing_frames = 0
        self._action_gap_after_press = False
        self._missing_since: float | None = None
        self._round_initial_key: tuple[int, int] | None = None
        self._last_message: tuple[int, int, str] | None = None
        self._bet_seen = 0
        self._bet_missing = 0
        self._bet_placed = False
        self._bet_confirmed = False
        self._zero_bet_seen = 0
        self._last_bet_value: int | None = None
        self._last_bet_at = 0.0
        self._bet_retry_count = 0
        self._bet_retry_exhausted = False
        self._awaiting_round_after_bet = False
        self._deal_clear_frames = 0
        self._deal_first_player: int | None = None
        self._hand_soft = False
        self._last_player_total: int | None = None
        self._dealer_total: int | None = None
        self._dealer_soft = False
        self._split_pending = False
        self._split_active = False
        # Majestic displays two totals after a split, but plays the second
        # displayed total first and the first displayed total second.
        self._split_hand_index = 1
        self._split_pair_rank: int | None = None
        self._split_hand_totals: list[int | None] = [None, None]
        # Once the second displayed hand is finished, its OCR value must not
        # be allowed to change while the first hand is being played.  Majestic
        # occasionally renders the inactive hand with a transient total
        # (e.g. 21 -> 29); keeping this bit of state prevents that frame from
        # corrupting settlement telemetry.
        self._split_hand_closed = [False, False]
        self._split_hand_soft = [False, False]
        self._split_hand_decisions = [0, 0]
        self._round: dict[str, Any] | None = None
        self._pending_final: tuple[int, int] | tuple[int, int, int] | None = None
        self._stats: dict[str, int | float] = {
            "hands": 0,
            "bets": 0,
            "completed": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "unknown": 0,
            "decisions": 0,
            "hits": 0,
            "stands": 0,
            "doubles": 0,
            "splits": 0,
            "split_hands": 0,
            "wagered": 0,
            "accounted_wagered": 0,
            "net": 0,
            "roi_pct": 0.0,
            "win_rate_pct": 0.0,
            "peak_net": 0,
            "max_drawdown": 0,
            "risk_stopped": 0,
            "cards_inferred": 0,
            "running_count": 0,
            "ocr_misses": 0,
            "capture_errors": 0,
            "input_blocks": 0,
            "action_retries": 0,
            "action_unacknowledged": 0,
            "bet_retries": 0,
            "bet_unacknowledged": 0,
        }
        self._telemetry = BlackjackSessionTelemetry(config)
        self._telemetry.event(
            "strategy_configured",
            strategy_name="majestic_effective48_s17_enhc_max_ev",
            version=11,
            objective="max_expected_value_100_plus",
            rules_model=(
                "effective48_iid_s17_european_no_hole_card_one_split_das"
                if self._double_policy != "never"
                else "effective48_iid_s17_european_no_hole_card_one_split_no_das"
            ),
            rank_mix_model="three_ten_value_ranks_plus_nine_single_ranks",
            ten_value_probability=0.25,
            modeled_initial_ev_pct=-2.2116,
            soft_hand_inference=True,
            bet_policy="flat_minimum_ev",
            bet_amount=self._bet_amount,
            double_policy=self._double_policy,
            double_headroom_reserved=self._double_policy != "never",
            split_enabled=self._split_enabled,
            split_policy=(
                "effective48_s17_enhc_one_split_das_v2"
                if self._double_policy != "never"
                else "effective48_s17_enhc_one_split_no_das_v1"
            ),
            double_after_split=self._double_policy != "never",
            split_order="second_total_then_first_total",
            card_tracking="inferred_totals_v2",
            count_system="majestic_effective48_balanced_v2",
            count_used_for_betting=False,
            focus_recovery=bool(config.get("screen", {}).get("auto_focus_game_for_input", False)),
            action_retry_sec=self._action_retry_sec,
            max_action_retries=self._max_action_retries,
            bet_retry_sec=self._bet_retry_sec,
            max_bet_retries=self._max_bet_retries,
            max_session_loss=self._max_session_loss,
            max_drawdown_from_peak=self._max_drawdown,
            max_hands_per_session=self._max_hands,
        )
        self._debug_path = Path(config.get("debug", {}).get("live_crop", "logs/blackjack_live.png"))
        debug = config.get("debug", {})
        self._save_event_screenshots = bool(debug.get("save_event_screenshots", True))
        screenshot_root = Path(
            str(debug.get("screenshot_directory", "logs/debug/blackjack_screenshots"))
        )
        if not screenshot_root.is_absolute():
            screenshot_root = ROOT_DIR / screenshot_root
        self._screenshot_directory = screenshot_root / self._telemetry.session_id
        self._max_event_screenshots = max(0, int(debug.get("max_event_screenshots", 200)))
        self._event_screenshot_count = 0
        self._last_saved_state: tuple[str, int, int, int] | None = None
        self._last_capture_error_at = 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        return dict(self._stats)

    def _reset_split_state(self) -> None:
        self._split_pending = False
        self._split_active = False
        self._split_hand_index = 1
        self._split_pair_rank = None
        self._split_hand_totals = [None, None]
        self._split_hand_closed = [False, False]
        self._split_hand_soft = [False, False]
        self._split_hand_decisions = [0, 0]

    def _active_player_total(self, score: BlackjackScore) -> int:
        if (
            getattr(self, "_split_active", False)
            and getattr(self, "_split_hand_index", 0) == 1
            and score.player_second is not None
        ):
            return score.player_second
        return score.player

    def _active_soft(self) -> bool:
        if getattr(self, "_split_active", False):
            index = int(getattr(self, "_split_hand_index", 0))
            return bool(self._split_hand_soft[index])
        return bool(self._hand_soft)

    def _decision_key(self, score: BlackjackScore) -> tuple[int, ...]:
        total = self._active_player_total(score)
        if getattr(self, "_split_active", False):
            return (total, score.dealer, int(self._split_hand_index))
        return (total, score.dealer)

    def _initial_pair_rank(self) -> int | None:
        if self._round is None:
            return None
        cards = self._round.get("player_cards_inferred", [])
        if (
            isinstance(cards, list)
            and len(cards) == 2
            and cards[0] == cards[1]
            and int(cards[0]) in CARD_VALUES
        ):
            return int(cards[0])
        return None

    def _activate_split(self, score: BlackjackScore) -> None:
        """Start with the second displayed total once split totals are visible."""
        if score.player_second is None or getattr(self, "_split_active", False):
            return
        self._split_pending = False
        self._split_active = True
        self._split_hand_index = 1
        totals = [score.player, score.player_second]
        self._split_hand_totals = totals.copy()
        self._split_hand_closed = [False, False]
        self._split_hand_decisions = [0, 0]
        rank = self._split_pair_rank
        self._split_hand_soft = [infer_initial_soft(rank, total) for total in totals]
        if self._round is not None:
            self._round["split_confirmed"] = True
            self._round["split_initial_totals"] = totals.copy()
            self._round["split_stake_multipliers"] = [1, 1]
            # Each split hand receives one new card. The original pair was
            # already recorded by the ordinary initial-hand tracker.
            if rank is not None:
                first_total, first_soft = (11, True) if rank == 11 else (rank, False)
                for index, total in enumerate(totals):
                    inferred = infer_card_transition(first_total, first_soft, total)
                    if inferred is not None:
                        self._record_inferred_card(
                            "player",
                            inferred[0],
                            f"split_{'first' if index == 0 else 'second'}_total_deal",
                        )
            self._telemetry.event(
                "split_confirmed",
                round=self._round.get("round"),
                pair_rank=rank,
                left=totals[0],
                right=totals[1],
                dealer=score.dealer,
                active_hand="second_total",
            )

    def _apply_split_transition(
        self,
        score: BlackjackScore,
        index: int,
        *,
        source: str | None = None,
    ) -> None:
        total = score.player if index == 0 else score.player_second
        if total is None:
            return
        closed = getattr(self, "_split_hand_closed", [False, False])
        if index < len(closed) and closed[index]:
            # The inactive hand is still visible, but its OCR value is no
            # longer authoritative.  Record the observation for diagnostics
            # and keep the last confirmed total for settlement.
            self._telemetry.event(
                "inactive_split_score_ignored",
                round=self._round.get("round") if self._round else None,
                hand="first_total" if index == 0 else "second_total",
                stored_total=self._split_hand_totals[index],
                observed_total=total,
            )
            return
        previous = self._split_hand_totals[index]
        previous_soft = self._split_hand_soft[index]
        if previous is not None:
            inferred = infer_card_transition(previous, previous_soft, total)
            phase = source or ("deal" if self._split_hand_decisions[index] == 0 else "hit")
            if inferred is not None:
                self._record_inferred_card(
                    "player",
                    inferred[0],
                    f"split_{'first' if index == 0 else 'second'}_total_{phase}",
                )
                self._split_hand_soft[index] = inferred[1]
            else:
                self._split_hand_soft[index] = soft_after_hit(previous, total, previous_soft)
                self._telemetry.event(
                    "card_inference_miss",
                    round=self._round.get("round") if self._round else None,
                    owner=f"player_{'first' if index == 0 else 'second'}_total",
                    previous=previous,
                    total=total,
                    phase=f"split_{phase}",
                )
        self._split_hand_totals[index] = total
        if self._round is not None and self._split_hand_decisions[index] == 0:
            initial_totals = self._round.setdefault("split_initial_totals", [None, None])
            initial_totals[index] = total

    def _advance_split_hand(self, score: BlackjackScore, reason: str) -> None:
        if not getattr(self, "_split_active", False) or self._split_hand_index != 1:
            return
        closed = getattr(self, "_split_hand_closed", [False, False])
        if len(closed) < 2:
            closed = [False, False]
        closed[1] = True
        self._split_hand_closed = closed
        self._split_hand_index = 0
        self._split_hand_totals[0] = score.player
        self._telemetry.event(
            "split_hand_advanced",
            round=self._round.get("round") if self._round else None,
            from_hand="second_total",
            to_hand="first_total",
            reason=reason,
            first_total=score.player,
            second_total=score.player_second,
            dealer=score.dealer,
        )

    def _split_final_totals(self, score: BlackjackScore) -> tuple[int | None, int | None]:
        """Return split totals with closed hands preferred over raw OCR."""
        if not getattr(self, "_split_active", False):
            return score.player, score.player_second
        closed = getattr(self, "_split_hand_closed", [False, False])
        first = (
            self._split_hand_totals[0]
            if closed[0] and self._split_hand_totals[0] is not None
            else score.player
        )
        second = (
            self._split_hand_totals[1]
            if closed[1] and self._split_hand_totals[1] is not None
            else score.player_second
        )
        return first, second

    def _region(self, name: str, fallback: list[int]) -> list[int]:
        # GameCapture owns all scaling. Returning config-space coordinates here
        # avoids scaling the same rectangle twice on non-1920x1080 desktops.
        return list(self._config.get("regions", {}).get(name, fallback))

    @staticmethod
    def _outcome(player: int, dealer: int) -> str:
        if player > 21:
            return "loss"
        if dealer > 21:
            return "win"
        if player > dealer:
            return "win"
        if player < dealer:
            return "loss"
        return "push"

    @staticmethod
    def _result_notice(crop: np.ndarray) -> str | None:
        """Detect the green natural-blackjack payout notification."""
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.array([35, 120, 120], dtype=np.uint8),
            np.array([95, 255, 255], dtype=np.uint8),
        )
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(green)
        areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)]
        # Recorded notices contain a ~16x16 green check and a tall green
        # accent line. Requiring both enough total color and one coherent
        # component rejects isolated green pixels from the table felt.
        if sum(areas) >= 180 and max(areas, default=0) >= 90:
            return "blackjack_win"
        return None

    def _begin_round(self, score: BlackjackScore) -> None:
        if self._round is not None:
            return
        number = int(self._stats["hands"]) + 1
        self._stats["hands"] = number
        self._round = {
            "round": number,
            "started_at": BlackjackSessionTelemetry.timestamp(),
            "bet_amount": self._bet_amount,
            "initial_player": score.player,
            "dealer_up": score.dealer,
            "decisions": [],
            "stake_multiplier": 1,
            "player_cards_inferred": [],
            "dealer_cards_inferred": [],
            "dealer_totals_observed": [],
            "running_count_before": int(self._stats.get("running_count", 0)),
        }
        self._pending_final = None
        self._telemetry.event("round_started", **self._round)

    def _update_rates(self) -> None:
        wagered = float(self._stats.get("accounted_wagered", 0))
        resolved = int(self._stats.get("resolved", 0))
        self._stats["roi_pct"] = (
            round(float(self._stats["net"]) / wagered * 100, 2) if wagered else 0.0
        )
        self._stats["win_rate_pct"] = (
            round(int(self._stats["wins"]) / resolved * 100, 2) if resolved else 0.0
        )
        net = int(self._stats["net"])
        self._stats["peak_net"] = max(int(self._stats.get("peak_net", 0)), net)
        drawdown = int(self._stats["peak_net"]) - net
        self._stats["max_drawdown"] = max(int(self._stats.get("max_drawdown", 0)), drawdown)

    def _risk_allows_bet(self) -> bool:
        if self._risk_stop_reason is not None:
            return False
        net = int(self._stats["net"])
        drawdown = int(self._stats.get("peak_net", 0)) - net
        # Do not enter a hand that cannot afford an optimal double later.
        # Falling back from DOUBLE to HIT solely because the session is close
        # to a cap lowers expected value.  Reserving the full possible stake
        # keeps every started hand on the configured strategy while preserving
        # the exact same loss/drawdown limits.
        possible_hands = 2 if getattr(self, "_split_enabled", False) else 1
        possible_multiplier = 2 if self._double_policy != "never" else 1
        prospective_loss = self._bet_amount * possible_hands * possible_multiplier
        reason: str | None = None
        if self._max_session_loss and net - prospective_loss < -self._max_session_loss:
            reason = f"лимит убытка {self._max_session_loss}"
        elif self._max_drawdown and drawdown + prospective_loss > self._max_drawdown:
            reason = f"лимит просадки {self._max_drawdown}"
        elif self._max_hands and int(self._stats["hands"]) >= self._max_hands:
            reason = f"лимит раздач {self._max_hands}"
        if reason is None:
            return True
        self._stop_for_risk(reason)
        return False

    def _stop_for_risk(self, reason: str) -> None:
        if self._risk_stop_reason is not None:
            return
        net = int(self._stats["net"])
        drawdown = int(self._stats.get("peak_net", 0)) - net
        self._risk_stop_reason = reason
        self._stats["risk_stopped"] = 1
        self._telemetry.event(
            "risk_limit_reached",
            reason=reason,
            net=net,
            peak_net=int(self._stats.get("peak_net", 0)),
            drawdown=drawdown,
        )
        self._telemetry.update_summary(self.stats)
        print(
            f"[21] Новые ставки остановлены: {reason}. Итог {net:+d}, просадка {drawdown}.",
            flush=True,
        )

    def _enforce_reached_risk_limits(self) -> None:
        """Stop immediately when a completed hand reaches a configured cap."""
        if self._risk_stop_reason is not None:
            return
        net = int(self._stats["net"])
        drawdown = int(self._stats.get("peak_net", 0)) - net
        reason: str | None = None
        if self._max_session_loss and net <= -self._max_session_loss:
            reason = f"лимит убытка {self._max_session_loss}"
        elif self._max_drawdown and drawdown >= self._max_drawdown:
            reason = f"лимит просадки {self._max_drawdown}"
        elif self._max_hands and int(self._stats["completed"]) >= self._max_hands:
            reason = f"лимит раздач {self._max_hands}"
        if reason is not None:
            self._stop_for_risk(reason)

    def _double_within_risk(self) -> bool:
        """Whether a full doubled-hand loss stays inside session limits."""
        net = int(self._stats["net"])
        drawdown = int(self._stats.get("peak_net", 0)) - net
        # A split can expose two independently doubled stakes. New-hand risk
        # admission reserved this full amount so an optimal play is never
        # downgraded halfway through the round.
        doubled_stake = self._bet_amount * (4 if getattr(self, "_split_active", False) else 2)
        if self._max_session_loss and net - doubled_stake < -self._max_session_loss:
            return False
        if self._max_drawdown and drawdown + doubled_stake > self._max_drawdown:
            return False
        return True

    def _finish_round(self) -> None:
        if self._round is None:
            return
        pending = self._pending_final
        if pending is None:
            player, dealer, player_second = 0, 0, None
        elif len(pending) == 3:
            player, dealer, player_second = pending
        else:
            player, dealer = pending
            player_second = None
        notice = str(self._round.get("result_notice", ""))
        natural_notice = notice == "blackjack_win" and not self._round.get("split_confirmed")
        self._initialize_terminal_card_tracking(
            player=21 if natural_notice else player,
            dealer=dealer,
        )
        # A natural can be identified from the inferred opening cards even
        # when the green result toast was missed or the final score was
        # already captured.  This fixes the common A+10 settlement being
        # recorded as an ordinary 1:1 win.
        inferred_player_cards = self._round.get("player_cards_inferred", [])
        natural_from_cards = (
            not self._round.get("split_confirmed")
            and isinstance(inferred_player_cards, list)
            and len(inferred_player_cards) == 2
            and sorted(int(card) for card in inferred_player_cards) == [10, 11]
        )
        natural_blackjack = natural_notice or natural_from_cards
        if natural_blackjack:
            player = 21
        is_split = player_second is not None or bool(self._round.get("split_confirmed"))
        if natural_blackjack:
            # A dealer natural pushes a player natural; otherwise the table
            # pays 3:2.  The dealer score may be unavailable for a toast-only
            # result, in which case the existing natural-win behaviour is
            # retained.
            multiplier = int(self._round.get("stake_multiplier", 1))
            stake = self._bet_amount * multiplier
            if dealer == 21:
                outcomes = ["push"]
                outcome = "push"
                net = 0
            else:
                outcomes = ["win"]
                outcome = "win"
                net = int(round(stake * 1.5))
            hand_nets = [net]
        elif player_second is not None:
            outcomes = [
                self._outcome(player, dealer),
                self._outcome(player_second, dealer),
            ]
            multipliers = list(self._round.get("split_stake_multipliers", [1, 1]))[:2]
            while len(multipliers) < 2:
                multipliers.append(1)
            stakes = [self._bet_amount * max(1, int(value)) for value in multipliers]
            hand_nets = [
                stake if hand_outcome == "win" else -stake if hand_outcome == "loss" else 0
                for hand_outcome, stake in zip(outcomes, stakes)
            ]
            stake = sum(stakes)
            net = sum(hand_nets)
            outcome = outcomes[0] if outcomes[0] == outcomes[1] else "mixed"
        elif pending is not None:
            outcomes = [self._outcome(player, dealer)]
            multiplier = int(self._round.get("stake_multiplier", 1))
            stake = self._bet_amount * multiplier
            net = stake if outcomes[0] == "win" else -stake if outcomes[0] == "loss" else 0
            hand_nets = [net]
            outcome = outcomes[0]
        elif natural_blackjack:
            outcomes = ["win"]
            outcome = "win"
            multiplier = int(self._round.get("stake_multiplier", 1))
            stake = self._bet_amount * multiplier
            net = int(round(stake * 1.5))
            hand_nets = [net]
        elif is_split:
            outcomes = ["unknown", "unknown"]
            outcome = "unknown"
            multipliers = list(self._round.get("split_stake_multipliers", [1, 1]))[:2]
            while len(multipliers) < 2:
                multipliers.append(1)
            stake = sum(self._bet_amount * max(1, int(value)) for value in multipliers)
            net = 0
            hand_nets = [0, 0]
        else:
            outcomes = ["unknown"]
            outcome = "unknown"
            multiplier = int(self._round.get("stake_multiplier", 1))
            stake = self._bet_amount * multiplier
            net = 0
            hand_nets = [0]
        self._stats["completed"] = int(self._stats["completed"]) + 1
        self._stats["wagered"] = int(self._stats["wagered"]) + stake
        if is_split:
            self._stats["split_hands"] = int(self._stats.get("split_hands", 0)) + 2
        if "unknown" not in outcomes:
            self._stats["resolved"] = int(self._stats.get("resolved", 0)) + len(outcomes)
            self._stats["accounted_wagered"] = int(self._stats.get("accounted_wagered", 0)) + stake
            self._stats["net"] = int(self._stats["net"]) + net
        for hand_outcome in outcomes:
            key = {
                "win": "wins",
                "loss": "losses",
                "push": "pushes",
                "unknown": "unknown",
            }[hand_outcome]
            self._stats[key] = int(self._stats[key]) + 1
        self._update_rates()
        completed = {
            **self._round,
            "completed_at": BlackjackSessionTelemetry.timestamp(),
            "final_player": player or None,
            "final_player_second": player_second,
            "final_dealer": dealer or None,
            "outcome": outcome,
            "hand_outcomes": outcomes,
            "hand_nets": hand_nets,
            "natural_blackjack": natural_blackjack,
            "stake": stake,
            "net": net,
            "running_count_after": int(self._stats.get("running_count", 0)),
        }
        self._telemetry.record_round(completed, self.stats)
        self._enforce_reached_risk_limits()
        labels = {
            "win": "ПОБЕДА",
            "loss": "ПОРАЖЕНИЕ",
            "push": "НИЧЬЯ",
            "mixed": "РАЗНЫЙ ИТОГ РУК",
            "unknown": "БЕЗ РЕЗУЛЬТАТА",
        }
        player_label = (
            f"{player}/{player_second}" if player_second is not None else str(player or "?")
        )
        print(
            f"[21] Раунд {completed['round']}: {labels[outcome]} "
            f"({player_label}:{dealer or '?'}, итог {net:+d}, ROI {self._stats['roi_pct']}%)",
            flush=True,
        )
        self._round = None
        self._pending_final = None
        self._reset_split_state()

    def _initialize_terminal_card_tracking(self, *, player: int, dealer: int) -> None:
        """Recover visible cards for hands that never offered an action."""
        if self._round is None or self._round.get("card_tracking_started"):
            return
        self._round["card_tracking_started"] = True

        dealer_up = int(self._round.get("dealer_up", 0) or 0)
        if dealer_up in CARD_VALUES:
            self._round["dealer_totals_observed"] = [dealer_up]
            self._record_inferred_card("dealer", dealer_up, "up_card_terminal")

        first = getattr(self, "_deal_first_player", None)
        if first is None:
            initial = int(self._round.get("initial_player", 0) or 0)
            first = initial if initial in CARD_VALUES else None
        if first is not None and 2 <= player <= 31:
            first_total, first_soft = (11, True) if first == 11 else (first, False)
            second = infer_card_transition(first_total, first_soft, player)
            if second is not None:
                self._record_inferred_card("player", first, "first_dealt_terminal")
                self._record_inferred_card("player", second[0], "second_card_terminal")

        # A terminal 21 against an ace/ten is a dealer blackjack and therefore
        # exposes exactly one hole card.  Other final totals can contain
        # several draws and cannot be reconstructed safely from one jump.
        if dealer == 21 and dealer_up in (10, 11):
            hole = 11 if dealer_up == 10 else 10
            self._record_inferred_card("dealer", hole, "blackjack_hole_terminal")

    def _record_inferred_card(self, owner: str, card: int, source: str) -> None:
        if self._round is None or card not in CARD_VALUES:
            return
        key = f"{owner}_cards_inferred"
        cards = self._round.setdefault(key, [])
        cards.append(card)
        # The observed 3x-ten effective-48 mix makes standard Hi-Lo drift
        # upward even for independent cards.  Keep a mix-balanced diagnostic
        # count instead; it is deliberately never used to size wagers.
        count_delta = majestic_balanced_count_value(card)
        self._stats["cards_inferred"] = int(self._stats["cards_inferred"]) + 1
        self._stats["running_count"] = int(self._stats["running_count"]) + count_delta
        self._telemetry.event(
            "card_inferred",
            round=self._round.get("round"),
            owner=owner,
            value=card,
            source=source,
            count_delta=count_delta,
            running_count=int(self._stats["running_count"]),
        )

    def _initialize_card_tracking(self, score: BlackjackScore) -> None:
        if self._round is None or self._round.get("card_tracking_started"):
            return
        self._round["card_tracking_started"] = True
        self._dealer_total = score.dealer
        self._dealer_soft = score.dealer == 11
        self._round["dealer_totals_observed"] = [score.dealer]
        self._record_inferred_card("dealer", score.dealer, "up_card")

        first = self._deal_first_player
        if first is None:
            return
        first_total, first_soft = (11, True) if first == 11 else (first, False)
        second = infer_card_transition(first_total, first_soft, score.player)
        if second is None:
            self._telemetry.event(
                "card_inference_miss",
                round=self._round.get("round"),
                owner="player",
                previous=first_total,
                total=score.player,
                phase="initial",
            )
            return
        self._record_inferred_card("player", first, "first_dealt_total")
        self._record_inferred_card("player", second[0], "second_card")

    def _apply_player_transition(self, total: int, *, source: str = "hit") -> None:
        if self._last_player_total is None:
            return
        previous_total = self._last_player_total
        previous_soft = self._hand_soft
        inferred = infer_card_transition(previous_total, previous_soft, total)
        if inferred is not None:
            card, new_soft = inferred
            self._record_inferred_card("player", card, source)
            self._hand_soft = new_soft
        else:
            self._hand_soft = soft_after_hit(previous_total, total, previous_soft)
            self._telemetry.event(
                "card_inference_miss",
                round=self._round.get("round") if self._round else None,
                owner="player",
                previous=previous_total,
                total=total,
                phase=source,
            )
        self._last_player_total = total

    def _record_terminal_player_transition(self, score: BlackjackScore) -> None:
        """Record the last HIT/DOUBLE when the action panel never returns.

        A bust, automatic 21, or double immediately hands control to the
        dealer.  Earlier versions only inferred the new card after the player
        prompt returned, which systematically omitted these terminal cards.
        The caller invokes this once on a confirmed final frame.
        """
        action = self._last_pressed_action
        if self._pending_final is not None or action not in (HIT, DOUBLE):
            return
        source = f"terminal_{action}"
        if self._split_active:
            index = self._split_hand_index
            total = score.player if index == 0 else score.player_second
            if total is None or self._split_hand_totals[index] is None:
                return
            previous = self._split_hand_totals[index]
            previous_soft = self._split_hand_soft[index]
            if total == previous and not infer_card_transition(previous, previous_soft, total):
                self._telemetry.event(
                    "action_transition_missing",
                    round=self._round.get("round") if self._round else None,
                    player=total,
                    dealer=score.dealer,
                    action=action,
                    split_hand=index,
                    previous=previous,
                )
                if not getattr(self, "_action_retry_exhausted", False):
                    self._stats["action_unacknowledged"] = (
                        int(self._stats.get("action_unacknowledged", 0)) + 1
                    )
                self._last_pressed_action = None
                self._last_action_key = None
                self._action_gap_after_press = False
                self._action_missing_frames = 0
                return
            self._apply_split_transition(score, index, source=source)
        else:
            if self._last_player_total is None:
                return
            previous = self._last_player_total
            if score.player == previous and not infer_card_transition(
                previous, self._hand_soft, score.player
            ):
                self._telemetry.event(
                    "action_transition_missing",
                    round=self._round.get("round") if self._round else None,
                    player=score.player,
                    dealer=score.dealer,
                    action=action,
                    previous=previous,
                )
                if not getattr(self, "_action_retry_exhausted", False):
                    self._stats["action_unacknowledged"] = (
                        int(self._stats.get("action_unacknowledged", 0)) + 1
                    )
                self._last_pressed_action = None
                self._last_action_key = None
                self._action_gap_after_press = False
                self._action_missing_frames = 0
                return
            self._apply_player_transition(score.player, source=source)
        # There cannot be another player decision in this round. Clearing the
        # acknowledgement state also prevents duplicate inference while the
        # result remains visible for several frames.
        self._last_pressed_action = None
        self._last_action_key = None
        self._action_gap_after_press = False
        self._action_missing_frames = 0

    def _player_transition_acknowledged(self, score: BlackjackScore) -> bool:
        """Whether the last HIT/DOUBLE produced a defensible new total."""
        action = self._last_pressed_action
        if action not in (HIT, DOUBLE):
            return True
        if self._split_active:
            index = self._split_hand_index
            total = score.player if index == 0 else score.player_second
            previous = self._split_hand_totals[index]
            if total is None or previous is None:
                return False
            if total != previous:
                return True
            return infer_card_transition(previous, self._split_hand_soft[index], total) is not None
        if self._last_player_total is None:
            return False
        if score.player != self._last_player_total:
            return True
        return (
            infer_card_transition(self._last_player_total, self._hand_soft, score.player)
            is not None
        )

    def _observe_dealer_total(self, total: int) -> None:
        if self._round is None or total <= 0:
            return
        observed = self._round.setdefault("dealer_totals_observed", [])
        if not observed or observed[-1] != total:
            observed.append(total)
        if self._dealer_total is None:
            self._dealer_total = total
            self._dealer_soft = total == 11
            return
        if total == self._dealer_total:
            return
        previous = self._dealer_total
        inferred = infer_card_transition(previous, self._dealer_soft, total)
        if inferred is not None:
            card, self._dealer_soft = inferred
            self._record_inferred_card("dealer", card, "hole_or_draw")
        else:
            self._telemetry.event(
                "dealer_total_jump",
                round=self._round.get("round"),
                previous=previous,
                total=total,
            )
            self._dealer_soft = False
        self._dealer_total = total

    def _save_event_screenshot(
        self,
        frame: np.ndarray,
        event: str,
        score: BlackjackScore | None = None,
    ) -> None:
        if (
            not self._save_event_screenshots
            or self._event_screenshot_count >= self._max_event_screenshots
        ):
            return
        state = (
            event,
            score.player if score else -1,
            score.player_second if score and score.player_second is not None else -1,
            score.dealer if score else -1,
        )
        if state == self._last_saved_state:
            return
        self._last_saved_state = state
        self._event_screenshot_count += 1
        self._screenshot_directory.mkdir(parents=True, exist_ok=True)
        # Event-only capture includes the physical cards on the table. The
        # fast polling frame intentionally covers only UI regions.
        debug_region = self._region("blackjack_debug", [620, 340, 1300, 740])
        try:
            crop = self._capture.grab_region(debug_region)
        except Exception:
            # Retain at least the already captured UI if an extra BitBlt grab
            # is unavailable in exclusive fullscreen mode.
            crop = self._capture.crop_from_frame(frame, debug_region)
        round_number = self._round.get("round", 0) if self._round else 0
        totals = ""
        if score is not None:
            second = f"_{score.player_second}" if score.player_second is not None else ""
            totals = f"_{score.player}{second}_{score.dealer}"
        filename = f"{self._event_screenshot_count:03d}_r{round_number:03d}_{event}{totals}.jpg"
        cv2.imwrite(str(self._screenshot_directory / filename), crop)
        self._telemetry.event(
            "screenshot_saved",
            event_name=event,
            path=str(self._screenshot_directory / filename),
        )

    def _press_action(self, action: str, score: BlackjackScore) -> bool:
        if self._input.press_key_named(self._keys[action]):
            self._last_input_blocked_key = None
            return True
        round_number = self._round.get("round") if self._round else None
        active_total = self._active_player_total(score)
        blocked_key = (round_number, active_total, score.dealer, action)
        if blocked_key != self._last_input_blocked_key:
            self._last_input_blocked_key = blocked_key
            self._stats["input_blocks"] = int(self._stats.get("input_blocks", 0)) + 1
            foreground = str(getattr(self._input, "last_foreground_identity", ""))
            self._telemetry.event(
                "input_blocked",
                round=round_number,
                player=active_total,
                split_hand=(
                    "second_total"
                    if getattr(self, "_split_active", False) and self._split_hand_index == 1
                    else "first_total"
                    if getattr(self, "_split_active", False)
                    else None
                ),
                dealer=score.dealer,
                action=action,
                foreground=foreground,
            )
            print(
                f"[21] Ход {action} не отправлен: окно GTA не получило фокус "
                f"({foreground or 'окно неизвестно'}).",
                flush=True,
            )
        return False

    def _retry_pending_action(
        self,
        action: str,
        score: BlackjackScore,
        now: float,
    ) -> bool:
        """Retry a press when the unchanged action prompt never acknowledged it."""
        key = self._decision_key(score)
        if self._last_action_key != key:
            return False
        if now - self._last_action_at < self._action_retry_sec:
            return True
        if self._action_retry_count >= self._max_action_retries:
            if not self._action_retry_exhausted:
                self._action_retry_exhausted = True
                self._stats["action_unacknowledged"] = (
                    int(self._stats.get("action_unacknowledged", 0)) + 1
                )
                self._telemetry.event(
                    "action_unacknowledged",
                    round=self._round.get("round") if self._round else None,
                    player=self._active_player_total(score),
                    dealer=score.dealer,
                    action=action,
                    retries=self._action_retry_count,
                )
            return True
        if self._press_action(action, score):
            self._action_retry_count += 1
            self._last_action_at = now
            self._stats["action_retries"] = int(self._stats.get("action_retries", 0)) + 1
            self._telemetry.event(
                "action_retried",
                round=self._round.get("round") if self._round else None,
                player=self._active_player_total(score),
                dealer=score.dealer,
                action=action,
                retry=self._action_retry_count,
            )
            print(
                f"[21] Повторяю {action}: игровая панель не подтвердила прошлое нажатие "
                f"({self._action_retry_count}/{self._max_action_retries}).",
                flush=True,
            )
        return True

    def _handle_bet_screen(
        self,
        allow_action: bool,
        bet_value: tuple[int, float] | None,
        frame: np.ndarray | None = None,
    ) -> None:
        if self._round is not None and self._pending_final is None and frame is not None:
            notice_region = self._region("result_notice", [740, 980, 450, 85])
            notice_crop = self._capture.crop_from_frame(frame, notice_region)
            notice = self._result_notice(notice_crop)
            if notice is not None:
                self._round["result_notice"] = notice
                self._telemetry.event(
                    "result_notice_detected",
                    round=self._round.get("round"),
                    result=notice,
                )
            self._save_event_screenshot(frame, "unknown_result")
        self._finish_round()
        self._reset_split_state()
        self._seen = _Seen()
        self._last_action_key = None
        self._last_message = None
        self._round_initial_key = None
        self._last_pressed_action = None
        self._last_input_blocked_key = None
        self._action_retry_count = 0
        self._action_retry_exhausted = False
        self._action_missing_frames = 0
        self._action_gap_after_press = False
        self._bet_seen += 1
        self._bet_missing = 0
        # Once risk protection has stopped wagering, keep the process idle.
        # Reading the same rectangle would interpret the reward timer as a
        # constantly changing bet value and flood telemetry indefinitely.
        if self._risk_stop_reason is not None:
            return
        if bet_value is not None:
            value, confidence = bet_value
            if value != self._last_bet_value:
                self._telemetry.event(
                    "bet_value_observed", value=value, confidence=round(confidence, 3)
                )
                self._last_bet_value = value
            if value > 0:
                self._bet_confirmed = True
                self._zero_bet_seen = 0
                self._bet_retry_count = 0
                self._bet_retry_exhausted = False
                return
            self._zero_bet_seen += 1
        else:
            # Never retry blindly: an unreadable non-zero value could turn a
            # valid wager into an accidental second wager.
            self._zero_bet_seen = 0

        retry_due = (
            self._bet_placed
            and not self._bet_confirmed
            and self._zero_bet_seen >= self._bet_confirm_frames
            and time.perf_counter() - self._last_bet_at >= self._bet_retry_sec
        )
        if retry_due and self._bet_retry_count >= self._max_bet_retries:
            if not self._bet_retry_exhausted:
                self._bet_retry_exhausted = True
                self._stats["bet_unacknowledged"] = (
                    int(self._stats.get("bet_unacknowledged", 0)) + 1
                )
                self._telemetry.event(
                    "bet_unacknowledged",
                    amount=self._bet_amount,
                    retries=self._bet_retry_count,
                    observed_value=0,
                )
                print(
                    f"[21] Ставка {self._bet_amount} не подтверждена после "
                    f"{self._bet_retry_count} повторов. Автоставки приостановлены "
                    "до начала раздачи или перезапуска бота.",
                    flush=True,
                )
            return
        if (
            not self._auto_bet
            or not allow_action
            or not self._risk_allows_bet()
            or (self._bet_placed and not retry_due)
            or self._bet_seen < self._bet_confirm_frames
            or time.perf_counter() - self._last_bet_at < self._cooldown
        ):
            return
        if self._input.press_key_named(self._bet_key):
            self._bet_placed = True
            self._awaiting_round_after_bet = True
            self._deal_clear_frames = 0
            self._deal_first_player = None
            self._hand_soft = False
            self._last_player_total = None
            self._dealer_total = None
            self._dealer_soft = False
            self._reset_split_state()
            self._last_pressed_action = None
            self._action_missing_frames = 0
            self._action_gap_after_press = False
            self._last_bet_at = time.perf_counter()
            self._zero_bet_seen = 0
            if retry_due:
                self._bet_retry_count += 1
                self._stats["bet_retries"] = int(self._stats.get("bet_retries", 0)) + 1
            else:
                self._bet_retry_count = 0
                self._bet_retry_exhausted = False
                self._stats["bets"] = int(self._stats["bets"]) + 1
            self._telemetry.event(
                "bet_pressed", amount=self._bet_amount, key=self._bet_key, retry=retry_due
            )
            verb = "повторена" if retry_due else "отправлена"
            print(
                f"[21] Ставка {self._bet_amount}: {verb} клавишей {self._bet_key.upper()}",
                flush=True,
            )

    def _observe_no_bet_screen(self) -> None:
        self._bet_seen = 0
        self._bet_missing += 1
        # A disappearing panel is part of the dealing animation. It must not
        # re-arm a fresh bet until at least one score from that wager was seen.
        if self._bet_missing >= self._bet_confirm_frames and not self._awaiting_round_after_bet:
            self._bet_placed = False
            self._bet_confirmed = False
            self._zero_bet_seen = 0
            self._last_bet_value = None
            self._bet_retry_count = 0
            self._bet_retry_exhausted = False

    def _accept_pending_bet(self, score: BlackjackScore) -> None:
        """Arm a new hand after the first score from an accepted wager.

        The blank dealing animation increments ``_bet_missing``.  Resetting it
        here is essential for hands that never show the action panel (natural
        blackjack or dealer blackjack); otherwise the first score immediately
        clears ``_bet_placed`` and the whole hand is omitted from telemetry.
        """
        self._awaiting_round_after_bet = False
        self._deal_clear_frames = 0
        self._bet_missing = 0
        self._bet_retry_count = 0
        self._bet_retry_exhausted = False
        self._deal_first_player = score.player if 1 <= score.player <= 11 else None
        self._telemetry.event("bet_accepted", player=score.player, dealer=score.dealer)

    def tick(self, allow_action: bool = True) -> BlackjackScore | None:
        observation = self._read_table()
        if observation is None:
            return None
        frame = observation.frame
        score = observation.score
        action_available = observation.action_available
        bet_available = observation.bet_available
        bet_value = observation.bet_value
        if not action_available and self._last_action_key is not None:
            self._action_missing_frames += 1
            if self._action_missing_frames >= self._confirm_frames:
                self._action_gap_after_press = True
        if score is None:
            self._seen = _Seen()
            if bet_available:
                self._handle_bet_screen(allow_action, bet_value, frame)
            else:
                if self._awaiting_round_after_bet:
                    self._deal_clear_frames += 1
                self._observe_no_bet_screen()
            now = time.perf_counter()
            if self._missing_since is None:
                self._missing_since = now
            elif now - self._missing_since >= self._round_reset_sec:
                self._last_action_key = None
                self._last_message = None
                self._round_initial_key = None
            self._stats["ocr_misses"] += 1
            return None

        self._missing_since = None
        if self._awaiting_round_after_bet:
            # The previous result can remain visible briefly after Enter. A
            # genuine new deal always passes through an unreadable/0:0 clear
            # phase, so never turn stale totals into a second ghost round.
            if self._deal_clear_frames < self._bet_confirm_frames:
                self._seen = _Seen()
                return score
            self._accept_pending_bet(score)
        self._observe_no_bet_screen()
        if self._split_pending and score.player_second is not None:
            self._activate_split(score)
        if not action_available:
            self._observe_inactive_turn(score, frame, allow_action)
            return score

        prepared = self._prepare_player_turn(score, frame, allow_action)
        if prepared is None:
            return score
        key, active_total = prepared
        recommendation = self._recommend_action(score, key, active_total)
        self._report_decision(score, active_total, recommendation)

        self._execute_decision(score, key, active_total, recommendation, allow_action)
        return score

    def _read_table(self) -> _TableObservation | None:
        """Capture once, then derive all scores and panel observations."""
        score_region = self._region("score", [760, 735, 430, 110])
        action_region = self._region("action_prompt", [1575, 1015, 345, 65])
        bet_region = self._region("bet_panel", [1640, 935, 270, 48])
        bet_value_region = self._region("bet_value", [1810, 978, 90, 48])
        result_notice_region = self._region("result_notice", [740, 980, 450, 85])
        try:
            frame = self._capture.grab_union(
                [
                    score_region,
                    action_region,
                    bet_region,
                    bet_value_region,
                    result_notice_region,
                ]
            )
        except Exception as exc:
            self._stats["capture_errors"] += 1
            now = time.perf_counter()
            if now - self._last_capture_error_at >= 2.0:
                print(f"[21] Ошибка захвата экрана, повторяю: {exc}", flush=True)
                self._last_capture_error_at = now
            return None
        crop = self._capture.crop_from_frame(frame, score_region)
        action_crop = self._capture.crop_from_frame(frame, action_region)
        bet_crop = self._capture.crop_from_frame(frame, bet_region)
        bet_value_crop = self._capture.crop_from_frame(frame, bet_value_region)
        if self._config.get("debug", {}).get("save_live_crop", True):
            self._debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self._debug_path), crop)
        score = self._detector.detect(crop)
        bet_gray = cv2.cvtColor(bet_crop, cv2.COLOR_BGR2GRAY)
        bet_bright_pixels = int(np.count_nonzero(bet_gray >= 200))
        # The always-visible white food button overlaps this rectangle and has
        # ~3,500 bright pixels. The real wager panel is dark and contains only
        # its light text, so both a lower and upper bound are required.
        bet_available = (
            self._min_bet_bright_pixels <= bet_bright_pixels <= self._max_bet_bright_pixels
        )
        bet_value = self._detector.detect_number(
            bet_value_crop,
            maximum=max(self._bet_amount * 100, 10_000),
            min_confidence=float(
                self._config.get("blackjack", {}).get("min_bet_ocr_confidence", 0.58)
            ),
        )
        action_gray = cv2.cvtColor(action_crop, cv2.COLOR_BGR2GRAY)
        action_available = int(np.count_nonzero(action_gray >= 200)) >= int(
            self._config.get("blackjack", {}).get("min_action_bright_pixels", 80)
        )
        return _TableObservation(frame, score, action_available, bet_available, bet_value)

    def _observe_inactive_turn(
        self, score: BlackjackScore, frame: np.ndarray, allow_action: bool
    ) -> None:
        """Track dealing and settlement while the player prompt is absent."""
        # Totals remain on screen while the dealer draws and while the
        # result is shown. Those are observations, never player turns.
        self._seen = _Seen()
        # A natural 21 (or dealer blackjack) may skip the action panel
        # completely. A just-confirmed bet proves this is a new hand.
        if self._round is None and self._bet_placed:
            self._begin_round(score)
        # A split transition can briefly hide the action prompt.  Advance
        # the internal hand pointer as soon as the completed second hand
        # is no longer actionable; if the prompt returns on the next
        # frame, the first hand will be decided normally.  This avoids
        # treating the dealer animation as proof that the first hand was
        # already played.
        if (
            self._round is not None
            and self._split_active
            and self._split_hand_index == 1
            and self._action_gap_after_press
            and self._last_pressed_action in (HIT, STAND, DOUBLE)
            and score.dealer <= 11
            and (
                self._last_pressed_action in (STAND, DOUBLE)
                or score.player_second is not None
                and score.player_second >= 21
            )
        ):
            split_transition_action = self._last_pressed_action
            if split_transition_action in (HIT, DOUBLE):
                # Record the card while the second hand is still active;
                # switching the pointer first would attribute it to the
                # first hand.
                self._record_terminal_player_transition(score)
            self._advance_split_hand(
                score,
                f"{split_transition_action.lower()}_prompt_missing",
            )
        # If the panel disappeared without a defensible total change,
        # retry the same action while the dealer is still showing only
        # the up-card.  This catches the three observed terminal HIT
        # misses without sending a blind click once dealer draws begin.
        if (
            self._auto_action
            and allow_action
            and self._round is not None
            and self._action_gap_after_press
            and self._last_pressed_action in (HIT, DOUBLE)
            and not self._action_retry_exhausted
            and not self._player_transition_acknowledged(score)
            and 2 <= score.dealer <= 11
            and 2 <= score.player <= 21
        ):
            self._retry_pending_action(
                self._last_pressed_action,
                score,
                time.perf_counter(),
            )
        # Dealer must draw to 17. Lower totals are animation frames, not a
        # completed result. During a split, Majestic plays the second
        # displayed total first; the first total is the final active hand.
        player_finished_by_bust = self._active_player_total(score) > 21
        round_finished = player_finished_by_bust or score.dealer >= 17
        if self._round is not None and round_finished:
            self._record_terminal_player_transition(score)
        if self._round is not None and self._round.get("card_tracking_started"):
            self._observe_dealer_total(score.dealer)
        if self._round is not None and score.player > 0 and score.dealer > 0 and round_finished:
            if self._split_active and score.player_second is not None:
                final_player, final_second = self._split_final_totals(score)
                self._pending_final = (
                    final_player if final_player is not None else score.player,
                    score.dealer,
                    final_second if final_second is not None else score.player_second,
                )
            else:
                self._pending_final = (score.player, score.dealer)
            self._save_event_screenshot(frame, "result", score)

    def _prepare_player_turn(
        self, score: BlackjackScore, frame: np.ndarray, allow_action: bool
    ) -> tuple[tuple[int, ...], int] | None:
        """Confirm OCR and reconcile the previous action before choosing a move."""
        # During the player's turn only the dealer's up-card is visible. Any
        # value above 11 is necessarily a transient or an OCR error. Do not
        # consume a completed action cycle on an invalid OCR frame.
        active_total = self._active_player_total(score)
        split_score_valid = not self._split_active or score.player_second is not None
        # A hit can bust the second displayed hand. The game immediately opens
        # the prompt for the first displayed hand, so allow that terminal value
        # through long enough to advance the split state before validating the
        # newly active total.
        split_second_finished_by_hit = (
            self._split_active
            and self._split_hand_index == 1
            and self._action_gap_after_press
            and self._last_pressed_action == HIT
            and score.player_second is not None
            and score.player_second >= 21
        )
        if (
            not (1 <= score.dealer <= 11)
            or not split_score_valid
            or (not (2 <= active_total <= 21) and not split_second_finished_by_hit)
        ):
            self._seen = _Seen()
            return None

        transition_processed = False
        if self._action_gap_after_press:
            if self._split_active:
                index = self._split_hand_index
                if self._last_pressed_action in (HIT, DOUBLE):
                    self._apply_split_transition(score, index)
                    transition_processed = True
                active_split_total = score.player if index == 0 else score.player_second
                if index == 1 and (
                    self._last_pressed_action in (STAND, DOUBLE)
                    or (
                        self._last_pressed_action == HIT
                        and active_split_total is not None
                        and active_split_total >= 21
                    )
                ):
                    self._advance_split_hand(score, self._last_pressed_action or "automatic")
            # The prompt disappeared while a requested card was dealt and is
            # now back. A soft hand can keep the same displayed total when a
            # ten-value card demotes its ace (soft 18 -> hard 18).
            elif self._last_pressed_action == HIT and self._last_player_total is not None:
                self._apply_player_transition(score.player)
                transition_processed = True
            self._last_action_key = None
            self._last_message = None
            self._seen = _Seen()
            self._action_gap_after_press = False
            self._action_retry_count = 0
            self._action_retry_exhausted = False
        self._action_missing_frames = 0

        active_total = self._active_player_total(score)
        if not (2 <= active_total <= 21):
            self._seen = _Seen()
            return None

        # A disappearing prompt after F is the dealing animation. Wait for
        # the three bracketed totals before making any hand decision.
        if self._split_pending and score.player_second is None:
            self._seen = _Seen()
            if self._auto_action and allow_action:
                self._retry_pending_action(SPLIT, score, time.perf_counter())
            return None

        active_total = self._active_player_total(score)
        key = self._decision_key(score)
        if self._round_initial_key is None:
            self._round_initial_key = key
        self._begin_round(score)
        if self._seen.key == key:
            self._seen.count += 1
        else:
            self._seen = _Seen(key, 1)
        if self._seen.count < self._confirm_frames:
            return None

        self._save_event_screenshot(frame, "decision", score)

        if self._split_active:
            index = self._split_hand_index
            if self._split_hand_totals[index] != active_total and not transition_processed:
                self._apply_split_transition(score, index)
        elif self._last_player_total is None:
            first = self._deal_first_player
            self._hand_soft = infer_initial_soft(first, score.player)
            if self._round is not None:
                self._round["initial_soft"] = self._hand_soft
                self._round["first_card_value"] = first
                self._initialize_card_tracking(score)
        elif score.player != self._last_player_total and not transition_processed:
            self._apply_player_transition(score.player)
        if not self._split_active:
            self._last_player_total = score.player

        return key, active_total

    def _recommend_action(
        self, score: BlackjackScore, key: tuple[int, ...], active_total: int
    ) -> _ActionRecommendation:
        double_is_available = (
            self._split_hand_decisions[self._split_hand_index] == 0
            if self._split_active
            else key == self._round_initial_key
        )
        active_soft = self._active_soft()
        policy_allows_double = double_allowed_by_policy(
            self._double_policy,
            active_total,
            soft=active_soft,
        )
        pair_rank = (
            self._initial_pair_rank()
            if self._split_enabled
            and not self._split_active
            and not self._split_pending
            and key == self._round_initial_key
            else None
        )
        split_recommended = pair_rank is not None and should_split_pair(
            pair_rank,
            score.dealer,
            can_double_after_split=self._double_policy != "never",
        )
        unrestricted_action = (
            SPLIT
            if split_recommended
            else basic_strategy(
                active_total,
                score.dealer,
                can_double=double_is_available,
                soft=active_soft,
            )
        )
        policy_blocked = unrestricted_action == DOUBLE and not policy_allows_double
        action = (
            SPLIT
            if split_recommended
            else basic_strategy(
                active_total,
                score.dealer,
                can_double=double_is_available and policy_allows_double,
                soft=active_soft,
            )
        )
        double_blocked = action == DOUBLE and not self._double_within_risk()
        if double_blocked:
            action = basic_strategy(
                active_total,
                score.dealer,
                can_double=False,
                soft=active_soft,
            )
        return _ActionRecommendation(action, active_soft, pair_rank, policy_blocked, double_blocked)

    def _report_decision(
        self, score: BlackjackScore, active_total: int, recommendation: _ActionRecommendation
    ) -> None:
        action = recommendation.action
        active_soft = recommendation.soft
        policy_blocked = recommendation.policy_blocked
        double_blocked = recommendation.double_blocked
        message = (active_total, score.dealer, action)
        if message != self._last_message:
            if policy_blocked:
                self._telemetry.event(
                    "double_blocked_by_policy",
                    round=self._round.get("round") if self._round else None,
                    player=active_total,
                    dealer=score.dealer,
                    soft=active_soft,
                    policy=self._double_policy,
                    fallback=action,
                )
            if double_blocked:
                self._telemetry.event(
                    "double_blocked_by_risk",
                    round=self._round.get("round") if self._round else None,
                    player=active_total,
                    dealer=score.dealer,
                    net=int(self._stats["net"]),
                )
            labels = {
                HIT: "ЕЩЁ",
                STAND: "ДОСТАТОЧНО",
                DOUBLE: "ДАБЛ",
                SPLIT: "СПЛИТ",
            }
            mode = "автонажатие" if self._auto_action else "совет"
            hand_label = (
                "вторая сумма"
                if self._split_active and self._split_hand_index == 1
                else "первая сумма"
                if self._split_active
                else "рука"
            )
            print(
                f"[21] {hand_label} {active_total} против {score.dealer} "
                f"-> {labels[action]} ({mode}, OCR {score.confidence:.0%})",
                flush=True,
            )
            self._last_message = message

    def _execute_decision(
        self,
        score: BlackjackScore,
        key: tuple[int, ...],
        active_total: int,
        recommendation: _ActionRecommendation,
        allow_action: bool,
    ) -> None:
        action = recommendation.action
        active_soft = recommendation.soft
        pair_rank = recommendation.pair_rank
        now = time.perf_counter()
        if not self._auto_action or not allow_action:
            return
        if self._retry_pending_action(action, score, now):
            return
        if now - self._last_action_at < self._cooldown:
            return
        if self._press_action(action, score):
            self._stats["decisions"] += 1
            self._stats[
                {
                    HIT: "hits",
                    STAND: "stands",
                    DOUBLE: "doubles",
                    SPLIT: "splits",
                }[action]
            ] += 1
            self._last_action_key = key
            self._last_action_at = now
            self._last_pressed_action = action
            self._action_retry_count = 0
            self._action_retry_exhausted = False
            self._action_missing_frames = 0
            self._action_gap_after_press = False
            if self._round is not None:
                if action == DOUBLE:
                    if self._split_active:
                        multipliers = self._round.setdefault("split_stake_multipliers", [1, 1])
                        multipliers[self._split_hand_index] = 2
                    else:
                        self._round["stake_multiplier"] = 2
                elif action == SPLIT:
                    self._split_pending = True
                    self._split_pair_rank = pair_rank
                    self._round["split_requested"] = True
                    self._round["split_pair_rank"] = pair_rank
                if self._split_active:
                    self._split_hand_decisions[self._split_hand_index] += 1
                decision = {
                    "timestamp": BlackjackSessionTelemetry.timestamp(),
                    "player": active_total,
                    "player_left": score.player,
                    "player_right": score.player_second,
                    "dealer": score.dealer,
                    "action": action,
                    "soft": active_soft,
                    "split_hand": (
                        "second_total"
                        if self._split_active and self._split_hand_index == 1
                        else "first_total"
                        if self._split_active
                        else None
                    ),
                    "double_policy": self._double_policy,
                }
                self._round["decisions"].append(decision)
                self._telemetry.event("decision", round=self._round["round"], **decision)
        return

    def cleanup(self) -> None:
        try:
            if self._round is not None and self._pending_final is not None:
                self._finish_round()
            elif self._round is not None:
                self._telemetry.event(
                    "round_interrupted",
                    round=self._round.get("round"),
                    decisions=self._round.get("decisions", []),
                )
                self._round = None
            self._telemetry.finish(self.stats)
        finally:
            self._input.release_all_keys()
