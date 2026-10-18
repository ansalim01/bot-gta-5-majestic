"""Round-by-round blackjack telemetry and live session summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bot.config_loader import ROOT_DIR
from bot.telemetry.streams import close_event_stream


class BlackjackSessionTelemetry:
    def __init__(self, config: dict[str, Any]) -> None:
        settings = config.get("telemetry", {})
        self.enabled = bool(settings.get("enabled", True))
        configured = Path(str(settings.get("directory", "logs/blackjack_sessions")))
        self.directory = configured if configured.is_absolute() else ROOT_DIR / configured
        now = datetime.now().astimezone()
        self.session_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.started_at = now
        self.events_path = self.directory / f"{self.session_id}.jsonl"
        self.summary_path = self.directory / f"{self.session_id}_summary.json"
        self._stream = None
        self._rounds: list[dict[str, Any]] = []
        self._strategy: dict[str, Any] = {}
        self._finished = False
        if not self.enabled:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.events_path, "a", encoding="utf-8")
            self.event("session_started")
        except OSError as exc:
            self.enabled = False
            self._close_events()
            print(f"[21] Аналитика отключена: {exc}", flush=True)

    @staticmethod
    def timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def event(self, name: str, **data: Any) -> None:
        if name == "strategy_configured":
            # Keep the exact decision model beside every summary.  Without
            # this, historical v2-v10 sessions are indistinguishable from the
            # effective-48 v11 strategy during aggregate analysis.
            self._strategy = dict(data)
        if not self.enabled or self._stream is None or self._finished:
            return
        record = {"timestamp": self.timestamp(), "event": name, "data": data}
        try:
            self._stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._stream.flush()
        except OSError as exc:
            self.enabled = False
            self._close_events()
            print(f"[21] Ошибка записи аналитики: {exc}", flush=True)

    def record_round(self, round_data: dict[str, Any], stats: dict[str, Any]) -> None:
        item = dict(round_data)
        self._rounds.append(item)
        self.event("round_completed", **item)
        self._write_summary(stats, finished=False)

    def update_summary(self, stats: dict[str, Any]) -> None:
        """Persist live state changes that happen outside round completion."""
        self._write_summary(stats, finished=False)

    def _summary(self, stats: dict[str, Any], finished: bool) -> dict[str, Any]:
        inferred_cards = [
            int(card)
            for round_data in self._rounds
            for key in ("player_cards_inferred", "dealer_cards_inferred")
            for card in round_data.get(key, [])
            if isinstance(card, (int, float)) and 2 <= int(card) <= 11
        ]
        rank_counts = {str(rank): inferred_cards.count(rank) for rank in range(2, 12)}
        positive_count_next = [
            int(round_data["dealer_up"])
            for round_data in self._rounds
            if int(round_data.get("running_count_before", 0)) >= 2
            and isinstance(round_data.get("dealer_up"), (int, float))
            and 2 <= int(round_data["dealer_up"]) <= 11
        ]
        negative_count_next = [
            int(round_data["dealer_up"])
            for round_data in self._rounds
            if int(round_data.get("running_count_before", 0)) <= -2
            and isinstance(round_data.get("dealer_up"), (int, float))
            and 2 <= int(round_data["dealer_up"]) <= 11
        ]

        def negative_value_rate(cards: list[int]) -> float | None:
            if not cards:
                return None
            return round(sum(card in (9, 10, 11) for card in cards) / len(cards), 4)

        positive_rate = negative_value_rate(positive_count_next)
        negative_rate = negative_value_rate(negative_count_next)
        enough_shoe_samples = (
            len(inferred_cards) >= 500
            and len(positive_count_next) >= 25
            and len(negative_count_next) >= 25
        )
        if not enough_shoe_samples:
            shoe_test = "collecting"
        else:
            rate_delta = float(positive_rate) - float(negative_rate)
            if rate_delta >= 0.06:
                shoe_test = "possible_persistent_shoe"
            elif abs(rate_delta) <= 0.03:
                shoe_test = "no_persistence_signal"
            else:
                shoe_test = "inconclusive"
        card_analysis = {
            "status": "ready_for_shoe_test" if enough_shoe_samples else "collecting",
            "cards_inferred": len(inferred_cards),
            "rounds_with_cards": sum(
                1
                for round_data in self._rounds
                if round_data.get("player_cards_inferred")
                or round_data.get("dealer_cards_inferred")
            ),
            "inferred_per_completed_hand": round(len(inferred_cards) / len(self._rounds), 2)
            if self._rounds
            else 0.0,
            "rank_value_counts": rank_counts,
            "running_count": int(stats.get("running_count", 0)),
            "count_system": self._strategy.get("count_system", "unknown_historical_count"),
            "next_card_source": "dealer_up",
            "negative_count_values": [9, 10, 11],
            "shoe_test": shoe_test,
            "positive_count_next_samples": len(positive_count_next),
            "negative_count_next_samples": len(negative_count_next),
            "positive_count_next_negative_value_rate": positive_rate,
            "negative_count_next_negative_value_rate": negative_rate,
            "used_for_betting": bool(self._strategy.get("count_used_for_betting", False)),
            "deck_model_conclusion": "effective_iid_or_frequently_reshuffled",
            "note": (
                "Счёт сбалансирован под наблюдаемую effective-48 смесь и основан только "
                "на однозначно восстановленных картах. История не показала пригодной "
                "для ставок памяти колоды, поэтому счёт остаётся диагностическим."
            ),
        }
        return {
            "schema_version": 2,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "updated_at": self.timestamp(),
            "finished": finished,
            "strategy": dict(self._strategy),
            "stats": dict(stats),
            "rounds": list(self._rounds),
            "card_analysis": card_analysis,
            "events_file": str(self.events_path),
            "accounting_note": (
                "Расчёт основан на настроенном размере ставки и выплате 1:1; "
                "натуральный blackjack учитывается как 3:2 по зелёному уведомлению о выигрыше. "
                "После сплита левая и правая руки учитываются независимо, включая отдельный дабл. "
                "Раздачи без распознанного итога не включаются в net и accounted_wagered."
            ),
        }

    def _write_summary(self, stats: dict[str, Any], finished: bool) -> None:
        if not self.enabled:
            return
        summary = self._summary(stats, finished)
        try:
            content = json.dumps(summary, ensure_ascii=False, indent=2)
            self.summary_path.write_text(content, encoding="utf-8")
            (self.directory / "latest_summary.json").write_text(content, encoding="utf-8")
            text = self._summary_text(summary)
            (self.directory / "latest_summary.txt").write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"[21] Ошибка сохранения отчёта: {exc}", flush=True)

    @staticmethod
    def _summary_text(summary: dict[str, Any]) -> str:
        values = summary["stats"]
        strategy = summary.get("strategy", {})
        strategy_name = strategy.get("strategy_name", "не указана")
        strategy_version = strategy.get("version")
        strategy_label = (
            f"{strategy_name} v{strategy_version}"
            if strategy_version is not None
            else str(strategy_name)
        )
        return (
            f"Блэкджек — сессия {summary['session_id']}\n"
            f"Стратегия: {strategy_label}\n"
            f"Раздач: {values.get('completed', 0)} | "
            f"Побед: {values.get('wins', 0)} | "
            f"Поражений: {values.get('losses', 0)} | "
            f"Ничьих: {values.get('pushes', 0)}\n"
            f"Поставлено: {values.get('wagered', 0)} | "
            f"Итог: {values.get('net', 0)} | ROI: {values.get('roi_pct', 0)}%\n"
            f"{summary['accounting_note']}\n"
        )

    def finish(self, stats: dict[str, Any]) -> None:
        try:
            if not self.enabled or self._finished:
                return
            self.event("session_finished", stats=stats)
            self._write_summary(stats, finished=True)
        finally:
            self._finished = True
            self._close_events()

    def _close_events(self) -> None:
        stream, self._stream = self._stream, None
        close_event_stream(stream)
