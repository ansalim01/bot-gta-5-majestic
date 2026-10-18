"""Lightweight JSONL telemetry and post-session fishing diagnostics."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bot.config_loader import ROOT_DIR
from bot.telemetry.streams import close_event_stream


class FishingSessionTelemetry:
    def __init__(self, config: dict[str, Any]) -> None:
        settings = config.get("telemetry", {})
        self.enabled = bool(settings.get("enabled", True))
        self._sample_interval_sec = max(
            0.25,
            float(settings.get("sample_interval_sec", 2.0)),
        )
        configured_dir = Path(str(settings.get("directory", "logs/fishing_sessions")))
        self.directory = (
            configured_dir if configured_dir.is_absolute() else ROOT_DIR / configured_dir
        )
        now = datetime.now().astimezone()
        self.session_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.started_at = now
        self._started_perf = time.perf_counter()
        self._last_state = ""
        self._last_state_at = self._started_perf
        self._state_durations: dict[str, float] = defaultdict(float)
        self._event_counts: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        self._recent_failures: list[dict[str, Any]] = []
        self._perf_samples: list[tuple[float, float, float]] = []
        self._finished = False
        self._stream = None
        self.events_path = self.directory / f"{self.session_id}.jsonl"
        self.summary_path = self.directory / f"{self.session_id}_summary.json"

        if not self.enabled:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.events_path, "a", encoding="utf-8")
            self.event(
                "session_started",
                mode=config.get("mode", "cast_and_reel"),
                auto_cast=bool(config.get("fishing", {}).get("auto_cast", False)),
                auto_start_activity=bool(
                    config.get("fishing", {}).get("auto_start_activity", True)
                ),
                sustenance_enabled=bool(config.get("fishing", {}).get("sustenance_enabled", True)),
                target_fish_count=max(
                    0,
                    int(config.get("fishing", {}).get("target_fish_count", 0)),
                ),
            )
        except OSError as exc:
            self.enabled = False
            self._close_events()
            self._stream = None
            print(f"[BOT] WARNING: fishing telemetry disabled: {exc}", flush=True)

    @property
    def sample_interval_sec(self) -> float:
        return self._sample_interval_sec

    def event(self, name: str, **data: Any) -> None:
        if not self.enabled or self._stream is None or self._finished:
            return
        self._event_counts[name] += 1
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_sec": round(time.perf_counter() - self._started_perf, 3),
            "event": name,
            "state": self._last_state or None,
            "data": data,
        }
        try:
            self._stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._stream.flush()
        except OSError as exc:
            self.enabled = False
            self._close_events()
            print(f"[BOT] WARNING: telemetry write failed: {exc}", flush=True)

    def failure(self, reason: str, **data: Any) -> None:
        self._failures[reason] += 1
        failure = {
            "reason": reason,
            "elapsed_sec": round(time.perf_counter() - self._started_perf, 3),
            **data,
        }
        self._recent_failures.append(failure)
        self._recent_failures = self._recent_failures[-20:]
        self.event("failure", **failure)

    def observe_state(self, state: str) -> None:
        if not self.enabled or self._finished or state == self._last_state:
            return
        now = time.perf_counter()
        if self._last_state:
            self._state_durations[self._last_state] += now - self._last_state_at
        previous = self._last_state or None
        self._last_state = state
        self._last_state_at = now
        self.event("state_changed", previous=previous, current=state)

    def performance(self, average_ms: float, maximum_ms: float, fps: float) -> None:
        if not self.enabled or self._finished:
            return
        sample = (float(average_ms), float(maximum_ms), float(fps))
        self._perf_samples.append(sample)
        self._perf_samples = self._perf_samples[-1800:]
        self.event(
            "performance",
            average_tick_ms=round(sample[0], 2),
            maximum_tick_ms=round(sample[1], 2),
            fps=round(sample[2], 2),
        )

    def finish(self, stats: dict[str, Any]) -> dict[str, Any] | None:
        try:
            if not self.enabled or self._finished:
                return None
            now_perf = time.perf_counter()
            if self._last_state:
                self._state_durations[self._last_state] += now_perf - self._last_state_at

            ended_at = datetime.now().astimezone()
            duration = max(0.0, now_perf - self._started_perf)
            performance = self._performance_summary()
            conclusions = self._build_conclusions(stats, duration, performance)
            summary = {
                "session_id": self.session_id,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "ended_at": ended_at.isoformat(timespec="seconds"),
                "duration_sec": round(duration, 2),
                "stats": dict(stats),
                "event_counts": dict(self._event_counts),
                "failure_reasons": dict(self._failures),
                "state_durations_sec": {
                    key: round(value, 2) for key, value in self._state_durations.items()
                },
                "performance": performance,
                "recent_failures": list(self._recent_failures),
                "conclusions": conclusions,
                "events_file": str(self.events_path),
            }
            self.event("session_finished", stats=stats, conclusions=conclusions)

            try:
                self.summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                latest_json = self.directory / "latest_summary.json"
                latest_json.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                latest_text = self.directory / "latest_summary.txt"
                latest_text.write_text(
                    self._format_text_summary(summary),
                    encoding="utf-8",
                )
            except OSError as exc:
                print(f"[BOT] WARNING: telemetry summary write failed: {exc}", flush=True)
            return summary
        finally:
            self._finished = True
            self._close_events()

    def _performance_summary(self) -> dict[str, float]:
        if not self._perf_samples:
            return {}
        averages = [sample[0] for sample in self._perf_samples]
        maxima = [sample[1] for sample in self._perf_samples]
        fps_values = [sample[2] for sample in self._perf_samples]
        return {
            "average_tick_ms": round(sum(averages) / len(averages), 2),
            "maximum_tick_ms": round(max(maxima), 2),
            "average_fps": round(sum(fps_values) / len(fps_values), 2),
            "samples": len(self._perf_samples),
        }

    def _build_conclusions(
        self,
        stats: dict[str, Any],
        duration: float,
        performance: dict[str, float],
    ) -> list[dict[str, str]]:
        conclusions: list[dict[str, str]] = []

        def add(level: str, title: str, recommendation: str) -> None:
            conclusions.append(
                {
                    "level": level,
                    "title": title,
                    "recommendation": recommendation,
                }
            )

        if self._failures["setup_start_prompt_timeout"]:
            add(
                "error",
                "Не найдена подсказка начала рыбалки",
                "Проверьте, что персонаж стоит в точке рыбалки, HUD включён, а регион fishing_start_prompt совпадает с экраном.",
            )
        if self._failures["setup_equipment_prompt_timeout"]:
            add(
                "error",
                "Не подтверждён экран снаряжения",
                "Проверьте наличие удочки, лески и наживки, затем откалибруйте fishing_equipment_prompt.",
            )
        retry_limit = self._failures["cast_start_retry_limit_reached"]
        if retry_limit:
            add(
                "error",
                f"Достигнут лимит неудачных запусков заброса: {retry_limit}",
                "Бот безопасно остановлен без движения S/W. Проверьте экран, подойдите к точке рыбалки и возобновите F9 только у стартовой подсказки.",
            )
        focus_losses = self._failures["game_window_not_focused"]
        if focus_losses:
            add(
                "warning",
                f"Ввод приостанавливался из-за потери фокуса GTA: {focus_losses}",
                "Верните GTA на передний план: бот продолжит работу, не отправляя клавиши в другие приложения.",
            )

        inventory_failures = self._failures["sustenance_inventory_not_open"]
        water_missing = self._failures["sustenance_water_not_found"]
        food_missing = self._failures["sustenance_food_not_found"]
        if inventory_failures:
            add(
                "error",
                f"Не открылся инвентарь для еды и воды: {inventory_failures}",
                "Проверьте клавишу sustenance.inventory_key, задержку inventory_open_wait_sec и шаблон inventory_header.",
            )
        if water_missing:
            add(
                "warning",
                f"В инвентаре не найдена вода: {water_missing}",
                "Положите настроенный напиток в инвентарь или обновите шаблоны templates.water.",
            )
        if food_missing:
            add(
                "warning",
                f"В инвентаре не найдена еда: {food_missing}",
                "Положите настроенную еду в инвентарь или обновите шаблоны templates.food.",
            )

        timing_failures = sum(
            self._failures[key]
            for key in (
                "timing_bar_not_seen",
                "timing_bar_disappeared",
                "timing_timeout",
                "timing_ghost_bar",
                "timing_lost",
            )
        )
        if timing_failures:
            add(
                "warning",
                f"Ошибки тайминга заброса: {timing_failures}",
                "Проверьте ROI timing_bar, зелёный HSV-диапазон и press_lead_ms; при скачках уменьшите нагрузку от отладочных скриншотов.",
            )

        bite_timeouts = self._failures["bite_timeout"]
        if bite_timeouts:
            add(
                "warning",
                f"Потеряно ожидание поклёвки: {bite_timeouts}",
                "Проверьте шаблоны ожидания и поклёвки, а также bite_text и hook_prompt на актуальном HUD.",
            )

        reel_timeout = self._failures["reel_timeout"]
        fish_escaped = self._failures["fish_escaped"]
        reel_missing = self._failures["reel_ui_never_appeared"]
        reel_modal = self._failures["reel_modal_missing"]
        if reel_timeout or fish_escaped:
            add(
                "error",
                f"Рыба сорвалась при вываживании: {reel_timeout + fish_escaped}",
                "Проверьте направление A/D, invert_pan_steer и чувствительность pan_velocity_min_px_sec.",
            )
        if reel_missing:
            add(
                "error",
                f"Мини-игра A/D не появилась: {reel_missing}",
                "Вероятна поздняя или неверная подсечка; сравните bite_text score и hook_delay_ms в событиях сессии.",
            )
        if reel_modal:
            add(
                "warning",
                f"Не найдено окно пойманной рыбы: {reel_modal}",
                "Бот безопасно остановлен до ручной проверки экрана. Проверьте регион catch_modal и задержку reel_modal_wait_sec.",
            )
        if self._failures["keep_click_missed"]:
            add(
                "warning",
                "Кнопка сохранения улова нажалась не с первого раза",
                "Проверьте keep_button или увеличьте keep_click_delay_ms.",
            )
        if self._failures["unhandled_exception"]:
            add(
                "error",
                "Рабочий процесс завершился с исключением",
                "Откройте recent_failures и последние строки JSONL: там сохранены тип исключения, сообщение и этап, на котором произошёл сбой.",
            )

        hooks = int(stats.get("hooks", 0) or 0)
        caught = int(stats.get("fish_caught", 0) or 0)
        if hooks >= 3 and caught / max(hooks, 1) < 0.5:
            add(
                "error",
                f"Низкая успешность: {caught} из {hooks} подсечек",
                "Основная проблема после подсечки: изучите reel_* ошибки и последние метрики направления в JSONL.",
            )
        casts = int(stats.get("casts", 0) or 0)
        failed = sum(self._failures.values()) > 0
        if casts == 0 and (duration >= 5 or failed):
            add(
                "error",
                "За сессию не выполнено ни одного заброса",
                "Проверьте стартовые подсказки, наличие снаряжения и распознавание timing_bar.",
            )

        if performance.get("average_tick_ms", 0) > 300:
            add(
                "warning",
                f"Медленный цикл: {performance['average_tick_ms']:.0f} мс",
                "Отключите show_overlay/save_screenshots и проверьте скорость захвата окна GTA.",
            )
        if not conclusions:
            add(
                "ok",
                "Критических ошибок не обнаружено",
                "Сессия завершилась без диагностируемых сбоев; данные сохранены для сравнения со следующими запусками.",
            )
        return conclusions

    @staticmethod
    def _format_text_summary(summary: dict[str, Any]) -> str:
        stats = summary["stats"]
        lines = [
            "ОТЧЁТ ПО РЫБАЛКЕ",
            f"Сессия: {summary['session_id']}",
            f"Длительность: {summary['duration_sec']} сек",
            f"Забросы: {stats.get('casts', 0)}",
            f"Подсечки: {stats.get('hooks', 0)}",
            f"Поймано: {stats.get('fish_caught', 0)}",
            f"Без окна / возможный мусор: {stats.get('unidentified_catches', 0)}",
            f"Таймауты: {stats.get('timeouts', 0)}",
            f"Паузы на еду/воду: {stats.get('sustenance_pauses', 0)}",
            f"Выпито: {stats.get('drinks', 0)}",
            f"Съедено: {stats.get('meals', 0)}",
            "",
            "ВЫВОДЫ:",
        ]
        for item in summary["conclusions"]:
            lines.append(f"- [{item['level'].upper()}] {item['title']}")
            lines.append(f"  {item['recommendation']}")
        lines.extend(("", f"Подробные события: {summary['events_file']}"))
        return "\n".join(lines) + "\n"

    def _close_events(self) -> None:
        stream, self._stream = self._stream, None
        close_event_stream(stream)
