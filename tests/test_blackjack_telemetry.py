import io
import json
from datetime import datetime
from pathlib import Path

from bot.telemetry.blackjack_session import BlackjackSessionTelemetry


def test_strategy_metadata_and_balanced_count_diagnostics_reach_summary():
    stats = {"running_count": 0, "completed": 2}
    telemetry = BlackjackSessionTelemetry.__new__(BlackjackSessionTelemetry)
    telemetry.enabled = True
    telemetry._stream = io.StringIO()
    telemetry._finished = False
    telemetry._strategy = {}
    telemetry._rounds = [
        {
            "round": 1,
            "dealer_up": 9,
            "running_count_before": 2,
            "player_cards_inferred": [2],
            "dealer_cards_inferred": [9],
        },
        {
            "round": 2,
            "dealer_up": 2,
            "running_count_before": -2,
            "player_cards_inferred": [10],
            "dealer_cards_inferred": [2],
        },
    ]
    telemetry.session_id = "test-v11"
    telemetry.started_at = datetime.now().astimezone()
    telemetry.events_path = Path("test-v11.jsonl")
    telemetry.event(
        "strategy_configured",
        strategy_name="majestic_effective48_s17_enhc_max_ev",
        version=11,
        rules_model="effective48_iid_s17_european_no_hole_card_one_split_das",
        rank_mix_model="three_ten_value_ranks_plus_nine_single_ranks",
        count_system="majestic_effective48_balanced_v2",
        count_used_for_betting=False,
    )

    event_record = json.loads(telemetry._stream.getvalue())
    summary = telemetry._summary(stats, finished=True)
    latest_text = telemetry._summary_text(summary)

    assert summary["schema_version"] == 2
    assert event_record["event"] == "strategy_configured"
    assert event_record["data"]["version"] == 11
    assert summary["strategy"]["version"] == 11
    assert summary["strategy"]["strategy_name"] == ("majestic_effective48_s17_enhc_max_ev")
    assert summary["card_analysis"]["count_system"] == ("majestic_effective48_balanced_v2")
    assert summary["card_analysis"]["next_card_source"] == "dealer_up"
    assert summary["card_analysis"]["positive_count_next_negative_value_rate"] == 1.0
    assert summary["card_analysis"]["negative_count_next_negative_value_rate"] == 0.0
    assert summary["card_analysis"]["used_for_betting"] is False
    assert "majestic_effective48_s17_enhc_max_ev v11" in latest_text
