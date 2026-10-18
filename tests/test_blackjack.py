from bot.jobs.blackjack import (
    DOUBLE,
    HIT,
    STAND,
    add_card_to_total,
    basic_strategy,
    double_allowed_by_policy,
    hi_lo_value,
    infer_card_transition,
    infer_initial_soft,
    majestic_balanced_count_value,
    should_split_pair,
    soft_after_hit,
)


def test_blackjack_runtime_config_uses_full_strategy_without_session_stops():
    from bot.config_loader import BLACKJACK_CONFIG_PATH, load_config

    config = load_config(
        BLACKJACK_CONFIG_PATH,
        include_user_overrides=False,
    )

    assert config["blackjack"]["double_policy"] == "basic"
    assert config["blackjack"]["split_enabled"] is True
    assert config["blackjack"]["max_bet_retries"] == 2
    assert config["screen"]["auto_focus_game_for_input"] is True
    assert config["blackjack"]["risk"]["max_session_loss"] == 0
    assert config["blackjack"]["risk"]["max_drawdown_from_peak"] == 0
    assert config["blackjack"]["risk"]["max_hands_per_session"] == 0


def test_blocked_blackjack_action_is_logged_once_and_can_retry():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class FakeInput:
        def __init__(self):
            self.allowed = False
            self.last_foreground_identity = "gamebot control panel [python]"

        def press_key_named(self, _key):
            return self.allowed

    class FakeTelemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._input = FakeInput()
    job._keys = {HIT: "enter", STAND: "space", DOUBLE: "tab"}
    job._round = {"round": 7}
    job._stats = {"input_blocks": 0}
    job._last_input_blocked_key = None
    job._telemetry = FakeTelemetry()
    score = BlackjackScore(player=11, dealer=9, confidence=0.99)

    assert job._press_action(HIT, score) is False
    assert job._press_action(HIT, score) is False
    assert job._stats["input_blocks"] == 1
    assert [name for name, _data in job._telemetry.events] == ["input_blocked"]

    job._input.allowed = True
    assert job._press_action(HIT, score) is True
    assert job._last_input_blocked_key is None


def test_unacknowledged_blackjack_action_is_retried_without_duplicate_decision():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class FakeInput:
        def __init__(self):
            self.presses = []

        def press_key_named(self, key):
            self.presses.append(key)
            return True

    class FakeTelemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._input = FakeInput()
    job._keys = {HIT: "enter", STAND: "space", DOUBLE: "tab"}
    job._round = {"round": 4}
    job._stats = {"action_retries": 0, "action_unacknowledged": 0}
    job._last_input_blocked_key = None
    job._last_action_key = (16, 10)
    job._last_action_at = 10.0
    job._action_retry_sec = 1.25
    job._max_action_retries = 2
    job._action_retry_count = 0
    job._action_retry_exhausted = False
    job._telemetry = FakeTelemetry()
    score = BlackjackScore(player=16, dealer=10, confidence=0.99)

    assert job._retry_pending_action(HIT, score, 10.5) is True
    assert job._input.presses == []

    assert job._retry_pending_action(HIT, score, 11.3) is True
    assert job._retry_pending_action(HIT, score, 12.6) is True
    assert job._input.presses == ["enter", "enter"]
    assert job._stats["action_retries"] == 2

    assert job._retry_pending_action(HIT, score, 14.0) is True
    assert job._retry_pending_action(HIT, score, 16.0) is True
    assert job._input.presses == ["enter", "enter"]
    assert job._stats["action_unacknowledged"] == 1
    assert [name for name, _data in job._telemetry.events] == [
        "action_retried",
        "action_retried",
        "action_unacknowledged",
    ]


def test_score_detector_on_ui_sized_text():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from bot.detectors.blackjack_score import BlackjackScoreDetector

    image = Image.new("RGB", (430, 110), (30, 70, 45))
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    ImageDraw.Draw(image).text((60, 35), "You [18], dealer [2]", font=font, fill=(210, 210, 210))
    frame = np.asarray(image)[:, :, ::-1].copy()
    score = BlackjackScoreDetector(min_confidence=0.3).detect(frame)
    assert score is not None
    assert (score.player, score.dealer) == (18, 2)


def test_score_detector_reads_left_right_and_dealer_after_split():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from bot.detectors.blackjack_score import BlackjackScoreDetector

    image = Image.new("RGB", (430, 110), (30, 70, 45))
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    ImageDraw.Draw(image).text(
        (35, 35),
        "You [16] and [22], dealer [10]",
        font=font,
        fill=(210, 210, 210),
    )
    frame = np.asarray(image)[:, :, ::-1].copy()
    score = BlackjackScoreDetector(min_confidence=0.3).detect(frame)

    assert score is not None
    assert (score.player, score.player_second, score.dealer) == (16, 22, 10)


def test_score_detector_accepts_dealer_bust_total():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from bot.detectors.blackjack_score import BlackjackScoreDetector

    image = Image.new("RGB", (430, 110), (30, 70, 45))
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    ImageDraw.Draw(image).text((60, 35), "You [14], dealer [26]", font=font, fill=(210, 210, 210))
    frame = np.asarray(image)[:, :, ::-1].copy()
    score = BlackjackScoreDetector(min_confidence=0.3).detect(frame)
    assert score is not None
    assert (score.player, score.dealer) == (14, 26)


def test_isolated_bet_value_ocr():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from bot.detectors.blackjack_score import BlackjackScoreDetector

    image = Image.new("RGB", (90, 48), (25, 28, 32))
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    ImageDraw.Draw(image).text((35, 10), "10", font=font, fill=(225, 225, 225))
    frame = np.asarray(image)[:, :, ::-1].copy()
    result = BlackjackScoreDetector(min_confidence=0.3).detect_number(frame, min_confidence=0.3)
    assert result is not None
    assert result[0] == 10


def test_hard_total_basic_strategy_boundaries():
    dealers = tuple(range(2, 12))
    expected = {
        8: (HIT,) * 10,
        9: (HIT, HIT, HIT, HIT, DOUBLE, HIT, HIT, HIT, HIT, HIT),
        10: (DOUBLE,) * 7 + (HIT, HIT, HIT),
        11: (DOUBLE,) * 7 + (HIT, HIT, HIT),
        12: (HIT,) * 10,
        13: (HIT,) * 10,
        14: (HIT, HIT, STAND, STAND, STAND, HIT, HIT, HIT, HIT, HIT),
        15: (STAND, STAND, STAND, STAND, STAND, HIT, HIT, HIT, HIT, HIT),
        16: (STAND, STAND, STAND, STAND, STAND, HIT, HIT, HIT, HIT, HIT),
        17: (STAND,) * 10,
    }

    for player, actions in expected.items():
        assert tuple(basic_strategy(player, dealer) for dealer in dealers) == actions


def test_double_can_be_disabled():
    assert basic_strategy(10, 6, can_double=False) == HIT
    assert basic_strategy(9, 6, can_double=False) == HIT
    assert basic_strategy(11, 8, can_double=False) == HIT
    assert basic_strategy(15, 6, can_double=False, soft=True) == HIT
    assert basic_strategy(16, 5, can_double=False, soft=True) == HIT
    assert basic_strategy(17, 6, can_double=False, soft=True) == HIT
    assert basic_strategy(18, 5, can_double=False, soft=True) == STAND
    assert basic_strategy(18, 6, can_double=False, soft=True) == STAND


def test_soft_total_basic_strategy_boundaries():
    dealers = tuple(range(2, 12))
    expected = {
        13: (HIT,) * 10,
        14: (HIT,) * 10,
        15: (HIT, HIT, HIT, HIT, DOUBLE, HIT, HIT, HIT, HIT, HIT),
        16: (HIT, HIT, HIT, DOUBLE, DOUBLE, HIT, HIT, HIT, HIT, HIT),
        17: (HIT, HIT, HIT, DOUBLE, DOUBLE, HIT, HIT, HIT, HIT, HIT),
        18: (STAND, STAND, STAND, DOUBLE, DOUBLE, STAND, STAND, HIT, HIT, HIT),
        19: (STAND,) * 10,
    }

    for player, actions in expected.items():
        assert tuple(basic_strategy(player, dealer, soft=True) for dealer in dealers) == actions


def test_enhc_pair_split_strategy_boundaries():
    expected_split_dealers = {
        2: {3, 4, 5, 6, 7},
        3: {3, 4, 5, 6, 7, 8},
        4: set(),
        5: set(),
        6: {5, 6},
        7: {2, 3, 4, 5, 6, 7},
        8: {2, 3, 4, 5, 6, 7, 8, 9},
        9: {4, 5, 6, 8, 9},
        10: set(),
        11: set(range(2, 12)),
    }

    for rank, split_dealers in expected_split_dealers.items():
        for dealer in range(2, 12):
            assert should_split_pair(rank, dealer) is (dealer in split_dealers)


def test_enhc_pair_split_strategy_without_double_after_split():
    expected_split_dealers = {
        2: {4, 5, 6, 7},
        3: {5, 6, 7},
        4: set(),
        5: set(),
        6: {6},
        7: {2, 3, 4, 5, 6, 7},
        8: {2, 3, 4, 5, 6, 7, 8, 9},
        9: {5, 6, 8, 9},
        10: set(),
        11: set(range(2, 12)),
    }

    for rank, split_dealers in expected_split_dealers.items():
        for dealer in range(2, 12):
            assert should_split_pair(
                rank,
                dealer,
                can_double_after_split=False,
            ) is (dealer in split_dealers)


def test_split_state_plays_second_displayed_total_before_first():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def event(self, *_args, **_kwargs):
            pass

    job = BlackjackJob.__new__(BlackjackJob)
    job._split_active = True
    job._split_hand_index = 1
    job._split_hand_totals = [16, 22]
    job._round = {"round": 3}
    job._telemetry = Telemetry()
    score = BlackjackScore(
        player=16,
        player_second=22,
        dealer=10,
        confidence=0.99,
    )

    assert job._active_player_total(score) == 22
    assert job._decision_key(score) == (22, 10, 1)
    job._advance_split_hand(score, STAND)
    assert job._active_player_total(score) == 16
    assert job._decision_key(score) == (16, 10, 0)


def test_split_activation_selects_second_displayed_total():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._split_pending = True
    job._split_active = False
    job._split_pair_rank = None
    job._round = {"round": 12}
    job._telemetry = Telemetry()
    score = BlackjackScore(player=7, player_second=18, dealer=6, confidence=0.99)

    job._activate_split(score)

    assert job._split_hand_index == 1
    assert job._active_player_total(score) == 18
    assert job._round["split_initial_totals"] == [7, 18]
    assert job._telemetry.events[-1][1]["active_hand"] == "second_total"


def test_split_round_259_uses_changing_second_total_not_stale_first_total():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def event(self, *_args, **_kwargs):
            pass

    job = BlackjackJob.__new__(BlackjackJob)
    job._split_active = True
    job._split_hand_index = 1
    job._split_hand_totals = [7, 5]
    job._round = {"round": 259}
    job._telemetry = Telemetry()

    first = BlackjackScore(player=7, player_second=5, dealer=4, confidence=0.99)
    after_hit = BlackjackScore(player=7, player_second=14, dealer=4, confidence=0.99)

    assert job._active_player_total(first) == 5
    assert basic_strategy(job._active_player_total(first), first.dealer) == HIT
    assert job._active_player_total(after_hit) == 14
    assert basic_strategy(job._active_player_total(after_hit), after_hit.dealer) == STAND

    job._advance_split_hand(after_hit, STAND)
    assert job._active_player_total(after_hit) == 7
    assert basic_strategy(job._active_player_total(after_hit), after_hit.dealer) == HIT


def test_split_round_250_switches_from_second_seven_to_first_fifteen():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def event(self, *_args, **_kwargs):
            pass

    job = BlackjackJob.__new__(BlackjackJob)
    job._split_active = True
    job._split_hand_index = 1
    job._split_hand_totals = [15, 7]
    job._round = {"round": 250}
    job._telemetry = Telemetry()
    score = BlackjackScore(player=15, player_second=7, dealer=2, confidence=0.99)

    assert job._active_player_total(score) == 7
    assert basic_strategy(job._active_player_total(score), score.dealer) == HIT

    job._advance_split_hand(score, STAND)
    assert job._active_player_total(score) == 15
    assert basic_strategy(job._active_player_total(score), score.dealer) == STAND


def test_closed_split_hand_ignores_late_ocr_total_changes():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._split_active = True
    job._split_hand_index = 1
    job._split_hand_totals = [17, 21]
    job._split_hand_closed = [False, False]
    job._split_hand_soft = [False, False]
    job._split_hand_decisions = [1, 1]
    job._round = {"round": 447}
    job._telemetry = Telemetry()

    score = BlackjackScore(player=17, player_second=21, dealer=10, confidence=0.99)
    job._advance_split_hand(score, STAND)
    job._apply_split_transition(
        BlackjackScore(player=17, player_second=29, dealer=17, confidence=0.99),
        1,
    )

    assert job._split_hand_totals == [17, 21]
    assert job._split_final_totals(
        BlackjackScore(player=17, player_second=29, dealer=17, confidence=0.99)
    ) == (17, 21)
    assert [name for name, _data in job._telemetry.events] == [
        "split_hand_advanced",
        "inactive_split_score_ignored",
    ]


def test_soft_hand_is_inferred_and_updated_from_totals():
    assert infer_initial_soft(11, 18) is True
    assert infer_initial_soft(7, 18) is True
    assert infer_initial_soft(8, 18) is False
    assert infer_initial_soft(None, 18) is False
    assert soft_after_hit(18, 18, True) is False
    assert soft_after_hit(17, 20, True) is True
    assert soft_after_hit(10, 21, False) is True
    assert soft_after_hit(16, 17, False) is False


def test_card_transitions_support_sequence_telemetry():
    assert add_card_to_total(10, False, 11) == (21, True)
    assert add_card_to_total(16, True, 9) == (15, False)
    assert add_card_to_total(12, True, 11) == (13, True)
    assert infer_card_transition(11, True, 15) == (4, True)
    assert infer_card_transition(16, True, 15) == (9, False)
    assert infer_card_transition(18, True, 18) == (10, False)
    assert infer_card_transition(10, False, 21) == (11, True)
    assert hi_lo_value(2) == 1
    assert hi_lo_value(8) == 0
    assert hi_lo_value(10) == -1
    assert majestic_balanced_count_value(2) == 1
    assert majestic_balanced_count_value(5) == 1
    assert majestic_balanced_count_value(6) == 1
    assert majestic_balanced_count_value(8) == 0
    assert majestic_balanced_count_value(9) == -1
    assert majestic_balanced_count_value(10) == -1
    assert majestic_balanced_count_value(11) == -1


def test_effective48_diagnostic_count_is_balanced_for_modeled_rank_mix():
    # The effective model has one copy of every non-ten value and three
    # ten-valued ranks.  Its expected running-count drift must be exactly 0.
    modeled_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11]
    assert sum(majestic_balanced_count_value(card) for card in modeled_values) == 0


def _terminal_transition_job(*, action, player, soft=False):
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._round = {
        "round": 1,
        "player_cards_inferred": [],
        "dealer_cards_inferred": [],
    }
    job._stats = {"cards_inferred": 0, "running_count": 0}
    job._telemetry = Telemetry()
    job._pending_final = None
    job._last_pressed_action = action
    job._last_action_key = (player, 6)
    job._last_player_total = player
    job._hand_soft = soft
    job._split_active = False
    job._action_gap_after_press = True
    job._action_missing_frames = 3
    return job


def test_terminal_hit_and_double_cards_are_recorded_once():
    from bot.detectors.blackjack_score import BlackjackScore

    hit_job = _terminal_transition_job(action=HIT, player=16)
    hit_result = BlackjackScore(player=26, dealer=18, confidence=0.99)
    hit_job._record_terminal_player_transition(hit_result)
    hit_job._record_terminal_player_transition(hit_result)

    assert hit_job._round["player_cards_inferred"] == [10]
    assert hit_job._stats["running_count"] == -1
    assert hit_job._telemetry.events[0][1]["source"] == "terminal_hit"

    double_job = _terminal_transition_job(action=DOUBLE, player=11)
    double_result = BlackjackScore(player=20, dealer=19, confidence=0.99)
    double_job._record_terminal_player_transition(double_result)

    assert double_job._round["player_cards_inferred"] == [9]
    assert double_job._stats["running_count"] == -1
    assert double_job._telemetry.events[0][1]["source"] == "terminal_double"


def test_terminal_soft_and_split_transitions_preserve_hand_state():
    from bot.detectors.blackjack_score import BlackjackScore

    soft_job = _terminal_transition_job(action=HIT, player=18, soft=True)
    soft_job._record_terminal_player_transition(
        BlackjackScore(player=18, dealer=20, confidence=0.99)
    )
    assert soft_job._round["player_cards_inferred"] == [10]
    assert soft_job._hand_soft is False

    split_job = _terminal_transition_job(action=HIT, player=12)
    split_job._split_active = True
    split_job._split_hand_index = 0
    split_job._split_hand_totals = [12, 18]
    split_job._split_hand_soft = [False, False]
    split_job._split_hand_decisions = [1, 1]
    split_job._record_terminal_player_transition(
        BlackjackScore(
            player=22,
            player_second=18,
            dealer=20,
            confidence=0.99,
        )
    )

    assert split_job._round["player_cards_inferred"] == [10]
    assert split_job._split_hand_totals == [22, 18]
    assert split_job._telemetry.events[0][1]["source"] == ("split_first_total_terminal_hit")


def test_ambiguous_terminal_transition_is_logged_without_inventing_a_card():
    from bot.detectors.blackjack_score import BlackjackScore

    job = _terminal_transition_job(action=HIT, player=10)
    job._record_terminal_player_transition(BlackjackScore(player=30, dealer=18, confidence=0.99))

    assert job._round["player_cards_inferred"] == []
    assert job._stats["cards_inferred"] == 0
    assert job._telemetry.events == [
        (
            "card_inference_miss",
            {
                "round": 1,
                "owner": "player",
                "previous": 10,
                "total": 30,
                "phase": "terminal_hit",
            },
        )
    ]


def test_unchanged_hard_total_marks_a_missing_terminal_action():
    from bot.detectors.blackjack_score import BlackjackScore

    job = _terminal_transition_job(action=HIT, player=16)
    job._stats["action_unacknowledged"] = 0
    job._record_terminal_player_transition(BlackjackScore(player=16, dealer=18, confidence=0.99))

    assert job._stats["action_unacknowledged"] == 1
    assert job._round["player_cards_inferred"] == []
    assert job._telemetry.events[0][0] == "action_transition_missing"


def test_inferred_natural_with_final_score_gets_three_to_two():
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.rounds = []

        def event(self, *_args, **_kwargs):
            pass

        def record_round(self, round_data, _stats):
            self.rounds.append(round_data)

        @staticmethod
        def timestamp():
            return "now"

    job = BlackjackJob.__new__(BlackjackJob)
    job._round = {
        "round": 433,
        "stake_multiplier": 1,
        "player_cards_inferred": [11, 10],
    }
    job._pending_final = (21, 18)
    job._bet_amount = 10
    job._stats = {
        "completed": 0,
        "wagered": 0,
        "accounted_wagered": 0,
        "resolved": 0,
        "net": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "unknown": 0,
        "peak_net": 0,
        "max_drawdown": 0,
        "running_count": 0,
    }
    job._telemetry = Telemetry()
    job._enforce_reached_risk_limits = lambda: None

    job._finish_round()

    assert job._stats["net"] == 15
    assert job._telemetry.rounds[0]["natural_blackjack"] is True


def test_conservative_double_policy_only_keeps_hard_ten_and_eleven():
    assert double_allowed_by_policy("conservative", 10, soft=False) is True
    assert double_allowed_by_policy("conservative", 11, soft=False) is True
    assert double_allowed_by_policy("conservative", 9, soft=False) is False
    assert double_allowed_by_policy("conservative", 16, soft=True) is False
    assert double_allowed_by_policy("basic", 9, soft=False) is True
    assert double_allowed_by_policy("never", 11, soft=False) is False


def test_outcome_classification():
    from bot.jobs.blackjack import BlackjackJob

    assert BlackjackJob._outcome(22, 18) == "loss"
    assert BlackjackJob._outcome(18, 22) == "win"
    assert BlackjackJob._outcome(20, 19) == "win"
    assert BlackjackJob._outcome(18, 20) == "loss"
    assert BlackjackJob._outcome(18, 18) == "push"


def test_green_natural_blackjack_notice_is_detected():
    import cv2
    import numpy as np

    from bot.jobs.blackjack import BlackjackJob

    crop = np.zeros((85, 450, 3), dtype=np.uint8)
    cv2.circle(crop, (25, 45), 8, (80, 210, 100), -1)
    cv2.rectangle(crop, (420, 20), (422, 65), (80, 210, 100), -1)

    assert BlackjackJob._result_notice(crop) == "blackjack_win"
    assert BlackjackJob._result_notice(np.zeros_like(crop)) is None


def test_natural_blackjack_notice_is_accounted_at_three_to_two():
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.rounds = []

        def record_round(self, round_data, _stats):
            self.rounds.append(round_data)

        @staticmethod
        def timestamp():
            return "now"

    job = BlackjackJob.__new__(BlackjackJob)
    job._round = {
        "round": 1,
        "stake_multiplier": 1,
        "result_notice": "blackjack_win",
    }
    job._pending_final = None
    job._bet_amount = 10
    job._stats = {
        "completed": 0,
        "wagered": 0,
        "accounted_wagered": 0,
        "resolved": 0,
        "net": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "unknown": 0,
        "peak_net": 0,
        "max_drawdown": 0,
        "running_count": 0,
    }
    job._telemetry = Telemetry()
    job._enforce_reached_risk_limits = lambda: None

    job._finish_round()

    assert job._stats["wins"] == 1
    assert job._stats["net"] == 15
    assert job._stats["wagered"] == 10
    assert job._telemetry.rounds[0]["natural_blackjack"] is True
    assert job._telemetry.rounds[0]["net"] == 15


def test_split_hands_are_settled_independently_with_separate_stakes():
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.rounds = []

        def record_round(self, round_data, _stats):
            self.rounds.append(round_data)

        @staticmethod
        def timestamp():
            return "now"

    job = BlackjackJob.__new__(BlackjackJob)
    job._round = {
        "round": 2,
        "split_confirmed": True,
        "split_stake_multipliers": [1, 2],
        "card_tracking_started": True,
    }
    # Left 20 wins against 18. Right 22 loses its doubled stake.
    job._pending_final = (20, 18, 22)
    job._bet_amount = 10
    job._stats = {
        "completed": 0,
        "wagered": 0,
        "accounted_wagered": 0,
        "resolved": 0,
        "net": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "unknown": 0,
        "split_hands": 0,
        "peak_net": 0,
        "max_drawdown": 0,
        "running_count": 0,
    }
    job._telemetry = Telemetry()
    job._enforce_reached_risk_limits = lambda: None

    job._finish_round()

    assert job._stats["wins"] == 1
    assert job._stats["losses"] == 1
    assert job._stats["resolved"] == 2
    assert job._stats["split_hands"] == 2
    assert job._stats["wagered"] == 30
    assert job._stats["net"] == -10
    completed = job._telemetry.rounds[0]
    assert completed["hand_outcomes"] == ["win", "loss"]
    assert completed["hand_nets"] == [10, -20]
    assert completed["outcome"] == "mixed"


def test_risk_limit_blocks_double_before_it_can_overshoot_stop_loss():
    from bot.jobs.blackjack import BlackjackJob

    job = BlackjackJob.__new__(BlackjackJob)
    job._bet_amount = 10
    job._max_session_loss = 100
    job._max_drawdown = 100
    job._stats = {"net": -90, "peak_net": 0}
    assert job._double_within_risk() is False

    job._stats["net"] = -80
    assert job._double_within_risk() is True


def test_new_hand_reserves_room_for_an_optimal_double():
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def event(self, *_args, **_kwargs):
            pass

        def update_summary(self, *_args, **_kwargs):
            pass

    job = BlackjackJob.__new__(BlackjackJob)
    job._bet_amount = 10
    job._double_policy = "basic"
    job._max_session_loss = 100
    job._max_drawdown = 40
    job._max_hands = 500
    job._risk_stop_reason = None
    job._stats = {
        "net": -5,
        "peak_net": 20,
        "hands": 22,
        "risk_stopped": 0,
    }
    job._telemetry = Telemetry()

    assert job._risk_allows_bet() is False
    assert job._risk_stop_reason is not None

    job._risk_stop_reason = None
    job._stats["risk_stopped"] = 0
    job._double_policy = "never"
    assert job._risk_allows_bet() is True


def test_terminal_blackjacks_contribute_unambiguous_cards_to_count():
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def event(self, *_args, **_kwargs):
            pass

    job = BlackjackJob.__new__(BlackjackJob)
    job._round = {
        "initial_player": 10,
        "dealer_up": 11,
        "player_cards_inferred": [],
        "dealer_cards_inferred": [],
        "dealer_totals_observed": [],
    }
    job._deal_first_player = 10
    job._stats = {"cards_inferred": 0, "running_count": 0}
    job._telemetry = Telemetry()

    job._initialize_terminal_card_tracking(player=21, dealer=21)

    assert job._round["player_cards_inferred"] == [10, 11]
    assert job._round["dealer_cards_inferred"] == [11, 10]
    assert job._stats["cards_inferred"] == 4
    assert job._stats["running_count"] == -4


def test_completed_hand_stops_immediately_at_drawdown_limit():
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

        def update_summary(self, stats):
            self.summary = stats

    job = BlackjackJob.__new__(BlackjackJob)
    job._max_session_loss = 50
    job._max_drawdown = 50
    job._max_hands = 500
    job._risk_stop_reason = None
    job._stats = {
        "net": -10,
        "peak_net": 40,
        "completed": 12,
        "risk_stopped": 0,
    }
    job._telemetry = Telemetry()

    job._enforce_reached_risk_limits()

    assert job._risk_stop_reason == "лимит просадки 50"
    assert job._stats["risk_stopped"] == 1
    assert job._telemetry.events[0][0] == "risk_limit_reached"


def test_accepted_bet_survives_first_score_without_action_panel():
    from bot.detectors.blackjack_score import BlackjackScore
    from bot.jobs.blackjack import BlackjackJob

    class Telemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._awaiting_round_after_bet = True
    job._deal_clear_frames = 8
    job._bet_missing = 8
    job._bet_seen = 0
    job._bet_placed = True
    job._bet_confirmed = True
    job._zero_bet_seen = 0
    job._last_bet_value = 10
    job._bet_confirm_frames = 3
    job._deal_first_player = None
    job._telemetry = Telemetry()

    score = BlackjackScore(player=11, dealer=4, confidence=0.9)
    job._accept_pending_bet(score)
    job._observe_no_bet_screen()

    assert job._awaiting_round_after_bet is False
    assert job._deal_first_player == 11
    assert job._bet_missing == 1
    assert job._bet_placed is True
    assert job._telemetry.events == [("bet_accepted", {"player": 11, "dealer": 4})]


def test_unacknowledged_bet_stops_after_configured_retries():
    from bot.jobs.blackjack import BlackjackJob

    class FakeInput:
        def __init__(self):
            self.presses = []

        def press_key_named(self, key):
            self.presses.append(key)
            return True

    class Telemetry:
        def __init__(self):
            self.events = []

        def event(self, name, **data):
            self.events.append((name, data))

    job = BlackjackJob.__new__(BlackjackJob)
    job._round = None
    job._input = FakeInput()
    job._telemetry = Telemetry()
    job._risk_stop_reason = None
    job._auto_bet = True
    job._risk_allows_bet = lambda: True
    job._bet_amount = 10
    job._bet_key = "enter"
    job._bet_confirm_frames = 3
    job._bet_seen = 2
    job._bet_missing = 0
    job._bet_placed = True
    job._bet_confirmed = False
    job._zero_bet_seen = 2
    job._last_bet_value = 0
    job._last_bet_at = 0.0
    job._bet_retry_sec = 0.0
    job._max_bet_retries = 2
    job._bet_retry_count = 0
    job._bet_retry_exhausted = False
    job._cooldown = 0.0
    job._awaiting_round_after_bet = True
    job._stats = {"bets": 1, "bet_retries": 0, "bet_unacknowledged": 0}

    for _ in range(10):
        job._handle_bet_screen(True, (0, 0.99))

    assert job._input.presses == ["enter", "enter"]
    assert job._stats["bets"] == 1
    assert job._stats["bet_retries"] == 2
    assert job._stats["bet_unacknowledged"] == 1
    assert [name for name, _data in job._telemetry.events] == [
        "bet_pressed",
        "bet_pressed",
        "bet_unacknowledged",
    ]
