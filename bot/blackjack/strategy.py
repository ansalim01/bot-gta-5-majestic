"""Blackjack rules without capture, input, telemetry, or UI dependencies."""

from __future__ import annotations

HIT = "hit"
STAND = "stand"
DOUBLE = "double"
SPLIT = "split"
CARD_VALUES = tuple(range(2, 12))


def basic_strategy(
    player: int,
    dealer: int,
    *,
    can_double: bool = True,
    soft: bool = False,
) -> str:
    """Maximum-EV strategy for Majestic's observed effective-48 rank mix.

    Across more than 2,000 reliable dealer up-cards, ten-valued cards occur
    with probability 3/12 while every other value occurs with probability
    1/12.  The sequence behaves like an infinite/frequently reshuffled deck.
    The remaining observed rules are S17 and European no-hole-card dealing.
    Pair splitting is handled separately by :func:`should_split_pair`.
    ``soft`` means that an ace is currently counted as 11.
    """
    if soft:
        if player <= 14:
            return HIT
        if player == 15:
            return DOUBLE if can_double and dealer == 6 else HIT
        if player in (16, 17):
            return DOUBLE if can_double and 5 <= dealer <= 6 else HIT
        if player == 18:
            if can_double and 5 <= dealer <= 6:
                return DOUBLE
            return STAND if 2 <= dealer <= 8 else HIT
        return STAND

    if player >= 17:
        return STAND
    if player <= 8:
        return HIT
    if player == 9:
        return DOUBLE if can_double and dealer == 6 else HIT
    if player in (10, 11):
        return DOUBLE if can_double and 2 <= dealer <= 8 else HIT
    if player in (12, 13):
        return HIT
    if player == 14:
        return STAND if 4 <= dealer <= 6 else HIT
    return STAND if 2 <= dealer <= 6 else HIT


def should_split_pair(
    rank: int,
    dealer: int,
    *,
    can_double_after_split: bool = True,
) -> bool:
    """Maximum-EV effective-48 one-split policy for Majestic.

    The table assumes an effectively infinite deck, S17, European no-hole
    card dealing, double after split, and no re-split (the UI removes ``F``
    after the first split).  In particular, 8s are hit rather than split
    against 10/ace because both split stakes lose to a later dealer blackjack.
    The alternate no-DAS rows keep the ``never`` double-policy internally
    consistent; the default full-EV policy uses DAS.
    """
    if not can_double_after_split:
        if rank == 2:
            return 4 <= dealer <= 7
        if rank == 3:
            return 5 <= dealer <= 7
        if rank in (4, 5, 10):
            return False
        if rank == 6:
            return dealer == 6
        if rank == 7:
            return 2 <= dealer <= 7
        if rank == 8:
            return 2 <= dealer <= 9
        if rank == 9:
            return dealer in (5, 6, 8, 9)
        if rank == 11:
            return 2 <= dealer <= 11
        return False
    if rank == 2:
        return 3 <= dealer <= 7
    if rank == 3:
        return 3 <= dealer <= 8
    if rank == 4:
        return False
    if rank == 5:
        return False
    if rank == 6:
        return dealer in (5, 6)
    if rank == 7:
        return 2 <= dealer <= 7
    if rank == 8:
        return 2 <= dealer <= 9
    if rank == 9:
        return dealer in (4, 5, 6, 8, 9)
    if rank == 10:
        return False
    if rank == 11:
        return 2 <= dealer <= 11
    return False


def infer_initial_soft(first_card: int | None, total: int) -> bool:
    """Infer whether the two-card total contains an ace counted as 11."""
    if first_card is None:
        return False
    return (first_card == 11 and 12 <= total <= 21) or (
        1 <= first_card <= 10 and total == first_card + 11
    )


def soft_after_hit(previous_total: int, total: int, was_soft: bool) -> bool:
    """Update usable-ace state after a confirmed hit has completed."""
    inferred = infer_card_transition(previous_total, was_soft, total)
    if inferred is not None:
        return inferred[1]
    if was_soft:
        # If the total stayed level or fell, the ace was demoted from 11 to 1.
        return total > previous_total
    # A jump by 11 means the drawn ace can still be valued as 11.
    return total - previous_total == 11 and total <= 21


def add_card_to_total(total: int, soft: bool, card: int) -> tuple[int, bool]:
    """Apply one rank value (ace=11, faces=10) to a blackjack total."""
    if card == 11:
        if soft:
            return total + 1, True
        if total + 11 <= 21:
            return total + 11, True
        return total + 1, False
    updated = total + card
    if updated > 21 and soft:
        return updated - 10, False
    return updated, soft


def infer_card_transition(
    previous_total: int,
    previous_soft: bool,
    total: int,
) -> tuple[int, bool] | None:
    """Infer one newly dealt card from consecutive displayed totals."""
    matches = [
        (card, new_soft)
        for card in CARD_VALUES
        for new_total, new_soft in [add_card_to_total(previous_total, previous_soft, card)]
        if new_total == total
    ]
    return matches[0] if len(matches) == 1 else None


def double_allowed_by_policy(
    policy: str,
    player: int,
    *,
    soft: bool,
) -> bool:
    """Restrict variance without changing the hit/stand strategy table."""
    normalized = policy.strip().lower()
    if normalized == "never":
        return False
    if normalized == "conservative":
        return not soft and player in (10, 11)
    return True


def hi_lo_value(card: int) -> int:
    if 2 <= card <= 6:
        return 1
    if card in (10, 11):
        return -1
    return 0


def majestic_balanced_count_value(card: int) -> int:
    """Balanced diagnostic count for the rank mix observed on Majestic.

    Long-session telemetry contains roughly three ten-valued ranks for every
    one rank of each other value.  Classic Hi-Lo (2-6 versus 10/ace) therefore
    has a positive drift even for independently shuffled cards.  Balancing
    2-6 against 9/10/ace removes that structural drift.  The count remains
    diagnostic only and must not drive wager sizes without evidence of a
    persistent shoe.
    """
    if 2 <= card <= 6:
        return 1
    if card in (9, 10, 11):
        return -1
    return 0
