"""Pure scoring functions: raw match stats -> FPL points.

These reproduce the official scoring table exactly and are the ground truth used to
sanity-check ingested data and to convert simulated match events into points.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import Position, ScoringRules


@dataclass(frozen=True)
class MatchStats:
    """One player's raw stats in one match."""

    minutes: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheet: bool = False  # team kept CS while player was on for 60+
    goals_conceded: int = 0  # conceded while player on the pitch
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    bonus: int = 0  # awarded bonus (0-3), from BPS ranking
    defensive_contribution: int = 0  # CBIT for DEF, CBIRT for MID/FWD


def appearance_points(minutes: int, rules: ScoringRules) -> int:
    if minutes <= 0:
        return 0
    return rules.played_60_plus if minutes >= 60 else rules.played_upto_59


def clean_sheet_points(stats: MatchStats, position: Position, rules: ScoringRules) -> int:
    # A clean sheet only counts with 60+ minutes played (official rule).
    if stats.clean_sheet and stats.minutes >= 60:
        return rules.clean_sheet_points[position]
    return 0


def goals_conceded_points(stats: MatchStats, position: Position, rules: ScoringRules) -> int:
    if position in rules.goals_conceded_penalised_positions and stats.minutes > 0:
        return -(stats.goals_conceded // rules.goals_conceded_per_minus)
    return 0


def defensive_contribution_points(
    stats: MatchStats, position: Position, rules: ScoringRules
) -> int:
    if not rules.defcon_enabled or position not in rules.defcon_threshold:
        return 0
    if stats.defensive_contribution >= rules.defcon_threshold[position]:
        return rules.defcon_points
    return 0


def match_points(stats: MatchStats, position: Position, rules: ScoringRules) -> int:
    """Total FPL points for one player in one match (bonus included as given)."""
    if stats.minutes <= 0:
        # No appearance: no points of any kind accrue.
        return 0
    pts = appearance_points(stats.minutes, rules)
    pts += stats.goals * rules.goal_points[position]
    pts += stats.assists * rules.assist_points
    pts += clean_sheet_points(stats, position, rules)
    pts += goals_conceded_points(stats, position, rules)
    if position == Position.GK:
        pts += stats.saves // rules.saves_per_point
        pts += stats.penalties_saved * rules.penalty_save_points
    pts += stats.penalties_missed * rules.penalty_miss_points
    pts += stats.yellow_cards * rules.yellow_card_points
    pts += stats.red_cards * rules.red_card_points
    pts += stats.own_goals * rules.own_goal_points
    pts += defensive_contribution_points(stats, position, rules)
    pts += stats.bonus
    return pts


def award_bonus(bps_by_player: dict[int, int], rules: ScoringRules) -> dict[int, int]:
    """Award bonus points from BPS scores for one match, with official tie handling.

    Ties share the higher award and consume the lower slots, e.g. two players tied
    on the top BPS both get 3 and the next gets 1; three tied at the top all get 3
    and nobody gets 2 or 1.
    """
    if not bps_by_player:
        return {}
    awards: dict[int, int] = {}
    distinct = sorted(set(bps_by_player.values()), reverse=True)
    slot = 0  # index into rules.bonus_points
    for score in distinct:
        if slot >= len(rules.bonus_points):
            break
        tied = [p for p, s in bps_by_player.items() if s == score]
        for p in tied:
            awards[p] = rules.bonus_points[slot]
        slot += len(tied)
    return awards
