"""Rules-engine scoring tests, hand-computed from the official FPL scoring table."""

from fpl_engine.rules import MatchStats, Position, award_bonus, match_points, rules_for

RULES = rules_for("2025-26").scoring


def test_no_minutes_no_points():
    stats = MatchStats(minutes=0, goals=2, bonus=3)
    assert match_points(stats, Position.FWD, RULES) == 0


def test_sub_appearance_only():
    assert match_points(MatchStats(minutes=20), Position.MID, RULES) == 1
    assert match_points(MatchStats(minutes=60), Position.MID, RULES) == 2
    assert match_points(MatchStats(minutes=59), Position.MID, RULES) == 1


def test_forward_brace():
    # 90 min (2) + 2 goals (8) + 1 assist (3) + bonus 3 = 16
    stats = MatchStats(minutes=90, goals=2, assists=1, bonus=3)
    assert match_points(stats, Position.FWD, RULES) == 16


def test_midfielder_goal_and_clean_sheet():
    # 90 min (2) + goal (5) + CS (1) = 8
    stats = MatchStats(minutes=90, goals=1, clean_sheet=True)
    assert match_points(stats, Position.MID, RULES) == 8


def test_defender_clean_sheet_needs_60_minutes():
    on_58 = MatchStats(minutes=58, clean_sheet=True)
    on_60 = MatchStats(minutes=60, clean_sheet=True)
    assert match_points(on_58, Position.DEF, RULES) == 1  # appearance only
    assert match_points(on_60, Position.DEF, RULES) == 6  # 2 + 4


def test_goalkeeper_full_house():
    # 90 (2) + CS (4) + 6 saves (2) + pen save (5) = 13
    stats = MatchStats(minutes=90, clean_sheet=True, saves=6, penalties_saved=1)
    assert match_points(stats, Position.GK, RULES) == 13


def test_goals_conceded_deduction():
    # DEF, 90 min, 4 conceded: 2 - 2 = 0
    stats = MatchStats(minutes=90, goals_conceded=4)
    assert match_points(stats, Position.DEF, RULES) == 0
    # 3 conceded: 2 - 1 = 1
    assert match_points(MatchStats(minutes=90, goals_conceded=3), Position.DEF, RULES) == 1
    # MID never penalised for conceding
    assert match_points(MatchStats(minutes=90, goals_conceded=5), Position.MID, RULES) == 2


def test_cards_own_goals_pen_miss():
    stats = MatchStats(minutes=90, yellow_cards=1, own_goals=1, penalties_missed=1)
    # 2 - 1 - 2 - 2 = -3
    assert match_points(stats, Position.FWD, RULES) == -3
    assert match_points(MatchStats(minutes=45, red_cards=1), Position.MID, RULES) == -2


def test_defensive_contribution():
    # DEF with 10+ CBIT gets +2; 9 gets nothing.
    assert match_points(MatchStats(minutes=90, defensive_contribution=10), Position.DEF, RULES) == 4
    assert match_points(MatchStats(minutes=90, defensive_contribution=9), Position.DEF, RULES) == 2
    # MID threshold is 12.
    assert match_points(MatchStats(minutes=90, defensive_contribution=12), Position.MID, RULES) == 4
    assert match_points(MatchStats(minutes=90, defensive_contribution=11), Position.MID, RULES) == 2
    # GK not eligible.
    assert match_points(MatchStats(minutes=90, defensive_contribution=15), Position.GK, RULES) == 2


class TestBonusAwards:
    def test_no_ties(self):
        assert award_bonus({1: 30, 2: 25, 3: 20, 4: 10}, RULES) == {1: 3, 2: 2, 3: 1}

    def test_tie_for_first(self):
        # Official: both get 3, third gets 1.
        assert award_bonus({1: 30, 2: 30, 3: 20}, RULES) == {1: 3, 2: 3, 3: 1}

    def test_tie_for_second(self):
        # Official: first gets 3, both tied get 2, nobody gets 1.
        assert award_bonus({1: 30, 2: 25, 3: 25, 4: 20}, RULES) == {1: 3, 2: 2, 3: 2}

    def test_three_way_tie_for_first(self):
        assert award_bonus({1: 30, 2: 30, 3: 30, 4: 29}, RULES) == {1: 3, 2: 3, 3: 3}

    def test_tie_for_third(self):
        assert award_bonus({1: 30, 2: 25, 3: 20, 4: 20}, RULES) == {1: 3, 2: 2, 3: 1, 4: 1}
