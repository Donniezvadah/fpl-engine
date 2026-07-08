"""Squad legality, autosubs, transfers, and chip tests."""

import pytest

from fpl_engine.rules import (
    ChipState,
    Position,
    Squad,
    SquadPlayer,
    apply_autosubs,
    available_chips,
    can_play_chip,
    effective_captain,
    hit_points,
    next_free_transfers,
    rules_for,
    sell_price,
    validate_selection,
    validate_squad_composition,
)

RULES = rules_for("2025-26")


def make_squad_players(price: int = 50) -> list[SquadPlayer]:
    """A legal 15: ids 1-2 GK, 3-7 DEF, 8-12 MID, 13-15 FWD, all different clubs mod 20."""
    players = []
    pid = 1
    for pos, n in [(Position.GK, 2), (Position.DEF, 5), (Position.MID, 5), (Position.FWD, 3)]:
        for _ in range(n):
            players.append(SquadPlayer(pid, pos, club_id=pid % 20, price=price))
            pid += 1
    return players


def default_squad() -> Squad:
    # XI: GK 1, DEF 3-6 (4), MID 8-11 (4), FWD 13-14 (2). Bench: 2 (GK), 7, 12, 15.
    return Squad(
        players=tuple(make_squad_players()),
        starting_ids=frozenset({1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14}),
        captain_id=8,
        vice_captain_id=13,
        bench_order=(7, 12, 15),
    )


class TestComposition:
    def test_legal_squad(self):
        assert validate_squad_composition(make_squad_players(), RULES) == []

    def test_budget_exceeded(self):
        players = make_squad_players(price=67)  # 15 * 67 = 1005 > 1000
        errs = validate_squad_composition(players, RULES)
        assert any("budget" in e for e in errs)

    def test_budget_exact_ok(self):
        # 15 players * 66 = 990; add 10 to one player => exactly 1000.
        players = make_squad_players(price=66)
        players[0] = SquadPlayer(1, Position.GK, club_id=1, price=76)
        assert validate_squad_composition(players, RULES) == []

    def test_wrong_position_counts(self):
        players = make_squad_players()
        players[0] = SquadPlayer(99, Position.DEF, club_id=19, price=50)  # 1 GK, 6 DEF
        errs = validate_squad_composition(players, RULES)
        assert any("GK count" in e for e in errs) and any("DEF count" in e for e in errs)

    def test_club_limit(self):
        players = make_squad_players()
        players[3] = SquadPlayer(4, Position.DEF, club_id=1, price=50)
        players[4] = SquadPlayer(5, Position.DEF, club_id=1, price=50)
        players[5] = SquadPlayer(6, Position.DEF, club_id=1, price=50)  # club 1 x4 (with pid 1)
        errs = validate_squad_composition(players, RULES)
        assert any("max 3" in e for e in errs)

    def test_duplicate_players(self):
        players = make_squad_players()
        players[1] = players[0]
        errs = validate_squad_composition(players, RULES)
        assert any("duplicate" in e for e in errs)


class TestSelection:
    def test_legal_selection(self):
        assert validate_selection(default_squad(), RULES) == []

    def test_two_gks_in_xi(self):
        s = default_squad()
        bad = Squad(s.players, frozenset(s.starting_ids - {14} | {2}), 8, 13, (7, 12, 15))
        errs = validate_selection(bad, RULES)
        assert any("GK" in e for e in errs)

    def test_too_few_defenders(self):
        s = default_squad()
        # Swap DEF 6 out for MID 12: 3 DEF... start with 4 DEF, removing one leaves 3 = legal.
        # Remove two DEF for MID 12 and FWD 15 -> 2 DEF = illegal.
        bad_ids = frozenset(s.starting_ids - {5, 6} | {12, 15})
        bad = Squad(s.players, bad_ids, 8, 13, (7,))
        errs = validate_selection(bad, RULES)
        assert any("DEF" in e for e in errs)

    def test_captain_not_in_xi(self):
        s = default_squad()
        bad = Squad(s.players, s.starting_ids, captain_id=15, vice_captain_id=13,
                    bench_order=s.bench_order)
        errs = validate_selection(bad, RULES)
        assert any("captain must be" in e for e in errs)

    def test_captain_equals_vice(self):
        s = default_squad()
        bad = Squad(s.players, s.starting_ids, 8, 8, s.bench_order)
        errs = validate_selection(bad, RULES)
        assert any("must differ" in e for e in errs)


class TestAutosubs:
    def full_minutes(self, exceptions: dict[int, int] | None = None) -> dict[int, int]:
        mins = {i: 90 for i in range(1, 16)}
        if exceptions:
            mins.update(exceptions)
        return mins

    def test_no_subs_needed(self):
        s = default_squad()
        assert apply_autosubs(s, self.full_minutes(), RULES) == s.starting_ids

    def test_gk_swap(self):
        s = default_squad()
        final = apply_autosubs(s, self.full_minutes({1: 0}), RULES)
        assert 2 in final and 1 not in final

    def test_gk_not_replaced_by_outfield(self):
        s = default_squad()
        # Both GKs blank: XI keeps the non-playing GK (no outfield replacement).
        final = apply_autosubs(s, self.full_minutes({1: 0, 2: 0}), RULES)
        assert 1 in final

    def test_first_bench_priority(self):
        s = default_squad()
        final = apply_autosubs(s, self.full_minutes({14: 0}), RULES)
        assert 7 in final and 14 not in final  # bench_order[0] comes in

    def test_formation_constraint_skips_bench_player(self):
        s = default_squad()
        # XI has FWD 13,14 (exactly formation_min=1 after one blank). If both bench
        # priority players can't keep formation legal... here DEF 7 replacing FWD 14
        # gives 5 DEF, 4 MID, 1 FWD = legal, so 7 comes in.
        # Instead: blank FWD 13 AND 14; only one outfield sub can restore, second FWD
        # slot must be filled by FWD 15 (12 is MID -> 0 FWD illegal for last one).
        final = apply_autosubs(s, self.full_minutes({13: 0, 14: 0}), RULES)
        assert 15 in final  # FWD 15 must come in to keep >=1 FWD
        assert 7 in final  # DEF 7 fills the other slot (5-4-1... wait 4+1=5 DEF)

    def test_bench_player_who_blanked_is_skipped(self):
        s = default_squad()
        final = apply_autosubs(s, self.full_minutes({14: 0, 7: 0}), RULES)
        assert 12 in final and 7 not in final and 14 not in final


class TestCaptaincy:
    def test_captain_plays(self):
        s = default_squad()
        assert effective_captain(s, {8: 90, 13: 90}) == 8

    def test_vice_steps_up(self):
        s = default_squad()
        assert effective_captain(s, {8: 0, 13: 90}) == 13

    def test_neither_plays(self):
        s = default_squad()
        assert effective_captain(s, {8: 0, 13: 0}) is None


class TestTransfers:
    def test_ft_banking_caps_at_five(self):
        ft = 1
        for _ in range(10):  # roll 10 GWs with no transfers
            ft = next_free_transfers(ft, 0, RULES.transfers)
        assert ft == 5

    def test_ft_spend_and_regain(self):
        assert next_free_transfers(2, 1, RULES.transfers) == 2  # 2-1+1
        assert next_free_transfers(1, 1, RULES.transfers) == 1
        assert next_free_transfers(5, 0, RULES.transfers) == 5  # capped

    def test_hits_do_not_go_negative(self):
        # 4 transfers with 1 FT: 3 hits, next week back to 1 FT.
        assert hit_points(4, 1, RULES.transfers) == 12
        assert next_free_transfers(1, 4, RULES.transfers) == 1

    def test_hit_points(self):
        assert hit_points(1, 1, RULES.transfers) == 0
        assert hit_points(2, 1, RULES.transfers) == 4
        assert hit_points(0, 1, RULES.transfers) == 0

    def test_sell_price_profit_rounds_down(self):
        assert sell_price(55, 58, RULES.transfers) == 56  # +0.3 -> keep 0.1
        assert sell_price(55, 56, RULES.transfers) == 55  # +0.1 -> keep 0.0
        assert sell_price(55, 57, RULES.transfers) == 56  # +0.2 -> keep 0.1

    def test_sell_price_losses_fully_absorbed(self):
        assert sell_price(55, 52, RULES.transfers) == 52
        assert sell_price(55, 55, RULES.transfers) == 55


class TestChips:
    def test_all_available_initially(self):
        assert set(available_chips(ChipState(), 1, RULES)) == {
            "wildcard", "bboost", "3xc", "freehit",
        }

    def test_used_chip_unavailable_same_half(self):
        state = ChipState(used=[("wildcard", 5)])
        assert "wildcard" not in available_chips(state, 10, RULES)

    def test_second_set_after_gw19(self):
        state = ChipState(used=[("wildcard", 5), ("bboost", 12)])
        assert set(available_chips(state, 20, RULES)) == {
            "wildcard", "bboost", "3xc", "freehit",
        }

    def test_one_chip_per_gw(self):
        state = ChipState()
        assert can_play_chip(state, "bboost", 10, RULES, chip_this_gw="3xc") is False
        assert can_play_chip(state, "bboost", 10, RULES) is True

    def test_first_half_chip_expires(self):
        # Unused first-half wildcard can't be "saved": in GW20+ you're using the
        # second-half one, which is exactly what available_chips models.
        state = ChipState(used=[("wildcard", 25)])
        assert "wildcard" not in available_chips(state, 30, RULES)
        assert "wildcard" in available_chips(state, 15, RULES)
