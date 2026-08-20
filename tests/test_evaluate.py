"""Squad evaluation: name resolution and scoring an arbitrary 15."""

import pandas as pd
import pytest

from fpl_engine.recommend.evaluate import evaluate_squad, resolve_players, suggest_upgrades
from fpl_engine.rules import rules_for

RULES = rules_for("2026-27")


def make_pool() -> pd.DataFrame:
    """A legal-squad-worth of players plus alternatives, priced so a 15 fits £100m.

    Expected points strictly decrease with player_id so there is exactly one best
    player overall and one best per position — ties would make "who is captain"
    ambiguous and the assertions meaningless.
    """
    rows = []
    pid = 1
    spec = [(1, 4, 45), (2, 10, 50), (3, 10, 60), (4, 6, 70)]
    for position, n, base in spec:
        for _ in range(n):
            rows.append(
                {
                    "player_id": pid,
                    "web_name": f"P{position}_{pid}",
                    "first_name": "First",
                    "second_name": f"Last{pid}",
                    "position": position,
                    "team_id": pid % 12,
                    "now_price": base,
                    "xp": 10.0 - 0.1 * pid,
                    "club_short": f"T{pid % 12:02d}",
                    "club": f"Club {pid % 12}",
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def legal_fifteen(pool: pd.DataFrame) -> list[int]:
    ids: list[int] = []
    for position, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
        ids += pool[pool["position"] == position]["player_id"].head(n).tolist()
    return ids


class TestNameResolution:
    @pytest.fixture
    def pool(self):
        return pd.DataFrame(
            [
                {
                    "player_id": 1, "web_name": "Gyökeres", "first_name": "Viktor",
                    "second_name": "Gyökeres", "position": 4, "team_id": 1,
                    "now_price": 75, "xp": 5.0, "club_short": "ARS", "club": "Arsenal",
                },
                {
                    "player_id": 2, "web_name": "Palmer", "first_name": "Cole",
                    "second_name": "Palmer", "position": 3, "team_id": 2,
                    "now_price": 95, "xp": 4.0, "club_short": "CHE", "club": "Chelsea",
                },
                {
                    "player_id": 3, "web_name": "Palmer", "first_name": "Alex",
                    "second_name": "Palmer", "position": 1, "team_id": 3,
                    "now_price": 40, "xp": 2.0, "club_short": "IPS", "club": "Ipswich",
                },
            ]
        )

    def test_accents_are_optional(self, pool):
        assert resolve_players(["Gyokeres"], pool)[0].player_id == 1
        assert resolve_players(["Gyökeres"], pool)[0].player_id == 1

    def test_ambiguous_surname_is_refused_not_guessed(self, pool):
        r = resolve_players(["Palmer"], pool)[0]
        assert not r.ok
        assert len(r.candidates) == 2

    def test_club_hint_disambiguates(self, pool):
        assert resolve_players(["Palmer (IPS)"], pool)[0].player_id == 3
        assert resolve_players(["Palmer (CHE)"], pool)[0].player_id == 2

    def test_numeric_id_accepted(self, pool):
        assert resolve_players(["2"], pool)[0].player_id == 2

    def test_unknown_name_reports_no_match(self, pool):
        r = resolve_players(["Nonexistent"], pool)[0]
        assert not r.ok and r.candidates == []


class TestEvaluation:
    def test_scores_a_legal_squad(self):
        pool = make_pool()
        ev = evaluate_squad(legal_fifteen(pool), pool, RULES)
        assert ev.legality == []
        assert len(ev.squad) == 15
        # XI is 11 players; captain is one of them and is counted once in xi_xp.
        assert (ev.squad["role"] != "bench").sum() == 11
        assert ev.total_xp == pytest.approx(ev.xi_xp + ev.squad["xp"].max())

    def test_captain_is_the_highest_xp_starter(self):
        pool = make_pool()
        ev = evaluate_squad(legal_fifteen(pool), pool, RULES)
        best = ev.squad.loc[ev.squad["xp"].idxmax(), "player_id"]
        assert ev.captain_id == int(best)

    def test_illegal_squad_is_reported(self):
        pool = make_pool()
        # Six defenders, four midfielders: wrong shape.
        ids = (
            pool[pool["position"] == 1]["player_id"].head(2).tolist()
            + pool[pool["position"] == 2]["player_id"].head(6).tolist()
            + pool[pool["position"] == 3]["player_id"].head(4).tolist()
            + pool[pool["position"] == 4]["player_id"].head(3).tolist()
        )
        ev = evaluate_squad(ids, pool, RULES)
        assert ev.legality != []

    def test_upgrades_are_affordable_and_positive(self):
        pool = make_pool()
        # Deliberately pick the *worst* players so upgrades must exist.
        ids: list[int] = []
        for position, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
            ids += pool[pool["position"] == position]["player_id"].tail(n).tolist()
        sel = pool[pool["player_id"].isin(ids)]
        ups = suggest_upgrades(sel, pool, RULES, bank=0)
        assert not ups.empty
        assert (ups["gain"] > 0).all()
        # Never suggests spending money that isn't there.
        assert (ups["spend"] <= 0).all()

    def test_upgrades_respect_the_club_limit(self):
        pool = make_pool()
        ids = legal_fifteen(pool)
        # Put three owned defenders on one club, and park an irresistible fourth
        # defender there too. Signing him would breach the 3-per-club rule.
        owned_defs = pool[pool["position"] == 2]["player_id"].head(3).tolist()
        blocked = int(pool[pool["position"] == 2]["player_id"].iloc[-1])
        pool.loc[pool["player_id"].isin([*owned_defs, blocked]), "team_id"] = 99
        pool.loc[pool["player_id"] == blocked, "xp"] = 99.0

        sel = pool[pool["player_id"].isin(ids)]
        assert (sel["team_id"] == 99).sum() == RULES.squad.max_per_club

        ups = suggest_upgrades(sel, pool, RULES, bank=0)
        blocked_name = pool.loc[pool["player_id"] == blocked, "web_name"].iloc[0]
        # He may replace a team-mate, which leaves the count at three, but never
        # anyone from another club — that would make four.
        replaces = set(ups[ups["in_name"] == blocked_name]["out_name"])
        assert replaces <= set(sel[sel["team_id"] == 99]["web_name"])
