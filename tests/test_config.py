"""Settings resolution: season rollover and tolerant team-id parsing."""

from datetime import date

import pytest

from fpl_engine.config import Settings, current_season


class TestCurrentSeason:
    @pytest.mark.parametrize(
        ("today", "expected"),
        [
            (date(2026, 7, 1), "2026-27"),  # new game opens in July
            (date(2026, 7, 25), "2026-27"),
            (date(2026, 8, 21), "2026-27"),  # season starts
            (date(2027, 1, 2), "2026-27"),  # midway, calendar year rolls over
            (date(2027, 5, 24), "2026-27"),  # season ends
            (date(2027, 6, 15), "2026-27"),  # close season still belongs to it
            (date(2027, 7, 1), "2027-28"),  # next game opens
        ],
    )
    def test_season_label_from_date(self, today, expected):
        assert current_season(today) == expected


class TestTeamId:
    def test_blank_means_unlinked(self, monkeypatch):
        # An unset CI variable expands to "" — that must mean draft mode, not a crash.
        monkeypatch.setenv("FPL_TEAM_ID", "")
        assert Settings().team_id == 0

    def test_whitespace_means_unlinked(self, monkeypatch):
        monkeypatch.setenv("FPL_TEAM_ID", "   ")
        assert Settings().team_id == 0

    def test_real_id_parses(self, monkeypatch):
        monkeypatch.setenv("FPL_TEAM_ID", "1234567")
        assert Settings().team_id == 1234567

    def test_garbage_still_rejected(self, monkeypatch):
        monkeypatch.setenv("FPL_TEAM_ID", "not-a-number")
        with pytest.raises(Exception):
            Settings()


class TestFreeTransferDerivation:
    """The API never publishes free transfers, so they are replayed from the
    official banking rule over the transfers actually made."""

    def _state(self, rows):
        import pandas as pd

        return pd.DataFrame(
            [{"event": e, "event_transfers": t, "active_chip": c} for e, t, c in rows]
        )

    def test_one_after_an_untouched_opening_gameweek(self):
        from fpl_engine.recommend.weekly import free_transfers_for_next_gw
        from fpl_engine.rules import rules_for

        # The opening squad is built with unlimited transfers, so the first free
        # transfer is only earned after GW1.
        state = self._state([(1, 0, None)])
        assert free_transfers_for_next_gw(state, rules_for("2026-27")) == 1

    def test_unused_transfers_accumulate_and_cap(self):
        from fpl_engine.recommend.weekly import free_transfers_for_next_gw
        from fpl_engine.rules import rules_for

        rules = rules_for("2026-27")
        state = self._state([(gw, 0, None) for gw in range(1, 4)])
        assert free_transfers_for_next_gw(state, rules) == 3
        state = self._state([(gw, 0, None) for gw in range(1, 12)])
        assert free_transfers_for_next_gw(state, rules) == rules.transfers.max_banked_free_transfers

    def test_spending_transfers_reduces_the_bank(self):
        from fpl_engine.recommend.weekly import free_transfers_for_next_gw
        from fpl_engine.rules import rules_for

        # Bank two, spend two, and you are back to one the following week.
        state = self._state([(1, 0, None), (2, 0, None), (3, 2, None)])
        assert free_transfers_for_next_gw(state, rules_for("2026-27")) == 1

    def test_hits_never_push_the_bank_negative(self):
        from fpl_engine.recommend.weekly import free_transfers_for_next_gw
        from fpl_engine.rules import rules_for

        state = self._state([(1, 0, None), (2, 6, None)])
        assert free_transfers_for_next_gw(state, rules_for("2026-27")) == 1

    def test_chip_transfers_do_not_consume_the_bank(self):
        from fpl_engine.recommend.weekly import free_transfers_for_next_gw
        from fpl_engine.rules import rules_for

        # A wildcard rebuild leaves the bank untouched: two banked going in,
        # still two after, plus the one earned for the following week.
        state = self._state([(1, 0, None), (2, 0, None), (3, 11, "wildcard")])
        assert free_transfers_for_next_gw(state, rules_for("2026-27")) == 3


class TestSellPriceEstimation:
    """The public picks endpoint omits purchase and selling price, so they are
    reconstructed from the price a player was worth when he entered the squad.
    Defaulting them to zero told the optimiser selling raised no money, and it
    could never afford a transfer."""

    def _frames(self, first_gw=1, bought=50, now=55):
        import pandas as pd

        picks = pd.DataFrame([{"player_id": 7, "event": first_gw}])
        squad_rows = pd.DataFrame([{"player_id": 7}])
        return picks, squad_rows, bought, now

    def test_profit_is_shared_with_the_game(self, monkeypatch):
        from fpl_engine.rules import rules_for
        from fpl_engine.rules.transfers import sell_price

        rules = rules_for("2026-27")
        # Bought at 5.0, now 5.5: keep half the 0.5 profit, rounded down per 0.1.
        assert sell_price(50, 55, rules.transfers) == 52

    def test_a_price_fall_is_absorbed_in_full(self):
        from fpl_engine.rules import rules_for
        from fpl_engine.rules.transfers import sell_price

        rules = rules_for("2026-27")
        assert sell_price(55, 50, rules.transfers) == 50

    def test_unchanged_price_sells_for_what_it_cost(self):
        from fpl_engine.rules import rules_for
        from fpl_engine.rules.transfers import sell_price

        rules = rules_for("2026-27")
        assert sell_price(55, 55, rules.transfers) == 55


class TestNextGameweek:
    """Once a deadline passes the squad is locked even though the matches are
    still being played, so planning must skip a gameweek already under way."""

    def _fixtures(self, rows):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "season": "2026-27",
                    "fixture_id": 100 + i,
                    "event": ev,
                    "kickoff_time": pd.Timestamp(ko),
                    "team_h": 1,
                    "team_a": 2,
                    "team_h_score": None,
                    "team_a_score": None,
                    "team_h_difficulty": 3,
                    "team_a_difficulty": 3,
                    "finished": fin,
                }
                for i, (ev, ko, fin) in enumerate(rows)
            ]
        )

    def test_skips_a_gameweek_already_kicked_off(self, monkeypatch, tmp_path):
        import pandas as pd

        from fpl_engine.recommend import weekly

        fx = self._fixtures(
            [
                (1, "2026-08-21T18:00:00", True),
                (2, "2026-08-28T19:00:00", False),  # under way, deadline gone
                (3, "2026-09-04T18:00:00", False),
            ]
        )
        monkeypatch.setattr(weekly.pd, "read_sql", lambda *a, **k: fx)
        now = pd.Timestamp("2026-08-31T19:30:00")
        assert weekly.next_gameweek("2026-27", now=now) == 3

    def test_returns_the_gameweek_still_open(self, monkeypatch):
        import pandas as pd

        from fpl_engine.recommend import weekly

        fx = self._fixtures(
            [
                (1, "2026-08-21T18:00:00", True),
                (2, "2026-08-28T19:00:00", False),
            ]
        )
        monkeypatch.setattr(weekly.pd, "read_sql", lambda *a, **k: fx)
        now = pd.Timestamp("2026-08-25T12:00:00")
        assert weekly.next_gameweek("2026-27", now=now) == 2
