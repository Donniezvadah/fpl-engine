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
