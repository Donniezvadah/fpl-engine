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
        monkeypatch.setenv("FPL_TEAM_ID", "994469")
        assert Settings().team_id == 994469

    def test_garbage_still_rejected(self, monkeypatch):
        monkeypatch.setenv("FPL_TEAM_ID", "not-a-number")
        with pytest.raises(Exception):
            Settings()
