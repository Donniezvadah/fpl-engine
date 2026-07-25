"""Feature pipeline: cross-season continuity and no-leakage guarantees."""

import pandas as pd

from fpl_engine.models.features import PLAYER_KEY, compute_features


def _fixtures(season_events: list[tuple[str, int, int]]) -> pd.DataFrame:
    """One finished fixture per (season, event) with a fixed 2-1 scoreline."""
    rows = []
    for season, event, fid in season_events:
        rows.append(
            {
                "season": season,
                "fixture_id": fid,
                "event": event,
                "team_h": 1,
                "team_a": 2,
                "team_h_score": 2,
                "team_a_score": 1,
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
                "finished": True,
                "kickoff_time": pd.Timestamp("2026-01-01"),
            }
        )
    return pd.DataFrame(rows)


def _pgw(season: str, player_id: int, code: int, event: int, fid: int, **stats) -> dict:
    row = {
        "season": season,
        "player_id": player_id,
        "code": code,
        "fixture_id": fid,
        "event": event,
        "team_id": 1,
        "opponent_team": 2,
        "was_home": True,
        "position": 3,
        "minutes": 90,
        "started": 1,
        "total_points": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bonus": 0,
        "bps": 0,
        "defensive_contribution": 0,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "expected_goal_involvements": 0.0,
        "expected_goals_conceded": 0.0,
        "influence": 0.0,
        "creativity": 0.0,
        "threat": 0.0,
        "ict_index": 0.0,
        "value": 50,
        "selected": 1000,
        "transfers_in": 0,
        "transfers_out": 0,
    }
    row.update(stats)
    return row


class TestCrossSeasonContinuity:
    """A player's element id changes between seasons but `code` does not, so
    last season's form must carry into the new season's first gameweek."""

    def build(self) -> pd.DataFrame:
        # Same player: id 100 in 2025-26 (prolific), id 7 in 2026-27 (no data yet).
        rows = [
            _pgw("2025-26", 100, 555, ev, ev, expected_goals=0.8, goals_scored=1)
            for ev in range(1, 9)
        ]
        rows.append(_pgw("2026-27", 7, 555, 1, 101, minutes=0))  # upcoming fixture
        fixtures = _fixtures(
            [("2025-26", ev, ev) for ev in range(1, 9)] + [("2026-27", 1, 101)]
        )
        return compute_features(pd.DataFrame(rows), fixtures)

    def test_new_season_row_inherits_last_season_form(self):
        feats = self.build()
        new = feats[(feats["season"] == "2026-27") & (feats["event"] == 1)].iloc[0]
        # Prolific last season (0.8 xG/90) must not be reset to the league prior.
        assert new["xg90"] > 0.5
        assert new["roll8_minutes"] == 90.0
        assert new["roll4_started"] == 1.0

    def test_identity_key_is_the_stable_code(self):
        feats = self.build()
        assert set(feats[PLAYER_KEY].unique()) == {555}

    def test_player_without_code_is_isolated(self):
        rows = [
            _pgw("2025-26", 100, 555, 1, 1, expected_goals=0.9),
            _pgw("2026-27", 7, None, 1, 101, minutes=0),
        ]
        feats = compute_features(
            pd.DataFrame(rows), _fixtures([("2025-26", 1, 1), ("2026-27", 1, 101)])
        )
        # Distinct keys: the code-less row must not inherit another player's form.
        assert feats[PLAYER_KEY].nunique() == 2
        new = feats[feats["season"] == "2026-27"].iloc[0]
        assert new["roll8_minutes"] == 0.0


class TestNoLeakage:
    def test_features_exclude_the_current_match(self):
        # A player who blanks then hauls: the haul row's features must not see it.
        rows = [
            _pgw("2025-26", 1, 11, 1, 1, goals_scored=0, expected_goals=0.0, total_points=2),
            _pgw("2025-26", 1, 11, 2, 2, goals_scored=3, expected_goals=2.5, total_points=20),
        ]
        feats = compute_features(
            pd.DataFrame(rows), _fixtures([("2025-26", 1, 1), ("2025-26", 2, 2)])
        )
        haul = feats[feats["event"] == 2].iloc[0]
        first = feats[feats["event"] == 1].iloc[0]
        # GW2's xg90 reflects only GW1 (0.0 xG), shrunk toward the prior — well
        # below what including its own 2.5 xG would produce.
        assert haul["xg90"] < 0.2
        assert first["roll8_minutes"] == 0.0  # nothing before the first match
