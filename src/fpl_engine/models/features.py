"""Feature engineering for the prediction models.

Builds a per-(player, gameweek-fixture) frame from the player_gameweeks table with
strictly backward-looking rolling features — every feature for a GW row uses only
information available before that GW's deadline (enforced with shift(1) before
rolling). This same code path serves training, backtesting, and live prediction,
which is what keeps the walk-forward backtest honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from ..db.models import Fixture, Player, PlayerGameweek, Team
from ..db.session import get_engine

# Trailing windows (in appearances/fixtures) used for form features.
SHORT_WIN = 4
LONG_WIN = 8

FEATURE_COLUMNS = [
    "position",
    "was_home",
    "value",
    "own_pct",
    # availability / role
    "last1_minutes",
    f"roll{SHORT_WIN}_minutes",
    f"roll{LONG_WIN}_minutes",
    f"roll{SHORT_WIN}_started",
    f"roll{LONG_WIN}_played60",
    # attacking form (per-90, trailing, minutes-weighted)
    "xg90",
    "xa90",
    "threat90",
    "creativity90",
    "goals90",
    "assists90",
    # defensive / keeper
    "saves90",
    "defcon90",
    "xgc90",
    # bonus propensity
    "bps90",
    "bonus90",
    "points90",
    # team & opponent context
    "team_goals_for_r6",
    "team_goals_against_r6",
    "opp_goals_for_r6",
    "opp_goals_against_r6",
    "fdr",
]

TARGET_COLUMNS = [
    "minutes",
    "started",
    "played",
    "played60",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
]


def load_player_gameweeks(seasons: list[str]) -> pd.DataFrame:
    engine = get_engine()
    q = select(PlayerGameweek).where(PlayerGameweek.season.in_(seasons))
    df = pd.read_sql(q, engine)
    return df


def load_fixtures(seasons: list[str]) -> pd.DataFrame:
    engine = get_engine()
    q = select(Fixture).where(Fixture.season.in_(seasons))
    return pd.read_sql(q, engine)


def load_players(season: str) -> pd.DataFrame:
    engine = get_engine()
    q = select(Player).where(Player.season == season)
    return pd.read_sql(q, engine)


def load_teams(seasons: list[str]) -> pd.DataFrame:
    engine = get_engine()
    q = select(Team).where(Team.season.in_(seasons))
    return pd.read_sql(q, engine)


def _per90_trailing(df: pd.DataFrame, stat: str, window: int, prior: float, prior_minutes: float = 450.0) -> pd.Series:
    """Minutes-weighted trailing per-90 rate with empirical-Bayes shrinkage.

    rate = (sum(stat over window) + prior * prior_minutes/90) / (sum(min) + prior_minutes) * 90
    Uses shift(1) so the current row's own match is excluded.
    """
    g = df.groupby(["season", "player_id"], sort=False)
    stat_sum = g[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    min_sum = g["minutes"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    stat_sum = stat_sum.fillna(0.0)
    min_sum = min_sum.fillna(0.0)
    return (stat_sum + prior * prior_minutes / 90.0) / (min_sum + prior_minutes) * 90.0


def _team_rolling_goals(fixtures: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Per (season, team, event): trailing mean goals for/against over finished matches."""
    fin = fixtures[fixtures["finished"] & fixtures["event"].notna()].copy()
    home = fin[["season", "event", "team_h", "team_h_score", "team_a_score"]].rename(
        columns={"team_h": "team", "team_h_score": "gf", "team_a_score": "ga"}
    )
    away = fin[["season", "event", "team_a", "team_a_score", "team_h_score"]].rename(
        columns={"team_a": "team", "team_a_score": "gf", "team_h_score": "ga"}
    )
    matches = pd.concat([home, away], ignore_index=True).sort_values(["season", "team", "event"])
    g = matches.groupby(["season", "team"], sort=False)
    matches["goals_for_r6"] = g["gf"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    matches["goals_against_r6"] = g["ga"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    league_gf = matches["gf"].mean()
    matches["goals_for_r6"] = matches["goals_for_r6"].fillna(league_gf)
    matches["goals_against_r6"] = matches["goals_against_r6"].fillna(league_gf)
    # A team can play twice in a GW; average the trailing values in that case.
    return (
        matches.groupby(["season", "team", "event"])[["goals_for_r6", "goals_against_r6"]]
        .mean()
        .reset_index()
    )


# Priors (league-typical per-90 rates) used for shrinkage of thin samples.
_PRIORS = {
    "expected_goals": 0.10,
    "expected_assists": 0.08,
    "threat": 25.0,
    "creativity": 12.0,
    "goals_scored": 0.10,
    "assists": 0.08,
    "saves": 2.5,
    "defensive_contribution": 6.0,
    "expected_goals_conceded": 1.4,
    "bps": 12.0,
    "bonus": 0.25,
    "total_points": 2.5,
}


def build_feature_frame(seasons: list[str]) -> pd.DataFrame:
    """The master modeling frame: one row per player-fixture with features + targets."""
    df = load_player_gameweeks(seasons)
    if df.empty:
        raise RuntimeError(f"no player_gameweeks rows for seasons {seasons}; run `fpl update`")
    fixtures = load_fixtures(seasons)
    return compute_features(df, fixtures)


def compute_features(df: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Compute backward-looking features on a player_gameweeks frame.

    `df` may include synthetic future rows (zeros for stats); since every feature
    is shift(1)-rolling, those rows receive features from real past matches only.
    """
    df = df.sort_values(["season", "player_id", "event", "fixture_id"]).reset_index(drop=True)

    # Targets.
    df["played"] = (df["minutes"] > 0).astype(int)
    df["played60"] = (df["minutes"] >= 60).astype(int)
    df["started"] = df["started"].fillna(df["minutes"] >= 60).astype(int)

    g = df.groupby(["season", "player_id"], sort=False)
    df["last1_minutes"] = g["minutes"].transform(lambda s: s.shift(1)).fillna(0.0)
    for win in (SHORT_WIN, LONG_WIN):
        df[f"roll{win}_minutes"] = (
            g["minutes"].transform(lambda s: s.shift(1).rolling(win, min_periods=1).mean()).fillna(0.0)
        )
    df[f"roll{SHORT_WIN}_started"] = (
        g["started"].transform(lambda s: s.shift(1).rolling(SHORT_WIN, min_periods=1).mean()).fillna(0.0)
    )
    df[f"roll{LONG_WIN}_played60"] = (
        g["played60"].transform(lambda s: s.shift(1).rolling(LONG_WIN, min_periods=1).mean()).fillna(0.0)
    )

    df["xg90"] = _per90_trailing(df, "expected_goals", LONG_WIN, _PRIORS["expected_goals"])
    df["xa90"] = _per90_trailing(df, "expected_assists", LONG_WIN, _PRIORS["expected_assists"])
    df["threat90"] = _per90_trailing(df, "threat", LONG_WIN, _PRIORS["threat"])
    df["creativity90"] = _per90_trailing(df, "creativity", LONG_WIN, _PRIORS["creativity"])
    df["goals90"] = _per90_trailing(df, "goals_scored", 2 * LONG_WIN, _PRIORS["goals_scored"])
    df["assists90"] = _per90_trailing(df, "assists", 2 * LONG_WIN, _PRIORS["assists"])
    df["saves90"] = _per90_trailing(df, "saves", LONG_WIN, _PRIORS["saves"])
    df["defcon90"] = _per90_trailing(df, "defensive_contribution", LONG_WIN, _PRIORS["defensive_contribution"])
    df["xgc90"] = _per90_trailing(df, "expected_goals_conceded", LONG_WIN, _PRIORS["expected_goals_conceded"])
    df["bps90"] = _per90_trailing(df, "bps", LONG_WIN, _PRIORS["bps"])
    df["bonus90"] = _per90_trailing(df, "bonus", 2 * LONG_WIN, _PRIORS["bonus"])
    df["points90"] = _per90_trailing(df, "total_points", LONG_WIN, _PRIORS["total_points"])

    # Ownership as a fraction of the max (selected counts scale with total entries).
    season_max = df.groupby("season")["selected"].transform("max").replace(0, 1)
    df["own_pct"] = df["selected"] / season_max

    # Team/opponent rolling goals.
    team_roll = _team_rolling_goals(fixtures)
    df = df.merge(
        team_roll.rename(
            columns={
                "team": "team_id",
                "goals_for_r6": "team_goals_for_r6",
                "goals_against_r6": "team_goals_against_r6",
            }
        ),
        on=["season", "team_id", "event"],
        how="left",
    )
    df = df.merge(
        team_roll.rename(
            columns={
                "team": "opponent_team",
                "goals_for_r6": "opp_goals_for_r6",
                "goals_against_r6": "opp_goals_against_r6",
            }
        ),
        on=["season", "opponent_team", "event"],
        how="left",
    )
    league_avg = 1.4
    for c in ("team_goals_for_r6", "team_goals_against_r6", "opp_goals_for_r6", "opp_goals_against_r6"):
        df[c] = df[c].fillna(league_avg)

    # Official fixture difficulty for the player's side.
    fx = fixtures[["season", "fixture_id", "team_h_difficulty", "team_a_difficulty"]]
    df = df.merge(fx, on=["season", "fixture_id"], how="left")
    df["fdr"] = np.where(df["was_home"], df["team_h_difficulty"], df["team_a_difficulty"])
    df["fdr"] = df["fdr"].fillna(3)
    df = df.drop(columns=["team_h_difficulty", "team_a_difficulty"])

    df["was_home"] = df["was_home"].astype(int)
    return df
