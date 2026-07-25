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

# Stable cross-season player identity. The API reassigns element ids every
# season but `code` is permanent, so this is what links a player's 2025-26 record
# to his 2026-27 one. Within-season rolling windows group on (SEASON_KEY,
# PLAYER_KEY); cross-season shrinkage targets group on PLAYER_KEY alone.
PLAYER_KEY = "player_key"
SEASON_KEY = "season"

# A player needs at least this many prior-season minutes before we shrink toward
# his own history rather than the league prior (~5 full matches).
_MIN_PRIOR_MINUTES = 450.0

# Pseudo-count controlling how fast this season's minutes data displaces the
# carried-over prior-season estimate in the availability features. Set to 0 to
# disable carryover entirely.
CARRYOVER_PSEUDOCOUNT = 6.0

# Carryover only applies to this many opening gameweeks, ramping to zero. Keeps
# the rest of the season identical to carrying nothing over.
EARLY_SEASON_GWS = 6

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


def available_seasons() -> list[str]:
    """Seasons with played-match data in the database, oldest first."""
    engine = get_engine()
    df = pd.read_sql(select(PlayerGameweek.season).distinct(), engine)
    return sorted(df["season"].tolist())


def training_seasons_for(target_season: str) -> list[str]:
    """Every season with data that precedes the target — the default training set."""
    return [s for s in available_seasons() if s < target_season]


def latest_backtestable_season() -> str:
    """Newest season that has finished matches — the right default to replay."""
    engine = get_engine()
    df = pd.read_sql(
        select(Fixture.season).where(Fixture.finished.is_(True)).distinct(), engine
    )
    if df.empty:
        raise RuntimeError("no finished fixtures in the database; run `fpl update` first")
    return sorted(df["season"].tolist())[-1]


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


def _trailing_sums(
    df: pd.DataFrame, stat: str, window: int, keys: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Trailing sums of `stat` and of minutes over `window`, grouped by `keys`.

    shift(1) excludes the current row's own match, so a feature can never see
    the outcome it is used to predict.
    """
    g = df.groupby(keys, sort=False)
    stat_sum = g[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    min_sum = g["minutes"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    return stat_sum.fillna(0.0), min_sum.fillna(0.0)


def _per90_trailing(
    df: pd.DataFrame,
    stat: str,
    window: int,
    prior: float,
    prior_minutes: float = 450.0,
) -> pd.Series:
    """Minutes-weighted trailing per-90 rate with empirical-Bayes shrinkage.

    The trailing window is within-season. The shrinkage target is the player's
    own rate in earlier seasons where we have one, falling back to the league
    prior otherwise — so a player with thin current-season data is pulled toward
    what he actually did last year rather than toward an average footballer.
    Once the within-season sample is large it dominates and the target stops
    mattering, which is why this helps early season without disturbing later
    gameweeks.
    """
    stat_sum, min_sum = _trailing_sums(df, stat, window, [SEASON_KEY, PLAYER_KEY])
    target = _prior_season_rate(df, stat, prior)
    return (stat_sum + target * prior_minutes / 90.0) / (min_sum + prior_minutes) * 90.0


def _blended_rolling_mean(
    df: pd.DataFrame, col: str, window: int, pseudocount: float
) -> pd.Series:
    """Trailing mean of `col`, blending within-season form with prior-season base.

    Availability features (minutes, starts) are raw means rather than shrunk
    rates, so carryover has to be explicit: with no matches played yet this
    season the value comes entirely from last season, and each new match shifts
    weight onto the current one. Without this a new season would open with every
    player looking like he never plays.

    The carried-over term is a whole-prior-season average, deliberately not a
    trailing window spanning the season break. The closing gameweeks of a season
    are full of dead-rubber rotation, so a trailing window straddling the boundary
    reads a rested starter as a benchwarmer -- it put Haaland, a 34-start season
    behind him, at a 53% chance of starting the opener.
    """
    within = df.groupby([SEASON_KEY, PLAYER_KEY], sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0.0)
    if pseudocount <= 0:
        return within

    carried = _prior_season_mean(df, col)
    n_this_season = df.groupby([SEASON_KEY, PLAYER_KEY], sort=False).cumcount()
    # Weight on carried history falls as this season's sample grows...
    w_carry = pseudocount / (n_this_season + pseudocount)
    # ...and is switched off entirely once the season is no longer young. Beyond
    # that point every appearance-thin player is thin for a reason (rotation,
    # injury, a new role) and last year's record misleads: allowing it through
    # cost 3.8% rank correlation across GW1-38 while helping only the opening
    # weeks. Gating on gameweek makes mid-season behaviour bit-identical to
    # carrying nothing over, so this can only help where it is switched on.
    ramp = ((EARLY_SEASON_GWS - (df["event"] - 1)) / EARLY_SEASON_GWS).clip(0.0, 1.0)
    w_carry = w_carry * ramp
    return (1.0 - w_carry) * within + w_carry * carried


def _prior_season_mean(df: pd.DataFrame, col: str, min_matches: int = 5) -> pd.Series:
    """Each row's per-match average of `col` over the player's earlier seasons.

    Constant within a season, so it carries no within-season leakage; zero for
    players with no meaningful prior record (newly promoted or newly signed).
    """
    by_ps = df.groupby([PLAYER_KEY, SEASON_KEY], sort=True)[col].agg(["sum", "count"])
    cum = by_ps.groupby(level=0).cumsum().groupby(level=0).shift(1)
    counts = cum["count"]
    mean = cum["sum"] / counts.where(counts > 0)
    mean = mean.where(counts >= min_matches)
    mapped = pd.MultiIndex.from_arrays([df[PLAYER_KEY], df[SEASON_KEY]]).map(mean)
    return pd.Series(mapped, index=df.index, dtype="float64").fillna(0.0)


def _prior_season_rate(df: pd.DataFrame, stat: str, league_prior: float) -> pd.Series:
    """Each row's per-90 rate for `stat` across the player's *earlier* seasons.

    Computed once per (player, season) from completed prior seasons only, so it
    is constant within a season and carries no within-season leakage.
    """
    by_ps = df.groupby([PLAYER_KEY, SEASON_KEY], sort=True)[[stat, "minutes"]].sum()
    # Cumulative totals over strictly earlier seasons for the same player.
    cum = by_ps.groupby(level=0).cumsum().groupby(level=0).shift(1)
    minutes = cum["minutes"]
    rate = cum[stat] / minutes.where(minutes > 0) * 90.0
    # Require a meaningful sample before trusting a player's own history.
    rate = rate.where(minutes >= _MIN_PRIOR_MINUTES)
    mapped = pd.MultiIndex.from_arrays([df[PLAYER_KEY], df[SEASON_KEY]]).map(rate)
    return pd.Series(mapped, index=df.index, dtype="float64").fillna(league_prior)


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


def build_feature_frame(
    seasons: list[str], carryover_pseudocount: float = CARRYOVER_PSEUDOCOUNT
) -> pd.DataFrame:
    """The master modeling frame: one row per player-fixture with features + targets."""
    df = load_player_gameweeks(seasons)
    if df.empty:
        raise RuntimeError(f"no player_gameweeks rows for seasons {seasons}; run `fpl update`")
    fixtures = load_fixtures(seasons)
    return compute_features(df, fixtures, carryover_pseudocount)


def compute_features(
    df: pd.DataFrame,
    fixtures: pd.DataFrame,
    carryover_pseudocount: float = CARRYOVER_PSEUDOCOUNT,
) -> pd.DataFrame:
    """Compute backward-looking features on a player_gameweeks frame.

    `df` may include synthetic future rows (zeros for stats); since every feature
    is shift(1)-rolling, those rows receive features from real past matches only.

    `carryover_pseudocount` controls how much last season's record informs a
    player before this season has data on him. Setting it to 0 disables
    carryover entirely (strict within-season features).
    """
    # Stable identity across seasons: the permanent `code` where present, else a
    # per-season fallback (negated so it can never collide with a real code).
    code = pd.to_numeric(df.get("code"), errors="coerce")
    df[PLAYER_KEY] = code.where(code.notna(), -df["player_id"]).astype("int64")

    # Chronological within player: season strings sort correctly ("2025-26" < "2026-27").
    df = df.sort_values([PLAYER_KEY, "season", "event", "fixture_id"]).reset_index(drop=True)

    # Targets.
    df["played"] = (df["minutes"] > 0).astype(int)
    df["played60"] = (df["minutes"] >= 60).astype(int)
    df["started"] = df["started"].fillna(df["minutes"] >= 60).astype(int)

    df["last1_minutes"] = _blended_rolling_mean(df, "minutes", 1, carryover_pseudocount)
    for win in (SHORT_WIN, LONG_WIN):
        df[f"roll{win}_minutes"] = _blended_rolling_mean(
            df, "minutes", win, carryover_pseudocount
        )
    df[f"roll{SHORT_WIN}_started"] = _blended_rolling_mean(
        df, "started", SHORT_WIN, carryover_pseudocount
    )
    df[f"roll{LONG_WIN}_played60"] = _blended_rolling_mean(
        df, "played60", LONG_WIN, carryover_pseudocount
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
