"""Synthesize prediction rows for upcoming fixtures.

For live recommendations we need feature rows for matches that haven't happened.
We append one synthetic player_gameweek row per (player, upcoming fixture) with
zeroed stats, run the shared feature pipeline (all features are shift(1)-rolling,
so synthetic rows inherit features from real past matches only), and return just
the synthetic rows, ready for PointsPredictor.predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import compute_features, load_fixtures, load_player_gameweeks, load_players


def synthesize_future_rows(
    season: str, players: pd.DataFrame, fixtures: pd.DataFrame, gws: list[int]
) -> pd.DataFrame:
    """One row per active player per fixture in the target gameweeks."""
    fx = fixtures[(fixtures["season"] == season) & (fixtures["event"].isin(gws))]
    rows = []
    for f in fx.itertuples(index=False):
        for side, opp, home in ((f.team_h, f.team_a, True), (f.team_a, f.team_h, False)):
            side_players = players[players["team_id"] == side]
            for p in side_players.itertuples(index=False):
                rows.append(
                    {
                        "season": season,
                        "player_id": p.player_id,
                        "fixture_id": f.fixture_id,
                        "event": f.event,
                        "code": p.code,
                        "team_id": side,
                        "opponent_team": opp,
                        "was_home": home,
                        "position": p.position,
                        "value": p.now_price,
                        "selected": 0,
                        "synthetic": True,
                    }
                )
    return pd.DataFrame(rows)


def build_prediction_rows(
    seasons: list[str],
    target_season: str,
    target_gws: list[int],
    cutoff_gw: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (prediction_rows_with_features, players_frame).

    `cutoff_gw` drops target-season rows from GW `cutoff_gw` onwards before
    feature building — an honest as-of-deadline view for retrospective runs.
    """
    history = load_player_gameweeks(seasons)
    fixtures = load_fixtures(seasons)
    players = load_players(target_season)
    if cutoff_gw is not None:
        history = history[
            (history["season"] != target_season) | (history["event"] < cutoff_gw)
        ]

    future = synthesize_future_rows(target_season, players, fixtures, target_gws)
    if future.empty:
        raise RuntimeError(f"no fixtures found for {target_season} GWs {target_gws}")

    history = history.assign(synthetic=False)
    combined = pd.concat([history, future], ignore_index=True, sort=False)

    # Numeric stat columns absent from synthetic rows default to 0.
    stat_cols = [
        c
        for c in history.columns
        if c not in future.columns and pd.api.types.is_numeric_dtype(history[c])
    ]
    combined[stat_cols] = combined[stat_cols].fillna(0)
    combined["started"] = combined["started"].fillna(0)

    # Ownership for synthetic rows from the players table (selected_by_percent).
    own = players.set_index("player_id")["selected_by_percent"] / 100.0
    features = compute_features(combined, fixtures)
    mask = features["synthetic"].fillna(False).astype(bool)
    target = features[mask].copy()
    target["own_pct"] = target["player_id"].map(own).fillna(0.0)

    # Keep only rows for the requested GWs (safety).
    target = target[(target["season"] == target_season) & (target["event"].isin(target_gws))]
    return target.reset_index(drop=True), players


def dedupe_to_gw_level(pred: pd.DataFrame) -> pd.DataFrame:
    """Sum per-fixture xP to per-(player, GW) — handles double gameweeks."""
    keys = ["season", "player_id", "event"]
    first_cols = {
        "team_id": "first",
        "position": "first",
        "value": "first",
        "p_start": "max",
        "p60": "max",
        "p_play": "max",
    }
    sums = {
        c: "sum"
        for c in pred.columns
        if c.startswith(("xp", "composed_", "direct_", "e_", "lambda_")) or c in ("p_cs",)
    }
    agg = {**first_cols, **sums}
    agg = {k: v for k, v in agg.items() if k in pred.columns}
    out = pred.groupby(keys, as_index=False).agg(agg)
    n_fixtures = pred.groupby(keys).size().rename("n_fixtures").reset_index()
    return out.merge(n_fixtures, on=keys)
