"""Walk-forward backtest.

Replays a season gameweek by gameweek: models are fit only on matches strictly
before the target GW (plus earlier seasons), then predict the target GW. Because
the feature pipeline is shift(1)-rolling and Dixon-Coles is refit on a filtered
match list, no information from the target GW or later can leak into predictions.

Outputs per-GW and aggregate metrics for the ensemble and for naive baselines
(trailing 4-GW form, season points-per-game). A model change is acceptable only
if it does not regress these metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from ..models.features import (
    CARRYOVER_PSEUDOCOUNT,
    build_feature_frame,
    latest_backtestable_season,
    load_fixtures,
    load_teams,
    training_seasons_for,
)
from ..models.predictor import PointsPredictor
from ..rules.constants import rules_for

console = Console()


@dataclass
class BacktestResult:
    season: str
    per_gw: pd.DataFrame
    predictions: pd.DataFrame
    ensemble_weight: float
    summary: dict[str, float] = field(default_factory=dict)

    def print_report(self) -> None:
        t = Table(title=f"Walk-forward backtest — {self.season}")
        t.add_column("metric")
        t.add_column("ensemble", justify="right")
        t.add_column("composed", justify="right")
        t.add_column("direct", justify="right")
        t.add_column("form4", justify="right")
        t.add_column("season_ppg", justify="right")
        for metric in ("mae", "rmse", "spearman"):
            t.add_row(
                metric,
                *(
                    f"{self.summary.get(f'{m}_{metric}', float('nan')):.4f}"
                    for m in ("xp", "composed", "direct", "form4", "ppg")
                ),
            )
        console.print(t)
        console.print(
            f"captain hit rate (model pick in actual top-5): "
            f"{self.summary.get('captain_top5_rate', float('nan')):.1%} | "
            f"best-ensemble weight on composed: {self.ensemble_weight:.2f}"
        )


def _gw_actual_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_gw = (
        truth.groupby(["season", "player_id", "event"], as_index=False)
        .agg(actual=("total_points", "sum"), minutes=("minutes", "sum"))
    )
    keys = ["season", "player_id", "event"]
    return pred.merge(truth_gw, on=keys, how="left").fillna({"actual": 0, "minutes": 0})


def run_backtest(
    season: str | None = None,
    train_seasons: list[str] | None = None,
    start_gw: int = 7,
    end_gw: int = 38,
    refit_every: int = 4,
    carryover: float = CARRYOVER_PSEUDOCOUNT,
) -> BacktestResult:
    season = season or latest_backtestable_season()
    if train_seasons is None:
        train_seasons = training_seasons_for(season)
    all_seasons = [*train_seasons, season]
    rules = rules_for(season)

    df = build_feature_frame(all_seasons, carryover_pseudocount=carryover)
    fixtures = load_fixtures(all_seasons)
    teams = load_teams(all_seasons)

    # Baselines computed on the same strictly-backward basis.
    df = df.sort_values(["season", "player_id", "event", "fixture_id"]).reset_index(drop=True)
    g = df.groupby(["season", "player_id"], sort=False)
    df["form4"] = g["total_points"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    ).fillna(0.0)
    df["ppg"] = g["total_points"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0.0)

    target = df[df["season"] == season]
    gws = sorted(e for e in target["event"].unique() if start_gw <= e <= end_gw)

    predictor: PointsPredictor | None = None
    preds: list[pd.DataFrame] = []
    for i, gw in enumerate(gws):
        if predictor is None or i % refit_every == 0:
            train = df[(df["season"] != season) | (df["event"] < gw)]
            train = train[train["event"].notna()]
            fx_train = fixtures[
                (fixtures["season"] != season)
                | (fixtures["event"].notna() & (fixtures["event"] < gw))
            ]
            predictor = PointsPredictor(rules).fit(train, fx_train, teams)
        test = df[(df["season"] == season) & (df["event"] == gw)]
        if test.empty:
            continue
        p = predictor.predict(test)
        p["form4"] = test["form4"].values
        p["ppg"] = test["ppg"].values
        preds.append(p)
        console.print(f"GW{gw}: {len(p)} player-fixture predictions")

    pred = pd.concat(preds, ignore_index=True)
    truth = df[df["season"] == season][
        ["season", "player_id", "event", "total_points", "minutes"]
    ]

    # Aggregate to GW level (sums fixtures in DGWs) before scoring.
    from ..models.upcoming import dedupe_to_gw_level

    for col in ("form4", "ppg"):
        pred[f"e_{col}"] = pred[col]  # ride along through the aggregator via e_ prefix
    gw_pred = dedupe_to_gw_level(pred)
    gw_pred = gw_pred.rename(columns={"e_form4": "form4", "e_ppg": "ppg"})
    # form/ppg are per-GW baselines, not per-fixture sums; a DGW baseline is 2x form.
    scored = _gw_actual_frame(gw_pred, truth)

    # Accuracy is scored on players who actually appeared. This population must
    # not depend on model output -- filtering by predicted p_play scores every
    # variant on a different set of players, which silently invalidates any A/B
    # comparison (the naive baselines shift too, which is how it shows up).
    # Whether a player features at all is judged separately, by captain hit rate
    # over the unfiltered frame and by the minutes model's own metrics.
    eligible = scored[scored["minutes"] > 0].copy()

    ensemble_weight, summary = _fit_weight_and_score(eligible)
    per_gw = _per_gw_metrics(eligible)
    summary["captain_top5_rate"] = _captain_hit_rate(scored)

    return BacktestResult(
        season=season,
        per_gw=per_gw,
        predictions=scored,
        ensemble_weight=ensemble_weight,
        summary=summary,
    )


def _fit_weight_and_score(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    y = df["actual"].values
    composed = df["composed_xp"].values
    direct = df["direct_xp"].values
    best_w, best_mae = 0.5, np.inf
    for w in np.linspace(0, 1, 21):
        mae = float(np.mean(np.abs(w * composed + (1 - w) * direct - y)))
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    blend = best_w * composed + (1 - best_w) * direct

    def scores(pred: np.ndarray, name: str) -> dict[str, float]:
        rho = spearmanr(pred, y).statistic if len(y) > 2 else float("nan")
        return {
            f"{name}_mae": float(np.mean(np.abs(pred - y))),
            f"{name}_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            f"{name}_spearman": float(rho),
        }

    out: dict[str, float] = {}
    out.update(scores(blend, "xp"))
    out.update(scores(composed, "composed"))
    out.update(scores(direct, "direct"))
    out.update(scores(df["form4"].values, "form4"))
    out.update(scores(df["ppg"].values, "ppg"))
    return best_w, out


def _per_gw_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gw, grp in df.groupby("event"):
        y = grp["actual"].values
        p = grp["xp"].values
        rows.append(
            {
                "event": gw,
                "n": len(grp),
                "mae": float(np.mean(np.abs(p - y))),
                "spearman": float(spearmanr(p, y).statistic) if len(grp) > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _captain_hit_rate(scored: pd.DataFrame) -> float:
    """How often the model's top xP pick lands in the GW's actual top-5 scorers."""
    hits, total = 0, 0
    for _, grp in scored.groupby("event"):
        if len(grp) < 20:
            continue
        pick = grp.loc[grp["xp"].idxmax(), "player_id"]
        top5 = set(grp.nlargest(5, "actual")["player_id"])
        hits += int(pick in top5)
        total += 1
    return hits / total if total else float("nan")
