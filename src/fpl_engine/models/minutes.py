"""Minutes / availability model.

Decomposes playing time into three states — start, cameo (sub appearance), absent —
because FPL scoring is highly non-linear in minutes (the 60-minute threshold gates
appearance and clean-sheet points).

Two LightGBM classifiers:
  P(start)              — trained on all rows
  P(60+ | started)      — trained on started rows
plus an empirical P(play | not started) cameo rate from trailing bench behaviour.

Availability flags from the API (status, chance_of_playing) scale the result at
prediction time — injuries/suspensions are facts, not things to learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS

# Average minutes conditional on state (league-typical; recomputed at fit time).
DEFAULT_START_60_MIN = 87.0
DEFAULT_START_SHORT_MIN = 38.0
DEFAULT_CAMEO_MIN = 18.0

_LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


@dataclass
class MinutesPrediction:
    p_start: np.ndarray
    p60: np.ndarray  # unconditional P(minutes >= 60)
    p_play: np.ndarray  # unconditional P(minutes > 0)
    e_minutes: np.ndarray


class MinutesModel:
    def __init__(self) -> None:
        self.start_model: lgb.Booster | None = None
        self.p60_model: lgb.Booster | None = None
        self.cameo_rate: float = 0.35
        self.start_60_min = DEFAULT_START_60_MIN
        self.start_short_min = DEFAULT_START_SHORT_MIN
        self.cameo_min = DEFAULT_CAMEO_MIN

    def fit(self, df: pd.DataFrame, n_rounds: int = 300) -> "MinutesModel":
        X = df[FEATURE_COLUMNS]
        self.start_model = lgb.train(
            _LGB_PARAMS, lgb.Dataset(X, label=df["started"]), num_boost_round=n_rounds
        )
        started = df[df["started"] == 1]
        self.p60_model = lgb.train(
            _LGB_PARAMS,
            lgb.Dataset(started[FEATURE_COLUMNS], label=started["played60"]),
            num_boost_round=n_rounds,
        )
        bench = df[df["started"] == 0]
        if len(bench):
            self.cameo_rate = float(bench["played"].mean())
        s60 = started[started["played60"] == 1]["minutes"]
        s_short = started[(started["played60"] == 0) & (started["minutes"] > 0)]["minutes"]
        cameo = bench[bench["played"] == 1]["minutes"]
        if len(s60):
            self.start_60_min = float(s60.mean())
        if len(s_short):
            self.start_short_min = float(s_short.mean())
        if len(cameo):
            self.cameo_min = float(cameo.mean())
        return self

    def predict(
        self, df: pd.DataFrame, availability: np.ndarray | None = None
    ) -> MinutesPrediction:
        """`availability` is P(fit to play) in [0,1] from status/chance flags."""
        assert self.start_model is not None and self.p60_model is not None, "fit first"
        X = df[FEATURE_COLUMNS]
        p_start = np.asarray(self.start_model.predict(X), dtype=float)
        p60_given_start = np.asarray(self.p60_model.predict(X), dtype=float)
        if availability is not None:
            p_start = p_start * availability

        p_cameo = (1.0 - p_start) * self.cameo_rate * (
            availability if availability is not None else 1.0
        )
        p_play = p_start + p_cameo
        p60 = p_start * p60_given_start
        e_minutes = (
            p_start * p60_given_start * self.start_60_min
            + p_start * (1 - p60_given_start) * self.start_short_min
            + p_cameo * self.cameo_min
        )
        return MinutesPrediction(p_start=p_start, p60=p60, p_play=p_play, e_minutes=e_minutes)


def availability_from_flags(status: pd.Series, chance_next_round: pd.Series) -> np.ndarray:
    """Map API status + chance-of-playing to a multiplier on P(start).

    status: a=available, d=doubtful, i=injured, s=suspended, u=unavailable, n=not in squad.
    chance_next_round is 0/25/50/75/100 or NaN.
    """
    avail = np.ones(len(status), dtype=float)
    st = status.fillna("a").values
    chance = pd.to_numeric(chance_next_round, errors="coerce").values
    for i, s in enumerate(st):
        if s in ("i", "s", "u", "n"):
            avail[i] = 0.0
        elif s == "d":
            avail[i] = (chance[i] / 100.0) if not np.isnan(chance[i]) else 0.5
        elif not np.isnan(chance[i]):
            avail[i] = chance[i] / 100.0
    return avail
