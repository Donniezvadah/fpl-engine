"""Direct gradient-boosted points model.

A LightGBM regressor straight from features to total_points. It complements the
Poisson-composed model: composition encodes the scoring rules exactly but relies
on its component estimates; the direct model finds interactions the composition
misses (e.g. bonus clustering, defcon-heavy roles). The ensemble blends them.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS

# Heavily regularised on purpose: weekly points are mostly noise, and the earlier
# 63-leaf / 500-round setup overfit it. On the 2025-26 walk-forward backtest (run
# as two halves, GW1-19 and GW20-38) this cut the direct model's MAE by about 0.03
# to 0.04 and lifted its Spearman by about 0.01 in both halves; the ensemble's MAE
# and RMSE improved or stayed flat in both.
_LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 400,
    "lambda_l2": 20.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


class DirectPointsModel:
    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        # Residual std per predicted-points bucket, for uncertainty estimates.
        self.bucket_edges: np.ndarray | None = None
        self.bucket_std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, n_rounds: int = 400) -> "DirectPointsModel":
        X = df[FEATURE_COLUMNS]
        y = df["total_points"].astype(float)
        self.model = lgb.train(_LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=n_rounds)

        pred = np.asarray(self.model.predict(X), dtype=float)
        resid = y.values - pred
        self.bucket_edges = np.quantile(pred, np.linspace(0, 1, 11))
        self.bucket_edges[0], self.bucket_edges[-1] = -np.inf, np.inf
        stds = []
        for lo, hi in zip(self.bucket_edges[:-1], self.bucket_edges[1:]):
            mask = (pred >= lo) & (pred < hi)
            stds.append(float(resid[mask].std()) if mask.sum() > 20 else float(resid.std()))
        self.bucket_std = np.array(stds)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, variance)."""
        assert self.model is not None, "fit first"
        mean = np.asarray(self.model.predict(df[FEATURE_COLUMNS]), dtype=float)
        if self.bucket_edges is not None and self.bucket_std is not None:
            idx = np.clip(
                np.searchsorted(self.bucket_edges, mean, side="right") - 1,
                0,
                len(self.bucket_std) - 1,
            )
            var = self.bucket_std[idx] ** 2
        else:
            var = np.full_like(mean, 6.25)
        return mean, var
