"""Price-change prediction.

FPL's price algorithm is unpublished; what is observable is that changes are
driven by net transfers relative to ownership, with rises capped at one 0.1 step
per day and various dampeners. We model P(rise tonight) / P(fall tonight) with a
logistic curve on the net-transfer ratio, calibrated on observed (value-change,
net-transfer) pairs from stored gameweek history whenever data is available, and
fall back to community-standard thresholds otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fallback thresholds: net transfers as a fraction of owners needed for a change.
RISE_THRESHOLD = 0.011
FALL_THRESHOLD = -0.009
SLOPE = 450.0


@dataclass
class PriceForecast:
    player_id: np.ndarray
    p_rise: np.ndarray
    p_fall: np.ndarray
    expected_delta: np.ndarray  # in tenths, over the next ~24h


class PriceModel:
    def __init__(self, rise_threshold: float = RISE_THRESHOLD, fall_threshold: float = FALL_THRESHOLD):
        self.rise_threshold = rise_threshold
        self.fall_threshold = fall_threshold

    def calibrate(self, pgw: pd.DataFrame) -> "PriceModel":
        """Fit thresholds from history: rows where value changed vs their net-transfer ratio."""
        df = pgw.sort_values(["season", "player_id", "event"]).copy()
        g = df.groupby(["season", "player_id"], sort=False)
        df["value_change"] = g["value"].diff()
        df["net_ratio"] = (df["transfers_in"] - df["transfers_out"]) / df["selected"].clip(lower=1)
        rises = df[df["value_change"] > 0]["net_ratio"]
        falls = df[df["value_change"] < 0]["net_ratio"]
        if len(rises) > 50:
            self.rise_threshold = float(rises.quantile(0.25))
        if len(falls) > 50:
            self.fall_threshold = float(falls.quantile(0.75))
        return self

    def predict(self, players: pd.DataFrame) -> PriceForecast:
        """`players`: current players table (transfers_in_event / out / selected_by_percent)."""
        # selected_by_percent is a percentage of total entries; approximate owner count
        # is irrelevant because the ratio uses the same scale for numerator via events.
        selected = players["selected_by_percent"].clip(lower=0.05).values
        # Net transfers this event, normalized by an ownership proxy. Total entries
        # cancels: transfers are absolute, so scale by (pct/100 * entries). We don't
        # know entries; use transfers relative to each other via robust scaling.
        net = (players["transfers_in_event"] - players["transfers_out_event"]).values.astype(float)
        entries_proxy = max(np.abs(net).max() / 0.02, 1.0)  # crude but monotone
        ratio = net / (selected / 100.0 * entries_proxy)

        p_rise = 1.0 / (1.0 + np.exp(-SLOPE * (ratio - self.rise_threshold)))
        p_fall = 1.0 / (1.0 + np.exp(SLOPE * (ratio - self.fall_threshold)))
        return PriceForecast(
            player_id=players["player_id"].values,
            p_rise=p_rise,
            p_fall=p_fall,
            expected_delta=p_rise - p_fall,
        )
