"""End-to-end per-gameweek expected-points prediction.

`PointsPredictor.fit(train_df, fixtures, teams)` trains the minutes model, the
Dixon-Coles team model, and the direct LightGBM model. `predict(rows)` produces a
frame with per-fixture xP (mean + variance) from both model families plus the
ensemble blend. The same object serves the backtest (fit on past GWs, predict the
next) and live recommendations (fit on everything, predict upcoming fixtures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..rules.constants import SeasonRules
from .direct import DirectPointsModel
from .dixon_coles import DixonColesModel, matches_frame
from .minutes import MinutesModel, availability_from_flags
from .points import compose_xp

# Default blend; refit by the backtest (see backtest/ensemble_weights).
DEFAULT_COMPOSED_WEIGHT = 0.5


@dataclass
class PointsPredictor:
    rules: SeasonRules
    composed_weight: float = DEFAULT_COMPOSED_WEIGHT
    minutes_model: MinutesModel = field(default_factory=MinutesModel)
    direct_model: DirectPointsModel = field(default_factory=DirectPointsModel)
    dc_model: DixonColesModel = field(default_factory=DixonColesModel)
    team_code_by_id: dict[tuple[str, int], int] = field(default_factory=dict)

    def fit(
        self,
        train_df: pd.DataFrame,
        fixtures: pd.DataFrame,
        teams: pd.DataFrame,
        as_of: datetime | None = None,
    ) -> "PointsPredictor":
        self.minutes_model.fit(train_df)
        self.direct_model.fit(train_df)
        self.dc_model.fit(matches_frame(fixtures, teams), as_of=as_of)
        self.team_code_by_id = {
            (r.season, r.team_id): r.code for r in teams.itertuples(index=False)
        }
        return self

    def _codes(self, rows: pd.DataFrame, col: str) -> pd.Series:
        return rows.apply(lambda r: self.team_code_by_id.get((r["season"], r[col]), -1), axis=1)

    def predict(self, rows: pd.DataFrame, availability: np.ndarray | None = None) -> pd.DataFrame:
        """`rows`: feature-frame rows for the fixtures to predict (one per player-fixture)."""
        rows = rows.reset_index(drop=True)
        mins = self.minutes_model.predict(rows, availability=availability)

        team_codes = self._codes(rows, "team_id").values
        opp_codes = self._codes(rows, "opponent_team").values
        is_home = rows["was_home"].astype(bool).values

        lam_team = np.array(
            [
                self.dc_model.team_expected_goals(t, o, h)
                for t, o, h in zip(team_codes, opp_codes, is_home)
            ]
        )
        lam_opp = np.array(
            [
                self.dc_model.team_expected_conceded(t, o, h)
                for t, o, h in zip(team_codes, opp_codes, is_home)
            ]
        )
        baseline = np.array([self.dc_model.baseline_team_goals(t) for t in team_codes])
        fixture_adj = np.clip(lam_team / np.clip(baseline, 0.3, None), 0.4, 2.2)

        e_min_frac = mins.e_minutes / 90.0
        # Blend bookings-independent xG with realized scoring rate (finishing skill).
        goal_rate90 = 0.75 * rows["xg90"].values + 0.25 * rows["goals90"].values
        assist_rate90 = 0.75 * rows["xa90"].values + 0.25 * rows["assists90"].values
        e_goals = goal_rate90 * e_min_frac * fixture_adj
        e_assists = assist_rate90 * e_min_frac * fixture_adj

        # Saves scale with opponent attack relative to league average.
        opp_attack_adj = np.clip(lam_opp / self.dc_model.league_mu, 0.5, 2.0)
        e_saves = rows["saves90"].values * opp_attack_adj

        composed = compose_xp(
            position=rows["position"].values,
            p_play=mins.p_play,
            p60=mins.p60,
            e_minutes=mins.e_minutes,
            e_goals=e_goals,
            e_assists=e_assists,
            lambda_opp=lam_opp,
            e_saves=e_saves,
            defcon_rate=rows["defcon90"].values,
            e_bonus=rows["bonus90"].values,
            rules=self.rules,
        )

        direct_mean, direct_var = self.direct_model.predict(rows)
        # The direct model learned realized points including zeros from absences;
        # at prediction time availability flags must gate it too.
        if availability is not None:
            direct_mean = direct_mean * np.clip(availability, 0.0, 1.0)

        w = self.composed_weight
        xp = w * composed.total + (1 - w) * direct_mean
        xp_var = w**2 * composed.variance + (1 - w) ** 2 * direct_var

        out = rows[
            ["season", "player_id", "event", "fixture_id", "team_id", "opponent_team", "was_home", "position", "value"]
        ].copy()
        for k, v in composed.as_dict().items():
            out[f"composed_{k}" if k in ("xp", "xp_var") else k] = v
        out["direct_xp"] = direct_mean
        out["direct_xp_var"] = direct_var
        out["xp"] = xp
        out["xp_var"] = xp_var
        out["p_start"] = mins.p_start
        out["p60"] = mins.p60
        out["p_play"] = mins.p_play
        out["e_minutes"] = mins.e_minutes
        out["lambda_team"] = lam_team
        out["lambda_opp"] = lam_opp
        out["e_goals"] = e_goals
        out["e_assists"] = e_assists
        out["p_cs"] = np.exp(-lam_opp)
        return out


def availability_for_rows(rows: pd.DataFrame, players: pd.DataFrame) -> np.ndarray:
    """Availability multipliers for prediction rows from current player flags."""
    flags = players.set_index("player_id")[["status", "chance_next_round"]]
    joined = rows.join(flags, on="player_id")
    return availability_from_flags(joined["status"], joined["chance_next_round"])
