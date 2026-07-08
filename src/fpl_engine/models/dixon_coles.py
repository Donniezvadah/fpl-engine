"""Dixon-Coles team goal model.

Fits per-team attack/defence strengths and home advantage by maximizing a
time-decayed Poisson likelihood over finished matches (Dixon & Coles 1997, with
the low-score dependence term omitted — it matters for correct-score betting,
not for the expected-goal rates and clean-sheet probabilities we consume).

Teams are keyed by their stable FPL `code`, so history joins cleanly across
seasons and newly promoted sides simply start from the shrinkage prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DECAY_HALF_LIFE_DAYS = 120.0  # weight on a match halves every ~4 months
SHRINKAGE = 5.0  # ridge pull of strengths toward league average


@dataclass
class MatchRates:
    lambda_home: float
    lambda_away: float

    def clean_sheet_prob_home(self) -> float:
        return float(np.exp(-self.lambda_away))

    def clean_sheet_prob_away(self) -> float:
        return float(np.exp(-self.lambda_home))


class DixonColesModel:
    def __init__(self) -> None:
        self.attack: dict[int, float] = {}  # by team code
        self.defence: dict[int, float] = {}
        self.home_adv: float = 0.25
        self.league_mu: float = 1.35  # baseline goals per team per match

    def fit(self, matches: pd.DataFrame, as_of: datetime | None = None) -> "DixonColesModel":
        """`matches` needs: home_code, away_code, home_goals, away_goals, kickoff_time."""
        m = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if m.empty:
            raise ValueError("no finished matches to fit on")
        as_of = as_of or datetime.utcnow()
        age_days = (as_of - pd.to_datetime(m["kickoff_time"])).dt.total_seconds() / 86400.0
        w = np.power(0.5, np.clip(age_days, 0, None) / DECAY_HALF_LIFE_DAYS)

        codes = sorted(set(m["home_code"]) | set(m["away_code"]))
        idx = {c: i for i, c in enumerate(codes)}
        n = len(codes)
        hi = m["home_code"].map(idx).values
        ai = m["away_code"].map(idx).values
        hg = m["home_goals"].values.astype(float)
        ag = m["away_goals"].values.astype(float)
        wv = w.values if hasattr(w, "values") else np.asarray(w)

        self.league_mu = float(np.average(np.concatenate([hg, ag]), weights=np.tile(wv, 2)))

        def unpack(theta):
            atk = theta[:n]
            dfc = theta[n : 2 * n]
            home_adv = theta[2 * n]
            return atk, dfc, home_adv

        def nll(theta):
            atk, dfc, home_adv = unpack(theta)
            # identifiability: strengths centred at 0 via penalty below
            lam_h = self.league_mu * np.exp(atk[hi] + dfc[ai] + home_adv)
            lam_a = self.league_mu * np.exp(atk[ai] + dfc[hi])
            ll = wv * (hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a)
            penalty = SHRINKAGE * (np.sum(atk**2) + np.sum(dfc**2))
            penalty += 1e3 * (np.sum(atk) ** 2 + np.sum(dfc) ** 2)  # centre constraint
            return -np.sum(ll) + penalty

        theta0 = np.zeros(2 * n + 1)
        theta0[-1] = 0.25
        res = minimize(nll, theta0, method="L-BFGS-B")
        atk, dfc, home_adv = unpack(res.x)
        self.attack = {c: float(atk[idx[c]]) for c in codes}
        self.defence = {c: float(dfc[idx[c]]) for c in codes}
        self.home_adv = float(home_adv)
        return self

    def rates(self, home_code: int, away_code: int) -> MatchRates:
        atk_h = self.attack.get(home_code, 0.0)
        dfc_h = self.defence.get(home_code, 0.0)
        atk_a = self.attack.get(away_code, 0.0)
        dfc_a = self.defence.get(away_code, 0.0)
        return MatchRates(
            lambda_home=self.league_mu * np.exp(atk_h + dfc_a + self.home_adv),
            lambda_away=self.league_mu * np.exp(atk_a + dfc_h),
        )

    def team_expected_goals(self, team_code: int, opp_code: int, is_home: bool) -> float:
        r = self.rates(team_code, opp_code) if is_home else self.rates(opp_code, team_code)
        return r.lambda_home if is_home else r.lambda_away

    def team_expected_conceded(self, team_code: int, opp_code: int, is_home: bool) -> float:
        r = self.rates(team_code, opp_code) if is_home else self.rates(opp_code, team_code)
        return r.lambda_away if is_home else r.lambda_home

    def baseline_team_goals(self, team_code: int) -> float:
        """Expected goals vs an average opponent at a neutral-ish venue (half home adv)."""
        return self.league_mu * float(
            np.exp(self.attack.get(team_code, 0.0) + self.home_adv / 2.0)
        )


def matches_frame(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Join fixtures to stable team codes → the frame DixonColesModel.fit expects."""
    code_map = teams.set_index(["season", "team_id"])["code"]
    fin = fixtures[fixtures["finished"]].copy()
    fin["home_code"] = fin.set_index(["season", "team_h"]).index.map(code_map)
    fin["away_code"] = fin.set_index(["season", "team_a"]).index.map(code_map)
    return fin.rename(columns={"team_h_score": "home_goals", "team_a_score": "away_goals"})[
        ["home_code", "away_code", "home_goals", "away_goals", "kickoff_time", "season", "event"]
    ]
