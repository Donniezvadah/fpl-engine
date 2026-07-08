"""Compose per-player expected points (with variance) from component predictions.

Everything is vectorized numpy over aligned arrays. The composition mirrors the
official scoring table exactly (via rules constants), applied to probabilistic
inputs:

  appearance   2*P(60+) + 1*(P(play) - P(60+))
  goals        E[goals] * goal_value(position)
  assists      E[assists] * 3
  clean sheet  P(60+) * P(opp scores 0) * cs_value(position)
  conceded     -E[floor(conceded/2)] for GK/DEF, weighted by time on pitch
  saves        E[floor(saves/3)] for GK (Poisson)
  defcon       P(defcon count >= threshold) * 2 (Poisson tail)
  bonus        trailing bonus propensity scaled by minutes
  cards        league-average deduction scaled by minutes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..rules.constants import Position, SeasonRules

_MAX_GOALS = 12  # truncation for Poisson sums

# League-average expected deduction from yellow/red cards per 90 by position.
_CARD_PENALTY_90 = {1: -0.04, 2: -0.18, 3: -0.16, 4: -0.12}


@dataclass
class XPBreakdown:
    total: np.ndarray
    variance: np.ndarray
    appearance: np.ndarray
    goals: np.ndarray
    assists: np.ndarray
    clean_sheets: np.ndarray
    conceded: np.ndarray
    saves: np.ndarray
    defcon: np.ndarray
    bonus: np.ndarray
    cards: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "xp": self.total,
            "xp_var": self.variance,
            "xp_appearance": self.appearance,
            "xp_goals": self.goals,
            "xp_assists": self.assists,
            "xp_cs": self.clean_sheets,
            "xp_conceded": self.conceded,
            "xp_saves": self.saves,
            "xp_defcon": self.defcon,
            "xp_bonus": self.bonus,
            "xp_cards": self.cards,
        }


def _e_floor_div(lam: np.ndarray, divisor: int) -> np.ndarray:
    """E[floor(X / divisor)] for X ~ Poisson(lam), truncated at _MAX_GOALS*2."""
    ks = np.arange(0, _MAX_GOALS * 2 + 1)
    pmf = stats.poisson.pmf(ks[None, :], np.asarray(lam, dtype=float)[:, None])
    vals = ks // divisor
    return (pmf * vals[None, :]).sum(axis=1)


def compose_xp(
    position: np.ndarray,  # 1..4
    p_play: np.ndarray,
    p60: np.ndarray,
    e_minutes: np.ndarray,
    e_goals: np.ndarray,  # expected goals scored by the player this fixture
    e_assists: np.ndarray,
    lambda_opp: np.ndarray,  # opponent's expected goals in this fixture
    e_saves: np.ndarray,  # expected saves if GK plays full match
    defcon_rate: np.ndarray,  # expected defensive contributions per full match
    e_bonus: np.ndarray,  # expected bonus points per full match
    rules: SeasonRules,
) -> XPBreakdown:
    sc = rules.scoring
    pos = np.asarray(position, dtype=int)
    minutes_frac = np.clip(np.asarray(e_minutes, dtype=float) / 90.0, 0.0, 1.0)

    appearance = 2.0 * p60 + 1.0 * np.clip(p_play - p60, 0.0, 1.0)

    goal_val = np.select(
        [pos == p.value for p in Position], [sc.goal_points[p] for p in Position], default=4
    ).astype(float)
    goals = e_goals * goal_val
    assists = e_assists * sc.assist_points

    cs_val = np.select(
        [pos == p.value for p in Position],
        [sc.clean_sheet_points[p] for p in Position],
        default=0,
    ).astype(float)
    p_cs_team = np.exp(-np.asarray(lambda_opp, dtype=float))
    p_cs = p60 * p_cs_team
    clean_sheets = p_cs * cs_val

    conceded = np.zeros_like(goals)
    penalised = np.isin(pos, [p.value for p in sc.goals_conceded_penalised_positions])
    # Goals conceded while on the pitch ≈ lambda_opp * time fraction.
    lam_on_pitch = np.asarray(lambda_opp, dtype=float) * minutes_frac
    conceded[penalised] = -_e_floor_div(
        lam_on_pitch[penalised], sc.goals_conceded_per_minus
    ) * p_play[penalised]

    saves = np.zeros_like(goals)
    gk = pos == Position.GK.value
    if gk.any():
        saves[gk] = _e_floor_div(
            np.asarray(e_saves, dtype=float)[gk] * minutes_frac[gk], sc.saves_per_point
        ) * p_play[gk]

    defcon = np.zeros_like(goals)
    if sc.defcon_enabled:
        for p, threshold in sc.defcon_threshold.items():
            mask = pos == p.value
            if mask.any():
                lam = np.asarray(defcon_rate, dtype=float)[mask] * minutes_frac[mask]
                defcon[mask] = (
                    stats.poisson.sf(threshold - 1, np.clip(lam, 1e-9, None))
                    * sc.defcon_points
                    * p_play[mask]
                )

    bonus = np.asarray(e_bonus, dtype=float) * minutes_frac
    card_pen = np.select(
        [pos == p.value for p in Position], [_CARD_PENALTY_90[p.value] for p in Position]
    ).astype(float)
    cards = card_pen * minutes_frac

    total = appearance + goals + assists + clean_sheets + conceded + saves + defcon + bonus + cards

    # Variance: treat components as independent (goals/assists Poisson, CS Bernoulli,
    # appearance mixture). Understates tail correlation slightly; the Monte Carlo
    # simulator models the correlated version.
    var = (
        e_goals * goal_val**2
        + e_assists * float(sc.assist_points) ** 2
        + p_cs * (1 - p_cs) * cs_val**2
        + 4.0 * p60 * (1 - p60)  # appearance spread
        + np.abs(conceded)  # ~Poisson-scale
        + np.abs(saves)
        + defcon * (sc.defcon_points - defcon)  # = t²·p(1-p) for Bernoulli award t=2
        + np.maximum(bonus, 0.0) * 1.5
    )

    return XPBreakdown(
        total=total,
        variance=var,
        appearance=appearance,
        goals=goals,
        assists=assists,
        clean_sheets=clean_sheets,
        conceded=conceded,
        saves=saves,
        defcon=defcon,
        bonus=bonus,
        cards=cards,
    )
