"""Chip-timing evaluation.

Marginal EV of playing each chip in each candidate gameweek, relative to the
no-chip transfer plan:

- Triple Captain: + captain's xP in that GW (the extra 1x), from the base plan.
- Bench Boost:    + bench xP in that GW (the bench already contributes a small
                  autosub weight in the base objective; BB adds the rest).
- Free Hit:       from-scratch single-GW optimum with (team value + bank) budget,
                  minus the base plan's XI+captain xP that GW.
- Wildcard:       full re-solve of the multi-period MILP with unlimited free
                  transfers in the candidate GW (captures rebuild synergy, so it
                  is the expensive one — restrict candidates).

These are decision-support EVs under the mean prediction; the Monte Carlo layer
adds distributional context (a TC on a high-variance captain is worth more than
its mean suggests at extreme-rank objectives).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..rules.constants import SeasonRules
from .planner import BENCH_WEIGHT, DECAY, TransferPlan, TransferPlanner
from .squad import SquadOptimizer


@dataclass
class ChipEvaluation:
    chip: str
    event: int
    gain_pts: float
    detail: str
    plan: TransferPlan | None = None  # the full re-solved plan (wildcard only)


def evaluate_chips(
    rules: SeasonRules,
    xp_matrix: pd.DataFrame,
    horizon: list[int],
    current_squad: dict[int, int],
    bank: int,
    free_transfers: int,
    team_value: int,
    available: list[str],
    wildcard_candidates: list[int] | None = None,
    base_plan: TransferPlan | None = None,
) -> list[ChipEvaluation]:
    planner = TransferPlanner(rules)
    base = base_plan or planner.plan(
        xp_matrix, horizon, current_squad, bank, free_transfers
    )
    pool = base.pool
    xp_of = {
        (int(r["player_id"]), gw): float(r[f"xp_{gw}"])
        for _, r in pool.iterrows()
        for gw in horizon
    }

    evals: list[ChipEvaluation] = []

    for k, gw in enumerate(horizon):
        plan_gw = base.gameweeks[k]
        decay = DECAY**k

        if "3xc" in available:
            cap_xp = xp_of.get((plan_gw.captain_id, gw), 0.0)
            evals.append(
                ChipEvaluation(
                    "3xc", gw, decay * cap_xp,
                    f"captain xP {cap_xp:.2f} gets a third multiple",
                )
            )

        if "bboost" in available:
            bench_xp = sum(xp_of.get((i, gw), 0.0) for i in plan_gw.bench_ids)
            evals.append(
                ChipEvaluation(
                    "bboost", gw, decay * bench_xp * (1 - BENCH_WEIGHT),
                    f"bench xP {bench_xp:.2f} counts in full",
                )
            )

        if "freehit" in available:
            opt = SquadOptimizer(rules)
            gw_players = pool.rename(columns={f"xp_{gw}": "xp"})[
                ["player_id", "position", "team_id", "value", "xp"]
            ]
            fh = opt.optimize(gw_players, budget=team_value + bank)
            base_gw_xp = plan_gw.xp + xp_of.get((plan_gw.captain_id, gw), 0.0)
            fh_xp = sum(
                xp_of.get((i, gw), 0.0) for i in fh.squad.starting_ids
            ) + xp_of.get((fh.squad.captain_id, gw), 0.0)
            evals.append(
                ChipEvaluation(
                    "freehit", gw, decay * max(0.0, fh_xp - base_gw_xp),
                    f"one-week optimal {fh_xp:.1f} vs planned {base_gw_xp:.1f}",
                )
            )

    if "wildcard" in available:
        for gw in wildcard_candidates or horizon[:2]:
            wc = planner.plan(
                xp_matrix, horizon, current_squad, bank, free_transfers,
                forced_chip=("wildcard", gw),
            )
            evals.append(
                ChipEvaluation(
                    "wildcard", gw, wc.objective - base.objective,
                    f"full-rebuild plan objective {wc.objective:.1f} vs {base.objective:.1f}",
                    plan=wc,
                )
            )

    return sorted(evals, key=lambda e: -e.gain_pts)
