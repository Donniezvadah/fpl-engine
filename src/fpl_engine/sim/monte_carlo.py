"""Monte Carlo gameweek simulator.

Samples correlated outcomes: one Poisson scoreline per fixture per simulation,
with player goals/assists allocated multinomially within their team's total —
so teammates' hauls are positively correlated and a defender's clean sheet is
exactly the opponent scoring zero, the correlations that matter for captaincy,
bench boost, and differential decisions.

Vectorized numpy: 10k sims × ~800 player-fixtures runs in seconds; 100k is fine
for season-level questions.

Bonus points are approximated by ranking a simulated BPS proxy within each
fixture (goals/assists/CS dominate real BPS ranks); autosubs are not simulated —
their EV is already reflected in squad selection via the bench weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..rules.constants import Position, SeasonRules

_BPS_GOAL = {1: 34, 2: 32, 3: 26, 4: 24}
_BPS_ASSIST = 18
_BPS_CS = {1: 14, 2: 14, 3: 2, 4: 0}
_BPS_NOISE = 12.0  # spread from passes/tackles/etc. that we don't simulate


@dataclass
class SimulationResult:
    player_ids: np.ndarray  # (n_players,) — per-GW aggregation keys
    points: np.ndarray  # (n_sims, n_players) simulated GW points

    def mean(self) -> np.ndarray:
        return self.points.mean(axis=0)

    def quantile(self, q: float) -> np.ndarray:
        return np.quantile(self.points, q, axis=0)

    def p_at_least(self, threshold: float) -> np.ndarray:
        return (self.points >= threshold).mean(axis=0)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": self.player_ids,
                "sim_mean": self.mean(),
                "sim_p10": self.quantile(0.10),
                "sim_p90": self.quantile(0.90),
                "p_haul_10plus": self.p_at_least(10),
                "p_blank_le2": (self.points <= 2).mean(axis=0),
            }
        )


def simulate_gameweek(
    pred: pd.DataFrame, rules: SeasonRules, n_sims: int = 10_000, seed: int | None = None
) -> SimulationResult:
    """`pred`: PointsPredictor.predict output (per player-fixture rows)."""
    rng = np.random.default_rng(seed)
    sc = rules.scoring
    df = pred.reset_index(drop=True)
    n = len(df)

    pos = df["position"].values.astype(int)
    p_start = df["p_start"].values
    p60_given_play = np.divide(
        df["p60"].values, np.clip(df["p_play"].values, 1e-9, None)
    ).clip(0, 1)
    p_play = df["p_play"].values.clip(0, 1)
    lam_team = df["lambda_team"].values
    lam_opp = df["lambda_opp"].values
    e_goals = df["e_goals"].values
    e_assists = df["e_assists"].values

    # --- playing state ---------------------------------------------------
    u = rng.random((n_sims, n))
    plays = u < p_play[None, :]
    plays60 = plays & (rng.random((n_sims, n)) < p60_given_play[None, :])

    # --- fixture scorelines (shared within (fixture, side)) --------------
    side_key = df["fixture_id"].astype(str) + ":" + df["was_home"].astype(int).astype(str)
    sides, side_idx = np.unique(side_key.values, return_inverse=True)
    lam_team_side = np.zeros(len(sides))
    lam_opp_side = np.zeros(len(sides))
    for si in range(len(sides)):
        rows = np.where(side_idx == si)[0]
        lam_team_side[si] = lam_team[rows[0]]
        lam_opp_side[si] = lam_opp[rows[0]]
    goals_team_side = rng.poisson(lam_team_side[None, :], size=(n_sims, len(sides)))
    goals_opp_side = rng.poisson(lam_opp_side[None, :], size=(n_sims, len(sides)))

    team_goals = goals_team_side[:, side_idx]  # (n_sims, n)
    opp_goals = goals_opp_side[:, side_idx]

    # --- allocate goals/assists to players -------------------------------
    # Share of the team's goals proportional to conditional scoring rates.
    rate_g = np.where(plays, (e_goals / np.clip(p_play, 1e-9, None))[None, :], 0.0)
    rate_a = np.where(plays, (e_assists / np.clip(p_play, 1e-9, None))[None, :], 0.0)

    goals = _allocate(rng, team_goals, rate_g, lam_team_side, side_idx)
    # ~x0.75 of goals have an FPL assist; allocate that many.
    assist_count = rng.binomial(team_goals, 0.75)
    assists = _allocate(rng, assist_count, rate_a, lam_team_side, side_idx)

    # --- component points -------------------------------------------------
    pts = np.zeros((n_sims, n))
    pts += np.where(plays60, sc.played_60_plus, np.where(plays, sc.played_upto_59, 0))

    goal_val = np.array([0] + [sc.goal_points[Position(p)] for p in (1, 2, 3, 4)])[pos]
    pts += goals * goal_val[None, :]
    pts += assists * sc.assist_points

    cs_val = np.array([0] + [sc.clean_sheet_points[Position(p)] for p in (1, 2, 3, 4)])[pos]
    clean = (opp_goals == 0) & plays60
    pts += clean * cs_val[None, :]

    penalised = np.isin(pos, [p.value for p in sc.goals_conceded_penalised_positions])
    conceded_pen = -(opp_goals // sc.goals_conceded_per_minus)
    pts += np.where(plays60 & penalised[None, :], conceded_pen, 0)

    gk = pos == Position.GK.value
    if gk.any():
        e_saves = df["saves90"].values if "saves90" in df else np.full(n, 2.5)
        opp_adj = np.clip(lam_opp / max(np.mean(lam_opp), 1e-9), 0.5, 2.0)
        saves = rng.poisson(np.where(gk, e_saves * opp_adj, 0)[None, :] * plays)
        pts += (saves // sc.saves_per_point) * gk[None, :]

    if sc.defcon_enabled and "defcon90" in df:
        dc_rate = df["defcon90"].values
        for p, threshold in sc.defcon_threshold.items():
            mask = pos == p.value
            if mask.any():
                dc = rng.poisson(np.where(mask, dc_rate, 0)[None, :] * plays60)
                pts += np.where(mask[None, :] & (dc >= threshold), sc.defcon_points, 0)

    # --- bonus via BPS proxy ranked within fixture ------------------------
    bps_goal = np.array([0] + [_BPS_GOAL[p] for p in (1, 2, 3, 4)])[pos]
    bps_cs = np.array([0] + [_BPS_CS[p] for p in (1, 2, 3, 4)])[pos]
    proxy = (
        goals * bps_goal[None, :]
        + assists * _BPS_ASSIST
        + clean * bps_cs[None, :]
        + rng.normal(0, _BPS_NOISE, size=(n_sims, n)) * plays
    )
    proxy = np.where(plays, proxy, -np.inf)
    fixture_ids, fx_idx = np.unique(df["fixture_id"].values, return_inverse=True)
    for fi in range(len(fixture_ids)):
        cols = np.where(fx_idx == fi)[0]
        if len(cols) < 3:
            continue
        sub = proxy[:, cols]
        order = np.argsort(-sub, axis=1)
        rows = np.arange(n_sims)
        for rank, award in enumerate(sc.bonus_points):
            if rank >= len(cols):
                break
            sel = cols[order[:, rank]]
            valid = np.isfinite(proxy[rows, sel])  # never award bonus to non-players
            pts[rows[valid], sel[valid]] += award

    # --- aggregate per (player, GW) for DGWs -------------------------------
    pids, p_idx = np.unique(df["player_id"].values, return_inverse=True)
    agg = np.zeros((n_sims, len(pids)))
    np.add.at(agg.T, p_idx, pts.T)
    return SimulationResult(player_ids=pids, points=agg)


def _allocate(
    rng: np.random.Generator,
    totals: np.ndarray,  # (n_sims, n) team totals broadcast per player row
    rates: np.ndarray,  # (n_sims, n) per-player allocation weights (0 if absent)
    lam_side: np.ndarray,
    side_idx: np.ndarray,
) -> np.ndarray:
    """Multinomial thinning: each of the team's goals goes to player j with
    probability rate_j / sum(rates in side). Non-modelled scorers absorb the rest
    implicitly when rates sum below the side total (binomial thinning per player)."""
    n_sims, n = rates.shape
    out = np.zeros((n_sims, n), dtype=int)
    # Normalize rates by expected side total so each player's share ≈ rate/lam.
    lam = lam_side[side_idx][None, :]
    share = np.clip(rates / np.clip(lam, 1e-9, None), 0.0, 1.0)
    # Binomial thinning of the side's total per player (approximate multinomial;
    # matches marginal expectations and preserves team-level correlation).
    out = rng.binomial(totals, share)
    return out


def squad_points_distribution(
    sim: SimulationResult,
    starting_ids: list[int],
    captain_id: int,
    vice_id: int,
    bench_ids: list[int] | None = None,
    bench_boost: bool = False,
    captain_multiplier: int = 2,
) -> np.ndarray:
    """Distribution of a squad's GW total (captain doubled, vice fallback)."""
    index = {pid: i for i, pid in enumerate(sim.player_ids)}
    xi_cols = [index[i] for i in starting_ids if i in index]
    total = sim.points[:, xi_cols].sum(axis=1)

    cap_col = index.get(captain_id)
    vice_col = index.get(vice_id)
    if cap_col is not None:
        cap_pts = sim.points[:, cap_col]
        cap_played = cap_pts != 0
        extra = np.where(cap_played, cap_pts, 0)
        if vice_col is not None:
            extra = np.where(cap_played, cap_pts, sim.points[:, vice_col])
        total = total + (captain_multiplier - 1) * extra

    if bench_boost and bench_ids:
        bench_cols = [index[i] for i in bench_ids if i in index]
        total = total + sim.points[:, bench_cols].sum(axis=1)
    return total
