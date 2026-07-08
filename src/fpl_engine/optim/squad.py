"""Single-gameweek squad optimization (MILP, PuLP/CBC).

Solves: best legal 15 / starting XI / captain / bench order for one gameweek,
either from scratch (wildcard / free hit / initial draft) or as a transfer move
from an existing squad with a transfer budget and sell prices.

Every solution is re-validated by the independent rules engine before being
returned — a solver or formulation bug can never emit an illegal squad.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from ..rules.constants import Position, SeasonRules
from ..rules.squad import Squad, SquadPlayer, validate_selection, validate_squad_composition

BENCH_WEIGHT = 0.10  # bench xP is worth ~10% (autosub probability)
VICE_WEIGHT = 0.05


@dataclass
class OptimizedSquad:
    squad: Squad
    xp_total: float
    cost: int
    player_rows: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        df = self.player_rows.copy()
        df["role"] = "bench"
        df.loc[df["player_id"].isin(self.squad.starting_ids), "role"] = "XI"
        df.loc[df["player_id"] == self.squad.captain_id, "role"] = "captain"
        df.loc[df["player_id"] == self.squad.vice_captain_id, "role"] = "vice"
        return df.sort_values(["role", "xp"], ascending=[True, False])


class SquadOptimizer:
    def __init__(self, rules: SeasonRules):
        self.rules = rules

    def optimize(
        self,
        players: pd.DataFrame,
        budget: int | None = None,
        current_squad: dict[int, int] | None = None,  # player_id -> sell price
        max_transfers: int | None = None,
        bank: int = 0,
        captain_multiplier: int = 2,
        include: set[int] | None = None,
        exclude: set[int] | None = None,
        bench_weight: float = BENCH_WEIGHT,
    ) -> OptimizedSquad:
        """`players` needs: player_id, position, club (team_id), price (buy), xp.

        From-scratch mode: budget constraint on buy prices.
        Transfer mode (current_squad given): keeps/sells owned players at sell
        price, buys newcomers at market price, spend ≤ sell proceeds + bank,
        and at most `max_transfers` newcomers.
        """
        sq = self.rules.squad
        p = players.reset_index(drop=True)
        ids = p["player_id"].tolist()
        xp = dict(zip(p["player_id"], p["xp"].astype(float)))
        pos = dict(zip(p["player_id"], p["position"].astype(int)))
        club = dict(zip(p["player_id"], p["team_id"].astype(int)))
        price = dict(zip(p["player_id"], p["value"].astype(int)))

        prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
        s = pulp.LpVariable.dicts("s", ids, cat="Binary")  # in 15
        x = pulp.LpVariable.dicts("x", ids, cat="Binary")  # in XI
        c = pulp.LpVariable.dicts("c", ids, cat="Binary")  # captain
        v = pulp.LpVariable.dicts("v", ids, cat="Binary")  # vice

        prob += (
            pulp.lpSum(x[i] * xp[i] for i in ids)
            + (captain_multiplier - 1) * pulp.lpSum(c[i] * xp[i] for i in ids)
            + VICE_WEIGHT * pulp.lpSum(v[i] * xp[i] for i in ids)
            + bench_weight * pulp.lpSum((s[i] - x[i]) * xp[i] for i in ids)
        )

        prob += pulp.lpSum(s[i] for i in ids) == sq.squad_size
        prob += pulp.lpSum(x[i] for i in ids) == sq.xi_size
        for position, need in sq.position_counts.items():
            members = [i for i in ids if pos[i] == position.value]
            prob += pulp.lpSum(s[i] for i in members) == need
        for position in Position:
            members = [i for i in ids if pos[i] == position.value]
            prob += pulp.lpSum(x[i] for i in members) >= sq.formation_min[position]
            prob += pulp.lpSum(x[i] for i in members) <= sq.formation_max[position]
        for club_id in set(club.values()):
            members = [i for i in ids if club[i] == club_id]
            prob += pulp.lpSum(s[i] for i in members) <= sq.max_per_club

        for i in ids:
            prob += x[i] <= s[i]
            prob += c[i] <= x[i]
            prob += v[i] <= x[i]
            prob += c[i] + v[i] <= 1
        prob += pulp.lpSum(c[i] for i in ids) == 1
        prob += pulp.lpSum(v[i] for i in ids) == 1

        if current_squad is None:
            cap = sq.budget if budget is None else budget
            prob += pulp.lpSum(s[i] * price[i] for i in ids) <= cap
        else:
            owned = set(current_squad)
            newcomers = [i for i in ids if i not in owned]
            # Cash: buys ≤ bank + proceeds of sales (owned players not kept).
            prob += pulp.lpSum(s[i] * price[i] for i in newcomers) <= bank + pulp.lpSum(
                (1 - s[i]) * current_squad[i] for i in ids if i in owned
            )
            if max_transfers is not None:
                prob += pulp.lpSum(s[i] for i in newcomers) <= max_transfers

        for i in include or set():
            if i in s:
                prob += s[i] == 1
        for i in exclude or set():
            if i in s:
                prob += s[i] == 0

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"solver status: {pulp.LpStatus[status]}")

        chosen = [i for i in ids if s[i].value() > 0.5]
        xi = frozenset(i for i in ids if x[i].value() > 0.5)
        captain = next(i for i in ids if c[i].value() > 0.5)
        vice = next(i for i in ids if v[i].value() > 0.5)
        bench_out = sorted(
            (i for i in chosen if i not in xi and pos[i] != Position.GK.value),
            key=lambda i: -xp[i],
        )

        squad_players = tuple(
            SquadPlayer(i, Position(pos[i]), club[i], price[i]) for i in chosen
        )
        squad = Squad(
            players=squad_players,
            starting_ids=xi,
            captain_id=captain,
            vice_captain_id=vice,
            bench_order=tuple(bench_out),
        )

        # Independent legality check (composition budget check applies to from-scratch mode).
        comp_budget = None if current_squad is not None else budget
        errors = validate_selection(squad, self.rules)
        if current_squad is None:
            errors += validate_squad_composition(list(squad_players), self.rules, comp_budget)
        if errors:
            raise RuntimeError(f"optimizer produced illegal squad: {errors}")

        rows = p[p["player_id"].isin(chosen)].copy()
        return OptimizedSquad(
            squad=squad,
            xp_total=float(pulp.value(prob.objective)),
            cost=sum(price[i] for i in chosen),
            player_rows=rows,
        )
