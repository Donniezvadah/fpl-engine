"""Multi-gameweek transfer planner (MILP over a rolling horizon).

Decides transfers for the next H gameweeks jointly: who to buy/sell and when,
whether hits are +EV, when to bank free transfers — maximizing time-decayed
expected points minus hit costs, under all squad rules and cash-flow constraints.

Formulation follows the community-standard multi-period FPL MILP (cf. sertalpbilal's
fpl-optimization): binary squad/XI/captain variables per (player, GW), transfer
in/out flow variables linking consecutive squads, integer free-transfer bank with
the official cap, and hits charged where transfers exceed the bank.

The player pool is pruned to the current squad plus the top predicted scorers per
position — CBC solves the pruned problem in seconds with no practical loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from ..rules.constants import Position, SeasonRules

DECAY = 0.90  # weight on GW t is DECAY^t: near GWs count more (uncertainty grows)
BENCH_WEIGHT = 0.10
TERMINAL_FT_VALUE = 0.6  # end-of-horizon value of a banked FT, in points
HIT_EPS = 0.01  # tie-break so the solver never takes gratuitous hits

POOL_PER_POSITION = {Position.GK: 12, Position.DEF: 40, Position.MID: 45, Position.FWD: 25}


@dataclass
class PlannedGameweek:
    event: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    free_transfers_before: int
    starting_ids: list[int]
    captain_id: int
    bench_ids: list[int]
    xp: float


@dataclass
class TransferPlan:
    gameweeks: list[PlannedGameweek]
    objective: float
    pool: pd.DataFrame = field(repr=False, default=None)


def prune_pool(
    xp_matrix: pd.DataFrame, owned: set[int], horizon: list[int]
) -> pd.DataFrame:
    """Keep owned players + top scorers per position by decayed total xP."""
    m = xp_matrix.copy()
    weights = {gw: DECAY**k for k, gw in enumerate(horizon)}
    m["score"] = sum(m[f"xp_{gw}"] * w for gw, w in weights.items())
    keep: set[int] = set(owned)
    for position, n in POOL_PER_POSITION.items():
        top = m[m["position"] == position.value].nlargest(n, "score")["player_id"]
        keep.update(top)
    return m[m["player_id"].isin(keep)].reset_index(drop=True)


class TransferPlanner:
    def __init__(self, rules: SeasonRules):
        self.rules = rules

    def plan(
        self,
        xp_matrix: pd.DataFrame,
        horizon: list[int],
        current_squad: dict[int, int],  # player_id -> sell price
        bank: int,
        free_transfers: int,
        forced_chip: tuple[str, int] | None = None,  # e.g. ("wildcard", 7)
        captain_multiplier: int = 2,
        time_limit_s: int = 60,
    ) -> TransferPlan:
        """`xp_matrix`: player_id, position, team_id, value (buy price), and one
        xp_<gw> column per horizon GW (0 for blanks, summed for doubles)."""
        rules = self.rules
        sq = rules.squad
        pool = prune_pool(xp_matrix, set(current_squad), horizon)
        ids = pool["player_id"].tolist()
        pos = dict(zip(pool["player_id"], pool["position"].astype(int)))
        club = dict(zip(pool["player_id"], pool["team_id"].astype(int)))
        buy = dict(zip(pool["player_id"], pool["value"].astype(int)))
        sell = {i: current_squad.get(i, buy[i]) for i in ids}
        xp = {
            (i, gw): float(pool.loc[pool["player_id"] == i, f"xp_{gw}"].iloc[0])
            for i in ids
            for gw in horizon
        }
        owned = set(current_squad)
        chip_name, chip_gw = forced_chip if forced_chip else (None, None)

        prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
        T = horizon
        s = pulp.LpVariable.dicts("s", (ids, T), cat="Binary")
        x = pulp.LpVariable.dicts("x", (ids, T), cat="Binary")
        c = pulp.LpVariable.dicts("c", (ids, T), cat="Binary")
        tin = pulp.LpVariable.dicts("tin", (ids, T), cat="Binary")
        tout = pulp.LpVariable.dicts("tout", (ids, T), cat="Binary")
        ft = pulp.LpVariable.dicts(
            "ft", T, lowBound=0, upBound=rules.transfers.max_banked_free_transfers, cat="Integer"
        )
        hit = pulp.LpVariable.dicts("hit", T, lowBound=0, cat="Integer")

        mult = {gw: captain_multiplier for gw in T}
        if chip_name == "3xc":
            mult[chip_gw] = 3
        bench_w = {gw: BENCH_WEIGHT for gw in T}
        if chip_name == "bboost":
            bench_w[chip_gw] = 1.0

        prob += (
            pulp.lpSum(
                DECAY**k
                * (
                    pulp.lpSum(x[i][gw] * xp[i, gw] for i in ids)
                    + (mult[gw] - 1) * pulp.lpSum(c[i][gw] * xp[i, gw] for i in ids)
                    + bench_w[gw] * pulp.lpSum((s[i][gw] - x[i][gw]) * xp[i, gw] for i in ids)
                    - rules.transfers.hit_cost * hit[gw]
                    - HIT_EPS * hit[gw]
                )
                for k, gw in enumerate(T)
            )
            + TERMINAL_FT_VALUE * ft[T[-1]]
        )

        for k, gw in enumerate(T):
            prob += pulp.lpSum(s[i][gw] for i in ids) == sq.squad_size
            prob += pulp.lpSum(x[i][gw] for i in ids) == sq.xi_size
            prob += pulp.lpSum(c[i][gw] for i in ids) == 1
            for position, need in sq.position_counts.items():
                members = [i for i in ids if pos[i] == position.value]
                prob += pulp.lpSum(s[i][gw] for i in members) == need
            for position in Position:
                members = [i for i in ids if pos[i] == position.value]
                prob += pulp.lpSum(x[i][gw] for i in members) >= sq.formation_min[position]
                prob += pulp.lpSum(x[i][gw] for i in members) <= sq.formation_max[position]
            for club_id in set(club.values()):
                members = [i for i in ids if club[i] == club_id]
                prob += pulp.lpSum(s[i][gw] for i in members) <= sq.max_per_club
            for i in ids:
                prob += x[i][gw] <= s[i][gw]
                prob += c[i][gw] <= x[i][gw]

            # Squad evolution.
            for i in ids:
                prev = (1 if i in owned else 0) if k == 0 else s[i][T[k - 1]]
                prob += s[i][gw] == prev + tin[i][gw] - tout[i][gw]
                prob += tin[i][gw] + tout[i][gw] <= 1

            transfers_gw = pulp.lpSum(tin[i][gw] for i in ids)
            free_this_gw = ft[gw]
            if chip_name == "wildcard" and gw == chip_gw:
                # Wildcard GW: unlimited transfers, no hits.
                prob += hit[gw] == 0
            else:
                prob += hit[gw] >= transfers_gw - free_this_gw

            # FT bank dynamics (upper bounds; more FT is weakly better).
            if k == 0:
                prob += ft[gw] == min(free_transfers, rules.transfers.max_banked_free_transfers)
            else:
                prev_gw = T[k - 1]
                prev_transfers = pulp.lpSum(tin[i][prev_gw] for i in ids)
                if chip_name == "wildcard" and prev_gw == chip_gw:
                    # Wildcard transfers don't consume FTs, but the week's own FT
                    # goes into the chip: the bank carries over unchanged.
                    prob += ft[gw] <= ft[prev_gw]
                else:
                    prob += ft[gw] <= ft[prev_gw] - prev_transfers + hit[prev_gw] + 1

        # Cash flow: cumulative buys ≤ bank + cumulative sells, at every GW.
        for k, gw in enumerate(T):
            upto = T[: k + 1]
            spent = pulp.lpSum(tin[i][g] * buy[i] for i in ids for g in upto)
            raised = pulp.lpSum(tout[i][g] * sell[i] for i in ids for g in upto)
            prob += spent <= bank + raised

        if chip_name == "freehit":
            # Free Hit is a one-GW loan squad that fully reverts; modeling it inside
            # the multi-period MILP would need duplicated squad variables. It is
            # evaluated as a scenario comparison in optim/chips.py instead.
            raise ValueError("freehit is evaluated in chips.py, not inside the planner")

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
        if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
            raise RuntimeError(f"solver status: {pulp.LpStatus[status]}")

        gws: list[PlannedGameweek] = []
        for k, gw in enumerate(T):
            xi = [i for i in ids if x[i][gw].value() and x[i][gw].value() > 0.5]
            squad15 = [i for i in ids if s[i][gw].value() and s[i][gw].value() > 0.5]
            gws.append(
                PlannedGameweek(
                    event=gw,
                    transfers_in=[i for i in ids if tin[i][gw].value() and tin[i][gw].value() > 0.5],
                    transfers_out=[i for i in ids if tout[i][gw].value() and tout[i][gw].value() > 0.5],
                    hits=int(round(hit[gw].value() or 0)),
                    free_transfers_before=int(round(ft[gw].value() or 0)),
                    starting_ids=xi,
                    captain_id=next(
                        (i for i in ids if c[i][gw].value() and c[i][gw].value() > 0.5), xi[0]
                    ),
                    bench_ids=[i for i in squad15 if i not in xi],
                    xp=float(sum(xp[i, gw] for i in xi)),
                )
            )
        return TransferPlan(gameweeks=gws, objective=float(pulp.value(prob.objective)), pool=pool)
