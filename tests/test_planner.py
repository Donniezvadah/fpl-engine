"""Multi-gameweek transfer planner tests (small pools, fast solves)."""

import pandas as pd

from fpl_engine.optim.planner import TransferPlanner
from fpl_engine.optim.squad import SquadOptimizer
from fpl_engine.rules import rules_for

RULES = rules_for("2025-26")


def make_matrix(horizon: list[int]) -> pd.DataFrame:
    rows = []
    pid = 1
    for position, n in ((1, 4), (2, 8), (3, 8), (4, 6)):
        for k in range(n):
            row = {
                "player_id": pid,
                "position": position,
                "team_id": pid % 10,
                "value": 40 + 10 * (k % 4),
            }
            for gw in horizon:
                row[f"xp_{gw}"] = 2.0 + 0.3 * (pid % 4)
            rows.append(row)
            pid += 1
    return pd.DataFrame(rows)


def _base_squad(matrix: pd.DataFrame, horizon: list[int]) -> dict[int, int]:
    gw_players = matrix.rename(columns={f"xp_{horizon[0]}": "xp"})[
        ["player_id", "position", "team_id", "value", "xp"]
    ]
    squad = SquadOptimizer(RULES).optimize(gw_players)
    return {p.player_id: p.price for p in squad.squad.players}


def test_no_upgrades_means_no_transfers():
    horizon = [10, 11, 12]
    m = make_matrix(horizon)
    owned = _base_squad(m, horizon)
    plan = TransferPlanner(RULES).plan(m, horizon, owned, bank=0, free_transfers=1)
    # Squad already optimal and flat xP: taking hits can never pay.
    assert all(g.hits == 0 for g in plan.gameweeks)


def test_big_upgrade_triggers_transfer():
    horizon = [10, 11, 12]
    m = make_matrix(horizon)
    owned = _base_squad(m, horizon)
    outsider = int(m[(~m["player_id"].isin(owned)) & (m["position"] == 3)]["player_id"].iloc[0])
    for gw in horizon:
        m.loc[m["player_id"] == outsider, f"xp_{gw}"] = 15.0
    plan = TransferPlanner(RULES).plan(m, horizon, owned, bank=100, free_transfers=1)
    all_in = [i for g in plan.gameweeks for i in g.transfers_in]
    assert outsider in all_in
    # And he should come in immediately, not later.
    assert outsider in plan.gameweeks[0].transfers_in


def test_ft_banking_respects_cap():
    horizon = [10, 11, 12, 13]
    m = make_matrix(horizon)
    owned = _base_squad(m, horizon)
    plan = TransferPlanner(RULES).plan(m, horizon, owned, bank=0, free_transfers=5)
    assert all(
        g.free_transfers_before <= RULES.transfers.max_banked_free_transfers
        for g in plan.gameweeks
    )


def test_wildcard_gw_takes_no_hits():
    horizon = [10, 11]
    m = make_matrix(horizon)
    owned = _base_squad(m, horizon)
    # Make five outsiders attractive so a rebuild is worthwhile.
    outsiders = m[~m["player_id"].isin(owned)]["player_id"].head(5)
    for gw in horizon:
        m.loc[m["player_id"].isin(outsiders), f"xp_{gw}"] = 12.0
    plan = TransferPlanner(RULES).plan(
        m, horizon, owned, bank=0, free_transfers=1, forced_chip=("wildcard", 10)
    )
    assert plan.gameweeks[0].hits == 0
    assert len(plan.gameweeks[0].transfers_in) >= 3


def test_wildcard_week_does_not_add_a_free_transfer():
    horizon = [10, 11]
    m = make_matrix(horizon)
    owned = _base_squad(m, horizon)
    plan = TransferPlanner(RULES).plan(
        m, horizon, owned, bank=0, free_transfers=1, forced_chip=("wildcard", 10)
    )
    # The banked transfer carries over; the wildcard week's own is spent on the chip.
    assert plan.gameweeks[1].free_transfers_before == 1
