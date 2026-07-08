"""MILP optimizer tests on small synthetic pools (fast, deterministic)."""

import itertools

import pandas as pd
import pytest

from fpl_engine.optim.squad import SquadOptimizer
from fpl_engine.rules import rules_for
from fpl_engine.rules.squad import validate_squad_composition

RULES = rules_for("2025-26")


def make_pool(n_gk=4, n_def=8, n_mid=8, n_fwd=6) -> pd.DataFrame:
    """Synthetic pool: xp roughly increasing with price, unique clubs cycling."""
    rows = []
    pid = 1
    for position, n in ((1, n_gk), (2, n_def), (3, n_mid), (4, n_fwd)):
        for k in range(n):
            price = 40 + 10 * (k % 5)  # 4.0 to 8.0
            rows.append(
                {
                    "player_id": pid,
                    "position": position,
                    "team_id": pid % 10,
                    "value": price,
                    "xp": 2.0 + 0.05 * price + 0.3 * (pid % 3),
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_from_scratch_squad_is_legal():
    pool = make_pool()
    result = SquadOptimizer(RULES).optimize(pool)
    assert result.cost <= RULES.squad.budget
    assert validate_squad_composition(list(result.squad.players), RULES) == []


def test_budget_binds():
    pool = make_pool()
    tight = SquadOptimizer(RULES).optimize(pool, budget=700)
    assert tight.cost <= 700
    loose = SquadOptimizer(RULES).optimize(pool, budget=1000)
    assert loose.xp_total >= tight.xp_total - 1e-9


def test_captain_is_max_xp_starter():
    pool = make_pool()
    result = SquadOptimizer(RULES).optimize(pool)
    xp = dict(zip(pool["player_id"], pool["xp"]))
    xi_best = max(result.squad.starting_ids, key=lambda i: xp[i])
    assert abs(xp[result.squad.captain_id] - xp[xi_best]) < 1e-9


def test_include_exclude_respected():
    pool = make_pool()
    worst_gk = int(pool[pool["position"] == 1].nsmallest(1, "xp")["player_id"].iloc[0])
    best_mid = int(pool[pool["position"] == 3].nlargest(1, "xp")["player_id"].iloc[0])
    result = SquadOptimizer(RULES).optimize(
        pool, include={worst_gk}, exclude={best_mid}
    )
    squad_ids = {p.player_id for p in result.squad.players}
    assert worst_gk in squad_ids and best_mid not in squad_ids


def test_transfer_mode_limits_moves():
    pool = make_pool()
    base = SquadOptimizer(RULES).optimize(pool)
    owned = {p.player_id: p.price for p in base.squad.players}
    # Make one non-owned midfielder irresistible.
    pool2 = pool.copy()
    outsider = pool2[
        (~pool2["player_id"].isin(owned)) & (pool2["position"] == 3)
    ]["player_id"].iloc[0]
    pool2.loc[pool2["player_id"] == outsider, "xp"] = 25.0
    moved = SquadOptimizer(RULES).optimize(
        pool2, current_squad=owned, max_transfers=1, bank=0
    )
    new_ids = {p.player_id for p in moved.squad.players}
    assert len(new_ids - set(owned)) <= 1
    assert outsider in new_ids


def test_transfer_mode_cash_flow():
    pool = make_pool()
    base = SquadOptimizer(RULES).optimize(pool, budget=900)
    owned = {p.player_id: p.price for p in base.squad.players}
    moved = SquadOptimizer(RULES).optimize(pool, current_squad=owned, max_transfers=2, bank=5)
    price = dict(zip(pool["player_id"], pool["value"]))
    bought = [p.player_id for p in moved.squad.players if p.player_id not in owned]
    sold = [i for i in owned if i not in {p.player_id for p in moved.squad.players}]
    assert sum(price[i] for i in bought) <= 5 + sum(owned[i] for i in sold)


def test_matches_brute_force_on_tiny_pool():
    """XI selection from a fixed 15 must equal exhaustive search."""
    pool = make_pool(n_gk=2, n_def=5, n_mid=5, n_fwd=3)  # exactly 15 players
    # bench_weight=0 so the objective is XI + captain + 0.05*vice, mirrored below.
    result = SquadOptimizer(RULES).optimize(pool, budget=2000, bench_weight=0.0)
    xp = dict(zip(pool["player_id"], pool["xp"]))
    pos = dict(zip(pool["player_id"], pool["position"]))
    ids = pool["player_id"].tolist()

    best = -1.0
    for xi in itertools.combinations(ids, 11):
        counts = {p: sum(1 for i in xi if pos[i] == p) for p in (1, 2, 3, 4)}
        if counts[1] != 1 or not (3 <= counts[2] <= 5) or not (2 <= counts[3] <= 5) or not (1 <= counts[4] <= 3):
            continue
        ranked = sorted((xp[i] for i in xi), reverse=True)
        total = sum(ranked) + ranked[0] + 0.05 * ranked[1]
        best = max(best, total)

    assert result.xp_total == pytest.approx(best, abs=1e-6)
