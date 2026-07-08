"""xP composition arithmetic and Monte Carlo consistency."""

import numpy as np
import pandas as pd
import pytest

from fpl_engine.models.points import compose_xp
from fpl_engine.rules import rules_for
from fpl_engine.sim.monte_carlo import simulate_gameweek, squad_points_distribution

RULES = rules_for("2025-26")


def _one(**overrides):
    base = dict(
        position=np.array([3]),
        p_play=np.array([1.0]),
        p60=np.array([1.0]),
        e_minutes=np.array([90.0]),
        e_goals=np.array([0.0]),
        e_assists=np.array([0.0]),
        lambda_opp=np.array([1.0]),
        e_saves=np.array([0.0]),
        defcon_rate=np.array([0.0]),
        e_bonus=np.array([0.0]),
    )
    base.update(overrides)
    return compose_xp(rules=RULES, **base)


class TestCompose:
    def test_appearance_only(self):
        xp = _one()
        # 2 appearance + e^-1 CS point (MID) + card penalty
        expected = 2.0 + np.exp(-1.0) * 1 + (-0.16)
        assert xp.total[0] == pytest.approx(expected, abs=1e-6)

    def test_goal_value_by_position(self):
        mid = _one(e_goals=np.array([0.5]))
        fwd = _one(position=np.array([4]), e_goals=np.array([0.5]))
        assert mid.goals[0] == pytest.approx(2.5)  # 0.5 * 5
        assert fwd.goals[0] == pytest.approx(2.0)  # 0.5 * 4

    def test_defender_cs_and_conceded(self):
        d = _one(position=np.array([2]), lambda_opp=np.array([2.0]))
        assert d.clean_sheets[0] == pytest.approx(np.exp(-2.0) * 4, abs=1e-6)
        # E[floor(Poisson(2)/2)] > 0 deduction present
        assert d.conceded[0] < 0

    def test_zero_minutes_zero_everything(self):
        xp = _one(p_play=np.array([0.0]), p60=np.array([0.0]), e_minutes=np.array([0.0]))
        assert xp.total[0] == pytest.approx(0.0, abs=1e-9)

    def test_gk_saves(self):
        gk = _one(position=np.array([1]), e_saves=np.array([4.0]))
        assert gk.saves[0] > 0.5  # E[floor(Poisson(4)/3)] ≈ 1.0

    def test_defcon_threshold(self):
        d = _one(position=np.array([2]), defcon_rate=np.array([12.0]))
        assert 1.0 < d.defcon[0] <= 2.0  # P(Poisson(12) >= 10) is high


def _sim_frame() -> pd.DataFrame:
    """Two fixtures, four players, deterministic starters."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "fixture_id": [100, 100, 100, 200],
            "was_home": [True, True, False, True],
            "position": [3, 2, 4, 1],
            "p_start": [1.0, 1.0, 1.0, 1.0],
            "p60": [1.0, 1.0, 1.0, 1.0],
            "p_play": [1.0, 1.0, 1.0, 1.0],
            "e_minutes": [90.0, 90.0, 90.0, 90.0],
            "lambda_team": [1.8, 1.8, 1.1, 1.4],
            "lambda_opp": [1.1, 1.1, 1.8, 1.3],
            "e_goals": [0.5, 0.1, 0.4, 0.0],
            "e_assists": [0.3, 0.1, 0.2, 0.0],
            "saves90": [0.0, 0.0, 0.0, 3.0],
            "defcon90": [2.0, 8.0, 1.0, 0.0],
        }
    )


class TestSimulation:
    def test_shapes_and_determinism(self):
        sim = simulate_gameweek(_sim_frame(), RULES, n_sims=2000, seed=7)
        again = simulate_gameweek(_sim_frame(), RULES, n_sims=2000, seed=7)
        assert sim.points.shape == (2000, 4)
        assert np.array_equal(sim.points, again.points)

    def test_means_are_sane(self):
        sim = simulate_gameweek(_sim_frame(), RULES, n_sims=20_000, seed=1)
        means = sim.mean()
        assert (means > 0).all()
        # The attacking MID with 0.5 xG should out-average the keeper facing 1.3.
        assert means[0] > means[3] - 1.0

    def test_clean_sheet_correlation_same_team(self):
        """Defender (id 2) CS and midfielder (id 1) CS come from the same scoreline:
        their points must be positively correlated."""
        sim = simulate_gameweek(_sim_frame(), RULES, n_sims=20_000, seed=2)
        corr = np.corrcoef(sim.points[:, 0], sim.points[:, 1])[0, 1]
        assert corr > 0.05

    def test_squad_distribution_captain(self):
        sim = simulate_gameweek(_sim_frame(), RULES, n_sims=5_000, seed=3)
        plain = squad_points_distribution(sim, [1, 2, 3, 4], captain_id=1, vice_id=2)
        boosted = squad_points_distribution(
            sim, [1, 2, 3, 4], captain_id=1, vice_id=2, captain_multiplier=3
        )
        assert boosted.mean() > plain.mean()
