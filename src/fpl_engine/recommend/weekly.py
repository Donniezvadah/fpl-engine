"""Weekly recommendation assembly.

Pulls together predictions, the transfer planner, chip evaluation, and the Monte
Carlo simulator into one explained report: what to do this gameweek and why,
with the evidence behind every call. Works in two modes:

- linked mode: optimizes YOUR squad (transfers, hits, chips, captaincy);
- draft mode (no team linked / pre-season): best from-scratch squad.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select

from ..config import settings
from ..db.models import Fixture, MyTeamPick, MyTeamState
from ..db.session import get_engine, session_scope
from ..models.features import build_feature_frame, load_fixtures, load_teams
from ..models.predictor import PointsPredictor, availability_for_rows
from ..models.prices import PriceModel
from ..models.upcoming import build_prediction_rows, dedupe_to_gw_level
from ..optim.chips import ChipEvaluation, evaluate_chips
from ..optim.planner import TransferPlan, TransferPlanner
from ..optim.squad import OptimizedSquad, SquadOptimizer
from ..rules.chips import ChipState, available_chips
from ..rules.constants import rules_for
from ..sim.monte_carlo import SimulationResult, simulate_gameweek

DEFAULT_HORIZON = 6


@dataclass
class CaptainOption:
    player_id: int
    name: str
    xp: float
    p90: float  # ceiling (90th percentile simulated)
    p10: float
    p_haul: float
    ownership: float
    reasoning: str


@dataclass
class WeeklyReport:
    season: str
    next_gw: int
    mode: str  # "linked" | "draft"
    xp_table: pd.DataFrame
    plan: TransferPlan | None
    draft: OptimizedSquad | None
    captains: list[CaptainOption]
    chips: list[ChipEvaluation]
    differentials: pd.DataFrame
    price_watch: pd.DataFrame
    names: dict[int, str] = field(default_factory=dict)


def next_gameweek(season: str) -> int:
    engine = get_engine()
    fx = pd.read_sql(
        select(Fixture).where(Fixture.season == season), engine
    )
    unfinished = fx[~fx["finished"] & fx["event"].notna()]
    if unfinished.empty:
        raise RuntimeError(f"season {season} has no unfinished fixtures — season over?")
    return int(unfinished["event"].min())


def load_my_state(season: str, up_to_gw: int) -> tuple[dict[int, int], int, int, ChipState, int]:
    """Returns (squad {player_id: sell_price}, bank, free_transfers, chips, team_value)."""
    engine = get_engine()
    picks = pd.read_sql(
        select(MyTeamPick).where(
            MyTeamPick.season == season, MyTeamPick.event < up_to_gw
        ),
        engine,
    )
    state = pd.read_sql(
        select(MyTeamState).where(
            MyTeamState.season == season, MyTeamState.event < up_to_gw
        ),
        engine,
    )
    if picks.empty or state.empty:
        raise RuntimeError("no synced team data — run `fpl sync-team` (and set FPL_TEAM_ID)")
    last_gw = int(picks["event"].max())
    squad_rows = picks[picks["event"] == last_gw]
    squad = {
        int(r["player_id"]): int(r["selling_price"] or r["purchase_price"] or 0)
        for _, r in squad_rows.iterrows()
    }
    st = state[state["event"] == state["event"].max()].iloc[0]
    chips = ChipState(used=[tuple(c) for c in json.loads(st["chips_used"])])
    return squad, int(st["bank"]), int(st["free_transfers"]), chips, int(st["team_value"])


def build_weekly_report(
    season: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    n_sims: int = 10_000,
    train_seasons: list[str] | None = None,
    as_of_gw: int | None = None,
) -> WeeklyReport:
    """`as_of_gw` replays a past deadline: models see only GWs before it (for
    testing the full pipeline out of season)."""
    season = season or settings.season
    rules = rules_for(season)
    gw = as_of_gw if as_of_gw is not None else next_gameweek(season)
    gws = [g for g in range(gw, min(gw + horizon, rules.total_gameweeks + 1))]

    seasons = [*(train_seasons or []), season]
    rows, players = build_prediction_rows(seasons, season, gws, cutoff_gw=as_of_gw)
    availability = availability_for_rows(rows, players)

    train_df = build_feature_frame(seasons)
    train_df = train_df[train_df["minutes"].notna()]
    fixtures = load_fixtures(seasons)
    fx_train = fixtures[fixtures["finished"]]
    if as_of_gw is not None:
        train_df = train_df[
            (train_df["season"] != season) | (train_df["event"] < as_of_gw)
        ]
        fx_train = fx_train[
            (fx_train["season"] != season)
            | (fx_train["event"].notna() & (fx_train["event"] < as_of_gw))
        ]
    teams = load_teams(seasons)

    predictor = PointsPredictor(rules).fit(train_df, fx_train, teams)
    pred = predictor.predict(rows, availability=availability)
    # Carry rate features into the simulator.
    for c in ("saves90", "defcon90"):
        pred[c] = rows[c].values

    gw_xp = dedupe_to_gw_level(pred)
    names = dict(zip(players["player_id"], players["web_name"]))
    own = dict(zip(players["player_id"], players["selected_by_percent"]))

    # xP matrix for the planner: one row per player, one xp_<gw> column per GW.
    mat = gw_xp.pivot_table(index="player_id", values="xp", columns="event", fill_value=0.0)
    mat.columns = [f"xp_{c}" for c in mat.columns]
    for g in gws:  # blank GWs: ensure the column exists
        if f"xp_{g}" not in mat.columns:
            mat[f"xp_{g}"] = 0.0
    meta = players.set_index("player_id")[["position", "team_id", "now_price"]].rename(
        columns={"now_price": "value"}
    )
    xp_matrix = mat.join(meta, how="inner").reset_index()

    # --- mode: linked plan or from-scratch draft ---------------------------
    plan: TransferPlan | None = None
    draft: OptimizedSquad | None = None
    chips: list[ChipEvaluation] = []
    mode = "draft"
    try:
        squad, bank, fts, chip_state, team_value = load_my_state(season, gw)
        mode = "linked"
        planner = TransferPlanner(rules)
        plan = planner.plan(xp_matrix, gws, squad, bank, fts)
        chips = evaluate_chips(
            rules, xp_matrix, gws, squad, bank, fts, team_value,
            available=available_chips(chip_state, gw, rules),
            base_plan=plan,
        )
    except RuntimeError:
        first = f"xp_{gws[0]}"
        gw_players = xp_matrix.rename(columns={first: "xp"})[
            ["player_id", "position", "team_id", "value", "xp"]
        ]
        # Draft on the near horizon: weight future GWs into the score.
        weights = {g: 0.9**k for k, g in enumerate(gws)}
        gw_players = gw_players.copy()
        gw_players["xp"] = sum(
            xp_matrix[f"xp_{g}"] * w for g, w in weights.items()
        ) / sum(weights.values())
        draft = SquadOptimizer(rules).optimize(gw_players)

    # --- captaincy via simulation ------------------------------------------
    next_pred = pred[pred["event"] == gw]
    sim = simulate_gameweek(next_pred, rules, n_sims=n_sims)
    sim_df = sim.frame().set_index("player_id")

    xi_ids = (
        plan.gameweeks[0].starting_ids
        if plan
        else (list(draft.squad.starting_ids) if draft else [])
    )
    next_xp = gw_xp[gw_xp["event"] == gw].set_index("player_id")
    captains = []
    for pid in sorted(xi_ids, key=lambda i: -float(next_xp["xp"].get(i, 0.0)))[:5]:
        srow = sim_df.loc[pid] if pid in sim_df.index else None
        xp_val = float(next_xp["xp"].get(pid, 0.0))
        captains.append(
            CaptainOption(
                player_id=pid,
                name=names.get(pid, str(pid)),
                xp=xp_val,
                p90=float(srow["sim_p90"]) if srow is not None else float("nan"),
                p10=float(srow["sim_p10"]) if srow is not None else float("nan"),
                p_haul=float(srow["p_haul_10plus"]) if srow is not None else float("nan"),
                ownership=float(own.get(pid, 0.0)),
                reasoning=_captain_reason(pid, next_pred, names),
            )
        )

    # --- differentials & price watch ---------------------------------------
    table = gw_xp[gw_xp["event"] == gw].copy()
    table["name"] = table["player_id"].map(names)
    table["ownership"] = table["player_id"].map(own)
    differentials = (
        table[(table["ownership"] < 10.0) & (table["p_start"] > 0.7)]
        .nlargest(10, "xp")[["player_id", "name", "position", "value", "ownership", "xp", "p_cs", "e_goals", "e_assists"]]
    )

    price = PriceModel().predict(players)
    pw = pd.DataFrame(
        {
            "player_id": price.player_id,
            "p_rise": price.p_rise,
            "p_fall": price.p_fall,
        }
    )
    pw["name"] = pw["player_id"].map(names)
    price_watch = pd.concat(
        [pw.nlargest(8, "p_rise"), pw.nlargest(8, "p_fall")]
    ).drop_duplicates("player_id")

    return WeeklyReport(
        season=season,
        next_gw=gw,
        mode=mode,
        xp_table=table.sort_values("xp", ascending=False),
        plan=plan,
        draft=draft,
        captains=captains,
        chips=chips,
        differentials=differentials,
        price_watch=price_watch,
        names=names,
    )


def _captain_reason(pid: int, pred: pd.DataFrame, names: dict[int, str]) -> str:
    rows = pred[pred["player_id"] == pid]
    if rows.empty:
        return ""
    r = rows.iloc[0]
    venue = "home" if bool(r["was_home"]) else "away"
    parts = [
        f"{names.get(pid, pid)} ({venue}): E[goals] {r['e_goals']:.2f}, E[assists] {r['e_assists']:.2f}",
        f"team expected goals {r['lambda_team']:.2f} vs opponent {r['lambda_opp']:.2f}",
        f"start probability {r['p_start']:.0%}",
    ]
    if len(rows) > 1:
        parts.append(f"DOUBLE gameweek: {len(rows)} fixtures")
    return "; ".join(parts)
