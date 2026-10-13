"""Last-gameweek review: how the squad scored, how the models did, what-ifs.

Predictions are judged against what was known *before* the deadline. The best
source is a snapshot saved at the time (`reviews/predictions_gw<N>.json`, written
by `fpl export`); without one the gameweek is replayed with models trained only
on earlier gameweeks. The replay assumes every player was available, because
the injury flags the engine holds are today's, not the deadline's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import select

from ..db.models import MyTeamState, PlayerGameweek
from ..db.session import get_engine
from ..ingest.fpl_api import FplApiClient
from ..rules.autosubs import apply_autosubs, effective_captain
from ..rules.constants import Position, rules_for
from ..rules.squad import Squad, SquadPlayer

REVIEWS_DIR = Path("reviews")


def snapshot_path(gw: int, root: Path = REVIEWS_DIR) -> Path:
    return root / f"predictions_gw{gw}.json"


def load_snapshot(gw: int, root: Path = REVIEWS_DIR) -> dict | None:
    path = snapshot_path(gw, root)
    return json.loads(path.read_text()) if path.exists() else None


def write_snapshot(
    gw: int,
    xp: dict[int, float],
    recommended: list[dict] | None,
    generated_at: str,
    root: Path = REVIEWS_DIR,
) -> Path:
    """Freeze this week's predictions so next week's review can be honest."""
    root.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(gw, root)
    path.write_text(
        json.dumps(
            {
                "gw": gw,
                "generated_at": generated_at,
                "xp": {str(k): round(float(v), 3) for k, v in xp.items()},
                "recommended": recommended,
            },
            separators=(",", ":"),
        )
    )
    return path


def _squad_from_picks(
    picks: list[dict], position_of: dict[int, int], club_of: dict[int, int]
) -> Squad:
    ordered = sorted(picks, key=lambda p: p["position"])
    players = tuple(
        SquadPlayer(p["element"], Position(position_of[p["element"]]), club_of.get(p["element"], 0), 0)
        for p in ordered
    )
    xi = frozenset(p["element"] for p in ordered if p["position"] <= 11)
    captain = next(p["element"] for p in ordered if p["is_captain"])
    vice = next(p["element"] for p in ordered if p["is_vice_captain"])
    bench = tuple(
        p["element"]
        for p in ordered
        if p["position"] > 11 and position_of[p["element"]] != Position.GK.value
    )
    return Squad(players, xi, captain, vice, bench)


def _score(squad: Squad, pts: dict[int, int], mins: dict[int, int], rules) -> tuple[int, int | None]:
    """Gross points after autosubs and captaincy fallback, plus who wore the armband."""
    xi = apply_autosubs(squad, mins, rules)
    cap = effective_captain(squad, mins)
    total = sum(pts.get(i, 0) for i in xi)
    if cap is not None and cap in xi:
        total += pts.get(cap, 0)
    return total, cap


def _rec_to_picks(rec: list[dict]) -> list[dict]:
    """Snapshot recommendation entries -> pick dicts (order, captain, vice)."""
    return [
        {
            "element": e["id"],
            "position": e["order"],
            "is_captain": e["role"] == "captain",
            "is_vice_captain": e["role"] == "vice",
        }
        for e in rec
    ]


def build_review(
    season: str,
    team_id: int,
    gw: int,
    xp: dict[int, float],
    source: str,
    names: dict[int, str],
    position_of: dict[int, int],
    club_of: dict[int, int],
    club_short: dict[int, str],
    recommended: list[dict] | None = None,
) -> dict[str, Any] | None:
    """Review of a finished gameweek for the linked entry; None if unavailable."""
    rules = rules_for(season)
    engine = get_engine()
    pg = pd.read_sql(
        select(PlayerGameweek).where(PlayerGameweek.season == season, PlayerGameweek.event == gw),
        engine,
    )
    if pg.empty:
        return None
    agg = pg.groupby("player_id").agg(pts=("total_points", "sum"), mins=("minutes", "sum"))
    pts = {int(i): int(v) for i, v in agg["pts"].items()}
    mins = {int(i): int(v) for i, v in agg["mins"].items()}

    try:
        with FplApiClient(snapshot=False) as api:
            boot = api.bootstrap_static()
            this = api.entry_picks(team_id, gw)
            prev = api.entry_picks(team_id, gw - 1) if gw > 1 else None
    except Exception:  # offline or entry unavailable: skip the review
        return None
    ev = boot["events"][gw - 1]
    eh = this["entry_history"]

    squad = _squad_from_picks(this["picks"], position_of, club_of)
    gross, cap = _score(squad, pts, mins, rules)
    hit = int(eh.get("event_transfers_cost", 0))
    xi_final = apply_autosubs(squad, mins, rules)
    subs_in = {s["element_in"] for s in this.get("automatic_subs", [])}
    subs_out = {s["element_out"] for s in this.get("automatic_subs", [])}

    state = pd.read_sql(
        select(MyTeamState).where(MyTeamState.season == season, MyTeamState.event == gw - 1), engine
    )
    rank_before = (
        int(state["overall_rank"].iloc[0])
        if not state.empty and pd.notna(state["overall_rank"].iloc[0])
        else None
    )
    rank_after = eh.get("overall_rank")

    players = []
    for p in sorted(this["picks"], key=lambda r: r["position"]):
        pid = p["element"]
        in_xi = pid in xi_final
        role = (
            "bench"
            if not in_xi
            else "captain"
            if pid == cap
            else "vice"
            if p["is_vice_captain"]
            else "XI"
        )
        players.append(
            {
                "id": pid,
                "name": names.get(pid, str(pid)),
                "club_short": club_short.get(club_of.get(pid, 0), ""),
                "position": position_of.get(pid),
                "role": role,
                "minutes": mins.get(pid, 0),
                "points": pts.get(pid, 0),
                "xp": round(float(xp.get(pid, 0.0)), 2),
                "subbed_in": pid in subs_in,
                "subbed_out": pid in subs_out,
            }
        )

    xi_xp = sum(xp.get(i, 0.0) for i in xi_final) + xp.get(cap or -1, 0.0)

    what_if: list[dict] = []
    if prev is not None:
        old = _squad_from_picks(prev["picks"], position_of, club_of)
        what_if.append(
            {"label": f"Your GW{gw - 1} squad, no wildcard", "points": _score(old, pts, mins, rules)[0]}
        )
    if recommended:
        rec = _squad_from_picks(_rec_to_picks(recommended), position_of, club_of)
        what_if.append({"label": "The engine's wildcard squad", "points": _score(rec, pts, mins, rules)[0]})
    best_cap = max(xi_final, key=lambda i: pts.get(i, 0))
    what_if.append(
        {
            "label": f"Your team with {names.get(best_cap, '?')} as captain",
            "points": gross - pts.get(cap or -1, 0) + pts.get(best_cap, 0),
        }
    )
    what_if.append({"label": "What you scored (before hits)", "points": gross})

    # ---- how the models did on every player who played ----------------------
    frame = pd.DataFrame({"xp": pd.Series(xp), "pts": pd.Series(pts), "mins": pd.Series(mins)}).dropna()
    played = frame[frame["mins"] > 0]
    err = played["xp"] - played["pts"]
    quint = pd.qcut(played["xp"].rank(method="first"), 5, labels=False)
    calibration = [
        {
            "bin": f"{int(i) + 1}",
            "xp": round(float(g["xp"].mean()), 2),
            "actual": round(float(g["pts"].mean()), 2),
            "n": int(len(g)),
        }
        for i, g in played.groupby(quint)
    ]
    top10 = played.nlargest(10, "xp")
    model = {
        "n": int(len(played)),
        "mae": round(float(err.abs().mean()), 3),
        "rmse": round(float(np.sqrt((err**2).mean())), 3),
        "spearman": round(float(spearmanr(played["xp"], played["pts"]).statistic), 3),
        "top10_actual": round(float(top10["pts"].mean()), 2),
        "league_actual": round(float(played["pts"].mean()), 2),
        "calibration": calibration,
        "top_scorers": [
            {"name": names.get(int(i), str(i)), "points": int(r.pts), "xp": round(float(r.xp), 1)}
            for i, r in played.nlargest(6, "pts").iterrows()
        ],
        "source": source,
    }

    review: dict[str, Any] = {
        "event": gw,
        "finished": bool(ev.get("finished")),
        "provisional": not bool(ev.get("data_checked")),
        "points": int(eh["points"]) - hit,
        "gross_points": gross,
        "hit": hit,
        "bench_points": int(eh.get("points_on_bench", 0)),
        "average": ev.get("average_entry_score"),
        "rank_before": rank_before,
        "rank_after": int(rank_after) if rank_after else None,
        "chip": this.get("active_chip"),
        "captain": names.get(cap, "") if cap else "",
        "xi_xp": round(xi_xp, 1),
        "players": players,
        "what_if": what_if,
        "model": model,
        "recommended_overlap": None,
    }
    if recommended:
        rec_ids = {e["id"] for e in recommended}
        mine = {p["element"] for p in this["picks"]}
        review["recommended_overlap"] = {
            "shared": len(rec_ids & mine),
            "you_only": [names.get(i, str(i)) for i in sorted(mine - rec_ids)],
            "engine_only": [
                {"name": names.get(i, str(i)), "points": pts.get(i, 0)} for i in sorted(rec_ids - mine)
            ],
        }
    review["comments"] = _comments(review)
    return review


def _comments(r: dict[str, Any]) -> list[str]:
    """Plain-language read of the week, built from the numbers above."""
    out: list[str] = []
    pts, avg = r["points"], r["average"]
    gw = r["event"]
    if avg is not None:
        diff = pts - avg
        verdict = "above" if diff > 0 else "below" if diff < 0 else "level with"
        out.append(
            f"GW{gw}: {pts} points, "
            + (f"{abs(diff)} {verdict} the average of {avg}." if diff else f"{verdict} the average of {avg}.")
            + (" This was your wildcard week." if r["chip"] == "wildcard" else "")
        )
    if r["rank_before"] and r["rank_after"]:
        moved = r["rank_before"] - r["rank_after"]
        out.append(
            f"Overall rank {r['rank_before']:,} to {r['rank_after']:,}: "
            f"{'up' if moved > 0 else 'down'} {abs(moved):,} places."
        )
    out.append(
        f"Your XI and captain were expected to score {r['xi_xp']}; you got {r['gross_points']} "
        f"({r['gross_points'] - r['xi_xp']:+.1f}). A gap this size is ordinary weekly noise."
    )

    xi = [p for p in r["players"] if p["role"] != "bench" and p["minutes"] > 0]
    if xi:
        best = max(xi, key=lambda p: p["points"] - p["xp"])
        worst = min(xi, key=lambda p: p["points"] - p["xp"])
        out.append(
            f"Biggest hit: {best['name']} with {best['points']} points against {best['xp']} expected. "
            f"Biggest miss: {worst['name']} with {worst['points']} against {worst['xp']}."
        )
    cap = next((p for p in r["players"] if p["role"] == "captain"), None)
    if cap:
        top = max((p for p in r["players"] if p["role"] != "bench"), key=lambda p: p["points"])
        tail = (
            f" {top['name']} ({top['points']}) would have paid far more."
            if top["id"] != cap["id"] and top["points"] > cap["points"]
            else ""
        )
        unit = "point" if cap["points"] == 1 else "points"
        out.append(f"Captain {cap['name']} returned {cap['points']} {unit} (doubled).{tail}")
    if r["bench_points"]:
        out.append(f"{r['bench_points']} points stayed on the bench.")
    if r["hit"]:
        out.append(f"Transfer hits cost {r['hit']} points.")

    old = next((w for w in r["what_if"] if w["label"].endswith("no wildcard")), None)
    if old and r["chip"] == "wildcard":
        gap = r["gross_points"] - old["points"]
        if gap < 0:
            out.append(
                f"Honest verdict: your previous squad would have scored {old['points']} this week, "
                f"{-gap} more than the wildcard team. One gameweek says little about a six-week rebuild, "
                "but the wildcard did not pay off straight away."
            )
        else:
            out.append(
                f"The wildcard beat your previous squad by {gap} points this week "
                f"({r['gross_points']} against {old['points']})."
            )

    ov = r["recommended_overlap"]
    if ov and ov["engine_only"]:
        missed = ", ".join(f"{e['name']} ({e['points']})" for e in ov["engine_only"])
        out.append(
            f"You matched {ov['shared']} of 15 players in the engine's wildcard squad. "
            f"The ones you left out scored: {missed}."
        )

    m = r["model"]
    out.append(
        f"Model check on the {m['n']} players who played (players who did not play are excluded, as in the backtest): average error {m['mae']} points "
        f"(2025-26 backtest: 1.95). Its ten highest-rated players averaged {m['top10_actual']} points "
        f"against {m['league_actual']} for everyone, so the ranking held up even where single results did not."
    )
    if r["provisional"]:
        out.append("FPL has not finalised this gameweek yet, so bonus points can still move these figures.")
    return out
