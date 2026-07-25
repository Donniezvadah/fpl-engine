"""Export a weekly report as static JSON for the browser dashboard.

The published site is static: there is no server to query, so everything the page
can show has to be baked into these files. That includes per-gameweek expected
points across the whole horizon, so the browser can re-rank players over any
number of gameweeks without asking anything to recompute.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..rules.constants import Position
from .weekly import WeeklyReport, build_weekly_report


def _clean(value: Any) -> Any:
    """JSON-safe scalar: numpy types out, NaN/inf to None."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not np.isfinite(f) else round(f, 4)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_clean(v) for v in value]
    return value


def _records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    cols = [c for c in (columns or df.columns) if c in df.columns]
    return [{c: _clean(row[c]) for c in cols} for _, row in df[cols].iterrows()]


def build_payload(report: WeeklyReport) -> dict[str, Any]:
    names = report.names
    club_name = dict(zip(report.teams["team_id"], report.teams["name"]))
    club_short = dict(zip(report.teams["team_id"], report.teams["short_name"]))

    # Per-player horizon: {player_id: {gw: xp}}, plus the per-GW fixture count so
    # the site can show blanks and doubles.
    horizon = report.horizon_xp
    by_player: dict[int, dict[str, Any]] = {}
    if not horizon.empty:
        for row in horizon.itertuples(index=False):
            entry = by_player.setdefault(
                int(row.player_id),
                {"xp_by_gw": {}, "fixtures_by_gw": {}},
            )
            entry["xp_by_gw"][str(int(row.event))] = _clean(row.xp)
            entry["fixtures_by_gw"][str(int(row.event))] = int(
                getattr(row, "n_fixtures", 1) or 1
            )

    next_gw_rows = report.xp_table
    players: list[dict[str, Any]] = []
    for row in next_gw_rows.itertuples(index=False):
        pid = int(row.player_id)
        h = by_player.get(pid, {"xp_by_gw": {}, "fixtures_by_gw": {}})
        price = int(row.value)
        players.append(
            {
                "id": pid,
                "name": names.get(pid, str(pid)),
                "position": int(row.position),
                "position_name": Position(int(row.position)).name,
                "team_id": int(row.team_id),
                "club": club_name.get(int(row.team_id), ""),
                "club_short": club_short.get(int(row.team_id), ""),
                "price": price,
                "ownership": _clean(getattr(row, "ownership", None)),
                "xp": _clean(row.xp),
                "p_start": _clean(getattr(row, "p_start", None)),
                "p60": _clean(getattr(row, "p60", None)),
                "p_cs": _clean(getattr(row, "p_cs", None)),
                "e_goals": _clean(getattr(row, "e_goals", None)),
                "e_assists": _clean(getattr(row, "e_assists", None)),
                "xp_by_gw": h["xp_by_gw"],
                "fixtures_by_gw": h["fixtures_by_gw"],
            }
        )
    players.sort(key=lambda p: -(p["xp"] or 0))

    squad = None
    if report.draft is not None:
        squad = {
            "kind": "draft",
            "cost": int(report.draft.cost),
            "players": [
                {
                    "id": int(r["player_id"]),
                    "name": names.get(int(r["player_id"]), ""),
                    "role": str(r["role"]),
                    "position": int(r["position"]),
                    "position_name": Position(int(r["position"])).name,
                    "price": int(r["value"]),
                    "xp": _clean(r["xp"]),
                }
                for _, r in report.draft.summary().iterrows()
            ],
        }
    elif report.plan is not None:
        first = report.plan.gameweeks[0]
        squad = {
            "kind": "linked",
            "cost": None,
            "players": [
                {
                    "id": pid,
                    "name": names.get(pid, str(pid)),
                    "role": (
                        "captain"
                        if pid == first.captain_id
                        else ("XI" if pid in first.starting_ids else "bench")
                    ),
                    "position": None,
                    "position_name": "",
                    "price": None,
                    "xp": None,
                }
                for pid in [*first.starting_ids, *first.bench_ids]
            ],
        }

    return {
        "meta": {
            "season": report.season,
            "next_gw": report.next_gw,
            "mode": report.mode,
            "horizon_gws": [int(g) for g in report.horizon_gws],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "player_count": len(players),
        },
        "captains": [
            {
                "id": c.player_id,
                "name": c.name,
                "xp": _clean(c.xp),
                "p10": _clean(c.p10),
                "p90": _clean(c.p90),
                "p_haul": _clean(c.p_haul),
                "ownership": _clean(c.ownership),
                "reasoning": c.reasoning,
            }
            for c in report.captains
        ],
        "chips": [
            {
                "chip": e.chip,
                "event": int(e.event),
                "gain": _clean(e.gain_pts),
                "detail": e.detail,
            }
            for e in report.chips
        ],
        "plan": (
            [
                {
                    "event": int(g.event),
                    "transfers_in": [names.get(i, str(i)) for i in g.transfers_in],
                    "transfers_out": [names.get(i, str(i)) for i in g.transfers_out],
                    "hits": int(g.hits),
                    "free_transfers": int(g.free_transfers_before),
                    "captain": names.get(g.captain_id, ""),
                    "xi_xp": _clean(g.xp),
                }
                for g in report.plan.gameweeks
            ]
            if report.plan
            else None
        ),
        "squad": squad,
        "differentials": _records(
            report.differentials,
            ["player_id", "name", "position", "value", "ownership", "xp"],
        ),
        "price_watch": _records(report.price_watch, ["player_id", "name", "p_rise", "p_fall"]),
        "players": players,
    }


def export_site_data(
    out_dir: Path,
    horizon: int = 6,
    n_sims: int = 10_000,
    as_of_gw: int | None = None,
    report: WeeklyReport | None = None,
) -> dict[str, Any]:
    """Write data.json into `out_dir` and return a small summary."""
    report = report or build_weekly_report(
        horizon=horizon, n_sims=n_sims, as_of_gw=as_of_gw
    )
    payload = build_payload(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "data.json"
    target.write_text(json.dumps(payload, separators=(",", ":")))
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "players": len(payload["players"]),
        "season": payload["meta"]["season"],
        "gw": payload["meta"]["next_gw"],
    }
