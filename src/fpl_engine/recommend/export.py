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


def _my_team_payload(report: WeeklyReport, players: list[dict[str, Any]]) -> dict | None:
    """Current squad, chip status, rank history and the full Wildcard rebuild."""
    mt = report.my_team
    if mt is None:
        return None
    by_id = {p["id"]: p for p in players}
    gws = [str(g) for g in report.horizon_gws]

    def xp_h(pid: int) -> float:
        p = by_id.get(pid)
        return round(sum((p["xp_by_gw"].get(g) or 0.0) for g in gws), 3) if p else 0.0

    def entry(pid: int, role: str, order: int, sell: int | None = None) -> dict:
        p = by_id.get(pid, {})
        return {
            "id": pid,
            "name": p.get("name") or report.names.get(pid, str(pid)),
            "position": p.get("position"),
            "club_short": p.get("club_short", ""),
            "price": p.get("price"),
            "sell": sell,
            "role": role,  # captain | vice | XI | bench
            "order": order,
            "xp_next": p.get("xp"),
            "xp_horizon": xp_h(pid),
        }

    current = []
    for r in mt.picks.sort_values("pick_order").itertuples(index=False):
        role = "captain" if r.is_captain else "vice" if r.is_vice_captain else (
            "XI" if r.pick_order <= 11 else "bench"
        )
        if role in ("captain", "vice") and r.pick_order > 11:
            role = "bench"
        current.append(entry(int(r.player_id), role, int(r.pick_order), mt.squad.get(int(r.player_id))))

    wildcard = None
    if mt.wildcard is not None and mt.wildcard.plan is not None:
        wc = mt.wildcard
        g0 = wc.plan.gameweeks[0]
        xi = sorted(g0.starting_ids, key=lambda i: -(by_id.get(i, {}).get("xp") or 0.0))
        vice = next((i for i in xi if i != g0.captain_id), None)
        squad_wc = []
        for n, pid in enumerate(g0.starting_ids, start=1):
            role = "captain" if pid == g0.captain_id else "vice" if pid == vice else "XI"
            squad_wc.append(entry(pid, role, n))
        for n, pid in enumerate(g0.bench_ids, start=12):
            squad_wc.append(entry(pid, "bench", n))

        wc_ids = {e["id"] for e in squad_wc}
        out_ids = [i for i in mt.squad if i not in wc_ids]
        in_ids = [e["id"] for e in squad_wc if e["id"] not in mt.squad]
        moves = []
        for pos in (1, 2, 3, 4):
            outs = sorted(
                (i for i in out_ids if (by_id.get(i, {}).get("position")) == pos),
                key=lambda i: -(by_id.get(i, {}).get("price") or 0),
            )
            ins = sorted(
                (i for i in in_ids if (by_id.get(i, {}).get("position")) == pos),
                key=lambda i: -(by_id.get(i, {}).get("price") or 0),
            )
            for o, i in zip(outs, ins):
                moves.append(
                    {
                        "out": entry(o, "out", 0, mt.squad.get(o)),
                        "in": entry(i, "in", 0),
                        "delta_xp": round(xp_h(i) - xp_h(o), 2),
                    }
                )
        moves.sort(key=lambda m: -m["delta_xp"])
        wildcard = {
            "event": int(wc.event),
            "gain": _clean(wc.gain_pts),
            "detail": wc.detail,
            "squad": squad_wc,
            "moves": moves,
            "cost": sum((by_id.get(e["id"], {}).get("price") or 0) for e in squad_wc),
            "budget": mt.team_value + mt.bank,
            "follow_on": [
                {
                    "event": int(g.event),
                    "transfers_in": [report.names.get(i, str(i)) for i in g.transfers_in],
                    "transfers_out": [report.names.get(i, str(i)) for i in g.transfers_out],
                    "hits": int(g.hits),
                    "captain": report.names.get(g.captain_id, ""),
                }
                for g in wc.plan.gameweeks[1:]
            ],
        }

    return {
        "bank": mt.bank,
        "free_transfers": mt.free_transfers,
        "team_value": mt.team_value,
        "chips_used": [{"chip": c, "event": int(g)} for c, g in mt.chips_used],
        "chips_available": mt.chips_available,
        "current": current,
        "wildcard": wildcard,
        "history": _records(mt.history, ["event", "overall_points", "overall_rank", "event_transfers", "active_chip"]),
    }


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

    flags = (
        report.flags.set_index("player_id").to_dict("index") if not report.flags.empty else {}
    )
    fixture_labels: dict[tuple[int, int], list[str]] = {}
    for fr in report.fixture_rows.itertuples(index=False):
        label = f"{club_short.get(int(fr.opponent_team), '?')} ({'H' if fr.was_home else 'A'})"
        fixture_labels.setdefault((int(fr.player_id), int(fr.event)), []).append(label)

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
                "opp_by_gw": {
                    str(gw): " + ".join(fixture_labels.get((pid, gw), []))
                    for gw in report.horizon_gws
                    if (pid, gw) in fixture_labels
                },
                "status": flags.get(pid, {}).get("status", "a"),
                "chance": _clean(flags.get(pid, {}).get("chance_next_round")),
                "news": flags.get(pid, {}).get("news", "") or "",
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
        "my_team": _my_team_payload(report, players),
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
