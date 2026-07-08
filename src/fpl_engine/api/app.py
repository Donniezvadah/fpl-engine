"""FastAPI backend.

Thin JSON layer over the same functions the CLI uses; the Phase-2 web dashboard
consumes these endpoints. The heavy report build is cached in-process per GW.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import settings
from ..recommend.weekly import WeeklyReport, build_weekly_report

app = FastAPI(title="fpl-engine", version="0.1.0")


def _records(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None}).to_dict(orient="records")


@lru_cache(maxsize=1)
def _cached_report() -> WeeklyReport:
    return build_weekly_report()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "season": settings.season}


@app.post("/refresh")
def refresh() -> dict:
    _cached_report.cache_clear()
    return {"status": "cache cleared"}


@app.get("/report")
def report() -> dict:
    try:
        r = _cached_report()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {
        "season": r.season,
        "next_gw": r.next_gw,
        "mode": r.mode,
        "captains": [c.__dict__ for c in r.captains],
        "chips": [e.__dict__ for e in r.chips],
        "differentials": _records(r.differentials),
        "price_watch": _records(r.price_watch),
        "plan": (
            [
                {
                    "event": g.event,
                    "transfers_in": [r.names.get(i, str(i)) for i in g.transfers_in],
                    "transfers_out": [r.names.get(i, str(i)) for i in g.transfers_out],
                    "hits": g.hits,
                    "free_transfers": g.free_transfers_before,
                    "captain": r.names.get(g.captain_id),
                    "xi_xp": g.xp,
                }
                for g in r.plan.gameweeks
            ]
            if r.plan
            else None
        ),
        "draft": (_records(r.draft.summary()) if r.draft else None),
    }


@app.get("/players")
def players_table(gw: int | None = None) -> list[dict]:
    try:
        r = _cached_report()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    table = r.xp_table
    cols = [c for c in ("player_id", "name", "position", "value", "ownership", "xp", "p_start", "p_cs", "e_goals", "e_assists", "n_fixtures") if c in table.columns]
    return _records(table[cols].head(300))
