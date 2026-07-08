"""Cross-season player identity.

The FPL API's per-season element ids are NOT stable, but every player carries a
permanent `code`. Both the live ingest and the historical archive store `code`
on each row, so identity resolution is a lookup — with a name+position fuzzy
fallback for archive rows that predate reliable codes.
"""

from __future__ import annotations

import unicodedata

import pandas as pd
from sqlalchemy import select

from ..db.models import Player
from ..db.session import get_engine


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def code_map(season: str) -> dict[int, int]:
    """player_id -> stable code for one season."""
    engine = get_engine()
    df = pd.read_sql(select(Player.player_id, Player.code).where(Player.season == season), engine)
    return dict(zip(df["player_id"], df["code"]))


def link_players_across_seasons(seasons: list[str]) -> pd.DataFrame:
    """One row per stable code with the per-season element ids that share it."""
    engine = get_engine()
    df = pd.read_sql(
        select(Player.season, Player.player_id, Player.code, Player.web_name, Player.position)
        .where(Player.season.in_(seasons)),
        engine,
    )
    wide = df.pivot_table(
        index="code", columns="season", values="player_id", aggfunc="first"
    ).reset_index()
    latest = df.sort_values("season").groupby("code").last().reset_index()
    return wide.merge(latest[["code", "web_name", "position"]], on="code", how="left")
