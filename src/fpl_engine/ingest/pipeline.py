"""Ingestion orchestration: `fpl update` and one-time backfills."""

from __future__ import annotations

from rich.console import Console
from rich.progress import track

from ..config import settings
from ..db.session import session_scope
from . import store
from .fpl_api import FplApiClient

console = Console()


def detect_season(bootstrap: dict) -> str:
    """Derive the season label ("2026-27") from GW1's deadline in bootstrap.

    This makes `fpl update` roll over automatically when the new game launches,
    without a config change.
    """
    events = bootstrap.get("events") or []
    if not events:
        return settings.season
    first_deadline = events[0].get("deadline_time", "")
    year = int(first_deadline[:4])
    return f"{year}-{str(year + 1)[-2:]}"


def update_current_season(season: str | None = None, with_player_history: bool = True) -> dict:
    """Pull bootstrap, fixtures, and (optionally) every player's per-fixture history.

    The player-history pass is ~1 request per player (rate-limited); it's the
    canonical per-fixture dataset for the current season including xG and
    defensive contribution.
    """
    counts: dict[str, int] = {}
    with FplApiClient() as api:
        bootstrap = api.bootstrap_static()
        fixtures = api.fixtures()
        season = season or detect_season(bootstrap)
        if season != settings.season:
            console.print(
                f"[yellow]API is serving season {season} (config says {settings.season}); "
                f"ingesting as {season}. Update FPL_SEASON and review rules/constants.py "
                f"for new-season rule changes.[/yellow]"
            )

        with session_scope() as session:
            counts["teams"] = store.upsert_teams(session, season, bootstrap["teams"])
            counts["players"] = store.upsert_players(session, season, bootstrap["elements"])
            counts["fixtures"] = store.upsert_fixtures(session, season, fixtures)

        if with_player_history:
            elements = bootstrap["elements"]
            # Skip players with zero minutes: they have no history rows worth fetching.
            with_minutes = [e for e in elements if e.get("minutes", 0) > 0]
            n_rows = 0
            with session_scope() as session:
                for e in track(with_minutes, description="player histories"):
                    summary = api.element_summary(e["id"])
                    n_rows += store.upsert_player_history(
                        session,
                        season=season,
                        player_id=e["id"],
                        code=e["code"],
                        position=e["element_type"],
                        team_id=e["team"],
                        history_rows=summary.get("history", []),
                    )
            counts["player_gameweek_rows"] = n_rows

    return counts


def sync_my_team(season: str | None = None) -> dict:
    """Pull the user's entry: picks for every finished GW + bank/chip state."""
    season = season or settings.season
    if not settings.team_id:
        raise RuntimeError(
            "FPL_TEAM_ID is not set. Add it to .env (from your Points-page URL)."
        )
    with FplApiClient() as api:
        history = api.entry_history(settings.team_id)
        gws = [row["event"] for row in history.get("current", [])]
        with session_scope() as session:
            for gw in track(gws, description="entry picks"):
                picks = api.entry_picks(settings.team_id, gw)
                store.upsert_my_team(session, season, gw, picks, history)
    return {"gameweeks": len(gws)}


def backfill_history(seasons: list[str]) -> dict:
    """One-time load of older seasons from the vaastav GitHub archive."""
    from .history import load_season, load_season_fixtures_teams

    counts = {}
    for s in seasons:
        console.print(f"loading {s} from GitHub archive…")
        n_fx, n_teams = load_season_fixtures_teams(s)
        counts[s] = {
            "player_rows": load_season(s),
            "fixtures": n_fx,
            "teams": n_teams,
        }
    return counts
