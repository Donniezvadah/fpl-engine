"""Persist API payloads into SQLite (upsert semantics)."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..db.models import Fixture, MyTeamPick, MyTeamState, Player, PlayerGameweek, Team


def _parse_kickoff(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def upsert_teams(session: Session, season: str, teams: list[dict]) -> int:
    for t in teams:
        session.merge(
            Team(
                season=season,
                team_id=t["id"],
                code=t["code"],
                name=t["name"],
                short_name=t["short_name"],
                strength_overall_home=t.get("strength_overall_home", 0),
                strength_overall_away=t.get("strength_overall_away", 0),
                strength_attack_home=t.get("strength_attack_home", 0),
                strength_attack_away=t.get("strength_attack_away", 0),
                strength_defence_home=t.get("strength_defence_home", 0),
                strength_defence_away=t.get("strength_defence_away", 0),
            )
        )
    return len(teams)


def _chance(value) -> int | None:
    return None if value is None else int(value)


def upsert_players(session: Session, season: str, elements: list[dict]) -> int:
    for e in elements:
        session.merge(
            Player(
                season=season,
                player_id=e["id"],
                code=e["code"],
                web_name=e["web_name"],
                first_name=e.get("first_name", ""),
                second_name=e.get("second_name", ""),
                team_id=e["team"],
                position=e["element_type"],
                now_price=e["now_cost"],
                selected_by_percent=float(e.get("selected_by_percent", 0) or 0),
                status=e.get("status", "a"),
                chance_next_round=_chance(e.get("chance_of_playing_next_round")),
                news=e.get("news", "") or "",
                transfers_in_event=e.get("transfers_in_event", 0),
                transfers_out_event=e.get("transfers_out_event", 0),
                total_points=e.get("total_points", 0),
                minutes=e.get("minutes", 0),
                penalties_order=e.get("penalties_order"),
                corners_order=e.get("corners_and_indirect_freekicks_order"),
                freekicks_order=e.get("direct_freekicks_order"),
                updated_at=datetime.utcnow(),
            )
        )
    return len(elements)


def upsert_fixtures(session: Session, season: str, fixtures: list[dict]) -> int:
    for f in fixtures:
        session.merge(
            Fixture(
                season=season,
                fixture_id=f["id"],
                event=f.get("event"),
                kickoff_time=_parse_kickoff(f.get("kickoff_time")),
                team_h=f["team_h"],
                team_a=f["team_a"],
                team_h_score=f.get("team_h_score"),
                team_a_score=f.get("team_a_score"),
                team_h_difficulty=f.get("team_h_difficulty", 3),
                team_a_difficulty=f.get("team_a_difficulty", 3),
                finished=bool(f.get("finished", False)),
            )
        )
    return len(fixtures)


def completed_fixture_ids(fixtures: list[dict]) -> set[int]:
    """Fixtures that have reached full time.

    Once a gameweek is under way, element-summary lists every one of its fixtures
    in each player's history -- including matches not yet kicked off, as rows of
    zeros. Stored as they come, those read as games the player sat out: Arsenal's
    unplayed GW7 match took Raya's start probability from 0.98 to 0.73. Only
    matches at full time count. `finished` lags the final whistle by up to a day,
    so `finished_provisional` (set at full time) is accepted too.
    """
    return {
        f["id"] for f in fixtures if f.get("finished") or f.get("finished_provisional")
    }


def delete_unplayed_history(session: Session, season: str, completed: set[int]) -> int:
    """Drop player rows for fixtures not yet completed (placeholders from earlier runs)."""
    result = session.execute(
        delete(PlayerGameweek).where(
            PlayerGameweek.season == season, PlayerGameweek.fixture_id.not_in(completed)
        )
    )
    return result.rowcount or 0


def upsert_player_history(
    session: Session,
    season: str,
    player_id: int,
    code: int,
    position: int,
    team_id: int,
    history_rows: list[dict],
    completed: set[int] | None = None,
) -> int:
    """Persist element-summary `history` rows (one per fixture played this season).

    Rows for fixtures outside `completed` are skipped; see `completed_fixture_ids`.
    """
    n = 0
    for h in history_rows:
        if completed is not None and h["fixture"] not in completed:
            continue
        session.merge(
            PlayerGameweek(
                season=season,
                player_id=player_id,
                fixture_id=h["fixture"],
                event=h["round"],
                code=code,
                team_id=team_id,
                opponent_team=h.get("opponent_team", 0),
                was_home=bool(h.get("was_home", True)),
                position=position,
                total_points=h.get("total_points", 0),
                minutes=h.get("minutes", 0),
                started=bool(h["starts"]) if h.get("starts") is not None else None,
                goals_scored=h.get("goals_scored", 0),
                assists=h.get("assists", 0),
                clean_sheets=h.get("clean_sheets", 0),
                goals_conceded=h.get("goals_conceded", 0),
                own_goals=h.get("own_goals", 0),
                penalties_saved=h.get("penalties_saved", 0),
                penalties_missed=h.get("penalties_missed", 0),
                yellow_cards=h.get("yellow_cards", 0),
                red_cards=h.get("red_cards", 0),
                saves=h.get("saves", 0),
                bonus=h.get("bonus", 0),
                bps=h.get("bps", 0),
                defensive_contribution=h.get("defensive_contribution", 0) or 0,
                expected_goals=float(h.get("expected_goals", 0) or 0),
                expected_assists=float(h.get("expected_assists", 0) or 0),
                expected_goal_involvements=float(h.get("expected_goal_involvements", 0) or 0),
                expected_goals_conceded=float(h.get("expected_goals_conceded", 0) or 0),
                influence=float(h.get("influence", 0) or 0),
                creativity=float(h.get("creativity", 0) or 0),
                threat=float(h.get("threat", 0) or 0),
                ict_index=float(h.get("ict_index", 0) or 0),
                value=h.get("value", 0),
                selected=h.get("selected", 0),
                transfers_in=h.get("transfers_in", 0),
                transfers_out=h.get("transfers_out", 0),
            )
        )
        n += 1
    return n


def upsert_my_team(
    session: Session, season: str, gw: int, picks_payload: dict, history_payload: dict
) -> None:
    """Persist entry picks + state for one GW from the entry endpoints."""
    entry_history = picks_payload.get("entry_history", {})
    for p in picks_payload.get("picks", []):
        session.merge(
            MyTeamPick(
                season=season,
                event=gw,
                player_id=p["element"],
                pick_order=p["position"],
                is_captain=bool(p.get("is_captain")),
                is_vice_captain=bool(p.get("is_vice_captain")),
                multiplier=p.get("multiplier", 1),
                purchase_price=p.get("purchase_price"),
                selling_price=p.get("selling_price"),
            )
        )
    chips_used = [
        (c["name"], c["event"]) for c in history_payload.get("chips", [])
    ]
    session.merge(
        MyTeamState(
            season=season,
            event=gw,
            bank=entry_history.get("bank", 0),
            team_value=entry_history.get("value", 0),
            event_transfers=entry_history.get("event_transfers", 0),
            chips_used=json.dumps(chips_used),
            active_chip=picks_payload.get("active_chip"),
            overall_points=entry_history.get("total_points", 0),
            overall_rank=entry_history.get("overall_rank"),
        )
    )
