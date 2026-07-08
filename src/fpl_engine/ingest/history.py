"""Historical season loader from the vaastav/Fantasy-Premier-League GitHub archive.

Used for seasons before the one served by the live API. We load merged_gw.csv
(one row per player per fixture) plus players_raw.csv (for the stable `code` and
element_type), and store into the same player_gameweeks table.

Only seasons since 2022-23 carry xG columns; earlier seasons still train the
minutes and points models on realized stats.
"""

from __future__ import annotations

import csv
import io

import httpx

from ..config import settings
from ..db.session import session_scope
from ..db.models import PlayerGameweek

POSITION_MAP = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4, "AM": 5}


def _fetch_csv(url: str) -> list[dict]:
    resp = httpx.get(url, timeout=settings.request_timeout_s, follow_redirects=True)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v in (None, "", "None"):
        return default
    return float(v)


def _i(row: dict, key: str, default: int = 0) -> int:
    return int(_f(row, key, float(default)))


def load_season_fixtures_teams(season: str) -> tuple[int, int]:
    """Ingest a historical season's fixtures.csv and teams.csv (needed by the
    Dixon-Coles model and team-form features during backtests)."""
    from ..db.models import Fixture, Team
    from datetime import datetime

    base = f"{settings.history_base_url}/{season}"
    fixtures = _fetch_csv(f"{base}/fixtures.csv")
    teams = _fetch_csv(f"{base}/teams.csv")

    def _kickoff(v: str | None):
        if not v or v == "None":
            return None
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)

    with session_scope() as session:
        for t in teams:
            session.merge(
                Team(
                    season=season,
                    team_id=int(t["id"]),
                    code=int(t["code"]),
                    name=t.get("name", ""),
                    short_name=t.get("short_name", ""),
                    strength_overall_home=_i(t, "strength_overall_home"),
                    strength_overall_away=_i(t, "strength_overall_away"),
                    strength_attack_home=_i(t, "strength_attack_home"),
                    strength_attack_away=_i(t, "strength_attack_away"),
                    strength_defence_home=_i(t, "strength_defence_home"),
                    strength_defence_away=_i(t, "strength_defence_away"),
                )
            )
        for f in fixtures:
            if not f.get("id"):
                continue
            session.merge(
                Fixture(
                    season=season,
                    fixture_id=int(f["id"]),
                    event=(int(float(f["event"])) if f.get("event") not in (None, "", "None") else None),
                    kickoff_time=_kickoff(f.get("kickoff_time")),
                    team_h=int(f["team_h"]),
                    team_a=int(f["team_a"]),
                    team_h_score=(_i(f, "team_h_score") if f.get("team_h_score") not in ("", "None", None) else None),
                    team_a_score=(_i(f, "team_a_score") if f.get("team_a_score") not in ("", "None", None) else None),
                    team_h_difficulty=_i(f, "team_h_difficulty", 3),
                    team_a_difficulty=_i(f, "team_a_difficulty", 3),
                    finished=str(f.get("finished", "False")).lower() in ("true", "1"),
                )
            )
    return len(fixtures), len(teams)


def load_season(season: str) -> int:
    """Ingest one historical season (e.g. "2023-24") from GitHub into SQLite."""
    base = f"{settings.history_base_url}/{season}"
    merged = _fetch_csv(f"{base}/gws/merged_gw.csv")
    players_raw = _fetch_csv(f"{base}/players_raw.csv")
    code_by_element = {int(p["id"]): int(p["code"]) for p in players_raw}
    type_by_element = {int(p["id"]): int(p["element_type"]) for p in players_raw}
    team_by_element = {int(p["id"]): int(p["team"]) for p in players_raw}

    n = 0
    with session_scope() as session:
        for row in merged:
            element = _i(row, "element")
            fixture = _i(row, "fixture")
            if not element or not fixture:
                continue
            position = type_by_element.get(element) or POSITION_MAP.get(
                row.get("position", ""), 0
            )
            starts_raw = row.get("starts")
            session.merge(
                PlayerGameweek(
                    season=season,
                    player_id=element,
                    fixture_id=fixture,
                    event=_i(row, "GW") or _i(row, "round"),
                    code=code_by_element.get(element),
                    team_id=team_by_element.get(element, 0),
                    opponent_team=_i(row, "opponent_team"),
                    was_home=str(row.get("was_home", "True")).lower() in ("true", "1"),
                    position=position,
                    total_points=_i(row, "total_points"),
                    minutes=_i(row, "minutes"),
                    started=(bool(int(float(starts_raw))) if starts_raw not in (None, "", "None") else None),
                    goals_scored=_i(row, "goals_scored"),
                    assists=_i(row, "assists"),
                    clean_sheets=_i(row, "clean_sheets"),
                    goals_conceded=_i(row, "goals_conceded"),
                    own_goals=_i(row, "own_goals"),
                    penalties_saved=_i(row, "penalties_saved"),
                    penalties_missed=_i(row, "penalties_missed"),
                    yellow_cards=_i(row, "yellow_cards"),
                    red_cards=_i(row, "red_cards"),
                    saves=_i(row, "saves"),
                    bonus=_i(row, "bonus"),
                    bps=_i(row, "bps"),
                    defensive_contribution=_i(row, "defensive_contribution"),
                    expected_goals=_f(row, "expected_goals"),
                    expected_assists=_f(row, "expected_assists"),
                    expected_goal_involvements=_f(row, "expected_goal_involvements"),
                    expected_goals_conceded=_f(row, "expected_goals_conceded"),
                    influence=_f(row, "influence"),
                    creativity=_f(row, "creativity"),
                    threat=_f(row, "threat"),
                    ict_index=_f(row, "ict_index"),
                    value=_i(row, "value"),
                    selected=_i(row, "selected"),
                    transfers_in=_i(row, "transfers_in"),
                    transfers_out=_i(row, "transfers_out"),
                )
            )
            n += 1
    return n
