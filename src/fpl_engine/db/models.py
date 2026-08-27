"""SQLAlchemy schema.

Two families of tables:
- Live-season state pulled from the official API (players, teams, fixtures, my team).
- Long history across seasons (player_gameweeks) used for model training. Rows are
  keyed by (season, ...) so multiple seasons coexist; cross-season player identity
  is resolved through the `player_codes` mapping (the API's stable `code` field).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    season: Mapped[str] = mapped_column(String(7), primary_key=True)
    team_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # API per-season id 1..20
    code: Mapped[int] = mapped_column(Integer, index=True)  # stable across seasons
    name: Mapped[str] = mapped_column(String(50))
    short_name: Mapped[str] = mapped_column(String(3))
    strength_overall_home: Mapped[int] = mapped_column(Integer, default=0)
    strength_overall_away: Mapped[int] = mapped_column(Integer, default=0)
    strength_attack_home: Mapped[int] = mapped_column(Integer, default=0)
    strength_attack_away: Mapped[int] = mapped_column(Integer, default=0)
    strength_defence_home: Mapped[int] = mapped_column(Integer, default=0)
    strength_defence_away: Mapped[int] = mapped_column(Integer, default=0)


class Player(Base):
    """Per-season player master row (current snapshot of the API `elements` entry)."""

    __tablename__ = "players"

    season: Mapped[str] = mapped_column(String(7), primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # API per-season element id
    code: Mapped[int] = mapped_column(Integer, index=True)  # stable across seasons
    web_name: Mapped[str] = mapped_column(String(60))
    first_name: Mapped[str] = mapped_column(String(60), default="")
    second_name: Mapped[str] = mapped_column(String(60), default="")
    team_id: Mapped[int] = mapped_column(Integer, index=True)
    position: Mapped[int] = mapped_column(Integer)  # 1 GK / 2 DEF / 3 MID / 4 FWD
    now_price: Mapped[int] = mapped_column(Integer)  # tenths
    selected_by_percent: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(1), default="a")  # a/d/i/s/u/n
    chance_next_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    news: Mapped[str] = mapped_column(Text, default="")
    transfers_in_event: Mapped[int] = mapped_column(Integer, default=0)
    transfers_out_event: Mapped[int] = mapped_column(Integer, default=0)
    total_points: Mapped[int] = mapped_column(Integer, default=0)
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    penalties_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corners_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    freekicks_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Fixture(Base):
    __tablename__ = "fixtures"

    season: Mapped[str] = mapped_column(String(7), primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)  # GW; None=TBD
    kickoff_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    team_h: Mapped[int] = mapped_column(Integer, index=True)
    team_a: Mapped[int] = mapped_column(Integer, index=True)
    team_h_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_a_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_h_difficulty: Mapped[int] = mapped_column(Integer, default=3)
    team_a_difficulty: Mapped[int] = mapped_column(Integer, default=3)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)


class PlayerGameweek(Base):
    """One player's stats in one gameweek fixture (training data + live results)."""

    __tablename__ = "player_gameweeks"

    season: Mapped[str] = mapped_column(String(7), primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event: Mapped[int] = mapped_column(Integer, index=True)
    code: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    team_id: Mapped[int] = mapped_column(Integer, default=0)
    opponent_team: Mapped[int] = mapped_column(Integer, default=0)
    was_home: Mapped[bool] = mapped_column(Boolean, default=True)
    position: Mapped[int] = mapped_column(Integer, default=0)

    total_points: Mapped[int] = mapped_column(Integer, default=0)
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    started: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    goals_scored: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, default=0)
    goals_conceded: Mapped[int] = mapped_column(Integer, default=0)
    own_goals: Mapped[int] = mapped_column(Integer, default=0)
    penalties_saved: Mapped[int] = mapped_column(Integer, default=0)
    penalties_missed: Mapped[int] = mapped_column(Integer, default=0)
    yellow_cards: Mapped[int] = mapped_column(Integer, default=0)
    red_cards: Mapped[int] = mapped_column(Integer, default=0)
    saves: Mapped[int] = mapped_column(Integer, default=0)
    bonus: Mapped[int] = mapped_column(Integer, default=0)
    bps: Mapped[int] = mapped_column(Integer, default=0)
    defensive_contribution: Mapped[int] = mapped_column(Integer, default=0)

    expected_goals: Mapped[float] = mapped_column(Float, default=0.0)
    expected_assists: Mapped[float] = mapped_column(Float, default=0.0)
    expected_goal_involvements: Mapped[float] = mapped_column(Float, default=0.0)
    expected_goals_conceded: Mapped[float] = mapped_column(Float, default=0.0)

    influence: Mapped[float] = mapped_column(Float, default=0.0)
    creativity: Mapped[float] = mapped_column(Float, default=0.0)
    threat: Mapped[float] = mapped_column(Float, default=0.0)
    ict_index: Mapped[float] = mapped_column(Float, default=0.0)

    value: Mapped[int] = mapped_column(Integer, default=0)  # price at the time, tenths
    selected: Mapped[int] = mapped_column(Integer, default=0)
    transfers_in: Mapped[int] = mapped_column(Integer, default=0)
    transfers_out: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (Index("ix_pgw_season_event", "season", "event"),)


class MyTeamPick(Base):
    """The user's squad as of a given gameweek (from entry/{id}/event/{gw}/picks)."""

    __tablename__ = "my_team_picks"

    season: Mapped[str] = mapped_column(String(7), primary_key=True)
    event: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pick_order: Mapped[int] = mapped_column(Integer)  # 1-15
    is_captain: Mapped[bool] = mapped_column(Boolean, default=False)
    is_vice_captain: Mapped[bool] = mapped_column(Boolean, default=False)
    multiplier: Mapped[int] = mapped_column(Integer, default=1)
    purchase_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selling_price: Mapped[int | None] = mapped_column(Integer, nullable=True)


class MyTeamState(Base):
    """Bank, chips, and transfer state for the user's entry per gameweek."""

    __tablename__ = "my_team_state"

    season: Mapped[str] = mapped_column(String(7), primary_key=True)
    event: Mapped[int] = mapped_column(Integer, primary_key=True)
    bank: Mapped[int] = mapped_column(Integer, default=0)  # tenths
    team_value: Mapped[int] = mapped_column(Integer, default=0)
    # Transfers MADE in this gameweek. Free transfers available are not
    # exposed by the API at all and must be derived by rolling the official
    # banking rule forward — see recommend.weekly.free_transfers_for_next_gw.
    event_transfers: Mapped[int] = mapped_column(Integer, default=0)
    chips_used: Mapped[str] = mapped_column(Text, default="[]")  # JSON [(chip, gw)]
    active_chip: Mapped[str | None] = mapped_column(String(20), nullable=True)
    overall_points: Mapped[int] = mapped_column(Integer, default=0)
    overall_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
