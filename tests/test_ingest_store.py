"""Ingest storage: only completed matches enter the player history."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from fpl_engine.db.models import Base, PlayerGameweek
from fpl_engine.ingest import store


def _history_row(fixture: int, minutes: int) -> dict:
    return {"fixture": fixture, "round": 7, "opponent_team": 2, "minutes": minutes}


def test_completed_means_full_time_not_just_finished():
    fixtures = [
        {"id": 1, "finished": True, "finished_provisional": True},
        # Full time, but `finished` waits for the data to be checked.
        {"id": 2, "finished": False, "finished_provisional": True},
        {"id": 3, "finished": False, "finished_provisional": False},
    ]
    assert store.completed_fixture_ids(fixtures) == {1, 2}


def test_unplayed_fixtures_are_not_stored_as_missed_games():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # A placeholder written by an older run, before the match was played.
        session.add(
            PlayerGameweek(
                season="2026-27", player_id=10, fixture_id=3, event=7, code=1,
                team_id=1, opponent_team=2, was_home=True, position=1, minutes=0,
            )
        )
        session.commit()

        dropped = store.delete_unplayed_history(session, "2026-27", completed={1, 2})
        stored = store.upsert_player_history(
            session, season="2026-27", player_id=10, code=1, position=1, team_id=1,
            history_rows=[_history_row(1, 90), _history_row(3, 0)],
            completed={1, 2},
        )
        session.commit()

        rows = session.scalars(select(PlayerGameweek)).all()
        assert dropped == 1
        assert stored == 1
        assert [(r.fixture_id, r.minutes) for r in rows] == [(1, 90)]
