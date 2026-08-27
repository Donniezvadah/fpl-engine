"""Engine/session factory for the local SQLite database."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings
from .models import Base

_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine(db_path: Path | None = None):
    global _engine, _SessionLocal
    if _engine is None:
        path = db_path or settings.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{path}",
            future=True,
            connect_args={"timeout": 60},  # ride out concurrent ingest writers
        )

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        Base.metadata.create_all(_engine)
        _heal_cached_schema(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


@contextmanager
def session_scope(db_path: Path | None = None) -> Iterator[Session]:
    get_engine(db_path)
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Tables that are a pure cache of the FPL API and can be rebuilt in seconds by
# re-syncing. Dropping one costs nothing; leaving it stale breaks every query
# against it.
_REBUILDABLE = ("my_team_state", "my_team_picks")


def _heal_cached_schema(engine) -> None:
    """Recreate cache tables whose columns no longer match the model.

    `create_all` only creates missing tables, never alters existing ones, so a
    database written by an older version keeps its old columns and every query
    against the renamed one fails. That is exactly what a cached CI database did
    after `free_transfers` became `event_transfers`. Only tables that can be
    refetched are touched; ingested match history is never dropped.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    for name in _REBUILDABLE:
        table = Base.metadata.tables.get(name)
        if table is None or name not in existing:
            continue
        on_disk = {c["name"] for c in inspector.get_columns(name)}
        expected = {c.name for c in table.columns}
        if on_disk != expected:
            table.drop(engine, checkfirst=True)
            table.create(engine)
