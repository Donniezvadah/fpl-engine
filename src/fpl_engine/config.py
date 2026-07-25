"""Application settings.

Values can be overridden via environment variables prefixed with FPL_
(e.g. FPL_TEAM_ID=1234567) or a .env file in the project root.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def current_season(today: date | None = None) -> str:
    """Season label for a date, e.g. "2026-27".

    FPL seasons run August-May and the new game opens in July, so July onwards
    belongs to the season starting that calendar year. Deriving this instead of
    hard-coding it means the engine rolls over on its own each summer.
    """
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPL_", env_file=PROJECT_ROOT / ".env")

    # The user's FPL entry (team) ID, from the Points-page URL. 0 = not linked yet.
    team_id: int = 0

    # Season the engine operates on; derived from today's date unless FPL_SEASON is set.
    season: str = Field(default_factory=current_season)

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models_store"

    api_base_url: str = "https://fantasy.premierleague.com/api"
    # vaastav/Fantasy-Premier-League raw GitHub content root.
    history_base_url: str = (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    )

    request_timeout_s: float = 30.0
    requests_per_second: float = 4.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "fpl.sqlite3"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"


settings = Settings()
