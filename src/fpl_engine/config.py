"""Application settings.

Values can be overridden via environment variables prefixed with FPL_
(e.g. FPL_TEAM_ID=1234567) or a .env file in the project root.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPL_", env_file=PROJECT_ROOT / ".env")

    # The user's FPL entry (team) ID, from the Points-page URL. 0 = not linked yet.
    team_id: int = 0

    # Season the engine is currently operating on, e.g. "2025-26".
    season: str = "2025-26"

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
