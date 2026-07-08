"""Official FPL API client.

Public, unauthenticated endpoints only. Responses are optionally snapshotted to
disk (data/snapshots/<date>/) so backtests can reconstruct "what was known when"
and re-runs don't hammer the API.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import settings

USER_AGENT = "fpl-engine/0.1 (personal analytics tool)"


class FplApiClient:
    def __init__(self, snapshot: bool = True):
        self._client = httpx.Client(
            base_url=settings.api_base_url,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.request_timeout_s,
            follow_redirects=True,
        )
        self._min_interval = 1.0 / settings.requests_per_second
        self._last_request_at = 0.0
        self._snapshot = snapshot

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FplApiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _get(self, path: str) -> Any:
        self._throttle()
        resp = self._client.get(path)
        resp.raise_for_status()
        data = resp.json()
        if self._snapshot:
            self._save_snapshot(path, data)
        return data

    def _save_snapshot(self, path: str, data: Any) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        safe = path.strip("/").replace("/", "__") or "root"
        out = settings.snapshots_dir / day / f"{safe}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, separators=(",", ":")))

    # ---- endpoints -------------------------------------------------------

    def bootstrap_static(self) -> dict:
        """Players, teams, events (gameweeks), game settings, chip config."""
        return self._get("/bootstrap-static/")

    def fixtures(self) -> list[dict]:
        return self._get("/fixtures/")

    def element_summary(self, player_id: int) -> dict:
        """One player's fixture-by-fixture history (this season) + past seasons."""
        return self._get(f"/element-summary/{player_id}/")

    def event_live(self, gw: int) -> dict:
        """All players' stats for one gameweek."""
        return self._get(f"/event/{gw}/live/")

    def entry(self, entry_id: int) -> dict:
        return self._get(f"/entry/{entry_id}/")

    def entry_history(self, entry_id: int) -> dict:
        """Per-GW points/rank/value/bank + chips used for an entry."""
        return self._get(f"/entry/{entry_id}/history/")

    def entry_picks(self, entry_id: int, gw: int) -> dict:
        return self._get(f"/entry/{entry_id}/event/{gw}/picks/")

    def entry_transfers(self, entry_id: int) -> list[dict]:
        return self._get(f"/entry/{entry_id}/transfers/")


def load_latest_snapshot(path: str) -> Any | None:
    """Read the most recent on-disk snapshot for an endpoint path, if any."""
    safe = path.strip("/").replace("/", "__") or "root"
    if not settings.snapshots_dir.exists():
        return None
    days = sorted((d for d in settings.snapshots_dir.iterdir() if d.is_dir()), reverse=True)
    for day in days:
        f: Path = day / f"{safe}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None
