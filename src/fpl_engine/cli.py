"""fpl — command-line interface to the engine."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="FPL decision-support engine", no_args_is_help=True)
console = Console()


@app.command()
def update(
    player_history: bool = typer.Option(
        True, help="Also fetch per-player fixture history (slower, ~4 req/s)."
    ),
) -> None:
    """Fetch current-season data from the official FPL API into SQLite."""
    from .ingest.pipeline import update_current_season

    counts = update_current_season(with_player_history=player_history)
    console.print({"updated": counts})


@app.command()
def backfill(
    seasons: list[str] = typer.Argument(
        ..., help='Historical seasons to load, e.g. 2022-23 2023-24 2024-25'
    ),
) -> None:
    """One-time load of past seasons from the vaastav GitHub archive."""
    from .ingest.pipeline import backfill_history

    counts = backfill_history(seasons)
    console.print({"rows": counts})


@app.command()
def sync_team() -> None:
    """Pull your own FPL entry (requires FPL_TEAM_ID in .env)."""
    from .ingest.pipeline import sync_my_team

    console.print(sync_my_team())


@app.command()
def backtest(
    season: str = typer.Option(
        None, help="Season to replay. Default: newest season with results."
    ),
    train_seasons: list[str] = typer.Option(
        [],
        "--train-season",
        help="Earlier seasons to train on. Default: every earlier season in the DB.",
    ),
    start_gw: int = typer.Option(7),
    end_gw: int = typer.Option(38),
    refit_every: int = typer.Option(4, help="Refit models every N gameweeks."),
    carryover: float = typer.Option(
        None,
        help="Prior-season carryover strength (0 disables). Default from features module.",
    ),
) -> None:
    """Walk-forward backtest: the empirical gate for the models."""
    from .backtest.harness import run_backtest
    from .models.features import CARRYOVER_PSEUDOCOUNT

    result = run_backtest(
        season=season,
        train_seasons=list(train_seasons) or None,
        start_gw=start_gw,
        end_gw=end_gw,
        refit_every=refit_every,
        carryover=CARRYOVER_PSEUDOCOUNT if carryover is None else carryover,
    )
    result.print_report()


@app.command()
def recommend(
    horizon: int = typer.Option(6, help="Gameweeks to plan ahead."),
    sims: int = typer.Option(10_000, help="Monte Carlo simulations."),
    train_seasons: list[str] = typer.Option(
        [],
        "--train-season",
        help="Seasons to train on. Default: every earlier season in the DB.",
    ),
    as_of_gw: int = typer.Option(
        None, help="Replay a past deadline (models see only earlier GWs)."
    ),
) -> None:
    """Weekly recommendations: transfers, captain, chips, differentials."""
    from .recommend.render import print_report
    from .recommend.weekly import build_weekly_report

    report = build_weekly_report(
        horizon=horizon,
        n_sims=sims,
        train_seasons=list(train_seasons) or None,
        as_of_gw=as_of_gw,
    )
    print_report(report)


@app.command()
def api(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the FastAPI app (backend for the web dashboard)."""
    import uvicorn

    uvicorn.run("fpl_engine.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
