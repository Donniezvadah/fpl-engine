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
    from rich.table import Table

    from .ingest.pipeline import sync_my_team

    result = sync_my_team()
    console.print(
        f"[bold]{result['team_name']}[/bold] ({result['manager']}) — "
        f"entry {result['team_id']}"
    )
    console.print(f"gameweeks synced: {result['gameweeks_synced']}")
    if result["note"]:
        console.print(f"[yellow]{result['note']}[/yellow]")

    if result["past_seasons"]:
        t = Table(title="Past seasons")
        t.add_column("season")
        t.add_column("points", justify="right")
        t.add_column("overall rank", justify="right")
        for s in result["past_seasons"]:
            t.add_row(s["season"], str(s["points"]), f"{s['rank']:,}")
        console.print(t)


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
def evaluate(
    players: list[str] = typer.Argument(
        None,
        help='Your 15 players, e.g. Haaland "Raya (ARS)" 411. Club hints break ties.',
    ),
    file: str = typer.Option(
        None, "--file", "-f", help="Read player names from a file, one per line."
    ),
    horizon: int = typer.Option(
        1, help="Gameweeks to score over (1 = the next one alone)."
    ),
    bank: int = typer.Option(0, help="Money left over, in tenths (5 == £0.5m)."),
) -> None:
    """Score a squad you already own against the model."""
    from pathlib import Path

    from rich.table import Table

    from .recommend.evaluate import build_pool, evaluate_squad, resolve_players
    from .recommend.weekly import build_weekly_report
    from .models.features import load_players
    from .rules.constants import Position, rules_for

    names = list(players or [])
    if file:
        names += [
            line.strip()
            for line in Path(file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not names:
        raise typer.BadParameter("give 15 player names, or --file with one per line")

    console.print(f"[dim]fitting models over a {horizon}-gameweek horizon…[/dim]")
    report = build_weekly_report(horizon=max(horizon, 2), n_sims=10_000)
    pool = build_pool(report, load_players(report.season), horizon=horizon)

    resolutions = resolve_players(names, pool)
    unresolved = [r for r in resolutions if not r.ok]
    if unresolved:
        console.print("[red]Could not identify these names:[/red]")
        for r in unresolved:
            hint = (
                "  candidates: " + "; ".join(r.candidates)
                if r.candidates
                else "  no match found"
            )
            console.print(f"  [bold]{r.query}[/bold]\n{hint}")
        raise typer.Exit(1)

    ids = [r.player_id for r in resolutions if r.player_id is not None]
    rules = rules_for(report.season)
    ev = evaluate_squad(ids, pool, rules, bank=bank, horizon_gws=report.horizon_gws[:horizon])

    span = (
        f"GW{report.next_gw}"
        if horizon == 1
        else f"GW{report.horizon_gws[0]}–GW{report.horizon_gws[horizon - 1]}"
    )
    console.print(
        f"\n[bold]Your squad — {span}[/bold]  ·  cost £{ev.cost / 10:.1f}m"
    )
    if ev.legality:
        console.print("[red]Not a legal squad:[/red] " + "; ".join(ev.legality))

    t = Table()
    for c in ("role", "player", "club", "pos", "price", "xP"):
        t.add_column(c, justify="right" if c in ("price", "xP") else "left")
    for r in ev.squad.itertuples():
        t.add_row(
            r.role,
            r.web_name,
            r.club_short,
            Position(int(r.position)).name,
            f"£{int(r.now_price) / 10:.1f}m",
            f"{r.xp:.2f}",
        )
    console.print(t)

    console.print(
        f"starting XI: [bold]{ev.xi_xp:.1f}[/bold] xP  ·  captain "
        f"[bold]{ev.captain_name}[/bold] (doubled) → total "
        f"[bold]{ev.total_xp:.1f}[/bold] xP  ·  bench {ev.bench_xp:.1f}"
    )
    if ev.optimal_xp == ev.optimal_xp:  # not NaN
        gap = ev.optimal_xp - ev.total_xp
        console.print(
            f"model's best squad at the same price: {ev.optimal_xp:.1f} xP "
            f"([{'red' if gap > 0 else 'green'}]{gap:+.1f}[/])"
        )

    if not ev.upgrades.empty:
        u = Table(title="Upgrades worth considering (same position, affordable)")
        for c in ("out", "in", "club", "spend", "gain xP"):
            u.add_column(c, justify="right" if c in ("spend", "gain xP") else "left")
        for r in ev.upgrades.itertuples():
            u.add_row(
                f"{r.out_name} (£{r.out_price / 10:.1f}m)",
                f"{r.in_name} (£{r.in_price / 10:.1f}m)",
                r.in_club,
                f"£{r.spend / 10:+.1f}m",
                f"{r.gain:+.2f}",
            )
        console.print(u)


@app.command()
def export(
    out: str = typer.Option(
        "web/public/data", help="Directory to write the static site's data.json into."
    ),
    horizon: int = typer.Option(6, help="Gameweeks of expected points to publish."),
    sims: int = typer.Option(10_000, help="Monte Carlo simulations."),
    as_of_gw: int = typer.Option(None, help="Replay a past deadline instead of the next one."),
) -> None:
    """Bake the current recommendation into static JSON for the web dashboard."""
    from pathlib import Path

    from .recommend.export import export_site_data

    result = export_site_data(
        Path(out), horizon=horizon, n_sims=sims, as_of_gw=as_of_gw
    )
    console.print(result)


@app.command()
def refresh(
    out: str = typer.Option("web/public/data", help="Where the dashboard reads data.json."),
    horizon: int = typer.Option(6, help="Gameweeks of expected points to publish."),
    sims: int = typer.Option(10_000, help="Monte Carlo simulations."),
) -> None:
    """One step for the weekly routine: update data, sync your team, re-export the site."""
    from pathlib import Path

    from .ingest.pipeline import sync_my_team, update_current_season
    from .recommend.export import export_site_data

    console.print(update_current_season())
    try:
        res = sync_my_team()
        console.print(f"synced {res['team_name']} — {res['gameweeks_synced']} gameweeks")
    except Exception as exc:  # unlinked or pre-season: still export draft mode
        console.print(f"[yellow]team sync skipped: {exc}[/yellow]")
    console.print(export_site_data(Path(out), horizon=horizon, n_sims=sims))


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
