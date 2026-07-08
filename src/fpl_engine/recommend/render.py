"""Console rendering of the weekly report (rich)."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .weekly import WeeklyReport

console = Console()


def _fmt_price(tenths: int | float) -> str:
    return f"£{tenths / 10:.1f}m"


def print_report(r: WeeklyReport) -> None:
    console.print(
        Panel.fit(
            f"[bold]GW{r.next_gw} — {r.season}[/bold]  mode: {r.mode}",
            title="FPL engine",
        )
    )

    if r.plan is not None:
        first = r.plan.gameweeks[0]
        t = Table(title=f"Transfer plan (next {len(r.plan.gameweeks)} GWs)")
        t.add_column("GW")
        t.add_column("in")
        t.add_column("out")
        t.add_column("hits", justify="right")
        t.add_column("FTs", justify="right")
        t.add_column("XI xP", justify="right")
        for g in r.plan.gameweeks:
            t.add_row(
                str(g.event),
                ", ".join(r.names.get(i, str(i)) for i in g.transfers_in) or "—",
                ", ".join(r.names.get(i, str(i)) for i in g.transfers_out) or "—",
                str(g.hits),
                str(g.free_transfers_before),
                f"{g.xp:.1f}",
            )
        console.print(t)
        console.print(
            f"GW{first.event} captain: [bold]{r.names.get(first.captain_id)}[/bold] | "
            f"bench: {', '.join(r.names.get(i, str(i)) for i in first.bench_ids)}"
        )

    if r.draft is not None:
        t = Table(title=f"Draft squad (cost {_fmt_price(r.draft.cost)})")
        t.add_column("role")
        t.add_column("player")
        t.add_column("pos", justify="right")
        t.add_column("price", justify="right")
        t.add_column("xP", justify="right")
        pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        for _, row in r.draft.summary().iterrows():
            t.add_row(
                row["role"],
                r.names.get(int(row["player_id"]), str(row["player_id"])),
                pos_names.get(int(row["position"]), "?"),
                _fmt_price(row["value"]),
                f"{row['xp']:.2f}",
            )
        console.print(t)

    if r.captains:
        t = Table(title="Captaincy (simulated)")
        for col in ("player", "xP", "floor p10", "ceiling p90", "P(10+)", "own%"):
            t.add_column(col, justify="right")
        for c in r.captains:
            t.add_row(
                c.name, f"{c.xp:.2f}", f"{c.p10:.0f}", f"{c.p90:.0f}",
                f"{c.p_haul:.0%}", f"{c.ownership:.1f}",
            )
        console.print(t)
        console.print(f"[dim]{r.captains[0].reasoning}[/dim]")

    if r.chips:
        t = Table(title="Chip timing (marginal EV, points)")
        t.add_column("chip")
        t.add_column("GW", justify="right")
        t.add_column("gain", justify="right")
        t.add_column("evidence")
        for e in r.chips[:8]:
            t.add_row(e.chip, str(e.event), f"{e.gain_pts:+.2f}", e.detail)
        console.print(t)

    if not r.differentials.empty:
        t = Table(title="Differentials (<10% owned, likely starters)")
        t.add_column("player")
        t.add_column("own%", justify="right")
        t.add_column("price", justify="right")
        t.add_column("xP", justify="right")
        for _, row in r.differentials.iterrows():
            t.add_row(
                str(row["name"]), f"{row['ownership']:.1f}",
                _fmt_price(row["value"]), f"{row['xp']:.2f}",
            )
        console.print(t)

    if not r.price_watch.empty:
        t = Table(title="Price watch (tonight-ish)")
        t.add_column("player")
        t.add_column("P(rise)", justify="right")
        t.add_column("P(fall)", justify="right")
        for _, row in r.price_watch.iterrows():
            t.add_row(str(row["name"]), f"{row['p_rise']:.0%}", f"{row['p_fall']:.0%}")
        console.print(t)
