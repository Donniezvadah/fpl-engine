"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Dataset,
  fmtNum,
  fmtPct,
  fmtPrice,
  loadDataset,
  POSITIONS,
} from "@/lib/api";
import { Badge, Card, StatTile, Table } from "@/components/ui";
import { CaptainChart, HorizonChart } from "@/components/charts";
import { PlayerExplorer } from "@/components/explorer";
import { MyTeamView } from "@/components/my-team";

export default function Dashboard() {
  const [data, setData] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"team" | "overview" | null>(null);

  useEffect(() => {
    loadDataset()
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <main className="mx-auto max-w-6xl px-4 py-10 w-full">
        <h1 className="text-xl font-semibold">FPL engine</h1>
        <div className="mt-6 rounded-xl border border-border-token bg-surface p-4 text-sm">
          <p className="font-medium text-serious">No data available</p>
          <p className="text-secondary mt-1">{error}</p>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="mx-auto max-w-6xl px-4 py-10 w-full">
        <h1 className="text-xl font-semibold">FPL engine</h1>
        <p className="text-sm text-secondary mt-4 animate-pulse">Loading…</p>
      </main>
    );
  }

  const view = tab ?? (data.my_team ? "team" : "overview");

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 w-full">
      <Header data={data} />
      {data.my_team && (
        <nav className="mt-4 flex gap-1 border-b border-border-token" aria-label="Views">
          {(
            [
              ["team", "My Team"],
              ["overview", "Overview & players"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              aria-current={view === key}
              className={`px-3 py-2 text-sm -mb-px border-b-2 ${
                view === key
                  ? "border-foreground font-medium"
                  : "border-transparent text-secondary hover:text-foreground"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      )}
      <div className="mt-6">
        {view === "team" ? <MyTeamView data={data} /> : <Content data={data} />}
      </div>
      <footer className="text-xs text-muted mt-8 pb-4">
        Generated {new Date(data.meta.generated_at).toLocaleString()} ·{" "}
        {data.meta.player_count} players · expected points carry uncertainty and
        pre-season estimates lean on priors — re-check near the deadline.
      </footer>
    </main>
  );
}

function Header({ data }: { data: Dataset }) {
  return (
    <header className="flex flex-wrap items-center gap-2">
      <h1 className="text-xl font-semibold mr-1">FPL engine</h1>
      <Badge>{data.meta.season}</Badge>
      <Badge tone="good">GW{data.meta.next_gw}</Badge>
      <Badge>{data.meta.mode === "linked" ? "your team" : "draft mode"}</Badge>
    </header>
  );
}

function Content({ data }: { data: Dataset }) {
  const gws = data.meta.horizon_gws;
  const topThree = useMemo(() => data.players.slice(0, 3), [data.players]);
  const squadCost = data.squad?.cost ?? null;

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatTile label="Next deadline" value={`GW${data.meta.next_gw}`} />
        <StatTile
          label="Best captain"
          value={data.captains[0]?.name ?? "—"}
          hint={
            data.captains[0]
              ? `${fmtNum(data.captains[0].xp, 1)} xP · P(10+) ${fmtPct(
                  data.captains[0].p_haul,
                )}`
              : undefined
          }
        />
        <StatTile
          label={data.squad?.kind === "linked" ? "Your squad" : "Draft squad cost"}
          value={squadCost != null ? fmtPrice(squadCost) : "—"}
        />
        <StatTile
          label="Top differential"
          value={data.differentials[0]?.name ?? "—"}
          hint={
            data.differentials[0]
              ? `${data.differentials[0].ownership.toFixed(1)}% owned`
              : undefined
          }
        />
      </div>

      <PlayerExplorer data={data} />

      <div className="grid lg:grid-cols-2 gap-4">
        <Card
          title="Captaincy"
          subtitle="expected points with simulated floor→ceiling (Monte Carlo)"
        >
          <CaptainChart captains={data.captains} />
          <Table
            head={["player", "xP", "p10", "p90", "P(10+)", "own %"]}
            rows={data.captains.map((c) => [
              c.name,
              fmtNum(c.xp),
              fmtNum(c.p10, 0),
              fmtNum(c.p90, 0),
              fmtPct(c.p_haul),
              c.ownership.toFixed(1),
            ])}
          />
          {data.captains[0] && (
            <p className="text-xs text-muted mt-2">{data.captains[0].reasoning}</p>
          )}
        </Card>

        <Card
          title="Fixture swing"
          subtitle={`expected points per gameweek — top 3 by GW${data.meta.next_gw}`}
        >
          <HorizonChart players={topThree} gws={gws} />
          <div className="flex flex-wrap gap-3 mt-2">
            {topThree.map((p, i) => (
              <span key={p.id} className="inline-flex items-center gap-1.5 text-xs">
                <span
                  aria-hidden
                  className="inline-block h-2 w-2 rounded-full"
                  style={{
                    background: ["var(--series-1)", "var(--series-2)", "var(--series-3)"][i],
                  }}
                />
                {p.name}
              </span>
            ))}
          </div>
        </Card>
      </div>

      {data.squad && (
        <Card
          title={data.squad.kind === "linked" ? "Your squad" : "Draft squad"}
          subtitle={
            squadCost != null
              ? `best legal 15 from scratch · ${fmtPrice(squadCost)}`
              : "current picks"
          }
        >
          <Table
            head={["role", "player", "pos", "price", "xP"]}
            rows={[...data.squad.players]
              .sort(
                (a, b) =>
                  ({ captain: 0, vice: 1, XI: 2, bench: 3 }[a.role] ?? 9) -
                    ({ captain: 0, vice: 1, XI: 2, bench: 3 }[b.role] ?? 9) ||
                  (b.xp ?? 0) - (a.xp ?? 0),
              )
              .map((s) => [
                s.role === "captain" ? <Badge key="r" tone="good">captain</Badge> : s.role,
                s.name,
                s.position_name || "—",
                fmtPrice(s.price),
                fmtNum(s.xp),
              ])}
          />
        </Card>
      )}

      {data.plan && (
        <Card title="Transfer plan" subtitle="multi-gameweek MILP: transfers, hits and banking">
          <Table
            head={["GW", "in", "out", "hits", "FTs", "captain", "XI xP"]}
            rows={data.plan.map((g) => [
              String(g.event),
              g.transfers_in.join(", ") || "—",
              g.transfers_out.join(", ") || "—",
              String(g.hits),
              String(g.free_transfers),
              g.captain || "—",
              fmtNum(g.xi_xp, 1),
            ])}
          />
        </Card>
      )}

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Chip timing" subtitle="marginal EV vs the no-chip plan">
          {data.chips.length ? (
            <Table
              head={["chip", "GW", "gain (pts)", "evidence"]}
              rows={data.chips.slice(0, 8).map((c) => [
                c.chip,
                String(c.event),
                <span key="g" className={c.gain > 0 ? "text-good" : "text-muted"}>
                  {c.gain >= 0 ? "+" : ""}
                  {fmtNum(c.gain)}
                </span>,
                <span key="d" className="text-xs text-secondary whitespace-normal">
                  {c.detail}
                </span>,
              ])}
            />
          ) : (
            <p className="text-sm text-muted">
              Chip evaluation needs a linked team with a published squad — available
              once the first deadline has passed.
            </p>
          )}
        </Card>

        <Card title="Differentials" subtitle="under 10% owned, likely starters">
          <Table
            head={["player", "pos", "price", "own %", "xP"]}
            rows={data.differentials.map((d) => [
              d.name,
              POSITIONS[d.position] ?? "?",
              fmtPrice(d.value),
              d.ownership.toFixed(1),
              fmtNum(d.xp),
            ])}
          />
        </Card>
      </div>
    </div>
  );
}
