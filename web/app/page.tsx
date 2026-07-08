"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Differential,
  DraftRow,
  fetchPlayers,
  fetchReport,
  fmtPct,
  fmtPrice,
  PlayerRow,
  POSITIONS,
  Report,
} from "@/lib/api";
import { Badge, Card, StatTile, Table } from "@/components/ui";
import { CaptainChart, TopXpChart } from "@/components/charts";

export default function Dashboard() {
  const [report, setReport] = useState<Report | null>(null);
  const [players, setPlayers] = useState<PlayerRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asOfGw, setAsOfGw] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const url = new URL(window.location.href);
    const gw = url.searchParams.get("gw");
    setAsOfGw(gw ? Number(gw) : undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([fetchReport(asOfGw), fetchPlayers(asOfGw)])
      .then(([r, p]) => {
        if (!cancelled) {
          setReport(r);
          setPlayers(p);
        }
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [asOfGw]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 w-full">
      <Header report={report} asOfGw={asOfGw} onReplay={setAsOfGw} />
      {loading && (
        <p className="text-sm text-secondary mt-8 animate-pulse">
          Building recommendations (models fit on demand — first load takes a minute)…
        </p>
      )}
      {error && (
        <div className="mt-8 rounded-xl border border-border-token bg-surface p-4 text-sm">
          <p className="font-medium text-serious">Engine not ready</p>
          <p className="text-secondary mt-1">{error}</p>
          <p className="text-muted mt-2 text-xs">
            Out of season? Try a replay: add <code>?gw=30</code> to the URL, or pick a
            gameweek above. In season, run <code>uv run fpl update</code> first.
          </p>
        </div>
      )}
      {report && players && !loading && (
        <ReportView report={report} players={players} />
      )}
    </main>
  );
}

function Header({
  report,
  asOfGw,
  onReplay,
}: {
  report: Report | null;
  asOfGw?: number;
  onReplay: (gw: number | undefined) => void;
}) {
  return (
    <header className="flex flex-wrap items-center gap-3">
      <h1 className="text-xl font-semibold">FPL engine</h1>
      {report && (
        <>
          <Badge>{report.season}</Badge>
          <Badge tone="good">GW{report.next_gw}</Badge>
          <Badge>{report.mode === "linked" ? "your team" : "draft mode"}</Badge>
          {report.as_of_replay && <Badge tone="serious">replay</Badge>}
        </>
      )}
      <div className="ml-auto flex items-center gap-2 text-xs text-secondary">
        <label htmlFor="gw">replay GW</label>
        <input
          id="gw"
          type="number"
          min={2}
          max={38}
          defaultValue={asOfGw}
          className="w-16 rounded-md border border-border-token bg-surface px-2 py-1 tabular-nums"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const v = (e.target as HTMLInputElement).valueAsNumber;
              onReplay(Number.isFinite(v) ? v : undefined);
            }
          }}
        />
      </div>
    </header>
  );
}

function ReportView({ report, players }: { report: Report; players: PlayerRow[] }) {
  const draftCost = useMemo(
    () => (report.draft ?? []).reduce((s, d) => s + d.value, 0),
    [report.draft],
  );

  return (
    <div className="mt-6 grid gap-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatTile label="Next deadline" value={`GW${report.next_gw}`} />
        <StatTile
          label="Best captain"
          value={report.captains[0]?.name ?? "—"}
          hint={
            report.captains[0]
              ? `${report.captains[0].xp.toFixed(1)} xP · P(10+) ${fmtPct(report.captains[0].p_haul)}`
              : undefined
          }
        />
        {report.plan ? (
          <StatTile
            label="Planned transfers"
            value={String(report.plan[0]?.transfers_in.length ?? 0)}
            hint={report.plan[0]?.hits ? `${report.plan[0].hits} hit(s)` : "no hits"}
          />
        ) : (
          <StatTile label="Draft squad cost" value={fmtPrice(draftCost)} />
        )}
        <StatTile
          label="Top differential"
          value={report.differentials[0]?.name ?? "—"}
          hint={
            report.differentials[0]
              ? `${report.differentials[0].ownership.toFixed(1)}% owned`
              : undefined
          }
        />
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card
          title="Captaincy"
          subtitle="expected points with simulated floor→ceiling (Monte Carlo)"
        >
          <CaptainChart captains={report.captains} />
          <Table
            head={["player", "xP", "p10", "p90", "P(10+)", "own %"]}
            rows={report.captains.map((c) => [
              c.name,
              c.xp.toFixed(2),
              c.p10.toFixed(0),
              c.p90.toFixed(0),
              fmtPct(c.p_haul),
              c.ownership.toFixed(1),
            ])}
          />
          {report.captains[0] && (
            <p className="text-xs text-muted mt-2">{report.captains[0].reasoning}</p>
          )}
        </Card>

        <Card title="Top expected points" subtitle={`GW${report.next_gw}, all players`}>
          <TopXpChart players={players} />
        </Card>
      </div>

      {report.plan && (
        <Card
          title="Transfer plan"
          subtitle="multi-gameweek MILP: transfers, hits and banking jointly optimized"
        >
          <Table
            head={["GW", "in", "out", "hits", "FTs", "captain", "XI xP"]}
            rows={report.plan.map((g) => [
              String(g.event),
              g.transfers_in.join(", ") || "—",
              g.transfers_out.join(", ") || "—",
              String(g.hits),
              String(g.free_transfers),
              g.captain ?? "—",
              g.xi_xp.toFixed(1),
            ])}
          />
        </Card>
      )}

      {report.draft && (
        <Card
          title="Draft squad"
          subtitle={`best legal 15 from scratch · ${fmtPrice(draftCost)}`}
        >
          <DraftTable rows={report.draft} />
        </Card>
      )}

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Chip timing" subtitle="marginal EV vs the no-chip plan">
          {report.chips.length ? (
            <Table
              head={["chip", "GW", "gain (pts)", "evidence"]}
              rows={report.chips.slice(0, 8).map((c) => [
                c.chip,
                String(c.event),
                <span key="g" className={c.gain_pts > 0 ? "text-good" : "text-muted"}>
                  {c.gain_pts >= 0 ? "+" : ""}
                  {c.gain_pts.toFixed(2)}
                </span>,
                <span key="d" className="text-xs text-secondary whitespace-normal">
                  {c.detail}
                </span>,
              ])}
            />
          ) : (
            <p className="text-sm text-muted">
              Chip evaluation needs a linked team (set FPL_TEAM_ID and run{" "}
              <code>fpl sync-team</code>).
            </p>
          )}
        </Card>

        <Card title="Differentials" subtitle="under 10% owned, likely starters">
          <DifferentialsTable rows={report.differentials} />
        </Card>
      </div>

      <Card title="Player expected points" subtitle="top 50 by xP">
        <Table
          head={["player", "pos", "price", "own %", "xP", "P(start)", "P(CS)", "fixtures"]}
          rows={players.slice(0, 50).map((p) => [
            p.name,
            POSITIONS[p.position] ?? "?",
            fmtPrice(p.value),
            p.ownership?.toFixed(1) ?? "—",
            p.xp.toFixed(2),
            p.p_start != null ? fmtPct(p.p_start) : "—",
            p.p_cs != null ? fmtPct(p.p_cs) : "—",
            String(p.n_fixtures ?? 1),
          ])}
        />
      </Card>
    </div>
  );
}

function DraftTable({ rows }: { rows: DraftRow[] }) {
  const order = { captain: 0, vice: 1, XI: 2, bench: 3 } as Record<string, number>;
  const sorted = [...rows].sort(
    (a, b) => order[a.role] - order[b.role] || b.xp - a.xp,
  );
  return (
    <Table
      head={["role", "player", "pos", "price", "xP"]}
      rows={sorted.map((d) => [
        d.role === "captain" ? <Badge key="r" tone="good">captain</Badge> : d.role,
        d.name,
        POSITIONS[d.position] ?? "?",
        fmtPrice(d.value),
        d.xp.toFixed(2),
      ])}
    />
  );
}

function DifferentialsTable({ rows }: { rows: Differential[] }) {
  return (
    <Table
      head={["player", "pos", "price", "own %", "xP"]}
      rows={rows.map((d) => [
        d.name,
        POSITIONS[d.position] ?? "?",
        fmtPrice(d.value),
        d.ownership.toFixed(1),
        d.xp.toFixed(2),
      ])}
    />
  );
}
