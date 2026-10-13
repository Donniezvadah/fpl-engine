"use client";

import { Dataset, ReviewPlayer, fmtNum } from "@/lib/api";
import { Badge, Card, StatTile, Table } from "@/components/ui";
import { CalibrationChart, WhatIfChart } from "@/components/charts";

const POS = ["", "GK", "DEF", "MID", "FWD"];
const ROLE_ORDER: Record<string, number> = { captain: 0, vice: 1, XI: 2, bench: 3 };

export function ReviewView({ data }: { data: Dataset }) {
  const r = data.review;
  if (!r) {
    return (
      <Card title="Last week" subtitle="nothing to review yet">
        <p className="text-sm text-secondary">
          The review appears once a gameweek has been played and your team is linked.
        </p>
      </Card>
    );
  }
  const vsAvg = r.average != null ? r.points - r.average : null;
  const moved = r.rank_before && r.rank_after ? r.rank_before - r.rank_after : null;
  const rows: ReviewPlayer[] = [...r.players].sort(
    (a, b) =>
      (ROLE_ORDER[a.role] ?? 9) - (ROLE_ORDER[b.role] ?? 9) ||
      (a.position ?? 0) - (b.position ?? 0),
  );

  return (
    <div className="grid grid-cols-1 gap-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatTile
          label={`GW${r.event} points`}
          value={String(r.points)}
          hint={
            vsAvg != null
              ? `${vsAvg >= 0 ? "+" : ""}${vsAvg} vs average ${r.average}${r.hit ? ` · hit −${r.hit}` : ""}`
              : undefined
          }
        />
        <StatTile
          label="Overall rank"
          value={r.rank_after ? r.rank_after.toLocaleString() : "—"}
          hint={
            moved != null
              ? `${moved >= 0 ? "▲" : "▼"} ${Math.abs(moved).toLocaleString()} places`
              : undefined
          }
        />
        <StatTile
          label="XI + captain: expected vs actual"
          value={`${fmtNum(r.xi_xp, 1)} → ${r.gross_points}`}
          hint="before hits"
        />
        <StatTile
          label="Model error this week"
          value={`${fmtNum(r.model.mae, 2)} pts`}
          hint={`${r.model.n} players who played`}
        />
      </div>

      <Card
        title={`How GW${r.event} went`}
        subtitle={r.provisional ? "bonus points not finalised yet, figures may move" : "final"}
      >
        <ul className="grid gap-2 text-sm">
          {r.comments.map((c, i) => (
            <li key={i} className={i === 0 ? "font-medium" : "text-secondary"}>
              {c}
            </li>
          ))}
        </ul>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title={`Your squad in GW${r.event}`} subtitle="expected points were set before the deadline">
          <Table
            head={["", "player", "pos", "mins", "xP", "pts", "vs xP"]}
            rows={rows.map((p) => {
              const diff = p.points - p.xp;
              return [
                p.role === "captain" ? (
                  <Badge key="r" tone="good">C</Badge>
                ) : p.role === "bench" ? (
                  <span key="r" className="text-muted text-xs">bench</span>
                ) : p.role === "vice" ? (
                  <Badge key="r">V</Badge>
                ) : (
                  ""
                ),
                <span key="n">
                  {p.name} <span className="text-muted text-xs">{p.club_short}</span>
                  {p.subbed_in && <span className="text-good text-xs"> ↑ auto-sub</span>}
                  {p.subbed_out && <span className="text-serious text-xs"> ↓</span>}
                </span>,
                POS[p.position ?? 0],
                String(p.minutes),
                fmtNum(p.xp, 1),
                <strong key="p">{p.points}</strong>,
                <span
                  key="d"
                  className={diff >= 2 ? "text-good" : diff <= -2 ? "text-serious" : "text-muted"}
                >
                  {diff >= 0 ? "+" : ""}
                  {fmtNum(diff, 1)}
                </span>,
              ];
            })}
          />
        </Card>

        <div className="grid grid-cols-1 gap-4 content-start">
          <Card title="What it would have scored" subtitle="same week, same results, different choices">
            <WhatIfChart items={r.what_if} />
          </Card>

          <Card
            title="Model scorecard"
            subtitle={`${
              r.model.source === "snapshot"
                ? "predictions saved before the deadline"
                : "replayed with data from before the deadline (assumes everyone was fit)"
            }`}
          >
            <div className="grid grid-cols-3 gap-3 mb-3 text-center">
              <div>
                <div className="text-xl font-semibold tabular-nums">{fmtNum(r.model.mae, 2)}</div>
                <div className="text-xs text-muted">average error</div>
              </div>
              <div>
                <div className="text-xl font-semibold tabular-nums">{fmtNum(r.model.spearman, 2)}</div>
                <div className="text-xs text-muted">rank correlation</div>
              </div>
              <div>
                <div className="text-xl font-semibold tabular-nums">
                  {fmtNum(r.model.top10_actual, 1)}
                  <span className="text-muted text-sm"> vs {fmtNum(r.model.league_actual, 1)}</span>
                </div>
                <div className="text-xs text-muted">top-10 rated vs all</div>
              </div>
            </div>
            <CalibrationChart bins={r.model.calibration} />
            <div className="flex gap-4 text-xs text-secondary mt-1">
              <span>
                <span className="inline-block h-2 w-2 rounded-sm bg-muted mr-1" />
                predicted
              </span>
              <span>
                <span className="inline-block h-2 w-2 rounded-sm bg-series-1 mr-1" />
                actual
              </span>
              <span className="text-muted">five equal groups of players, rated low to high</span>
            </div>
            <div className="text-xs font-medium text-secondary mt-4 mb-1">Top scorers of the week</div>
            <Table
              head={["player", "pts", "predicted"]}
              rows={r.model.top_scorers.map((t) => [t.name, String(t.points), fmtNum(t.xp, 1)])}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}
