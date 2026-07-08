"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CaptainOption, PlayerRow } from "@/lib/api";

const css = (name: string) =>
  typeof window === "undefined"
    ? "#2a78d6"
    : getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function ChartTooltip({
  active,
  payload,
  label,
  fmt,
}: {
  active?: boolean;
  payload?: { value: number }[];
  label?: string;
  fmt: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border-token bg-surface px-2.5 py-1.5 text-xs shadow-md">
      <div className="font-medium">{label}</div>
      <div className="text-secondary tabular-nums">{fmt(payload[0].value)}</div>
    </div>
  );
}

/** Horizontal bar chart of the top players by expected points (single series). */
export function TopXpChart({ players }: { players: PlayerRow[] }) {
  const data = players.slice(0, 15).map((p) => ({
    name: p.name,
    xp: Number(p.xp.toFixed(2)),
  }));
  const series = css("--series-1");
  const ink = css("--text-secondary");
  const grid = css("--border");
  return (
    <ResponsiveContainer width="100%" height={data.length * 26 + 40}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid horizontal={false} stroke={grid} strokeDasharray="2 4" />
        <XAxis
          type="number"
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={110}
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "transparent" }}
          content={<ChartTooltip fmt={(v) => `${v.toFixed(2)} xP`} />}
        />
        <Bar dataKey="xp" fill={series} barSize={12} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Captaincy: expected points with simulated floor→ceiling range per candidate. */
export function CaptainChart({ captains }: { captains: CaptainOption[] }) {
  const data = captains.map((c) => ({
    name: c.name,
    xp: Number(c.xp.toFixed(2)),
    // asymmetric error bar: [below, above] the mean
    range: [
      Math.max(0, Number((c.xp - c.p10).toFixed(2))),
      Math.max(0, Number((c.p90 - c.xp).toFixed(2))),
    ],
  }));
  const series = css("--series-1");
  const ink = css("--text-secondary");
  const grid = css("--border");
  return (
    <ResponsiveContainer width="100%" height={data.length * 34 + 40}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid horizontal={false} stroke={grid} strokeDasharray="2 4" />
        <XAxis
          type="number"
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          label={{
            value: "points (bar = expected, whisker = simulated p10–p90)",
            position: "insideBottom",
            offset: -2,
            fill: ink,
            fontSize: 10,
          }}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={110}
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "transparent" }}
          content={<ChartTooltip fmt={(v) => `${v.toFixed(2)} xP`} />}
        />
        <Bar dataKey="xp" fill={series} barSize={12} radius={[0, 4, 4, 0]}>
          <ErrorBar dataKey="range" width={5} strokeWidth={1.5} stroke={ink} direction="x" />
          {data.map((_, i) => (
            <Cell key={i} fill={series} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
