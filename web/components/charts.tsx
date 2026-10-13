"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CaptainOption, Player, Review } from "@/lib/api";

const css = (name: string, fallback: string) =>
  typeof window === "undefined"
    ? fallback
    : getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
      fallback;

function ChartTooltip({
  active,
  payload,
  label,
  suffix = "",
}: {
  active?: boolean;
  payload?: { value: number; name?: string }[];
  label?: string | number;
  suffix?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border-token bg-surface px-2.5 py-1.5 text-xs shadow-md">
      <div className="font-medium">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="text-secondary tabular-nums">
          {p.value?.toFixed(2)}
          {suffix}
        </div>
      ))}
    </div>
  );
}

/** Captaincy: expected points with the simulated floor-to-ceiling range. */
export function CaptainChart({ captains }: { captains: CaptainOption[] }) {
  const data = captains.map((c) => ({
    name: c.name,
    xp: Number(c.xp.toFixed(2)),
    range: [
      Math.max(0, Number((c.xp - c.p10).toFixed(2))),
      Math.max(0, Number((c.p90 - c.xp).toFixed(2))),
    ],
  }));
  const series = css("--series-1", "#2a78d6");
  const ink = css("--text-secondary", "#52514e");
  const grid = css("--border", "#e4e4e0");
  return (
    <ResponsiveContainer width="100%" height={data.length * 34 + 44}>
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
          width={104}
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip cursor={{ fill: "transparent" }} content={<ChartTooltip suffix=" xP" />} />
        <Bar dataKey="xp" fill={series} barSize={12} radius={[0, 4, 4, 0]}>
          <ErrorBar dataKey="range" width={5} strokeWidth={1.5} stroke={ink} direction="x" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Expected points per gameweek for the selected players — the fixture-swing view. */
export function HorizonChart({
  players,
  gws,
}: {
  players: Player[];
  gws: number[];
}) {
  const colors = [
    css("--series-1", "#2a78d6"),
    css("--series-2", "#1baf7a"),
    css("--series-3", "#eda100"),
  ];
  const ink = css("--text-secondary", "#52514e");
  const grid = css("--border", "#e4e4e0");
  const data = gws.map((gw) => {
    const row: Record<string, number | string> = { gw: `GW${gw}` };
    players.forEach((p) => {
      row[p.name] = Number((p.xp_by_gw[String(gw)] ?? 0).toFixed(2));
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={230}>
      <LineChart data={data} margin={{ left: 0, right: 12, top: 8 }}>
        <CartesianGrid stroke={grid} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="gw"
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={28}
        />
        <Tooltip content={<ChartTooltip suffix=" xP" />} />
        {players.map((p, i) => (
          <Line
            key={p.id}
            type="monotone"
            dataKey={p.name}
            stroke={colors[i % colors.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Points each alternative would have scored this week, as horizontal bars. */
export function WhatIfChart({ items }: { items: { label: string; points: number }[] }) {
  const ink = css("--text-secondary", "#52514e");
  const grid = css("--border", "#e4e4e0");
  const series = css("--series-1", "#2a78d6");
  const muted = css("--text-muted", "#8a887f");
  const actual = items[items.length - 1]?.points;
  const data = items.map((i) => ({ ...i, label: i.label.replace(" (before hits)", "") }));
  return (
    <ResponsiveContainer width="100%" height={data.length * 42 + 40}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 28 }}>
        <CartesianGrid horizontal={false} stroke={grid} strokeDasharray="2 4" />
        <XAxis type="number" tick={{ fill: ink, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis
          type="category"
          dataKey="label"
          width={190}
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip cursor={{ fill: "transparent" }} content={<ChartTooltip suffix=" pts" />} />
        <Bar dataKey="points" barSize={14} radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.points === actual && i === data.length - 1 ? series : muted} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Predicted vs actual average points for five equal groups of players, rated low to high. */
export function CalibrationChart({ bins }: { bins: Review["model"]["calibration"] }) {
  const ink = css("--text-secondary", "#52514e");
  const grid = css("--border", "#e4e4e0");
  const data = bins.map((b) => ({
    group: ["lowest", "low", "middle", "high", "highest"][Number(b.bin) - 1] ?? b.bin,
    Predicted: b.xp,
    Actual: b.actual,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 0, right: 12, top: 8 }}>
        <CartesianGrid stroke={grid} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="group" tick={{ fill: ink, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: ink, fontSize: 11 }} axisLine={false} tickLine={false} width={28} />
        <Tooltip cursor={{ fill: "transparent" }} />
        <Bar dataKey="Predicted" fill={css("--text-muted", "#8a887f")} barSize={16} radius={[4, 4, 0, 0]} />
        <Bar dataKey="Actual" fill={css("--series-1", "#2a78d6")} barSize={16} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
