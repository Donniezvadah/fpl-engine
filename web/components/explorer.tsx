"use client";

import { useMemo, useState } from "react";
import {
  Dataset,
  Player,
  fmtNum,
  fmtPct,
  fmtPrice,
  horizonFixtures,
  horizonXp,
} from "@/lib/api";

type SortKey = "xp" | "price" | "ownership" | "value" | "p_start" | "name";

const POSITION_FILTERS = [
  { label: "All", value: 0 },
  { label: "GK", value: 1 },
  { label: "DEF", value: 2 },
  { label: "MID", value: 3 },
  { label: "FWD", value: 4 },
];

/** Interactive player explorer: the reason the site is worth visiting rather
 *  than reading a static report. Everything runs client-side on baked JSON. */
export function PlayerExplorer({ data }: { data: Dataset }) {
  const gws = data.meta.horizon_gws;
  const [horizon, setHorizon] = useState(Math.min(3, gws.length));
  const [position, setPosition] = useState(0);
  const [query, setQuery] = useState("");
  const [maxPrice, setMaxPrice] = useState(155);
  const [maxOwnership, setMaxOwnership] = useState(100);
  const [sort, setSort] = useState<SortKey>("xp");
  const [starterOnly, setStarterOnly] = useState(true);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const scored = data.players
      .filter((p) => (position === 0 ? true : p.position === position))
      .filter((p) => p.price <= maxPrice)
      .filter((p) => (p.ownership ?? 0) <= maxOwnership)
      .filter((p) => (starterOnly ? (p.p_start ?? 0) >= 0.5 : true))
      .filter(
        (p) =>
          q === "" ||
          p.name.toLowerCase().includes(q) ||
          p.club.toLowerCase().includes(q) ||
          p.club_short.toLowerCase().includes(q),
      )
      .map((p) => {
        const total = horizonXp(p, gws, horizon);
        return {
          p,
          total,
          perMillion: p.price > 0 ? total / (p.price / 10) : 0,
          fixtures: horizonFixtures(p, gws, horizon),
        };
      });

    scored.sort((a, b) => {
      switch (sort) {
        case "price":
          return b.p.price - a.p.price;
        case "ownership":
          return (b.p.ownership ?? 0) - (a.p.ownership ?? 0);
        case "value":
          return b.perMillion - a.perMillion;
        case "p_start":
          return (b.p.p_start ?? 0) - (a.p.p_start ?? 0);
        case "name":
          return a.p.name.localeCompare(b.p.name);
        default:
          return b.total - a.total;
      }
    });
    return scored;
  }, [data.players, gws, horizon, position, query, maxPrice, maxOwnership, sort, starterOnly]);

  const maxTotal = rows.length ? Math.max(...rows.map((r) => r.total)) : 1;

  return (
    <section className="rounded-xl border border-border-token bg-surface p-4">
      <header className="mb-3">
        <h2 className="text-sm font-semibold tracking-wide">Player explorer</h2>
        <p className="text-xs text-muted mt-0.5">
          Expected points over the next {horizon} gameweek{horizon > 1 ? "s" : ""} (GW
          {gws.slice(0, horizon).join(", GW")}). {rows.length} of{" "}
          {data.players.length} players shown.
        </p>
      </header>

      <div className="flex flex-wrap gap-3 items-end mb-4">
        <label className="text-xs text-secondary">
          <span className="block mb-1">Search</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="player or club"
            className="w-44 rounded-md border border-border-token bg-surface-2 px-2 py-1 text-sm"
          />
        </label>

        <div className="text-xs text-secondary">
          <span className="block mb-1">Position</span>
          <div className="flex rounded-md overflow-hidden border border-border-token">
            {POSITION_FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => setPosition(f.value)}
                className={`px-2.5 py-1 text-sm ${
                  position === f.value
                    ? "bg-series-1 text-white"
                    : "bg-surface-2 hover:bg-surface"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        <label className="text-xs text-secondary">
          <span className="block mb-1">Horizon: {horizon} GW</span>
          <input
            type="range"
            min={1}
            max={gws.length}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="w-32 accent-series-1"
          />
        </label>

        <label className="text-xs text-secondary">
          <span className="block mb-1">Max price: {fmtPrice(maxPrice)}</span>
          <input
            type="range"
            min={38}
            max={155}
            step={1}
            value={maxPrice}
            onChange={(e) => setMaxPrice(Number(e.target.value))}
            className="w-32 accent-series-1"
          />
        </label>

        <label className="text-xs text-secondary">
          <span className="block mb-1">Max owned: {maxOwnership}%</span>
          <input
            type="range"
            min={0}
            max={100}
            value={maxOwnership}
            onChange={(e) => setMaxOwnership(Number(e.target.value))}
            className="w-32 accent-series-1"
          />
        </label>

        <label className="text-xs text-secondary flex items-center gap-1.5 pb-1">
          <input
            type="checkbox"
            checked={starterOnly}
            onChange={(e) => setStarterOnly(e.target.checked)}
            className="accent-series-1"
          />
          likely starters only
        </label>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted border-b border-border-token">
              <SortHeader label="Player" k="name" sort={sort} set={setSort} />
              <th className="py-1.5 pr-3 font-medium">Club</th>
              <th className="py-1.5 pr-3 font-medium">Pos</th>
              <SortHeader label="Price" k="price" sort={sort} set={setSort} right />
              <SortHeader label="Own %" k="ownership" sort={sort} set={setSort} right />
              <SortHeader label="P(start)" k="p_start" sort={sort} set={setSort} right />
              <th className="py-1.5 pr-3 font-medium text-right">Fix</th>
              <SortHeader label="xP/£m" k="value" sort={sort} set={setSort} right />
              <SortHeader label="xP" k="xp" sort={sort} set={setSort} right />
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 100).map(({ p, total, perMillion, fixtures }) => (
              <tr
                key={p.id}
                className="border-b border-border-token/60 last:border-0 hover:bg-surface-2/60"
              >
                <td className="py-1.5 pr-3 whitespace-nowrap font-medium">{p.name}</td>
                <td className="py-1.5 pr-3 whitespace-nowrap text-secondary">
                  {p.club_short}
                </td>
                <td className="py-1.5 pr-3 whitespace-nowrap text-secondary">
                  {p.position_name}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums">
                  {fmtPrice(p.price)}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-secondary">
                  {p.ownership == null ? "—" : p.ownership.toFixed(1)}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-secondary">
                  {fmtPct(p.p_start)}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-secondary">
                  {fixtures}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums">
                  {fmtNum(perMillion)}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums">
                  <span className="inline-flex items-center gap-2 justify-end w-full">
                    <span
                      aria-hidden
                      className="h-2 rounded-sm bg-series-1"
                      style={{
                        width: `${Math.max(2, (total / maxTotal) * 56)}px`,
                      }}
                    />
                    <span className="font-medium w-10 text-right">
                      {fmtNum(total, 1)}
                    </span>
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <p className="text-sm text-muted py-6 text-center">
            No players match these filters.
          </p>
        )}
        {rows.length > 100 && (
          <p className="text-xs text-muted pt-2">
            Showing the top 100 of {rows.length} matches.
          </p>
        )}
      </div>
    </section>
  );
}

function SortHeader({
  label,
  k,
  sort,
  set,
  right,
}: {
  label: string;
  k: SortKey;
  sort: SortKey;
  set: (k: SortKey) => void;
  right?: boolean;
}) {
  const active = sort === k;
  return (
    <th className={`py-1.5 pr-3 font-medium ${right ? "text-right" : ""}`}>
      <button
        onClick={() => set(k)}
        className={`hover:text-foreground ${active ? "text-foreground" : ""}`}
      >
        {label}
        {active ? " ↓" : ""}
      </button>
    </th>
  );
}
