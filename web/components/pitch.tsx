"use client";

import { Player, fmtNum, fmtPrice } from "@/lib/api";
import { Lineup } from "@/lib/squad";

type Availability = "ok" | "doubt" | "out";

export function availability(p: Player | undefined): Availability {
  if (!p) return "ok";
  if (p.status === "i" || p.status === "s" || p.status === "u" || p.status === "n") return "out";
  if (p.chance != null && p.chance <= 25) return "out";
  if (p.status === "d" || (p.chance != null && p.chance < 100)) return "doubt";
  return "ok";
}

const DOT: Record<Availability, string> = {
  ok: "",
  doubt: "bg-series-3",
  out: "bg-serious",
};

function PlayerCard({
  p,
  gw,
  role,
  tag,
  selected,
  onSelect,
}: {
  p: Player;
  gw: number;
  role?: "C" | "V";
  tag?: string;
  selected: boolean;
  onSelect?: (id: number) => void;
}) {
  const av = availability(p);
  const opp = p.opp_by_gw[String(gw)] ?? "—";
  const xp = p.xp_by_gw[String(gw)] ?? null;
  return (
    <button
      type="button"
      onClick={() => onSelect?.(p.id)}
      disabled={!onSelect}
      title={p.news || `${p.name} · ${p.club}`}
      className={`relative w-[74px] sm:w-[96px] rounded-lg bg-white/95 text-neutral-900 shadow-sm text-center px-1 py-1.5 transition ${
        onSelect ? "hover:-translate-y-0.5 cursor-pointer" : "cursor-default"
      } ${selected ? "ring-2 ring-series-1" : ""}`}
    >
      {role && (
        <span className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-neutral-900 text-white text-[10px] font-bold flex items-center justify-center">
          {role}
        </span>
      )}
      {av !== "ok" && (
        <span
          aria-label={av === "out" ? "unavailable" : "doubtful"}
          className={`absolute -top-1 -left-1 h-3 w-3 rounded-full border border-white ${DOT[av]}`}
        />
      )}
      {tag && (
        <span className="absolute -top-2 left-1/2 -translate-x-1/2 rounded-full bg-series-2 text-white text-[9px] font-semibold px-1.5 leading-4">
          {tag}
        </span>
      )}
      <div className="text-[10px] font-semibold text-neutral-500 leading-none">{p.club_short}</div>
      <div className="text-xs font-semibold truncate mt-0.5">{p.name}</div>
      <div className="text-[10px] text-neutral-500 tabular-nums">{opp}</div>
      <div className="text-xs font-bold tabular-nums mt-0.5">{fmtNum(xp, 1)} xP</div>
      <div className="text-[10px] text-neutral-500 tabular-nums">{fmtPrice(p.price)}</div>
    </button>
  );
}

/** Pitch view of a squad: GK at the back, forwards up front, bench underneath. */
export function Pitch({
  lineup,
  byId,
  gw,
  onSelect,
  selectedId,
  tags,
}: {
  lineup: Lineup;
  byId: Map<number, Player>;
  gw: number;
  onSelect?: (id: number) => void;
  selectedId?: number | null;
  tags?: Record<number, string>;
}) {
  const row = (pos: number) =>
    lineup.xi
      .map((id) => byId.get(id))
      .filter((p): p is Player => !!p && p.position === pos);

  const render = (p: Player) => (
    <PlayerCard
      key={p.id}
      p={p}
      gw={gw}
      role={p.id === lineup.captain ? "C" : p.id === lineup.vice ? "V" : undefined}
      tag={tags?.[p.id]}
      selected={selectedId === p.id}
      onSelect={onSelect}
    />
  );

  return (
    <div>
      <div
        className="rounded-xl px-2 py-4 flex flex-col gap-5"
        style={{
          background:
            "repeating-linear-gradient(180deg, #2f8f4e 0 44px, #2a8447 44px 88px)",
        }}
      >
        {[4, 3, 2, 1].map((pos) => (
          <div key={pos} className="flex justify-center gap-2 sm:gap-3">
            {row(pos).map(render)}
          </div>
        ))}
      </div>
      <div className="mt-2 rounded-xl bg-surface-2 px-2 py-3">
        <div className="text-[11px] text-muted mb-2 px-1">Bench (auto-sub order)</div>
        <div className="flex justify-center gap-2 sm:gap-3">
          {lineup.bench
            .map((id) => byId.get(id))
            .filter((p): p is Player => !!p)
            .map(render)}
        </div>
      </div>
    </div>
  );
}
