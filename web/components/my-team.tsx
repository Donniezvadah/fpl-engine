"use client";

import { ReactNode, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Dataset,
  Player,
  TeamEntry,
  fmtNum,
  fmtPrice,
  horizonXp,
} from "@/lib/api";
import { Badge, Card, StatTile, Table } from "@/components/ui";
import { Pitch, availability } from "@/components/pitch";
import {
  Lineup,
  Slot,
  TeamState,
  bestSwaps,
  pickLineup,
  planTransfers,
  validate,
  MAX_PER_CLUB,
} from "@/lib/squad";

const STORAGE_KEY = "fpl-engine:my-team:v1";
const POS = ["", "GK", "DEF", "MID", "FWD"];
const CHIP_LABEL: Record<string, string> = {
  wildcard: "Wildcard",
  bboost: "Bench Boost",
  freehit: "Free Hit",
  "3xc": "Triple Captain",
};

/** Player stub for squad members who have no row in the dataset (blank gameweek). */
function stub(e: TeamEntry, club = e.club_short): Player {
  return {
    id: e.id,
    name: e.name,
    position: e.position ?? 0,
    position_name: POS[e.position ?? 0] ?? "",
    team_id: -e.id,
    club,
    club_short: club,
    price: e.price ?? 0,
    ownership: null,
    xp: e.xp_next,
    p_start: null,
    p60: null,
    p_cs: null,
    e_goals: null,
    e_assists: null,
    xp_by_gw: {},
    fixtures_by_gw: {},
    opp_by_gw: {},
    status: "a",
    chance: null,
    news: "",
  };
}

function syncedState(data: Dataset): TeamState {
  const mt = data.my_team!;
  return {
    slots: mt.current.map((e) => ({ id: e.id, value: e.sell ?? e.price ?? 0 })),
    bank: mt.bank,
    freeTransfers: mt.free_transfers,
    captainId: mt.current.find((e) => e.role === "captain")?.id ?? null,
    syncedFrom: data.meta.generated_at,
    edited: false,
  };
}

function loadState(data: Dataset): TeamState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const saved = JSON.parse(raw) as TeamState;
      if (saved.slots?.length === 15 && saved.edited) return saved;
    }
  } catch {
    /* storage blocked or corrupt: fall through to the synced squad */
  }
  return syncedState(data);
}

function lineupOf(entries: TeamEntry[]): Lineup {
  const xi = entries.filter((e) => e.role !== "bench").sort((a, b) => a.order - b.order);
  const bench = entries.filter((e) => e.role === "bench").sort((a, b) => a.order - b.order);
  const n = (pos: number) => xi.filter((e) => e.position === pos).length;
  return {
    xi: xi.map((e) => e.id),
    bench: bench.map((e) => e.id),
    captain: entries.find((e) => e.role === "captain")?.id ?? null,
    vice: entries.find((e) => e.role === "vice")?.id ?? null,
    formation: `${n(2)}-${n(3)}-${n(4)}`,
    xp: 0,
  };
}

export function MyTeamView({ data }: { data: Dataset }) {
  const mt = data.my_team;
  if (!mt) {
    return (
      <Card title="My Team" subtitle="not linked yet">
        <p className="text-sm text-secondary">
          Put your numeric entry ID in <code>.env</code> as <code>FPL_TEAM_ID</code> (the number
          in your Points-page URL), then run <code>uv run fpl refresh</code>.
        </p>
      </Card>
    );
  }
  return <Linked data={data} />;
}

function Linked({ data }: { data: Dataset }) {
  const mt = data.my_team!;
  const gws = data.meta.horizon_gws;
  const gw = data.meta.next_gw;

  const [team, setTeam] = useState<TeamState>(() => loadState(data));
  const [n, setN] = useState(gws.length);
  const [editing, setEditing] = useState<number | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(team));
    } catch {
      /* private mode: edits simply won't persist */
    }
  }, [team]);

  const currentSquad = mt.current;
  const wildcardSquad = mt.wildcard?.squad;
  const byId = useMemo(() => {
    const m = new Map<number, Player>(data.players.map((p) => [p.id, p]));
    for (const e of [...currentSquad, ...(wildcardSquad ?? [])])
      if (!m.has(e.id)) m.set(e.id, stub(e));
    return m;
  }, [data.players, currentSquad, wildcardSquad]);

  const ids = useMemo(() => team.slots.map((s) => s.id), [team.slots]);
  const lineup = useMemo(
    () => pickLineup(ids, byId, gw, team.captainId),
    [ids, byId, gw, team.captainId],
  );
  const issues = useMemo(() => validate(ids, byId, team.bank), [ids, byId, team.bank]);
  const plan = useMemo(
    () => planTransfers(team, data.players, byId, gws, n),
    [team, data.players, byId, gws, n],
  );
  const swaps = useMemo(
    () => bestSwaps(team, data.players, byId, gws, n, 10),
    [team, data.players, byId, gws, n],
  );

  const syncedStale = team.edited && team.syncedFrom !== data.meta.generated_at;
  const last = mt.history[mt.history.length - 1];
  const wc = mt.wildcard;
  const wcAvailable = mt.chips_available.includes("wildcard");
  const squadValue = team.slots.reduce((t, s) => t + s.value, 0);

  const mutate = (patch: Partial<TeamState>) =>
    setTeam((t) => ({ ...t, ...patch, edited: true }));

  const swap = (outId: number, inP: Player) => {
    const slot = team.slots.find((s) => s.id === outId);
    if (!slot) return;
    mutate({
      slots: team.slots.map((s) => (s.id === outId ? { id: inP.id, value: inP.price } : s)),
      bank: team.bank + slot.value - inP.price,
      captainId: team.captainId === outId ? null : team.captainId,
    });
    setEditing(null);
    setQuery("");
  };

  const applyWildcard = () => {
    if (!wc) return;
    // Kept players stay at their sell value; new ones are bought at today's price.
    const held = new Map(team.slots.map((s) => [s.id, s.value]));
    const slots: Slot[] = wc.squad.map((e) => ({ id: e.id, value: held.get(e.id) ?? e.price ?? 0 }));
    const sold = team.slots.filter((s) => !wc.squad.some((e) => e.id === s.id));
    const bought = wc.squad.filter((e) => !held.has(e.id));
    mutate({
      slots,
      bank:
        team.bank +
        sold.reduce((t, s) => t + s.value, 0) -
        bought.reduce((t, e) => t + (e.price ?? 0), 0),
      captainId: wc.squad.find((e) => e.role === "captain")?.id ?? null,
    });
  };

  const reset = () => setTeam(syncedState(data));

  // ----- action list -------------------------------------------------------
  const actions: ReactNode[] = [];
  if (wcAvailable && wc && wc.gain > 8) {
    actions.push(
      <>
        <strong>Play your Wildcard in GW{wc.event}</strong> — rebuilding the whole squad is worth{" "}
        <strong className="text-good">+{fmtNum(wc.gain, 1)} xP</strong> over your best normal
        plan, changing {wc.moves.length} of your 15 players.
      </>,
    );
  }
  for (const id of ids) {
    const p = byId.get(id);
    const av = availability(p);
    if (p && av !== "ok")
      actions.push(
        <>
          <strong>{p.name}</strong> is {av === "out" ? "unavailable" : "a doubt"}
          {p.chance != null ? ` (${p.chance}% to play)` : ""}
          {p.news ? ` — ${p.news}` : ""}.{" "}
          {wcAvailable ? "The Wildcard team replaces him." : "Replace him before the deadline."}
        </>,
      );
  }
  if (lineup.captain != null) {
    const c = byId.get(lineup.captain);
    actions.push(
      <>
        Captain <strong>{c?.name}</strong> ({fmtNum(c?.xp_by_gw[String(gw)] ?? null, 1)} xP
        {c?.opp_by_gw[String(gw)] ? ` vs ${c.opp_by_gw[String(gw)]}` : ""}), vice{" "}
        <strong>{byId.get(lineup.vice ?? -1)?.name ?? "—"}</strong>.
      </>,
    );
  }
  if (!wcAvailable || !wc) {
    for (const m of plan)
      actions.push(
        <>
          Transfer {m.step}: <strong>{m.out.name}</strong> → <strong>{m.in.name}</strong>{" "}
          (+{fmtNum(m.delta, 1)} team xP over {n} GWs{m.hit ? `, before a −${m.hit} hit` : ", free"}).
        </>,
      );
  }

  const candidates = useMemo(() => {
    if (editing == null) return [];
    const out = byId.get(editing);
    const slot = team.slots.find((s) => s.id === editing);
    if (!out || !slot) return [];
    const owned = new Set(ids);
    const clubs = new Map<number, number>();
    for (const i of ids) {
      const p = byId.get(i);
      if (p) clubs.set(p.team_id, (clubs.get(p.team_id) ?? 0) + 1);
    }
    const q = query.trim().toLowerCase();
    return data.players
      .filter((p) => p.position === out.position && !owned.has(p.id))
      .filter((p) => p.price <= slot.value + team.bank)
      .filter(
        (p) => p.team_id === out.team_id || (clubs.get(p.team_id) ?? 0) < MAX_PER_CLUB,
      )
      .filter(
        (p) =>
          q === "" || p.name.toLowerCase().includes(q) || p.club_short.toLowerCase().includes(q),
      )
      .map((p) => ({ p, xp: horizonXp(p, gws, n) }))
      .sort((a, b) => b.xp - a.xp)
      .slice(0, 12);
  }, [editing, byId, team.slots, team.bank, ids, data.players, gws, n, query]);

  const editingPlayer = editing != null ? byId.get(editing) : undefined;
  const wcLineup = wc ? lineupOf(wc.squad) : null;
  const wcTags: Record<number, string> = {};
  if (wc) for (const m of wc.moves) wcTags[m.in.id] = "NEW";

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <StatTile
          label="Overall rank"
          value={last?.overall_rank ? last.overall_rank.toLocaleString() : "—"}
          hint={last ? `after GW${last.event}` : undefined}
        />
        <StatTile label="Points" value={String(last?.overall_points ?? "—")} />
        <StatTile label="Bank" value={fmtPrice(team.bank)} hint={`squad ${fmtPrice(squadValue)}`} />
        <StatTile label="Free transfers" value={String(team.freeTransfers)} />
        <div className="rounded-xl border border-border-token bg-surface px-4 py-3">
          <div className="text-xs text-secondary">Chips</div>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {mt.chips_available.map((c) => (
              <Badge key={c} tone="good">
                {CHIP_LABEL[c] ?? c}
              </Badge>
            ))}
            {mt.chips_used.map((c) => (
              <Badge key={c.chip + c.event}>
                {CHIP_LABEL[c.chip] ?? c.chip} · used GW{c.event}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      <Card
        title={`What to do before the GW${gw} deadline`}
        subtitle="generated from the models for your synced squad"
      >
        <ol className="grid gap-2 text-sm list-decimal pl-5">
          {actions.map((a, i) => (
            <li key={i} className="text-secondary [&_strong]:text-foreground">
              {a}
            </li>
          ))}
        </ol>
      </Card>

      {syncedStale && (
        <div className="rounded-xl border border-border-token bg-surface p-3 text-sm flex flex-wrap items-center gap-3">
          <span className="text-secondary">
            Fresh data has arrived since you edited this team.
          </span>
          <button className="underline" onClick={reset}>
            Re-sync to my FPL squad
          </button>
          <button className="underline" onClick={() => setTeam((t) => ({ ...t, syncedFrom: data.meta.generated_at }))}>
            Keep my edits
          </button>
        </div>
      )}

      <div className="grid xl:grid-cols-2 gap-4">
        <Card
          title="Your team"
          subtitle={`formation ${lineup.formation} · XI + captain ${fmtNum(lineup.xp, 1)} xP in GW${gw} · tap a player to swap or captain`}
        >
          <Pitch
            lineup={lineup}
            byId={byId}
            gw={gw}
            onSelect={(id) => setEditing(editing === id ? null : id)}
            selectedId={editing}
          />
          {issues.length > 0 && (
            <ul className="mt-3 text-xs text-serious list-disc pl-5">
              {issues.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          )}
          <div className="flex flex-wrap gap-3 mt-3 text-xs">
            <button className="underline" onClick={reset} disabled={!team.edited}>
              Reset to synced squad
            </button>
            {team.edited && <Badge>edited · saved in this browser</Badge>}
          </div>
        </Card>

        {wc && wcLineup ? (
          <Card
            title={`Wildcard team (GW${wc.event})`}
            subtitle={`+${fmtNum(wc.gain, 1)} xP vs your best normal plan · sell ${fmtPrice(
              wc.moves.reduce((t, m) => t + (m.out.sell ?? m.out.price ?? 0), 0),
            )}, buy ${fmtPrice(wc.moves.reduce((t, m) => t + (m.in.price ?? 0), 0))}`}
          >
            <Pitch lineup={wcLineup} byId={byId} gw={wc.event} tags={wcTags} />
            <div className="flex flex-wrap gap-3 mt-3 items-center">
              <button
                className="rounded-lg bg-foreground text-background text-sm font-medium px-3 py-1.5"
                onClick={applyWildcard}
              >
                Use this as my team
              </button>
              {!wcAvailable && <Badge tone="serious">Wildcard already used</Badge>}
            </div>
          </Card>
        ) : (
          <Card title="Wildcard team" subtitle="not available">
            <p className="text-sm text-muted">
              No Wildcard to play, so the transfer suggestions below are the way to improve.
            </p>
          </Card>
        )}
      </div>

      {editingPlayer && (
        <Card
          title={`Replace ${editingPlayer.name}`}
          subtitle={`${POS[editingPlayer.position]} · ${editingPlayer.club_short} · sells for ${fmtPrice(
            team.slots.find((s) => s.id === editingPlayer.id)?.value ?? null,
          )} · bank ${fmtPrice(team.bank)}`}
        >
          <div className="flex flex-wrap gap-2 mb-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name or club"
              className="rounded-lg border border-border-token bg-background px-2.5 py-1.5 text-sm"
            />
            {lineup.xi.includes(editingPlayer.id) && (
              <button
                className="rounded-lg border border-border-token px-2.5 py-1.5 text-sm"
                onClick={() => mutate({ captainId: editingPlayer.id })}
              >
                Make captain
              </button>
            )}
            <button
              className="rounded-lg border border-border-token px-2.5 py-1.5 text-sm"
              onClick={() => setEditing(null)}
            >
              Close
            </button>
          </div>
          <Table
            head={["", "player", "club", "price", `xP (${n} GW)`, "vs current"]}
            rows={candidates.map(({ p, xp }) => [
              <button
                key="b"
                className="rounded-md bg-foreground text-background px-2 py-0.5 text-xs"
                onClick={() => swap(editingPlayer.id, p)}
              >
                Swap in
              </button>,
              p.name,
              p.club_short,
              fmtPrice(p.price),
              fmtNum(xp, 1),
              <span key="d" className={xp - horizonXp(editingPlayer, gws, n) > 0 ? "text-good" : "text-muted"}>
                {fmtNum(xp - horizonXp(editingPlayer, gws, n), 1)}
              </span>,
            ])}
          />
        </Card>
      )}

      {wc && (
        <Card
          title="Wildcard transfers"
          subtitle={`${wc.moves.length} players out, ${wc.moves.length} in — expected points are summed over the next ${gws.length} gameweeks; the gain that counts is the whole-team figure above`}
        >
          <Table
            head={["out", "", "in", "pos", "price", "next 3 fixtures", "xP out", "xP in"]}
            rows={[...wc.moves]
              .sort((a, b) => b.in.xp_horizon - a.in.xp_horizon)
              .map((m) => [
              <span key="o" className="text-serious">
                {m.out.name}
                <span className="text-muted"> {m.out.club_short}</span>
              </span>,
              "→",
              <span key="i" className="font-medium text-good">
                {m.in.name}
                <span className="text-muted"> {m.in.club_short}</span>
              </span>,
              POS[m.in.position ?? 0],
              `${fmtPrice(m.out.sell ?? m.out.price)} → ${fmtPrice(m.in.price)}`,
              <span key="f" className="text-xs text-secondary">
                {gws
                  .slice(0, 3)
                  .map((g) => byId.get(m.in.id)?.opp_by_gw[String(g)] ?? "—")
                  .join(" · ")}
              </span>,
              <span key="xo" className="text-muted">
                {fmtNum(m.out.xp_horizon, 1)}
              </span>,
              <span key="xi" className="font-medium">
                {fmtNum(m.in.xp_horizon, 1)}
              </span>,
            ])}
          />
          {wc.follow_on.some((f) => f.transfers_in.length > 0) && (
            <div className="mt-4">
              <div className="text-xs font-medium text-secondary mb-1">
                After the Wildcard, the planner expects
              </div>
              <Table
                head={["GW", "in", "out", "hit", "captain"]}
                rows={wc.follow_on
                  .filter((f) => f.transfers_in.length > 0)
                  .map((f) => [
                    String(f.event),
                    f.transfers_in.join(", "),
                    f.transfers_out.join(", "),
                    f.hits ? `−${f.hits * 4}` : "free",
                    f.captain,
                  ])}
              />
            </div>
          )}
        </Card>
      )}

      <Card
        title="Transfer suggestions for your team"
        subtitle="recomputed from the squad above — edit the team and these update"
      >
        <div className="flex items-center gap-2 mb-3 text-xs text-secondary">
          Horizon
          {gws.map((_, i) => (
            <button
              key={i}
              onClick={() => setN(i + 1)}
              className={`rounded-full px-2 py-0.5 border ${
                n === i + 1 ? "bg-foreground text-background border-foreground" : "border-border-token"
              }`}
            >
              {i + 1} GW
            </button>
          ))}
        </div>
        <div className="text-xs font-medium text-secondary mb-1">
          Best sequence ({team.freeTransfers} free transfer{team.freeTransfers === 1 ? "" : "s"},
          −4 per extra)
        </div>
        {plan.length ? (
          <Table
            head={["#", "out", "in", "price", "team gain", "hit", "net", "running"]}
            rows={plan.map((m) => [
              String(m.step),
              m.out.name,
              m.in.name,
              `${fmtPrice(m.outValue)} → ${fmtPrice(m.in.price)}`,
              fmtNum(m.delta, 1),
              m.hit ? `−${m.hit}` : "free",
              <span key="n" className="text-good">+{fmtNum(m.net, 1)}</span>,
              fmtNum(m.cumulativeNet, 1),
            ])}
          />
        ) : (
          <p className="text-sm text-muted">No transfer beats its cost over this horizon.</p>
        )}
        <div className="text-xs font-medium text-secondary mt-4 mb-1">Best single swaps</div>
        <Table
          head={["", "out", "in", "price", `team gain (${n} GW)`]}
          rows={swaps.map((s) => [
            <button
              key="a"
              className="rounded-md bg-foreground text-background px-2 py-0.5 text-xs"
              onClick={() => swap(s.out.id, s.in)}
            >
              Apply
            </button>,
            `${s.out.name} (${s.out.club_short})`,
            `${s.in.name} (${s.in.club_short})`,
            `${fmtPrice(s.outValue)} → ${fmtPrice(s.in.price)}`,
            <span key="g" className="text-good">+{fmtNum(s.delta, 1)}</span>,
          ])}
        />
      </Card>

      <Card title="Rank history" subtitle="overall rank after each gameweek (higher on the chart is better)">
        <RankChart history={mt.history} />
      </Card>
    </div>
  );
}

function RankChart({ history }: { history: { event: number; overall_rank: number | null }[] }) {
  const data = history
    .filter((h) => h.overall_rank != null)
    .map((h) => ({ gw: `GW${h.event}`, rank: h.overall_rank as number }));
  const css = (name: string, fallback: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const ink = css("--text-secondary", "#52514e");
  const grid = css("--border", "#e4e4e0");
  const line = css("--series-1", "#2a78d6");
  const fmt = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${Math.round(v / 1e3)}k`);
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ left: 0, right: 12, top: 8 }}>
        <CartesianGrid stroke={grid} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="gw" tick={{ fill: ink, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis
          reversed
          domain={["dataMin - 200000", "dataMax + 200000"]}
          tickFormatter={fmt}
          tick={{ fill: ink, fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : String(v))} />
        <Line type="monotone" dataKey="rank" stroke={line} strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
