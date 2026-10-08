/** Client-side squad logic for the My Team page: rule checks, best-XI
 *  selection and transfer suggestions. Everything here is a pure function of
 *  the baked dataset, so edits to your team re-rank instantly with no backend. */

import { Player, horizonXp } from "@/lib/api";

export const SQUAD_SIZE = 15;
export const POSITION_COUNTS: Record<number, number> = { 1: 2, 2: 5, 3: 5, 4: 3 };
export const MAX_PER_CLUB = 3;
export const HIT_COST = 4;

/** A squad slot: the player and what you would get for selling him (tenths). */
export interface Slot {
  id: number;
  value: number;
}

export interface TeamState {
  slots: Slot[];
  bank: number;
  freeTransfers: number;
  captainId: number | null;
  /** data.meta.generated_at of the dataset the state was synced from. */
  syncedFrom: string;
  /** True once the user has changed anything; unedited state follows fresh data. */
  edited: boolean;
}

export interface Lineup {
  xi: number[];
  bench: number[];
  captain: number | null;
  vice: number | null;
  formation: string;
  xp: number;
}

export function validate(
  ids: number[],
  byId: Map<number, Player>,
  bank: number,
): string[] {
  const issues: string[] = [];
  if (ids.length !== SQUAD_SIZE) issues.push(`Squad has ${ids.length} players, needs ${SQUAD_SIZE}.`);
  const perPos: Record<number, number> = {};
  const perClub: Record<number, number> = {};
  for (const id of ids) {
    const p = byId.get(id);
    if (!p) continue;
    perPos[p.position] = (perPos[p.position] ?? 0) + 1;
    perClub[p.team_id] = (perClub[p.team_id] ?? 0) + 1;
  }
  for (const [pos, need] of Object.entries(POSITION_COUNTS)) {
    if ((perPos[Number(pos)] ?? 0) !== need)
      issues.push(`Needs ${need} of position ${pos}, has ${perPos[Number(pos)] ?? 0}.`);
  }
  for (const [club, n] of Object.entries(perClub)) {
    if (n > MAX_PER_CLUB) {
      const name = [...byId.values()].find((p) => p.team_id === Number(club))?.club ?? club;
      issues.push(`More than ${MAX_PER_CLUB} players from ${name}.`);
    }
  }
  if (bank < 0) issues.push(`Over budget by £${(-bank / 10).toFixed(1)}m.`);
  return issues;
}

const FORMATIONS: [number, number, number][] = [];
for (let d = 3; d <= 5; d++)
  for (let m = 2; m <= 5; m++)
    for (let f = 1; f <= 3; f++) if (d + m + f === 10) FORMATIONS.push([d, m, f]);

/** Best legal XI + ordered bench by expected points in `gw`, with captain and vice. */
export function pickLineup(
  ids: number[],
  byId: Map<number, Player>,
  gw: number,
  forcedCaptain: number | null = null,
): Lineup {
  const xp = (id: number) => byId.get(id)?.xp_by_gw[String(gw)] ?? 0;
  const by = (pos: number) =>
    ids
      .filter((i) => byId.get(i)?.position === pos)
      .sort((a, b) => xp(b) - xp(a));
  const gks = by(1);
  const defs = by(2);
  const mids = by(3);
  const fwds = by(4);
  const sum = (arr: number[], n: number) =>
    arr.slice(0, n).reduce((t, i) => t + xp(i), 0);

  let best: { f: [number, number, number]; score: number } | null = null;
  for (const f of FORMATIONS) {
    if (defs.length < f[0] || mids.length < f[1] || fwds.length < f[2]) continue;
    const score = sum(defs, f[0]) + sum(mids, f[1]) + sum(fwds, f[2]);
    if (!best || score > best.score) best = { f, score };
  }
  if (!best || gks.length === 0)
    return { xi: [], bench: [], captain: null, vice: null, formation: "—", xp: 0 };

  const [d, m, f] = best.f;
  const xi = [gks[0], ...defs.slice(0, d), ...mids.slice(0, m), ...fwds.slice(0, f)];
  const outfieldBench = [...defs.slice(d), ...mids.slice(m), ...fwds.slice(f)].sort(
    (a, b) => xp(b) - xp(a),
  );
  const bench = [...gks.slice(1), ...outfieldBench].sort((a, b) => {
    // Backup keeper always sits first on the bench by FPL convention.
    const ga = byId.get(a)?.position === 1 ? 1 : 0;
    const gb = byId.get(b)?.position === 1 ? 1 : 0;
    return gb - ga || xp(b) - xp(a);
  });
  const ranked = [...xi].sort((a, b) => xp(b) - xp(a));
  const captain =
    forcedCaptain != null && xi.includes(forcedCaptain) ? forcedCaptain : ranked[0] ?? null;
  const vice = ranked.find((i) => i !== captain) ?? null;
  return {
    xi,
    bench,
    captain,
    vice,
    formation: `${d}-${m}-${f}`,
    xp: sum(xi, xi.length) + (captain != null ? xp(captain) : 0),
  };
}

export interface Swap {
  out: Player;
  in: Player;
  outValue: number;
  delta: number; // gain in team expected points (best XI + captain) over the horizon
  cost: number; // change in bank (tenths, negative = spends money)
}

const BENCH_WEIGHT = 0.1; // same weight the engine's planner gives the bench

/** Expected points of a squad over the first `n` gameweeks: each week's best XI
 *  plus captain, with a small credit for bench depth. Bench-only upgrades
 *  therefore score near zero, as they should. */
export function teamScore(
  ids: number[],
  byId: Map<number, Player>,
  gws: number[],
  n: number,
): number {
  let total = 0;
  for (const gw of gws.slice(0, n)) {
    const l = pickLineup(ids, byId, gw);
    const xp = (i: number) => byId.get(i)?.xp_by_gw[String(gw)] ?? 0;
    total += l.xp + BENCH_WEIGHT * l.bench.reduce((t, i) => t + xp(i), 0);
  }
  return total;
}

const CANDIDATES_PER_POSITION = 30;

/** All legal single swaps from the current squad, best team gain first. */
export function bestSwaps(
  state: { slots: Slot[]; bank: number },
  all: Player[],
  byId: Map<number, Player>,
  gws: number[],
  n: number,
  limit = 12,
): Swap[] {
  const ids = state.slots.map((s) => s.id);
  const owned = new Set(ids);
  const clubCount = new Map<number, number>();
  for (const id of ids) {
    const p = byId.get(id);
    if (p) clubCount.set(p.team_id, (clubCount.get(p.team_id) ?? 0) + 1);
  }
  const base = teamScore(ids, byId, gws, n);
  const pool = new Map<number, Player[]>();
  for (const pos of [1, 2, 3, 4]) {
    pool.set(
      pos,
      all
        .filter((p) => p.position === pos && !owned.has(p.id) && (p.p_start ?? 0) >= 0.4)
        .sort((a, b) => horizonXp(b, gws, n) - horizonXp(a, gws, n))
        .slice(0, CANDIDATES_PER_POSITION),
    );
  }
  const swaps: Swap[] = [];
  for (const slot of state.slots) {
    const out = byId.get(slot.id);
    if (!out) continue;
    const budget = slot.value + state.bank;
    for (const cand of pool.get(out.position) ?? []) {
      if (cand.price > budget) continue;
      if (cand.team_id !== out.team_id && (clubCount.get(cand.team_id) ?? 0) >= MAX_PER_CLUB)
        continue;
      const next = ids.map((i) => (i === slot.id ? cand.id : i));
      const delta = teamScore(next, byId, gws, n) - base;
      if (delta <= 0.05) continue;
      swaps.push({ out, in: cand, outValue: slot.value, delta, cost: slot.value - cand.price });
    }
  }
  swaps.sort((a, b) => b.delta - a.delta);
  // Keep the best few per outgoing player so the list isn't one player repeated.
  const seen = new Map<number, number>();
  const result: Swap[] = [];
  for (const sw of swaps) {
    const c = seen.get(sw.out.id) ?? 0;
    if (c >= 2) continue;
    seen.set(sw.out.id, c + 1);
    result.push(sw);
    if (result.length >= limit) break;
  }
  return result;
}

export interface PlannedMove extends Swap {
  step: number;
  hit: number; // points cost of this move (0 while free transfers last)
  net: number; // delta minus hit
  cumulativeNet: number;
}

/** Greedy sequence of up to `max` transfers: each step takes the best remaining
 *  swap after the previous ones, and charges a 4-point hit once the free
 *  transfers run out. Stops when the next move is not worth its hit. */
export function planTransfers(
  state: { slots: Slot[]; bank: number; freeTransfers: number },
  all: Player[],
  byId: Map<number, Player>,
  gws: number[],
  n: number,
  max = 4,
): PlannedMove[] {
  let slots = state.slots.map((s) => ({ ...s }));
  let bank = state.bank;
  const moves: PlannedMove[] = [];
  let cumulative = 0;
  for (let step = 1; step <= max; step++) {
    const [top] = bestSwaps({ slots, bank }, all, byId, gws, n, 1);
    if (!top) break;
    const hit = step > state.freeTransfers ? HIT_COST : 0;
    const net = top.delta - hit;
    if (net <= 0) break;
    cumulative += net;
    moves.push({ ...top, step, hit, net, cumulativeNet: cumulative });
    slots = slots.map((s) => (s.id === top.out.id ? { id: top.in.id, value: top.in.price } : s));
    bank += top.cost;
  }
  return moves;
}
