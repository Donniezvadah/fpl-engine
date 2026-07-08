export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface CaptainOption {
  player_id: number;
  name: string;
  xp: number;
  p90: number;
  p10: number;
  p_haul: number;
  ownership: number;
  reasoning: string;
}

export interface ChipEval {
  chip: string;
  event: number;
  gain_pts: number;
  detail: string;
}

export interface PlannedGw {
  event: number;
  transfers_in: string[];
  transfers_out: string[];
  hits: number;
  free_transfers: number;
  captain: string | null;
  xi_xp: number;
}

export interface DraftRow {
  role: string;
  player_id: number;
  name: string;
  position: number;
  value: number;
  xp: number;
}

export interface Differential {
  player_id: number;
  name: string;
  position: number;
  value: number;
  ownership: number;
  xp: number;
  p_cs: number | null;
  e_goals: number | null;
  e_assists: number | null;
}

export interface PriceWatchRow {
  player_id: number;
  name: string;
  p_rise: number;
  p_fall: number;
}

export interface Report {
  season: string;
  next_gw: number;
  mode: "linked" | "draft";
  as_of_replay: boolean;
  captains: CaptainOption[];
  chips: ChipEval[];
  differentials: Differential[];
  price_watch: PriceWatchRow[];
  plan: PlannedGw[] | null;
  draft: DraftRow[] | null;
}

export interface PlayerRow {
  player_id: number;
  name: string;
  position: number;
  value: number;
  ownership: number | null;
  xp: number;
  p_start: number | null;
  p_cs: number | null;
  e_goals: number | null;
  e_assists: number | null;
  n_fixtures: number | null;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export function fetchReport(asOfGw?: number): Promise<Report> {
  return get<Report>(`/report${asOfGw ? `?as_of_gw=${asOfGw}` : ""}`);
}

export function fetchPlayers(asOfGw?: number): Promise<PlayerRow[]> {
  return get<PlayerRow[]>(`/players${asOfGw ? `?as_of_gw=${asOfGw}` : ""}`);
}

export const POSITIONS: Record<number, string> = {
  1: "GK",
  2: "DEF",
  3: "MID",
  4: "FWD",
};

export const fmtPrice = (tenths: number) => `£${(tenths / 10).toFixed(1)}m`;
export const fmtPct = (x: number) => `${Math.round(x * 100)}%`;
