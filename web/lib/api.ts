/** Types and loader for the baked static dataset.
 *
 * The published site is a static export with no backend, so the whole dataset
 * is fetched once as a single JSON file and every filter, sort and re-ranking
 * afterwards happens in the browser.
 */

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export interface Meta {
  season: string;
  next_gw: number;
  mode: "linked" | "draft";
  horizon_gws: number[];
  generated_at: string;
  player_count: number;
}

export interface Player {
  id: number;
  name: string;
  position: number;
  position_name: string;
  team_id: number;
  club: string;
  club_short: string;
  price: number;
  ownership: number | null;
  xp: number | null;
  p_start: number | null;
  p60: number | null;
  p_cs: number | null;
  e_goals: number | null;
  e_assists: number | null;
  xp_by_gw: Record<string, number | null>;
  fixtures_by_gw: Record<string, number>;
  opp_by_gw: Record<string, string>;
  status: string;
  chance: number | null;
  news: string;
}

export interface CaptainOption {
  id: number;
  name: string;
  xp: number;
  p10: number;
  p90: number;
  p_haul: number;
  ownership: number;
  reasoning: string;
}

export interface ChipEval {
  chip: string;
  event: number;
  gain: number;
  detail: string;
}

export interface PlannedGw {
  event: number;
  transfers_in: string[];
  transfers_out: string[];
  hits: number;
  free_transfers: number;
  captain: string;
  xi_xp: number;
}

export interface SquadPlayer {
  id: number;
  name: string;
  role: string;
  position: number | null;
  position_name: string;
  price: number | null;
  xp: number | null;
}

export interface Differential {
  player_id: number;
  name: string;
  position: number;
  value: number;
  ownership: number;
  xp: number;
}

export interface PriceWatchRow {
  player_id: number;
  name: string;
  p_rise: number;
  p_fall: number;
}

export interface TeamEntry {
  id: number;
  name: string;
  position: number | null;
  club_short: string;
  price: number | null;
  sell: number | null;
  role: string; // captain | vice | XI | bench | out | in
  order: number;
  xp_next: number | null;
  xp_horizon: number;
}

export interface WildcardMove {
  out: TeamEntry;
  in: TeamEntry;
  delta_xp: number;
}

export interface WildcardPlan {
  event: number;
  gain: number;
  detail: string;
  squad: TeamEntry[];
  moves: WildcardMove[];
  cost: number;
  budget: number;
  follow_on: {
    event: number;
    transfers_in: string[];
    transfers_out: string[];
    hits: number;
    captain: string;
  }[];
}

export interface RankPoint {
  event: number;
  overall_points: number;
  overall_rank: number | null;
  event_transfers: number;
  active_chip: string | null;
}

export interface MyTeamData {
  bank: number;
  free_transfers: number;
  team_value: number;
  chips_used: { chip: string; event: number }[];
  chips_available: string[];
  current: TeamEntry[];
  wildcard: WildcardPlan | null;
  history: RankPoint[];
}

export interface Dataset {
  my_team: MyTeamData | null;
  meta: Meta;
  captains: CaptainOption[];
  chips: ChipEval[];
  plan: PlannedGw[] | null;
  squad: { kind: string; cost: number | null; players: SquadPlayer[] } | null;
  differentials: Differential[];
  price_watch: PriceWatchRow[];
  players: Player[];
}

export async function loadDataset(): Promise<Dataset> {
  const res = await fetch(`${BASE_PATH}/data/data.json`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(
      `Could not load data.json (HTTP ${res.status}). Run \`uv run fpl export\` to generate it.`,
    );
  }
  return res.json();
}

/** Total expected points for a player over the first `n` gameweeks of the horizon. */
export function horizonXp(p: Player, gws: number[], n: number): number {
  return gws
    .slice(0, n)
    .reduce((sum, gw) => sum + (p.xp_by_gw[String(gw)] ?? 0), 0);
}

export function horizonFixtures(p: Player, gws: number[], n: number): number {
  return gws
    .slice(0, n)
    .reduce((sum, gw) => sum + (p.fixtures_by_gw[String(gw)] ?? 0), 0);
}

export const POSITIONS: Record<number, string> = {
  1: "GK",
  2: "DEF",
  3: "MID",
  4: "FWD",
};

export const fmtPrice = (tenths: number | null) =>
  tenths == null ? "—" : `£${(tenths / 10).toFixed(1)}m`;
export const fmtPct = (x: number | null) =>
  x == null ? "—" : `${Math.round(x * 100)}%`;
export const fmtNum = (x: number | null, dp = 2) =>
  x == null ? "—" : x.toFixed(dp);
