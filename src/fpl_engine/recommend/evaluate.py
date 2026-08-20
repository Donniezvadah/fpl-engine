"""Score an arbitrary 15-player squad against the model.

Exists because FPL keeps a manager's picks private until the gameweek deadline,
so before the season opens the only way to assess the squad someone has actually
drafted is for them to name it. Given 15 players this reports the best legal XI,
captain, total expected points, and the upgrades worth making — the same
machinery the optimiser uses, pointed at a squad it did not choose.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from ..optim.squad import BENCH_WEIGHT, SquadOptimizer
from ..rules.constants import Position, SeasonRules
from ..rules.squad import SquadPlayer, validate_squad_composition


def _normalize(name: str) -> str:
    """Fold accents and punctuation away so "Gyökeres", "Raya (ARS)" and
    "E.Le Fee" all reduce to plain space-separated lowercase tokens."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = "".join(ch if ch.isalnum() else " " for ch in s.lower())
    return " ".join(s.split())


def build_pool(report, players: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Candidate pool for evaluation: every player with an `xp` column.

    `horizon=1` scores the next gameweek alone; larger values sum expected points
    over that many gameweeks, which is the right lens for a squad you are about
    to lock in for a run of fixtures.
    """
    pool = players[
        ["player_id", "web_name", "first_name", "second_name", "position", "team_id", "now_price"]
    ].copy()

    gws = report.horizon_gws[:horizon]
    hx = report.horizon_xp
    if hx is not None and not hx.empty and gws:
        agg = (
            hx[hx["event"].isin(gws)]
            .groupby("player_id", as_index=False)["xp"]
            .sum()
        )
    else:
        agg = report.xp_table[["player_id", "xp"]]
    pool = pool.merge(agg, on="player_id", how="left")
    pool["xp"] = pool["xp"].fillna(0.0)

    teams = report.teams
    pool["club_short"] = pool["team_id"].map(dict(zip(teams["team_id"], teams["short_name"])))
    pool["club"] = pool["team_id"].map(dict(zip(teams["team_id"], teams["name"])))
    pool["club_short"] = pool["club_short"].fillna("")
    pool["club"] = pool["club"].fillna("")
    return pool


@dataclass
class Resolution:
    """Outcome of matching one user-supplied name to a player."""

    query: str
    player_id: int | None
    matched_name: str | None
    candidates: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.player_id is not None


def resolve_players(queries: list[str], players: pd.DataFrame) -> list[Resolution]:
    """Match free-text names to players, disambiguating by club when given.

    Accepts "Haaland", "Haaland (MCI)", "Man City Haaland" or a bare element id.
    Ambiguous surnames come back unresolved with the candidates listed rather
    than silently picking one — a wrong player would quietly corrupt the whole
    evaluation.
    """
    out: list[Resolution] = []
    by_id = set(players["player_id"])

    norm_web = players["web_name"].map(_normalize)
    norm_full = (players["first_name"] + " " + players["second_name"]).map(_normalize)
    club_short = players["club_short"].map(_normalize)
    club_full = players["club"].map(_normalize)

    for raw in queries:
        q = str(raw).strip()
        if not q:
            continue
        if q.isdigit() and int(q) in by_id:
            row = players[players["player_id"] == int(q)].iloc[0]
            out.append(Resolution(q, int(q), row["web_name"]))
            continue

        nq = _normalize(q)
        # A trailing/leading club hint narrows the search.
        club_mask = pd.Series(True, index=players.index)
        for token in nq.split():
            hit = (club_short == token) | (club_full == token)
            if hit.any():
                club_mask &= hit
                nq = " ".join(t for t in nq.split() if t != token)
        nq = nq.strip()

        exact = (norm_web == nq) & club_mask
        if exact.sum() == 1:
            row = players[exact].iloc[0]
            out.append(Resolution(q, int(row["player_id"]), row["web_name"]))
            continue

        full_exact = (norm_full == nq) & club_mask
        if full_exact.sum() == 1:
            row = players[full_exact].iloc[0]
            out.append(Resolution(q, int(row["player_id"]), row["web_name"]))
            continue

        contains = (
            norm_web.str.contains(nq, regex=False)
            | norm_full.str.contains(nq, regex=False)
        ) & club_mask
        matches = players[contains]
        if len(matches) == 1:
            row = matches.iloc[0]
            out.append(Resolution(q, int(row["player_id"]), row["web_name"]))
        else:
            cands = [
                f"{r.web_name} ({r.club_short}, {Position(int(r.position)).name}, "
                f"£{int(r.now_price) / 10:.1f}m)"
                for r in matches.head(8).itertuples()
            ]
            out.append(Resolution(q, None, None, cands))
    return out


@dataclass
class SquadEvaluation:
    squad: pd.DataFrame  # the 15, with xp and role
    xi_xp: float  # starting XI expected points
    captain_id: int
    captain_name: str
    total_xp: float  # XI + captain doubled
    bench_xp: float
    cost: int
    legality: list[str]
    upgrades: pd.DataFrame  # suggested swaps within budget
    optimal_xp: float  # what the unconstrained optimiser achieves
    horizon_gws: list[int]


def evaluate_squad(
    player_ids: list[int],
    pool: pd.DataFrame,
    rules: SeasonRules,
    bank: int = 0,
    horizon_gws: list[int] | None = None,
) -> SquadEvaluation:
    """`pool` needs: player_id, web_name, position, team_id, club_short, now_price, xp."""
    sel = pool[pool["player_id"].isin(player_ids)].copy()

    squad_players = [
        SquadPlayer(
            int(r.player_id), Position(int(r.position)), int(r.team_id), int(r.now_price)
        )
        for r in sel.itertuples()
    ]
    legality = validate_squad_composition(squad_players, rules, budget=1000 + bank)

    # Best legal XI from these 15, using the same optimiser (locked to this squad).
    xi_ids: set[int] = set()
    captain_id = int(sel.loc[sel["xp"].idxmax(), "player_id"])
    if not legality:
        opt = SquadOptimizer(rules)
        locked = sel.rename(columns={"now_price": "value"})[
            ["player_id", "position", "team_id", "value", "xp"]
        ]
        result = opt.optimize(locked, budget=10_000, bench_weight=BENCH_WEIGHT)
        xi_ids = set(result.squad.starting_ids)
        captain_id = result.squad.captain_id

    sel["role"] = [
        "captain"
        if int(r.player_id) == captain_id
        else ("XI" if int(r.player_id) in xi_ids else "bench")
        for r in sel.itertuples()
    ]
    xi = sel[sel["role"] != "bench"]
    bench = sel[sel["role"] == "bench"]
    xi_xp = float(xi["xp"].sum())
    cap_xp = float(sel.loc[sel["player_id"] == captain_id, "xp"].iloc[0])

    # What the optimiser would build with the same money, for context.
    optimal_xp = 0.0
    try:
        free = SquadOptimizer(rules).optimize(
            pool.rename(columns={"now_price": "value"})[
                ["player_id", "position", "team_id", "value", "xp"]
            ],
            budget=int(sel["now_price"].sum()) + bank,
        )
        free_rows = pool[pool["player_id"].isin(free.squad.starting_ids)]
        optimal_xp = float(free_rows["xp"].sum()) + float(
            pool.loc[pool["player_id"] == free.squad.captain_id, "xp"].iloc[0]
        )
    except Exception:
        optimal_xp = float("nan")

    return SquadEvaluation(
        squad=sel.sort_values(["role", "xp"], ascending=[True, False]),
        xi_xp=xi_xp,
        captain_id=captain_id,
        captain_name=str(sel.loc[sel["player_id"] == captain_id, "web_name"].iloc[0]),
        total_xp=xi_xp + cap_xp,
        bench_xp=float(bench["xp"].sum()),
        cost=int(sel["now_price"].sum()),
        legality=legality,
        upgrades=suggest_upgrades(sel, pool, rules, bank),
        optimal_xp=optimal_xp,
        horizon_gws=horizon_gws or [],
    )


def suggest_upgrades(
    sel: pd.DataFrame, pool: pd.DataFrame, rules: SeasonRules, bank: int, top_n: int = 8
) -> pd.DataFrame:
    """Single swaps that gain expected points and stay affordable and legal.

    Only same-position moves are considered: a cross-position swap changes the
    squad's shape and has to be judged as a pair, which is the transfer planner's
    job rather than a one-for-one comparison.
    """
    owned = set(sel["player_id"])
    club_counts = sel["team_id"].value_counts().to_dict()
    rows = []
    for out_p in sel.itertuples():
        budget = int(out_p.now_price) + bank
        same_pos = pool[
            (pool["position"] == out_p.position)
            & (~pool["player_id"].isin(owned))
            & (pool["now_price"] <= budget)
        ]
        for in_p in same_pos.nlargest(3, "xp").itertuples():
            # Club limit: bringing in a third-from-that-club is fine, a fourth is not.
            n_club = club_counts.get(int(in_p.team_id), 0)
            if int(in_p.team_id) != int(out_p.team_id) and n_club >= rules.squad.max_per_club:
                continue
            gain = float(in_p.xp) - float(out_p.xp)
            if gain <= 0.05:
                continue
            rows.append(
                {
                    # Not "in"/"out": `in` is a Python keyword, and pandas mangles
                    # such names into positional junk in itertuples().
                    "out_name": out_p.web_name,
                    "out_price": int(out_p.now_price),
                    "out_xp": float(out_p.xp),
                    "in_name": in_p.web_name,
                    "in_club": in_p.club_short,
                    "in_price": int(in_p.now_price),
                    "in_xp": float(in_p.xp),
                    "spend": int(in_p.now_price) - int(out_p.now_price),
                    "gain": gain,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("gain", ascending=False)
    # One suggestion per outgoing player keeps the list readable.
    return df.drop_duplicates("out_name").head(top_n).reset_index(drop=True)
