"""Automatic substitutions and captaincy fallback, per official rules.

Applied after all of a GW's matches finish:
- A starter who played 0 minutes is replaced by the first bench player (in bench
  order) who did play and keeps the formation legal. GKs swap only with the bench GK.
- If the captain played 0 minutes the vice-captain's score is doubled instead.
"""

from __future__ import annotations

from collections import Counter

from .constants import Position, SeasonRules
from .squad import Squad, SquadPlayer


def _formation_ok(positions: list[Position], rules: SeasonRules) -> bool:
    counts = Counter(positions)
    return all(
        rules.squad.formation_min[pos] <= counts.get(pos, 0) <= rules.squad.formation_max[pos]
        for pos in Position
    )


def apply_autosubs(
    squad: Squad, minutes_by_player: dict[int, int], rules: SeasonRules
) -> frozenset[int]:
    """Return the final scoring XI after automatic substitutions."""
    by_id: dict[int, SquadPlayer] = {p.player_id: p for p in squad.players}
    xi = list(squad.starting_ids)
    bench_ids = [p.player_id for p in squad.players if p.player_id not in squad.starting_ids]

    bench_gk = next((i for i in bench_ids if by_id[i].position == Position.GK), None)
    bench_outfield = list(squad.bench_order)

    # GK swap first: independent of formation.
    starting_gk = next(i for i in xi if by_id[i].position == Position.GK)
    if minutes_by_player.get(starting_gk, 0) == 0 and bench_gk is not None:
        if minutes_by_player.get(bench_gk, 0) > 0:
            xi[xi.index(starting_gk)] = bench_gk

    # Outfield: replace each non-playing starter with the first eligible bench player.
    used_bench: set[int] = set()
    for starter in [i for i in list(xi) if by_id[i].position != Position.GK]:
        if minutes_by_player.get(starter, 0) > 0:
            continue
        for cand in bench_outfield:
            if cand in used_bench or minutes_by_player.get(cand, 0) == 0:
                continue
            trial = [by_id[p].position for p in xi if p != starter] + [by_id[cand].position]
            if _formation_ok(trial, rules):
                xi[xi.index(starter)] = cand
                used_bench.add(cand)
                break

    return frozenset(xi)


def effective_captain(
    squad: Squad, minutes_by_player: dict[int, int]
) -> int | None:
    """Who receives the captain multiplier (None if neither played)."""
    if minutes_by_player.get(squad.captain_id, 0) > 0:
        return squad.captain_id
    if minutes_by_player.get(squad.vice_captain_id, 0) > 0:
        return squad.vice_captain_id
    return None
