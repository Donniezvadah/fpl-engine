"""Squad and starting-XI legality checks.

The optimizer emits squads; this module independently verifies them. Keeping the
checker separate from the MILP constraints means a solver bug cannot silently
produce an illegal team.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .constants import Position, SeasonRules


@dataclass(frozen=True)
class SquadPlayer:
    player_id: int
    position: Position
    club_id: int
    price: int  # tenths of a million (buy cost or sell value depending on context)


@dataclass(frozen=True)
class Squad:
    """A full 15-player squad with its selection details."""

    players: tuple[SquadPlayer, ...]
    starting_ids: frozenset[int]
    captain_id: int
    vice_captain_id: int
    bench_order: tuple[int, ...]  # outfield bench in autosub priority order (bench GK separate)


def validate_squad_composition(
    players: list[SquadPlayer], rules: SeasonRules, budget: int | None = None
) -> list[str]:
    """Return a list of rule violations (empty list == legal)."""
    errors: list[str] = []
    sq = rules.squad

    if len(players) != sq.squad_size:
        errors.append(f"squad has {len(players)} players, needs {sq.squad_size}")
    ids = [p.player_id for p in players]
    if len(set(ids)) != len(ids):
        errors.append("duplicate players in squad")

    by_pos = Counter(p.position for p in players)
    for pos, need in sq.position_counts.items():
        if by_pos.get(pos, 0) != need:
            errors.append(f"{pos.name} count is {by_pos.get(pos, 0)}, needs {need}")

    by_club = Counter(p.club_id for p in players)
    for club, n in by_club.items():
        if n > sq.max_per_club:
            errors.append(f"club {club} has {n} players, max {sq.max_per_club}")

    cap = sq.budget if budget is None else budget
    total = sum(p.price for p in players)
    if total > cap:
        errors.append(f"cost {total} exceeds budget {cap}")

    return errors


def validate_selection(squad: Squad, rules: SeasonRules) -> list[str]:
    """Validate starting XI, formation, captaincy and bench order."""
    errors: list[str] = []
    sq = rules.squad
    by_id = {p.player_id: p for p in squad.players}

    unknown = squad.starting_ids - by_id.keys()
    if unknown:
        errors.append(f"starting ids not in squad: {sorted(unknown)}")
        return errors

    if len(squad.starting_ids) != sq.xi_size:
        errors.append(f"XI has {len(squad.starting_ids)} players, needs {sq.xi_size}")

    xi_by_pos = Counter(by_id[i].position for i in squad.starting_ids)
    for pos in Position:
        n = xi_by_pos.get(pos, 0)
        if not (sq.formation_min[pos] <= n <= sq.formation_max[pos]):
            errors.append(
                f"formation has {n} {pos.name}, allowed "
                f"{sq.formation_min[pos]}-{sq.formation_max[pos]}"
            )

    if squad.captain_id not in squad.starting_ids:
        errors.append("captain must be in the starting XI")
    if squad.vice_captain_id not in squad.starting_ids:
        errors.append("vice-captain must be in the starting XI")
    if squad.captain_id == squad.vice_captain_id:
        errors.append("captain and vice-captain must differ")

    bench_ids = {p.player_id for p in squad.players} - squad.starting_ids
    bench_outfield = {i for i in bench_ids if by_id[i].position != Position.GK}
    if set(squad.bench_order) != bench_outfield:
        errors.append("bench_order must contain exactly the non-GK bench players")

    return errors
