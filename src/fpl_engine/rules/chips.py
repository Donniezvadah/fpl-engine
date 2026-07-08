"""Chip legality: which chips can be played in which gameweek."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import SeasonRules


@dataclass
class ChipState:
    """Chips already used this season, as {chip_name: gameweek_played}.

    With two_sets rules, the same chip can appear at most once per half; we track
    each use with the GW it was played in.
    """

    used: list[tuple[str, int]] = field(default_factory=list)

    def uses_of(self, chip: str) -> list[int]:
        return [gw for name, gw in self.used if name == chip]


def _half(gw: int, rules: SeasonRules) -> int:
    return 1 if gw <= rules.chips.first_half_last_gw else 2


def available_chips(state: ChipState, gw: int, rules: SeasonRules) -> list[str]:
    """Chips playable in `gw` given what has been used already."""
    out = []
    for chip in rules.chips.chips:
        uses = state.uses_of(chip)
        if rules.chips.two_sets:
            # One use allowed per half-season.
            if not any(_half(g, rules) == _half(gw, rules) for g in uses):
                out.append(chip)
        else:
            if not uses:
                out.append(chip)
    return out


def can_play_chip(
    state: ChipState, chip: str, gw: int, rules: SeasonRules, chip_this_gw: str | None = None
) -> bool:
    """Only one chip may be active per gameweek."""
    if chip_this_gw is not None and chip_this_gw != chip:
        return False
    return chip in available_chips(state, gw, rules)
