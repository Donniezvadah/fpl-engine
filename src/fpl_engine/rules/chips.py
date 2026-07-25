"""Chip legality: which chips can be played in which gameweek.

Availability is driven by the per-chip windows in `constants.ChipRules`, which
mirror the official bootstrap-static `chips` array. A chip is playable in a
gameweek when some window contains that gameweek and that window has not already
been used. Only one chip may be active in any gameweek.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ChipWindow, SeasonRules


@dataclass
class ChipState:
    """Chips already used this season, as (chip_name, gameweek_played) pairs."""

    used: list[tuple[str, int]] = field(default_factory=list)

    def uses_of(self, chip: str) -> list[int]:
        return [gw for name, gw in self.used if name == chip]


def _window_for(rules: SeasonRules, chip: str, gw: int) -> ChipWindow | None:
    """The window that would be consumed by playing `chip` in `gw`."""
    for w in rules.chips.windows_for(chip):
        if w.contains(gw):
            return w
    return None


def available_chips(state: ChipState, gw: int, rules: SeasonRules) -> list[str]:
    """Chips playable in `gw` given what has been used already."""
    out: list[str] = []
    for chip in rules.chips.chips:
        window = _window_for(rules, chip, gw)
        if window is None:
            continue  # chip has no window covering this gameweek
        if any(window.contains(used_gw) for used_gw in state.uses_of(chip)):
            continue  # this window has already been spent
        out.append(chip)
    return out


def can_play_chip(
    state: ChipState, chip: str, gw: int, rules: SeasonRules, chip_this_gw: str | None = None
) -> bool:
    """Whether `chip` may be played in `gw`. Only one chip is allowed per gameweek."""
    if chip_this_gw is not None and chip_this_gw != chip:
        return False
    return chip in available_chips(state, gw, rules)
