"""Season-versioned FPL rule constants.

Everything that the official game defines numerically lives here, keyed by season,
so a rule change (like 2025-26 adding defensive-contribution points and 5-transfer
banking) is a one-file update. Downstream code must never hard-code these numbers.

Positions follow the API's element_type: 1=GK, 2=DEF, 3=MID, 4=FWD.
Prices are integer tenths of a million (55 == £5.5m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Position(IntEnum):
    GK = 1
    DEF = 2
    MID = 3
    FWD = 4


@dataclass(frozen=True)
class ScoringRules:
    played_upto_59: int = 1
    played_60_plus: int = 2
    goal_points: dict[Position, int] = field(
        default_factory=lambda: {Position.GK: 6, Position.DEF: 6, Position.MID: 5, Position.FWD: 4}
    )
    assist_points: int = 3
    clean_sheet_points: dict[Position, int] = field(
        default_factory=lambda: {Position.GK: 4, Position.DEF: 4, Position.MID: 1, Position.FWD: 0}
    )
    saves_per_point: int = 3  # GK: 1 pt per 3 saves
    penalty_save_points: int = 5
    penalty_miss_points: int = -2
    goals_conceded_per_minus: int = 2  # GK/DEF: -1 per 2 conceded
    goals_conceded_penalised_positions: tuple[Position, ...] = (Position.GK, Position.DEF)
    yellow_card_points: int = -1
    red_card_points: int = -3
    own_goal_points: int = -2
    bonus_points: tuple[int, ...] = (3, 2, 1)  # top-3 BPS in a match
    # Defensive contribution (introduced 2025-26): flat award, max once per match.
    defcon_enabled: bool = True
    defcon_points: int = 2
    defcon_threshold: dict[Position, int] = field(
        default_factory=lambda: {Position.DEF: 10, Position.MID: 12, Position.FWD: 12}
    )  # DEF: CBIT >= 10; MID/FWD: CBIRT >= 12. GKs not eligible.


@dataclass(frozen=True)
class SquadRules:
    budget: int = 1000  # £100.0m in tenths
    squad_size: int = 15
    position_counts: dict[Position, int] = field(
        default_factory=lambda: {Position.GK: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
    )
    max_per_club: int = 3
    # Starting XI formation bounds (inclusive).
    xi_size: int = 11
    formation_min: dict[Position, int] = field(
        default_factory=lambda: {Position.GK: 1, Position.DEF: 3, Position.MID: 2, Position.FWD: 1}
    )
    formation_max: dict[Position, int] = field(
        default_factory=lambda: {Position.GK: 1, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
    )


@dataclass(frozen=True)
class TransferRules:
    free_transfers_per_gw: int = 1
    max_banked_free_transfers: int = 5  # raised from 2 in 2024-25
    hit_cost: int = 4  # points per extra transfer
    # Sell price: purchase + floor(profit / 2) where profit is in tenths.
    sell_on_fee_fraction: float = 0.5


@dataclass(frozen=True)
class ChipWindow:
    """One playable window for one chip, inclusive of both endpoints."""

    chip: str
    start_event: int
    stop_event: int

    def contains(self, gw: int) -> bool:
        return self.start_event <= gw <= self.stop_event


# Mirrors the `chips` array of the official bootstrap-static endpoint. Note that
# wildcard and free hit only open in GW2 — neither can be played in GW1, since
# the opening squad is already unlimited-transfer.
_TWO_SET_WINDOWS: tuple[ChipWindow, ...] = (
    ChipWindow("wildcard", 2, 19),
    ChipWindow("wildcard", 20, 38),
    ChipWindow("freehit", 2, 19),
    ChipWindow("freehit", 20, 38),
    ChipWindow("bboost", 1, 19),
    ChipWindow("bboost", 20, 38),
    ChipWindow("3xc", 1, 19),
    ChipWindow("3xc", 20, 38),
)


@dataclass(frozen=True)
class ChipRules:
    """Chip availability. Since 2025-26 there are two full sets of chips: one
    usable in the first half of the season, a second in the second half. Each
    window may be used once, and only one chip may be active per gameweek."""

    chips: tuple[str, ...] = ("wildcard", "bboost", "3xc", "freehit")
    windows: tuple[ChipWindow, ...] = _TWO_SET_WINDOWS

    def windows_for(self, chip: str) -> tuple[ChipWindow, ...]:
        return tuple(w for w in self.windows if w.chip == chip)


@dataclass(frozen=True)
class SeasonRules:
    season: str
    scoring: ScoringRules
    squad: SquadRules
    transfers: TransferRules
    chips: ChipRules
    total_gameweeks: int = 38


_2025_26 = SeasonRules(
    season="2025-26",
    scoring=ScoringRules(),
    squad=SquadRules(),
    transfers=TransferRules(),
    chips=ChipRules(),
)

# 2026-27 rules unpublished as of July 2026; start from 2025-26 and amend on relaunch.
_2026_27 = SeasonRules(
    season="2026-27",
    scoring=ScoringRules(),
    squad=SquadRules(),
    transfers=TransferRules(),
    chips=ChipRules(),
)

SEASON_RULES: dict[str, SeasonRules] = {r.season: r for r in (_2025_26, _2026_27)}


def rules_for(season: str) -> SeasonRules:
    try:
        return SEASON_RULES[season]
    except KeyError:
        raise KeyError(
            f"No rules registered for season {season!r}; add them to rules/constants.py"
        ) from None
