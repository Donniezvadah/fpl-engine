"""Free-transfer banking, hit costs, and sell-price arithmetic."""

from __future__ import annotations

from .constants import TransferRules


def next_free_transfers(current_ft: int, transfers_made: int, rules: TransferRules) -> int:
    """Free transfers available next GW, per the official banking rule.

    You gain one FT per gameweek, capped at the bank limit. Transfers made beyond
    your available FTs (hits) can't push the balance below zero.
    """
    remaining = max(0, current_ft - transfers_made)
    return min(remaining + rules.free_transfers_per_gw, rules.max_banked_free_transfers)


def hit_points(transfers_made: int, free_transfers: int, rules: TransferRules) -> int:
    """Point deduction for transfers beyond the available free transfers."""
    extra = max(0, transfers_made - free_transfers)
    return extra * rules.hit_cost


def sell_price(purchase_price: int, current_price: int, rules: TransferRules) -> int:
    """Official sell price in tenths: you keep 50% of profit, rounded down per 0.1.

    E.g. bought at 55, now 58 -> profit 3 tenths -> keep 1 tenth -> sell at 56.
    Falls are absorbed fully: bought at 55, now 52 -> sell at 52.
    """
    if current_price <= purchase_price:
        return current_price
    profit = current_price - purchase_price
    return purchase_price + int(profit * rules.sell_on_fee_fraction)
