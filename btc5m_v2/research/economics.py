from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FillEstimate:
    fillable: bool
    shares: float
    spent_usd: float
    average_price: float | None
    worst_price: float | None


def fee_per_share(
    price: float,
    fee_rate: float,
    enabled: bool,
    fee_exponent: float = 1.0,
) -> float:
    if not enabled:
        return 0.0
    p = float(price)
    if p < 0 or p > 1:
        raise ValueError("binary contract price must be between 0 and 1")
    rate = float(fee_rate)
    exponent = float(fee_exponent)
    if rate < 0:
        raise ValueError("fee_rate must be non-negative")
    if exponent < 0:
        raise ValueError("fee_exponent must be non-negative")
    return rate * (p * (1.0 - p)) ** exponent


def breakeven_probability(
    entry_price: float,
    fee_rate: float,
    enabled: bool,
    fee_exponent: float = 1.0,
) -> float:
    return float(entry_price) + fee_per_share(
        entry_price,
        fee_rate,
        enabled,
        fee_exponent,
    )


def expected_value_per_share(
    model_probability: float,
    entry_price: float,
    fee_rate: float,
    enabled: bool,
    fee_exponent: float = 1.0,
) -> float:
    q = float(model_probability)
    if q < 0 or q > 1:
        raise ValueError("model_probability must be between 0 and 1")
    return q - breakeven_probability(
        entry_price,
        fee_rate,
        enabled,
        fee_exponent,
    )


def executable_buy_fill(asks: Iterable[tuple[float, float]], notional_usd: float) -> FillEstimate:
    target = float(notional_usd)
    if target <= 0:
        raise ValueError("notional_usd must be positive")

    remaining = target
    shares = 0.0
    spent = 0.0
    worst = None
    for raw_price, raw_shares in sorted(asks, key=lambda level: float(level[0])):
        price = float(raw_price)
        available_shares = float(raw_shares)
        if price <= 0 or price > 1 or available_shares <= 0:
            continue
        level_usd = price * available_shares
        take_usd = min(remaining, level_usd)
        take_shares = take_usd / price
        shares += take_shares
        spent += take_usd
        remaining -= take_usd
        worst = price
        if remaining <= 1e-12:
            break

    fillable = remaining <= 1e-12
    average = spent / shares if shares > 0 else None
    return FillEstimate(fillable, shares, spent, average, worst)


def executable_sell_fill(bids: Iterable[tuple[float, float]], shares_to_sell: float) -> FillEstimate:
    target_shares = float(shares_to_sell)
    if target_shares <= 0:
        raise ValueError("shares_to_sell must be positive")

    remaining = target_shares
    shares = 0.0
    proceeds = 0.0
    worst = None
    for raw_price, raw_shares in sorted(bids, key=lambda level: float(level[0]), reverse=True):
        price = float(raw_price)
        available_shares = float(raw_shares)
        if price < 0 or price > 1 or available_shares <= 0:
            continue
        take_shares = min(remaining, available_shares)
        shares += take_shares
        proceeds += take_shares * price
        remaining -= take_shares
        worst = price
        if remaining <= 1e-12:
            break

    fillable = remaining <= 1e-12
    average = proceeds / shares if shares > 0 else None
    return FillEstimate(fillable, shares, proceeds, average, worst)
