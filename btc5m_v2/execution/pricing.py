from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FillLevel:
    price: float
    shares: float
    notional_usd: float


@dataclass(frozen=True)
class BuyFillEstimate:
    fillable: bool
    requested_notional_usd: float
    spent_usd: float
    shares: float
    average_price: float | None
    worst_price: float | None
    visible_ask_notional_usd: float
    participation_pct: float | None
    fills: tuple[FillLevel, ...]


@dataclass(frozen=True)
class SellFillEstimate:
    fillable: bool
    requested_shares: float
    sold_shares: float
    proceeds_usd: float
    average_price: float | None
    worst_price: float | None
    visible_bid_shares: float
    visible_bid_notional_usd: float
    participation_pct: float | None
    fills: tuple[FillLevel, ...]


def _normalized_levels(
    levels: Iterable[tuple[float, float]],
    *,
    reverse: bool,
) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for raw_price, raw_size in levels:
        price = float(raw_price)
        size = float(raw_size)
        if 0.0 < price <= 1.0 and size > 0.0:
            normalized.append((price, size))
    normalized.sort(key=lambda item: item[0], reverse=reverse)
    return normalized


def estimate_buy_fill(
    asks: Iterable[tuple[float, float]],
    *,
    notional_usd: float,
) -> BuyFillEstimate:
    """Estimate a taker buy by walking the selected side's visible asks."""

    requested = float(notional_usd)
    if requested <= 0.0:
        raise ValueError("notional_usd must be positive")

    levels = _normalized_levels(asks, reverse=False)
    visible = sum(price * size for price, size in levels)
    remaining = requested
    spent = 0.0
    shares = 0.0
    worst_price: float | None = None
    fills: list[FillLevel] = []

    for price, available_shares in levels:
        if remaining <= 1e-12:
            break
        level_notional = price * available_shares
        take_notional = min(remaining, level_notional)
        take_shares = take_notional / price
        if take_shares <= 0.0:
            continue
        fills.append(FillLevel(price=price, shares=take_shares, notional_usd=take_notional))
        spent += take_notional
        shares += take_shares
        remaining -= take_notional
        worst_price = price

    fillable = remaining <= 1e-9
    average = spent / shares if shares > 0.0 else None
    participation = (spent / visible * 100.0) if visible > 0.0 else None
    return BuyFillEstimate(
        fillable=fillable,
        requested_notional_usd=requested,
        spent_usd=spent,
        shares=shares,
        average_price=average,
        worst_price=worst_price,
        visible_ask_notional_usd=visible,
        participation_pct=participation,
        fills=tuple(fills),
    )


def estimate_sell_fill(
    bids: Iterable[tuple[float, float]],
    *,
    shares_to_sell: float,
) -> SellFillEstimate:
    """Estimate a taker sell by walking the selected token's visible bids."""

    requested = float(shares_to_sell)
    if requested <= 0.0:
        raise ValueError("shares_to_sell must be positive")

    levels = _normalized_levels(bids, reverse=True)
    visible_shares = sum(size for _, size in levels)
    visible_notional = sum(price * size for price, size in levels)
    remaining = requested
    sold = 0.0
    proceeds = 0.0
    worst_price: float | None = None
    fills: list[FillLevel] = []

    for price, available_shares in levels:
        if remaining <= 1e-12:
            break
        take_shares = min(remaining, available_shares)
        if take_shares <= 0.0:
            continue
        take_notional = price * take_shares
        fills.append(FillLevel(price=price, shares=take_shares, notional_usd=take_notional))
        sold += take_shares
        proceeds += take_notional
        remaining -= take_shares
        worst_price = price

    fillable = remaining <= 1e-9
    average = proceeds / sold if sold > 0.0 else None
    participation = (sold / visible_shares * 100.0) if visible_shares > 0.0 else None
    return SellFillEstimate(
        fillable=fillable,
        requested_shares=requested,
        sold_shares=sold,
        proceeds_usd=proceeds,
        average_price=average,
        worst_price=worst_price,
        visible_bid_shares=visible_shares,
        visible_bid_notional_usd=visible_notional,
        participation_pct=participation,
        fills=tuple(fills),
    )
