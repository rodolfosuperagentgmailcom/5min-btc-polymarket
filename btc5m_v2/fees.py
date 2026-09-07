from __future__ import annotations

from typing import Iterable, Protocol


class FillLike(Protocol):
    price: float
    shares: float


def taker_fee_usd(
    *,
    shares: float,
    price: float,
    fee_rate: float,
    fees_enabled: bool,
    fee_exponent: float = 1.0,
    is_taker: bool = True,
) -> float:
    """Continuous CLOB V2 platform fee for one execution level.

    CLOB V2 exposes a per-market fee schedule with rate ``r`` and exponent ``e``.
    The official V2 client applies ``r * (p * (1-p)) ** e`` per share. Venue
    rounding can be applied at accounting/reconciliation time; research keeps
    the continuous value so rounding choices do not create synthetic edge.
    """

    qty = float(shares)
    px = float(price)
    rate = float(fee_rate)
    exponent = float(fee_exponent)
    if qty < 0.0:
        raise ValueError("shares cannot be negative")
    if not 0.0 < px <= 1.0:
        raise ValueError("price must be in (0, 1]")
    if rate < 0.0:
        raise ValueError("fee_rate cannot be negative")
    if exponent < 0.0:
        raise ValueError("fee_exponent cannot be negative")
    if not fees_enabled or not is_taker or qty == 0.0 or rate == 0.0:
        return 0.0
    return qty * rate * (px * (1.0 - px)) ** exponent


def fee_for_fills(
    fills: Iterable[FillLike],
    *,
    fee_rate: float,
    fees_enabled: bool,
    fee_exponent: float = 1.0,
    is_taker: bool = True,
) -> float:
    return sum(
        taker_fee_usd(
            shares=float(fill.shares),
            price=float(fill.price),
            fee_rate=fee_rate,
            fee_exponent=fee_exponent,
            fees_enabled=fees_enabled,
            is_taker=is_taker,
        )
        for fill in fills
    )


def all_in_cost_per_share(
    *,
    spent_usd: float,
    shares: float,
    fee_usd: float,
) -> float | None:
    qty = float(shares)
    if qty <= 0.0:
        return None
    total = float(spent_usd) + float(fee_usd)
    if total < 0.0:
        raise ValueError("all-in cost cannot be negative")
    return total / qty
