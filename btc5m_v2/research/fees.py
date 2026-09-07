from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuyFeeEconomics:
    price: float
    fee_rate: float
    fees_enabled: bool
    gross_shares: float
    fee_usdc: float
    net_shares: float
    breakeven_probability: float


def _validate_price(price: float) -> float:
    value = float(price)
    if not 0 < value < 1:
        raise ValueError("price must be between 0 and 1")
    return value


def _validate_fee_rate(fee_rate: float) -> float:
    value = float(fee_rate)
    if value < 0:
        raise ValueError("fee_rate must be non-negative")
    return value


def taker_fee_usdc(
    shares: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
    round_to_5_decimals: bool = False,
) -> float:
    """Polymarket taker fee in USDC: C * feeRate * p * (1-p)."""

    quantity = float(shares)
    if quantity < 0:
        raise ValueError("shares must be non-negative")
    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    if not fees_enabled or quantity == 0 or rate == 0:
        return 0.0
    fee = quantity * rate * p * (1.0 - p)
    return round(fee, 5) if round_to_5_decimals else fee


def net_shares_after_taker_buy(
    gross_shares: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    """Net outcome shares received after a taker BUY.

    Polymarket collects a taker-buy fee in shares. Using the documented USDC
    fee formula, fee_shares = fee_usdc / price.
    """

    quantity = float(gross_shares)
    if quantity < 0:
        raise ValueError("gross_shares must be non-negative")
    p = _validate_price(price)
    fee = taker_fee_usdc(quantity, p, fee_rate, fees_enabled=fees_enabled)
    return quantity - fee / p


def hold_to_resolution_breakeven_probability(
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    """Exact continuous breakeven q for a taker buy held to binary settlement.

    For one gross share purchased at price p, cash cost is p. A taker fee on a
    buy reduces received shares to 1 - feeRate*(1-p). Expected settlement value
    is q times those net shares. Solve q * net_shares = p.
    """

    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    if not fees_enabled or rate == 0:
        return p
    net_share_factor = 1.0 - rate * (1.0 - p)
    if net_share_factor <= 0:
        raise ValueError("fee rate produces non-positive net shares")
    return p / net_share_factor


def expected_hold_pnl_per_gross_share(
    model_probability: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    q = float(model_probability)
    if not 0 <= q <= 1:
        raise ValueError("model_probability must be between 0 and 1")
    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    net_shares = net_shares_after_taker_buy(1.0, p, rate, fees_enabled=fees_enabled)
    return q * net_shares - p


def probability_edge_after_entry_fee(
    model_probability: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    return float(model_probability) - hold_to_resolution_breakeven_probability(
        price,
        fee_rate,
        fees_enabled=fees_enabled,
    )


def buy_fee_economics(
    gross_shares: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> BuyFeeEconomics:
    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    quantity = float(gross_shares)
    fee = taker_fee_usdc(quantity, p, rate, fees_enabled=fees_enabled)
    net = net_shares_after_taker_buy(quantity, p, rate, fees_enabled=fees_enabled)
    return BuyFeeEconomics(
        price=p,
        fee_rate=rate,
        fees_enabled=fees_enabled,
        gross_shares=quantity,
        fee_usdc=fee,
        net_shares=net,
        breakeven_probability=hold_to_resolution_breakeven_probability(
            p,
            rate,
            fees_enabled=fees_enabled,
        ),
    )
