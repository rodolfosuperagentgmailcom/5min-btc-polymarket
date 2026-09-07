from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuyFeeEconomics:
    price: float
    fee_rate: float
    fees_enabled: bool
    shares: float
    fee_usdc: float
    cash_cost_usdc: float
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
    """CLOB V2 taker fee in USDC: C * feeRate * p * (1-p).

    Since the April 28, 2026 CLOB V2 upgrade, fees are charged in USDC at
    match time. Research keeps the continuous value unless venue-style
    five-decimal rounding is explicitly requested.
    """

    quantity = float(shares)
    if quantity < 0:
        raise ValueError("shares must be non-negative")
    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    if not fees_enabled or quantity == 0 or rate == 0:
        return 0.0
    fee = quantity * rate * p * (1.0 - p)
    return round(fee, 5) if round_to_5_decimals else fee


def hold_to_resolution_breakeven_probability(
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    """Breakeven q for one taker-bought share held to binary settlement.

    CLOB V2 charges the fee in USDC, so one share costs p plus its USDC fee.
    A winning share still settles to $1; therefore q_break_even equals the
    all-in USDC cost per share.
    """

    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    fee = taker_fee_usdc(1.0, p, rate, fees_enabled=fees_enabled)
    return p + fee


def expected_hold_pnl_per_share(
    model_probability: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    q = float(model_probability)
    if not 0 <= q <= 1:
        raise ValueError("model_probability must be between 0 and 1")
    return q - hold_to_resolution_breakeven_probability(
        price,
        fee_rate,
        fees_enabled=fees_enabled,
    )


def probability_edge_after_entry_fee(
    model_probability: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> float:
    return expected_hold_pnl_per_share(
        model_probability,
        price,
        fee_rate,
        fees_enabled=fees_enabled,
    )


def buy_fee_economics(
    shares: float,
    price: float,
    fee_rate: float,
    *,
    fees_enabled: bool = True,
) -> BuyFeeEconomics:
    p = _validate_price(price)
    rate = _validate_fee_rate(fee_rate)
    quantity = float(shares)
    if quantity < 0:
        raise ValueError("shares must be non-negative")
    fee = taker_fee_usdc(quantity, p, rate, fees_enabled=fees_enabled)
    cash_cost = quantity * p + fee
    return BuyFeeEconomics(
        price=p,
        fee_rate=rate,
        fees_enabled=fees_enabled,
        shares=quantity,
        fee_usdc=fee,
        cash_cost_usdc=cash_cost,
        breakeven_probability=hold_to_resolution_breakeven_probability(
            p,
            rate,
            fees_enabled=fees_enabled,
        ),
    )
