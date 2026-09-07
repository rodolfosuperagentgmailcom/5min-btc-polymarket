from __future__ import annotations

from btc5m_v2.research.fees import (
    buy_fee_economics,
    expected_hold_pnl_per_share,
    hold_to_resolution_breakeven_probability,
    probability_edge_after_entry_fee,
    taker_fee_usdc,
)


def test_crypto_fee_formula_at_fifty_cents() -> None:
    # Current published crypto category rate is 0.07; 100 shares at $0.50
    # produces 100 * .07 * .5 * .5 = 1.75 USDC before rounding.
    assert round(taker_fee_usdc(100, 0.50, 0.07), 8) == 1.75


def test_clob_v2_breakeven_adds_usdc_fee_per_share() -> None:
    q = hold_to_resolution_breakeven_probability(0.70, 0.07)
    expected = 0.70 + 0.07 * 0.70 * 0.30
    assert round(q, 8) == round(expected, 8)
    assert q > 0.70
    assert abs(expected_hold_pnl_per_share(q, 0.70, 0.07)) < 1e-12


def test_buy_fee_economics_keeps_share_count_and_adds_cash_fee() -> None:
    economics = buy_fee_economics(100, 0.50, 0.07)
    assert economics.shares == 100
    assert round(economics.fee_usdc, 8) == 1.75
    assert round(economics.cash_cost_usdc, 8) == 51.75
    assert round(economics.breakeven_probability, 8) == 0.5175


def test_fee_disabled_market_has_price_as_breakeven() -> None:
    assert hold_to_resolution_breakeven_probability(0.70, 0.07, fees_enabled=False) == 0.70
    assert round(
        probability_edge_after_entry_fee(0.74, 0.70, 0.07, fees_enabled=False), 8
    ) == 0.04


def test_fee_raises_probability_hurdle() -> None:
    no_fee_edge = probability_edge_after_entry_fee(0.74, 0.70, 0.07, fees_enabled=False)
    fee_edge = probability_edge_after_entry_fee(0.74, 0.70, 0.07, fees_enabled=True)
    assert fee_edge < no_fee_edge
