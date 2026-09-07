from __future__ import annotations

from btc5m_v2.research.fees import (
    expected_hold_pnl_per_gross_share,
    hold_to_resolution_breakeven_probability,
    net_shares_after_taker_buy,
    probability_edge_after_entry_fee,
    taker_fee_usdc,
)


def test_crypto_fee_formula_at_fifty_cents() -> None:
    # Current published crypto category rate is 0.07; 100 shares at $0.50
    # produces 100 * .07 * .5 * .5 = 1.75 USDC before rounding.
    assert taker_fee_usdc(100, 0.50, 0.07) == 1.75
    assert net_shares_after_taker_buy(100, 0.50, 0.07) == 96.5


def test_breakeven_accounts_for_buy_fee_being_deducted_in_shares() -> None:
    q = hold_to_resolution_breakeven_probability(0.70, 0.07)
    assert round(q, 8) == round(0.70 / (1 - 0.07 * 0.30), 8)
    assert q > 0.70
    assert abs(expected_hold_pnl_per_gross_share(q, 0.70, 0.07)) < 1e-12


def test_fee_disabled_market_has_price_as_breakeven() -> None:
    assert hold_to_resolution_breakeven_probability(0.70, 0.07, fees_enabled=False) == 0.70
    assert probability_edge_after_entry_fee(0.74, 0.70, 0.07, fees_enabled=False) == 0.04


def test_fee_raises_probability_hurdle() -> None:
    no_fee_edge = probability_edge_after_entry_fee(0.74, 0.70, 0.07, fees_enabled=False)
    fee_edge = probability_edge_after_entry_fee(0.74, 0.70, 0.07, fees_enabled=True)
    assert fee_edge < no_fee_edge
