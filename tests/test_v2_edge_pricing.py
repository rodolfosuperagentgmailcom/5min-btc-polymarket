from __future__ import annotations

import math

from btc5m_v2.execution.pricing import estimate_buy_fill, estimate_sell_fill
from btc5m_v2.fees import fee_for_fills, taker_fee_usd
from btc5m_v2.strategy.edge import choose_best_edge, evaluate_buy_edge


def test_buy_fill_walks_selected_asks_and_returns_vwap():
    fill = estimate_buy_fill([(0.62, 10.0), (0.60, 5.0)], notional_usd=6.10)

    assert fill.fillable is True
    assert math.isclose(fill.spent_usd, 6.10)
    assert math.isclose(fill.shares, 10.0)
    assert math.isclose(fill.average_price or 0.0, 0.61)
    assert fill.worst_price == 0.62
    assert len(fill.fills) == 2
    assert math.isclose(fill.visible_ask_notional_usd, 9.20)
    assert math.isclose(fill.participation_pct or 0.0, 6.10 / 9.20 * 100.0)


def test_buy_fill_rejects_fake_full_fill_when_visible_depth_is_insufficient():
    fill = estimate_buy_fill([(0.50, 10.0)], notional_usd=8.0)

    assert fill.fillable is False
    assert fill.spent_usd == 5.0
    assert fill.shares == 10.0
    assert fill.average_price == 0.50


def test_sell_fill_walks_highest_bids_first():
    fill = estimate_sell_fill([(0.58, 10.0), (0.60, 5.0)], shares_to_sell=10.0)

    assert fill.fillable is True
    assert fill.sold_shares == 10.0
    assert math.isclose(fill.proceeds_usd, 5.90)
    assert math.isclose(fill.average_price or 0.0, 0.59)
    assert fill.worst_price == 0.58


def test_documented_fee_curve_is_applied_per_execution_level():
    fill = estimate_buy_fill([(0.60, 5.0), (0.62, 10.0)], notional_usd=6.10)
    fee = fee_for_fills(fill.fills, fee_rate=0.07, fees_enabled=True, is_taker=True)

    expected = 5.0 * 0.07 * 0.60 * 0.40 + 5.0 * 0.07 * 0.62 * 0.38
    assert math.isclose(fee, expected)
    assert math.isclose(fee, 0.16646)


def test_maker_and_fee_disabled_paths_charge_zero():
    assert taker_fee_usd(
        shares=10.0, price=0.50, fee_rate=0.07, fees_enabled=False, is_taker=True
    ) == 0.0
    assert taker_fee_usd(
        shares=10.0, price=0.50, fee_rate=0.07, fees_enabled=True, is_taker=False
    ) == 0.0


def test_all_in_breakeven_includes_fee_and_vwap():
    opportunity = evaluate_buy_edge(
        side="UP",
        model_probability=0.70,
        asks=[(0.60, 5.0), (0.62, 10.0)],
        notional_usd=6.10,
        fees_enabled=True,
        fee_rate=0.07,
    )

    assert opportunity.fill.fillable is True
    assert math.isclose(opportunity.fee_usd, 0.16646)
    assert math.isclose(opportunity.all_in_cost_per_share or 0.0, 0.626646)
    assert math.isclose(opportunity.breakeven_probability or 0.0, 0.626646)
    assert math.isclose(opportunity.edge_per_share or 0.0, 0.073354)
    assert math.isclose(opportunity.expected_value_usd or 0.0, 0.73354)


def test_edge_selection_uses_model_edge_not_highest_contract_ask():
    up = evaluate_buy_edge(
        side="UP",
        model_probability=0.80,
        asks=[(0.75, 100.0)],
        notional_usd=5.0,
        fees_enabled=True,
        fee_rate=0.07,
    )
    down = evaluate_buy_edge(
        side="DOWN",
        model_probability=0.64,
        asks=[(0.55, 100.0)],
        notional_usd=5.0,
        fees_enabled=True,
        fee_rate=0.07,
    )

    assert (up.fill.average_price or 0.0) > (down.fill.average_price or 0.0)
    assert (down.edge_per_share or 0.0) > (up.edge_per_share or 0.0)

    decision = choose_best_edge(
        up,
        down,
        minimum_model_probability=0.58,
        minimum_edge_per_share=0.03,
    )
    assert decision.action == "BUY"
    assert decision.side == "DOWN"


def test_book_participation_gate_blocks_oversized_order():
    opportunity = evaluate_buy_edge(
        side="UP",
        model_probability=0.90,
        asks=[(0.50, 20.0)],
        notional_usd=5.0,
        fees_enabled=False,
        fee_rate=0.0,
        max_book_participation_pct=20.0,
    )
    assert opportunity.fill.participation_pct == 50.0
    assert "book_participation_too_high" in opportunity.reasons


def test_choose_best_edge_returns_no_trade_below_threshold():
    up = evaluate_buy_edge(
        side="UP",
        model_probability=0.61,
        asks=[(0.60, 100.0)],
        notional_usd=5.0,
        fees_enabled=False,
        fee_rate=0.0,
    )
    down = evaluate_buy_edge(
        side="DOWN",
        model_probability=0.39,
        asks=[(0.40, 100.0)],
        notional_usd=5.0,
        fees_enabled=False,
        fee_rate=0.0,
    )

    decision = choose_best_edge(
        up,
        down,
        minimum_model_probability=0.58,
        minimum_edge_per_share=0.04,
    )
    assert decision.action == "NO_TRADE"
    assert decision.side is None
    assert "up:edge_below_minimum" in decision.reasons


def test_unfillable_side_cannot_be_selected_even_with_high_model_probability():
    up = evaluate_buy_edge(
        side="UP",
        model_probability=0.99,
        asks=[(0.60, 1.0)],
        notional_usd=5.0,
        fees_enabled=False,
        fee_rate=0.0,
    )
    down = evaluate_buy_edge(
        side="DOWN",
        model_probability=0.50,
        asks=[(0.50, 100.0)],
        notional_usd=5.0,
        fees_enabled=False,
        fee_rate=0.0,
    )

    decision = choose_best_edge(
        up,
        down,
        minimum_model_probability=0.58,
        minimum_edge_per_share=0.04,
    )
    assert decision.action == "NO_TRADE"
    assert "up:insufficient_visible_ask_depth" in decision.reasons
