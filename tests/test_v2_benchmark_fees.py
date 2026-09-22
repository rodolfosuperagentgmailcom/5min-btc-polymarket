from __future__ import annotations

from btc5m_v2.research.benchmark import (
    LegacyTrade,
    benchmark_legacy_threshold_net,
    fee_adjust_legacy_trade,
    first_legacy_trade_for_market,
)
from btc5m_v2.research.fees import (
    hold_to_resolution_breakeven_probability,
    taker_fee_usdc,
)


def resolved_row(slug: str, ts: int, *, seconds_left: float, up_ask: float, down_ask: float, winner: str):
    return {
        "slug": slug,
        "received_ts_ns": ts,
        "seconds_left": seconds_left,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "resolved": True,
        "winning_side": winner,
    }


def test_current_crypto_taker_fee_formula_at_50_and_70_cents():
    assert round(taker_fee_usdc(100, 0.50, 0.07), 8) == 1.75
    assert round(taker_fee_usdc(100, 0.70, 0.07), 8) == 1.47
    assert round(taker_fee_usdc(100, 0.30, 0.07), 8) == 1.47


def test_70_cent_hold_breakeven_is_7147_with_crypto_taker_fee():
    value = hold_to_resolution_breakeven_probability(0.70, 0.07)
    assert round(value, 8) == 0.7147


def test_legacy_trigger_uses_first_qualifying_row_and_stronger_ask():
    rows = [
        resolved_row("m1", 1, seconds_left=180, up_ask=0.69, down_ask=0.31, winner="UP"),
        resolved_row("m1", 2, seconds_left=170, up_ask=0.71, down_ask=0.72, winner="DOWN"),
        resolved_row("m1", 3, seconds_left=160, up_ask=0.90, down_ask=0.10, winner="UP"),
    ]
    trade = first_legacy_trade_for_market(rows)
    assert trade is not None
    assert trade.received_ts_ns == 2
    assert trade.side == "DOWN"
    assert trade.entry_ask == 0.72
    assert trade.won is True


def test_fee_adjustment_reduces_net_pnl_without_changing_signal():
    gross = LegacyTrade(
        slug="m1",
        received_ts_ns=1,
        seconds_left=120,
        side="UP",
        entry_ask=0.70,
        winning_side="UP",
        won=True,
        gross_pnl_per_share=0.30,
        gross_return_on_cost=0.30 / 0.70,
    )
    net = fee_adjust_legacy_trade(gross, fee_rate=0.07, fees_enabled=True)
    assert net is not None
    assert net.side == gross.side
    assert round(net.entry_fee_per_share, 8) == 0.0147
    assert round(net.cash_cost_per_share, 8) == 0.7147
    assert round(net.net_pnl_per_share, 8) == 0.2853


def test_optional_slippage_is_separate_from_fee():
    gross = LegacyTrade(
        slug="m1",
        received_ts_ns=1,
        seconds_left=120,
        side="UP",
        entry_ask=0.70,
        winning_side="UP",
        won=True,
        gross_pnl_per_share=0.30,
        gross_return_on_cost=0.30 / 0.70,
    )
    net = fee_adjust_legacy_trade(
        gross,
        fee_rate=0.07,
        fees_enabled=True,
        entry_slippage=0.005,
    )
    assert net is not None
    assert net.observed_ask == 0.70
    assert net.executed_price == 0.705
    assert net.slippage_per_share == 0.005
    assert net.net_pnl_per_share < 0.2853


def test_fee_aware_benchmark_reports_net_control_metrics():
    rows = [
        resolved_row("m1", 1, seconds_left=120, up_ask=0.70, down_ask=0.30, winner="UP"),
        resolved_row("m2", 2, seconds_left=120, up_ask=0.30, down_ask=0.70, winner="UP"),
    ]
    trades, summary = benchmark_legacy_threshold_net(rows, fee_rate=0.07)
    assert len(trades) == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["fees_included"] is True
    assert summary["fill_model_included"] is False
    assert summary["net_pnl_per_share_total"] < summary["gross_pnl_per_share_total"]
