from __future__ import annotations

from btc5m_v2.research.benchmarks import benchmark_settlement_pnl, legacy_threshold_decision
from btc5m_v2.research.economics import breakeven_probability, executable_buy_fill, executable_sell_fill, fee_per_share
from btc5m_v2.research.features import build_feature_row
from btc5m_v2.research.replay import ReplayRow


def replay(up_ask=0.72, down_ask=0.30, seconds_left=120.0, winner="UP"):
    row = {
        "slug": "btc-updown-5m-test",
        "received_ts_ns": 1,
        "seconds_left": seconds_left,
        "up_bid": max(0.0, up_ask - 0.02),
        "up_ask": up_ask,
        "up_spread": 0.02,
        "up_bid_depth_3_usd": 50.0,
        "up_ask_depth_3_usd": 50.0,
        "up_exchange_age_ms": 100.0,
        "down_bid": max(0.0, down_ask - 0.02),
        "down_ask": down_ask,
        "down_spread": 0.02,
        "down_bid_depth_3_usd": 50.0,
        "down_ask_depth_3_usd": 50.0,
        "down_exchange_age_ms": 100.0,
    }
    return ReplayRow(build_feature_row(row), winner, winner in {"UP", "DOWN"})


def test_fee_math_matches_binary_fee_formula():
    fee = fee_per_share(0.70, 0.07, True)
    assert abs(fee - 0.0147) < 1e-12
    assert abs(breakeven_probability(0.70, 0.07, True) - 0.7147) < 1e-12
    assert breakeven_probability(0.70, 0.07, False) == 0.70


def test_executable_buy_fill_uses_multiple_levels_and_rejects_insufficient_depth():
    fill = executable_buy_fill([(0.60, 5), (0.62, 10)], 6.0)
    assert fill.fillable is True
    assert fill.average_price is not None
    assert fill.average_price > 0.60
    assert fill.worst_price == 0.62

    partial = executable_buy_fill([(0.60, 2)], 5.0)
    assert partial.fillable is False


def test_executable_sell_fill_walks_best_bids_first():
    fill = executable_sell_fill([(0.58, 5), (0.60, 4)], 6)
    assert fill.fillable is True
    assert fill.worst_price == 0.58
    assert fill.average_price is not None
    assert fill.average_price < 0.60


def test_legacy_benchmark_reproduces_stronger_side_threshold_behavior():
    decision = legacy_threshold_decision(replay(up_ask=0.71, down_ask=0.74))
    assert decision.trade is True
    assert decision.side == "DOWN"
    assert decision.entry_price == 0.74


def test_legacy_benchmark_can_be_constrained_to_research_window():
    decision = legacy_threshold_decision(replay(seconds_left=200), seconds_left_min=90, seconds_left_max=150)
    assert decision.trade is False
    assert decision.reason == "too_early"


def test_settlement_pnl_is_per_share_and_fee_aware():
    decision = legacy_threshold_decision(replay(up_ask=0.72, down_ask=0.30, winner="UP"))
    fee = fee_per_share(decision.entry_price or 0.0, 0.07, True)
    pnl = benchmark_settlement_pnl(decision, winning_side="UP", fee_per_share_paid=fee)
    assert pnl is not None
    assert pnl < 0.28
    assert pnl > 0
