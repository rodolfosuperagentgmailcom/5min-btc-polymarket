from __future__ import annotations

from btc5m_v2.config import V2Config
from btc5m_v2.strategy.signal import decide_snapshot

EXPECTED_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def cfg() -> V2Config:
    return V2Config(
        {
            "resolution": {
                "expected_source": EXPECTED_SOURCE,
                "fail_if_missing": True,
                "fail_if_unexpected": True,
            },
            "entry": {"seconds_left_min": 90, "seconds_left_max": 150},
            "market_quality": {
                "max_selected_side_spread": 0.02,
                "min_top_3_ask_notional_usd": 30,
            },
            "freshness": {"clob_max_age_sec": 2, "max_consecutive_data_errors": 3},
            "signal": {
                "minimum_model_probability": 0.58,
                "minimum_edge_after_fees_and_slippage": 0.04,
            },
        }
    )


def row(**overrides):
    payload = {
        "seconds_left": 120,
        "resolution_source": EXPECTED_SOURCE,
        "up_bid": 0.58,
        "up_ask": 0.60,
        "up_spread": 0.02,
        "up_ask_depth_3_usd": 100,
        "up_exchange_age_ms": 100,
        "down_bid": 0.38,
        "down_ask": 0.40,
        "down_spread": 0.02,
        "down_ask_depth_3_usd": 100,
        "down_exchange_age_ms": 100,
    }
    payload.update(overrides)
    return payload


def test_decision_selects_positive_fee_adjusted_edge():
    decision = decide_snapshot(row(), q_up=0.70, cfg=cfg(), taker_fee_rate=0.07, fees_enabled=True)
    assert decision.trade is True
    assert decision.side == "UP"
    assert decision.edge is not None and decision.edge > 0.04


def test_decision_checks_selected_sides_own_spread():
    decision = decide_snapshot(
        row(up_spread=0.03, down_spread=0.01, up_ask=0.60, down_ask=0.40),
        q_up=0.70,
        cfg=cfg(),
    )
    assert decision.trade is False
    assert "spread_too_wide" in decision.up.reasons


def test_decision_fails_closed_on_stale_quote():
    decision = decide_snapshot(row(up_exchange_age_ms=2501), q_up=0.70, cfg=cfg())
    assert decision.trade is False
    assert "stale_clob_quote" in decision.up.reasons


def test_decision_fails_closed_on_changed_resolution_source():
    decision = decide_snapshot(
        row(resolution_source="https://example.com/wrong"),
        q_up=0.70,
        cfg=cfg(),
    )
    assert decision.trade is False
    assert "unexpected_resolution_source" in decision.reasons


def test_decision_blocks_small_edge_even_with_high_raw_probability():
    decision = decide_snapshot(row(up_ask=0.69, up_bid=0.67), q_up=0.72, cfg=cfg())
    assert decision.trade is False
    assert "edge_too_small" in decision.up.reasons


def test_down_side_is_evaluated_symmetrically():
    decision = decide_snapshot(
        row(up_ask=0.35, up_bid=0.33, down_ask=0.55, down_bid=0.53),
        q_up=0.25,
        cfg=cfg(),
    )
    assert decision.trade is True
    assert decision.side == "DOWN"


def test_entry_window_is_enforced():
    early = decide_snapshot(row(seconds_left=151), q_up=0.70, cfg=cfg())
    late = decide_snapshot(row(seconds_left=89), q_up=0.70, cfg=cfg())
    assert early.trade is False and "too_early" in early.reasons
    assert late.trade is False and "too_late" in late.reasons
