from __future__ import annotations

import math

from btc5m_v2.config import V2Config
from btc5m_v2.market import MarketInfo
from btc5m_v2.market_fees import MarketFeeSchedule
from btc5m_v2.online_paper import (
    PaperPosition,
    SessionRiskState,
    build_online_model_row,
    configured_signal_filter_reasons,
    evaluate_online_candidate,
    settle_position_payload,
)
from btc5m_v2.research.btc_features import BTCSample
from btc5m_v2.strategy.model import LogisticProbabilityModel

SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def cfg() -> V2Config:
    return V2Config(
        {
            "resolution": {"expected_source": SOURCE, "fail_if_missing": True, "fail_if_unexpected": True},
            "entry": {"seconds_left_min": 90, "seconds_left_max": 150},
            "market_quality": {
                "max_selected_side_spread": 0.02,
                "min_top_3_ask_notional_usd": 30,
                "max_order_book_participation_pct": 20,
            },
            "freshness": {"clob_max_age_sec": 2, "btc_max_age_sec": 3, "max_consecutive_data_errors": 3},
            "signal": {
                "minimum_absolute_btc_move_usd": 50,
                "minimum_impulse_z": 1.0,
                "require_30s_direction_agreement": True,
                "require_15s_direction_agreement": False,
                "minimum_model_probability": 0.58,
                "minimum_edge_after_fees_and_slippage": 0.04,
            },
            "risk": {
                "max_daily_loss_pct_equity": 3,
                "max_trades_per_day": 12,
                "max_consecutive_losses": 3,
            },
        }
    )


def snapshot(ts_ns: int, up_mid_shift: float = 0.0) -> dict:
    up_bid = 0.59 + up_mid_shift
    up_ask = 0.60 + up_mid_shift
    down_bid = 0.39 - up_mid_shift
    down_ask = 0.40 - up_mid_shift
    return {
        "slug": "btc-updown-5m-test",
        "condition_id": "cid",
        "resolution_source": SOURCE,
        "received_ts_ns": ts_ns,
        "seconds_left": 120.0,
        "up_bid": up_bid,
        "up_ask": up_ask,
        "up_spread": 0.01,
        "up_bid_depth_3_usd": 100.0,
        "up_ask_depth_3_usd": 100.0,
        "up_exchange_age_ms": 20.0,
        "down_bid": down_bid,
        "down_ask": down_ask,
        "down_spread": 0.01,
        "down_bid_depth_3_usd": 100.0,
        "down_ask_depth_3_usd": 100.0,
        "down_exchange_age_ms": 20.0,
    }


def test_online_feature_builder_is_causal_and_matches_training_names():
    now = 10_000_000_000_000
    history = [
        snapshot(now - 60_000_000_000, -0.04),
        snapshot(now - 30_000_000_000, -0.02),
        snapshot(now - 15_000_000_000, -0.01),
        snapshot(now - 5_000_000_000, -0.005),
        snapshot(now, 0.0),
    ]
    btc = [
        BTCSample(now - 60_000_000_000, 79_900.0),
        BTCSample(now - 30_000_000_000, 79_940.0),
        BTCSample(now - 15_000_000_000, 79_970.0),
        BTCSample(now, 80_000.0),
    ]
    row = build_online_model_row(history[-1], history, btc_samples=btc, reference_price=79_900.0)
    assert math.isclose(row["up_mid_change_60s"], 0.04, abs_tol=1e-12)
    assert math.isclose(row["up_mid_change_30s"], 0.02, abs_tol=1e-12)
    assert row["btc_delta_usd"] == 100.0
    assert row["btc_momentum_30s"] == 60.0
    assert row["btc_current"] == 80_000.0
    assert row["market_skew_up"] is not None

    future_history = history + [snapshot(now + 1_000_000_000, 0.25)]
    future_btc = btc + [BTCSample(now + 1_000_000_000, 90_000.0)]
    replay = build_online_model_row(
        history[-1], future_history, btc_samples=future_btc, reference_price=79_900.0
    )
    assert replay["up_mid_change_60s"] == row["up_mid_change_60s"]
    assert replay["btc_current"] == row["btc_current"]


def valid_row() -> dict:
    now = 10_000_000_000_000
    row = snapshot(now)
    row.update(
        {
            "btc_feed_age_ms": 100.0,
            "btc_delta_usd": 100.0,
            "btc_impulse_z": 1.5,
            "btc_momentum_15s": 15.0,
            "btc_momentum_30s": 30.0,
            "btc_momentum_60s": 60.0,
        }
    )
    return row


def test_btc_signal_quality_fails_closed():
    row = valid_row()
    assert configured_signal_filter_reasons(row, cfg()) == ()
    stale = dict(row, btc_feed_age_ms=4000.0)
    assert "stale_btc_feed" in configured_signal_filter_reasons(stale, cfg())
    weak = dict(row, btc_delta_usd=20.0)
    assert "btc_move_below_minimum" in configured_signal_filter_reasons(weak, cfg())
    reverse = dict(row, btc_momentum_30s=-10.0)
    assert "btc_momentum_30s_disagrees" in configured_signal_filter_reasons(reverse, cfg(), side="UP")


def test_online_candidate_uses_model_fee_and_full_book_edge():
    row = valid_row()
    model = LogisticProbabilityModel(
        feature_columns=("seconds_left",),
        means=(120.0,),
        scales=(1.0,),
        coefficients=(0.0,),
        intercept=2.2,
    )
    fees = MarketFeeSchedule(
        condition_id="cid",
        rate=0.07,
        exponent=1.0,
        taker_only=True,
        maker_base_fee_bps=0,
        taker_base_fee_bps=0,
    )
    decision, opportunity, reasons = evaluate_online_candidate(
        row,
        model=model,
        cfg=cfg(),
        fee_schedule=fees,
        up_asks=[(0.60, 100.0), (0.61, 100.0)],
        down_asks=[(0.40, 100.0)],
        notional_usd=5.0,
    )
    assert reasons == ()
    assert decision.trade and decision.side == "UP"
    assert opportunity is not None and opportunity.fill.fillable
    assert opportunity.edge_per_share is not None and opportunity.edge_per_share > 0.04
    assert opportunity.fee_usd > 0


def test_session_risk_and_settlement_accounting():
    state = SessionRiskState(starting_equity_usd=1000.0)
    state.trades_today = 12
    assert "max_trades_per_day" in state.gate_reasons(cfg())
    state.trades_today = 0
    state.realized_pnl_usd = -31.0
    assert "daily_loss_limit" in state.gate_reasons(cfg())
    state.realized_pnl_usd = 0.0
    state.consecutive_losses = 3
    assert "max_consecutive_losses" in state.gate_reasons(cfg())

    market = MarketInfo(
        slug="btc-updown-5m-test",
        condition_id="cid",
        up_token_id="up",
        down_token_id="down",
        end_iso="2030-01-01T00:05:00Z",
        end_ts=1_900_000_000.0,
        resolution_source=SOURCE,
    )
    model = LogisticProbabilityModel(
        feature_columns=("seconds_left",),
        means=(120.0,),
        scales=(1.0,),
        coefficients=(0.0,),
        intercept=2.2,
    )
    row = valid_row()
    fees = MarketFeeSchedule("cid", 0.0, 1.0, True, 0, 0)
    decision, opportunity, reasons = evaluate_online_candidate(
        row,
        model=model,
        cfg=cfg(),
        fee_schedule=fees,
        up_asks=[(0.60, 100.0)],
        down_asks=[(0.40, 100.0)],
        notional_usd=5.0,
    )
    assert not reasons and opportunity is not None
    position = PaperPosition.from_opportunity(
        market=market, row=row, model=model, decision=decision, opportunity=opportunity
    )
    settled = settle_position_payload(position, winning_side="UP")
    assert settled["status"] == "SETTLED"
    assert settled["won"] is True
    assert settled["net_pnl_usd"] > 0
    assert settled["order_path_used"] is False
