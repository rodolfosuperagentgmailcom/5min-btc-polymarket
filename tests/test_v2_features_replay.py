from __future__ import annotations

import json
import math

from btc5m_v2.config import V2Config
from btc5m_v2.research.replay import (
    attach_resolution_labels,
    build_feature_row,
    reference_observation_asof,
)
from btc5m_v2.strategy.features import (
    PriceObservation,
    build_btc_features,
    momentum_usd,
    orderbook_imbalance,
    realized_vol_usd_sqrt_sec,
)


def cfg() -> V2Config:
    return V2Config(
        {
            "resolution": {
                "expected_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
                "fail_if_missing": True,
                "fail_if_unexpected": True,
            },
            "entry": {"seconds_left_min": 90, "seconds_left_max": 150},
            "market_quality": {
                "max_selected_side_spread": 0.02,
                "min_top_3_ask_notional_usd": 30,
            },
            "freshness": {
                "clob_max_age_sec": 2.0,
                "btc_max_age_sec": 3.0,
                "max_consecutive_data_errors": 3,
            },
        }
    )


def make_prices(start_ns: int, now_ns: int) -> list[PriceObservation]:
    rows = [
        PriceObservation(
            ts_ns=start_ns - 2_000_000_000,
            price=100.0,
            source_ts_ms=(start_ns // 1_000_000) - 2000,
        )
    ]
    price = 101.0
    ts = start_ns
    while ts <= now_ns:
        rows.append(PriceObservation(ts_ns=ts, price=price, source_ts_ms=ts // 1_000_000))
        ts += 5_000_000_000
        price += 1.0
    return rows


def snapshot(start_sec: int, now_ns: int) -> dict:
    return {
        "slug": f"btc-updown-5m-{start_sec}",
        "condition_id": "condition-1",
        "resolution_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
        "market_end": "2033-05-18T03:38:20Z",
        "received_ts_ns": now_ns,
        "received_iso": "2033-05-18T03:35:20Z",
        "event_type": "price_change",
        "event_exchange_ts_ms": now_ns // 1_000_000,
        "seconds_left": 120.0,
        "up_bid": 0.59,
        "up_ask": 0.60,
        "up_spread": 0.01,
        "up_bid_depth_3_usd": 80.0,
        "up_ask_depth_3_usd": 20.0,
        "up_exchange_ts_ms": now_ns // 1_000_000,
        "up_exchange_age_ms": 20.0,
        "down_bid": 0.40,
        "down_ask": 0.41,
        "down_spread": 0.01,
        "down_bid_depth_3_usd": 20.0,
        "down_ask_depth_3_usd": 80.0,
        "down_exchange_ts_ms": now_ns // 1_000_000,
        "down_exchange_age_ms": 30.0,
    }


def metadata() -> dict:
    return {
        "condition_id": "condition-1",
        "up_token_id": "up-token",
        "down_token_id": "down-token",
        "resolution_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
        "market_end": "2033-05-18T03:38:20Z",
    }


def test_momentum_and_volatility_are_causal():
    now_ns = 2_000_000_120_000_000_000
    start_ns = now_ns - 120_000_000_000
    prices = make_prices(start_ns, now_ns)
    prices.append(PriceObservation(ts_ns=now_ns + 1_000_000_000, price=99999.0))

    assert momentum_usd(prices, now_ns=now_ns, horizon_sec=15) == 3.0
    assert momentum_usd(prices, now_ns=now_ns, horizon_sec=30) == 6.0
    assert momentum_usd(prices, now_ns=now_ns, horizon_sec=60) == 12.0

    vol = realized_vol_usd_sqrt_sec(prices, now_ns=now_ns, window_sec=60)
    assert vol is not None
    assert math.isclose(vol, math.sqrt(12.0 / 60.0), rel_tol=1e-9)


def test_future_price_does_not_change_current_features():
    now_ns = 2_000_000_120_000_000_000
    start_ns = now_ns - 120_000_000_000
    prices = make_prices(start_ns, now_ns)

    base = build_btc_features(
        prices,
        now_ns=now_ns,
        reference_price=100.0,
        reference_ts_ns=start_ns,
    )
    future = build_btc_features(
        prices + [PriceObservation(ts_ns=now_ns + 1, price=1_000_000.0)],
        now_ns=now_ns,
        reference_price=100.0,
        reference_ts_ns=start_ns,
    )
    assert base == future


def test_reference_must_have_been_received_by_snapshot():
    start_ns = 2_000_000_000_000_000_000
    obs = PriceObservation(
        ts_ns=start_ns + 1_000_000_000,
        price=100.0,
        source_ts_ms=start_ns // 1_000_000,
    )
    assert reference_observation_asof(
        [obs], market_start_ts_ns=start_ns, now_ns=start_ns + 500_000_000
    ) is None
    assert reference_observation_asof(
        [obs], market_start_ts_ns=start_ns, now_ns=start_ns + 2_000_000_000
    ) == obs


def test_orderbook_imbalance_is_side_specific():
    assert orderbook_imbalance(80.0, 20.0) == 0.6
    assert orderbook_imbalance(20.0, 80.0) == -0.6
    assert orderbook_imbalance(0.0, 0.0) is None


def test_feature_row_is_deterministic_and_label_free():
    start_sec = 2_000_000_000
    start_ns = start_sec * 1_000_000_000
    now_ns = start_ns + 180_000_000_000
    prices = make_prices(start_ns, now_ns)
    row1 = build_feature_row(snapshot(start_sec, now_ns), metadata=metadata(), btc_observations=prices, config=cfg())
    row2 = build_feature_row(snapshot(start_sec, now_ns), metadata=metadata(), btc_observations=prices, config=cfg())

    assert row1 == row2
    assert row1["feature_ready"] is True
    assert row1["label_up"] is None
    assert row1["resolution_outcome"] is None
    assert row1["up_obi_3"] == 0.6
    assert row1["down_obi_3"] == -0.6
    assert row1["btc_current"] == prices[-1].price
    assert row1["btc_move_usd"] > 0


def test_missing_or_stale_inputs_fail_closed_with_reason():
    start_sec = 2_000_000_000
    start_ns = start_sec * 1_000_000_000
    now_ns = start_ns + 180_000_000_000
    stale = snapshot(start_sec, now_ns)
    stale["up_exchange_age_ms"] = 5000.0

    row = build_feature_row(stale, metadata=metadata(), btc_observations=[], config=cfg())
    assert row["feature_ready"] is False
    reasons = set((row["skip_reason"] or "").split(","))
    assert "missing_btc_current" in reasons
    assert "missing_btc_reference" in reasons
    assert "stale_up_book" in reasons


def test_posthoc_label_join_does_not_mutate_causal_fields():
    start_sec = 2_000_000_000
    start_ns = start_sec * 1_000_000_000
    now_ns = start_ns + 180_000_000_000
    row = build_feature_row(
        snapshot(start_sec, now_ns),
        metadata=metadata(),
        btc_observations=make_prices(start_ns, now_ns),
        config=cfg(),
    )
    original = json.dumps(row, sort_keys=True)
    labelled = attach_resolution_labels(
        [row],
        {"resolved": True, "winning_side": "UP", "source": "polymarket_ws"},
    )[0]

    assert labelled["snapshot_id"] == row["snapshot_id"]
    assert labelled["label_up"] is True
    assert labelled["resolution_outcome"] == "UP"
    assert row["label_up"] is None

    causal_copy = dict(labelled)
    causal_copy["label_up"] = None
    causal_copy["resolution_outcome"] = None
    causal_copy["label_source"] = None
    assert json.dumps(causal_copy, sort_keys=True) == original
