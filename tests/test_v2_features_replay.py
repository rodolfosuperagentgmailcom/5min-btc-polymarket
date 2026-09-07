from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.features import build_feature_row, depth_imbalance, midpoint
from btc5m_v2.research.replay import iter_replay_rows


def sample_row(**overrides):
    row = {
        "slug": "btc-updown-5m-test",
        "received_ts_ns": 1000,
        "seconds_left": 120.0,
        "up_bid": 0.62,
        "up_ask": 0.64,
        "up_spread": 0.02,
        "up_bid_depth_3_usd": 80.0,
        "up_ask_depth_3_usd": 40.0,
        "up_exchange_age_ms": 100.0,
        "down_bid": 0.36,
        "down_ask": 0.38,
        "down_spread": 0.02,
        "down_bid_depth_3_usd": 30.0,
        "down_ask_depth_3_usd": 60.0,
        "down_exchange_age_ms": 120.0,
    }
    row.update(overrides)
    return row


def test_midpoint_and_depth_imbalance_are_bounded_and_causal():
    assert midpoint(0.60, 0.64) == 0.62
    assert midpoint(0.70, 0.60) is None
    assert round(depth_imbalance(75, 25) or 0, 6) == 0.5
    assert depth_imbalance(0, 0) is None


def test_feature_builder_uses_selected_sides_own_spread_and_depth():
    feature = build_feature_row(sample_row(up_ask=0.71, down_ask=0.31, up_spread=0.019, down_spread=0.005))
    assert feature.selected_side == "UP"
    assert feature.selected_ask == 0.71
    assert feature.selected_spread == 0.019
    assert feature.selected_depth_3_usd == 40.0
    assert feature.selected_exchange_age_ms == 100.0


def test_normalized_market_probabilities_sum_to_one():
    feature = build_feature_row(sample_row())
    assert feature.up_market_probability is not None
    assert feature.down_market_probability is not None
    assert abs(feature.up_market_probability + feature.down_market_probability - 1.0) < 1e-12


def test_equal_asks_do_not_force_a_direction():
    feature = build_feature_row(sample_row(up_ask=0.50, down_ask=0.50))
    assert feature.selected_side is None
    assert feature.selected_ask is None


def test_replay_filters_entry_window_and_attaches_only_final_label(tmp_path):
    market_dir = tmp_path / "2026-09-06" / "btc-updown-5m-test"
    market_dir.mkdir(parents=True)
    rows = [sample_row(received_ts_ns=1, seconds_left=151), sample_row(received_ts_ns=2, seconds_left=120), sample_row(received_ts_ns=3, seconds_left=89)]
    pq.write_table(pa.Table.from_pylist(rows), market_dir / "clob_snapshots_part-000000.parquet")
    (market_dir / "resolution.json").write_text(json.dumps({"resolved": True, "winning_side": "UP"}), encoding="utf-8")

    replay = list(iter_replay_rows(market_dir, seconds_left_min=90, seconds_left_max=150))
    assert len(replay) == 1
    assert replay[0].feature.received_ts_ns == 2
    assert replay[0].label_up == 1
    assert replay[0].selected_won is True


def test_unresolved_market_has_no_training_label(tmp_path):
    market_dir = tmp_path / "2026-09-06" / "btc-updown-5m-test"
    market_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([sample_row()]), market_dir / "clob_snapshots_part-000000.parquet")

    replay = list(iter_replay_rows(market_dir))
    assert len(replay) == 1
    assert replay[0].resolved is False
    assert replay[0].label_up is None
    assert replay[0].selected_won is None
