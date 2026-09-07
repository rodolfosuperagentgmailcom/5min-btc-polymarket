from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.btc_features import BTCSample, btc_features_at
from btc5m_v2.research.features import build_causal_features, build_feature_row, depth_imbalance, midpoint
from btc5m_v2.research.replay import (
    build_feature_dataset_rows,
    iter_replay_rows,
    write_feature_dataset,
)


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


def test_future_clob_rows_cannot_change_prior_causal_features():
    second = 1_000_000_000
    early = [
        sample_row(received_ts_ns=100 * second, up_bid=0.49, up_ask=0.51, down_bid=0.49, down_ask=0.51),
        sample_row(received_ts_ns=110 * second, up_bid=0.54, up_ask=0.56, down_bid=0.44, down_ask=0.46),
    ]
    future = sample_row(
        received_ts_ns=120 * second,
        up_bid=0.89,
        up_ask=0.91,
        down_bid=0.09,
        down_ask=0.11,
    )

    before = build_causal_features(early)
    after = build_causal_features([*early, future])

    assert after[:2] == before
    assert round(after[1]["up_mid_delta_5s"] or 0.0, 8) == 0.05
    assert round(after[1]["down_mid_delta_5s"] or 0.0, 8) == -0.05
    assert after[0]["up_mid_delta_5s"] is None


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


def test_btc_features_never_read_future_samples():
    base = 1_800_000_000_000_000_000
    samples = [
        BTCSample(base, 100.0),
        BTCSample(base + 15_000_000_000, 110.0),
        BTCSample(base + 30_000_000_000, 120.0),
        BTCSample(base + 45_000_000_000, 999.0),
    ]

    features = btc_features_at(
        samples,
        now_ns=base + 30_000_000_000,
        reference_price=90.0,
    )
    assert features["btc_price"] == 120.0
    assert features["btc_move_15s"] == 10.0
    assert features["btc_move_30s"] == 20.0
    assert features["btc_move_from_reference"] == 30.0
    assert features["btc_impulse_z"] is not None
    assert features["btc_impulse_z"] > 0


def test_feature_dataset_joins_btc_causally_then_attaches_terminal_label(tmp_path):
    base = 1_800_000_000_000_000_000
    market_dir = tmp_path / "2026-09-06" / "btc-updown-5m-test"
    market_dir.mkdir(parents=True)

    rows = [
        sample_row(received_ts_ns=base, seconds_left=150, up_bid=0.54, up_ask=0.55, down_bid=0.45, down_ask=0.46),
        sample_row(received_ts_ns=base + 15_000_000_000, seconds_left=135, up_bid=0.59, up_ask=0.60, down_bid=0.40, down_ask=0.41),
        sample_row(received_ts_ns=base + 30_000_000_000, seconds_left=120, up_bid=0.64, up_ask=0.65, down_bid=0.35, down_ask=0.36),
    ]
    pq.write_table(pa.Table.from_pylist(rows), market_dir / "clob_snapshots_part-000000.parquet")
    (market_dir / "resolution.json").write_text(
        json.dumps({"resolved": True, "winning_side": "UP"}),
        encoding="utf-8",
    )

    btc_samples = [
        BTCSample(base, 100_000.0),
        BTCSample(base + 15_000_000_000, 100_050.0),
        BTCSample(base + 30_000_000_000, 100_120.0),
        BTCSample(base + 45_000_000_000, 120_000.0),
    ]

    dataset = build_feature_dataset_rows(
        market_dir,
        seconds_left_min=90,
        seconds_left_max=150,
        btc_samples=btc_samples,
        btc_reference_price=100_000.0,
    )

    assert len(dataset) == 3
    assert dataset[-1]["btc_price"] == 100_120.0
    assert dataset[-1]["btc_move_15s"] == 70.0
    assert dataset[-1]["btc_move_30s"] == 120.0
    assert dataset[-1]["up_mid_delta_15s"] is not None
    assert dataset[-1]["target_up"] == 1
    assert dataset[-1]["winning_side"] == "UP"
    assert dataset[-1]["selected_side_by_mid_won"] is True

    output = write_feature_dataset(
        market_dir,
        btc_samples=btc_samples,
        btc_reference_price=100_000.0,
    )
    table = pq.read_table(output)
    assert table.num_rows == 3
    assert "btc_impulse_z" in table.column_names
    assert "target_up" in table.column_names
