from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.features import attach_resolution_labels, build_causal_features
from btc5m_v2.research.replay_v2 import build_market_replay


def snapshot(ts_sec: float, up_mid: float, down_mid: float) -> dict:
    ts_ns = int(ts_sec * 1_000_000_000)
    return {
        "slug": "btc-updown-5m-test",
        "condition_id": "condition-test",
        "resolution_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
        "market_end": "2030-01-01T00:05:00Z",
        "received_ts_ns": ts_ns,
        "received_iso": "2030-01-01T00:00:00Z",
        "event_type": "book",
        "event_exchange_ts_ms": int(ts_sec * 1000),
        "seconds_left": 120.0,
        "up_bid": up_mid - 0.01,
        "up_ask": up_mid + 0.01,
        "up_spread": 0.02,
        "up_bid_depth_3_usd": 60.0,
        "up_ask_depth_3_usd": 40.0,
        "up_exchange_ts_ms": int(ts_sec * 1000),
        "up_exchange_age_ms": 10.0,
        "down_bid": down_mid - 0.01,
        "down_ask": down_mid + 0.01,
        "down_spread": 0.02,
        "down_bid_depth_3_usd": 40.0,
        "down_ask_depth_3_usd": 60.0,
        "down_exchange_ts_ms": int(ts_sec * 1000),
        "down_exchange_age_ms": 12.0,
    }


def test_future_rows_do_not_change_past_features() -> None:
    prefix = [snapshot(100, 0.50, 0.50), snapshot(106, 0.55, 0.45)]
    future = snapshot(112, 0.95, 0.05)
    prefix_features = build_causal_features(prefix)
    full_features = build_causal_features(prefix + [future])
    assert prefix_features == full_features[:2]
    assert round(prefix_features[1]["up_mid_delta_5s"], 8) == 0.05
    assert round(prefix_features[1]["down_mid_delta_5s"], 8) == -0.05


def test_labels_are_attached_only_after_feature_generation() -> None:
    features = build_causal_features([snapshot(100, 0.60, 0.40)])
    assert "winning_side" not in features[0]
    assert "target_up" not in features[0]
    assert round(features[0]["up_book_imbalance_3"], 8) == 0.2
    assert round(features[0]["down_book_imbalance_3"], 8) == -0.2
    assert round(features[0]["sum_best_asks"], 8) == 1.02
    assert round(features[0]["sum_best_bids"], 8) == 0.98

    labeled = attach_resolution_labels(features, winning_side="UP")
    assert labeled[0]["winning_side"] == "UP"
    assert labeled[0]["target_up"] == 1
    assert labeled[0]["selected_side_by_mid_won"] is True


def test_lag_is_latest_snapshot_at_or_before_cutoff() -> None:
    rows = [snapshot(100, 0.50, 0.50), snapshot(103, 0.52, 0.48), snapshot(106, 0.56, 0.44)]
    features = build_causal_features(rows)
    assert round(features[2]["up_mid_delta_5s"], 8) == 0.06
    assert round(features[2]["up_mid_velocity_5s_per_sec"], 8) == 0.01


def test_replay_builder_writes_labeled_parquet(tmp_path: Path) -> None:
    market_dir = tmp_path / "2030-01-01" / "btc-updown-5m-test"
    market_dir.mkdir(parents=True)
    rows = [snapshot(100, 0.50, 0.50), snapshot(106, 0.55, 0.45)]
    pq.write_table(pa.Table.from_pylist(rows), market_dir / "clob_snapshots_part-000000.parquet")
    (market_dir / "resolution.json").write_text(
        json.dumps({"resolved": True, "winning_side": "UP", "source": "test"}),
        encoding="utf-8",
    )

    result = build_market_replay(market_dir)
    assert result.resolved is True
    assert result.winning_side == "UP"
    assert result.rows == 2
    output = pq.read_table(result.output_path).to_pylist()
    assert output[0]["target_up"] == 1
    assert round(output[1]["up_mid_delta_5s"], 8) == 0.05
