from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.dataset import (
    build_market_dataset_rows,
    write_dataset,
)

EXPECTED_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def _snapshot(ts_ns: int, *, up_mid: float, down_mid: float, seconds_left: float) -> dict:
    spread = 0.02
    return {
        "slug": "btc-updown-5m-test",
        "condition_id": "condition-test",
        "resolution_source": EXPECTED_SOURCE,
        "market_end": "2030-01-01T00:05:00Z",
        "received_ts_ns": ts_ns,
        "received_iso": "2030-01-01T00:00:00Z",
        "event_type": "book",
        "event_exchange_ts_ms": ts_ns // 1_000_000,
        "seconds_left": seconds_left,
        "up_bid": up_mid - spread / 2,
        "up_ask": up_mid + spread / 2,
        "up_spread": spread,
        "up_bid_depth_3_usd": 40.0,
        "up_ask_depth_3_usd": 60.0,
        "up_exchange_ts_ms": ts_ns // 1_000_000,
        "up_exchange_age_ms": 25.0,
        "down_bid": down_mid - spread / 2,
        "down_ask": down_mid + spread / 2,
        "down_spread": spread,
        "down_bid_depth_3_usd": 55.0,
        "down_ask_depth_3_usd": 45.0,
        "down_exchange_ts_ms": ts_ns // 1_000_000,
        "down_exchange_age_ms": 30.0,
    }


def _write_market(tmp_path: Path) -> Path:
    market_dir = tmp_path / "2030-01-01" / "btc-updown-5m-test"
    market_dir.mkdir(parents=True)
    start = 1_900_000_000_000_000_000
    rows = [
        _snapshot(start, up_mid=0.50, down_mid=0.50, seconds_left=150),
        _snapshot(start + 5_000_000_000, up_mid=0.55, down_mid=0.45, seconds_left=145),
        _snapshot(start + 10_000_000_000, up_mid=0.60, down_mid=0.40, seconds_left=140),
    ]
    pq.write_table(pa.Table.from_pylist(rows), market_dir / "clob_snapshots_part-000000.parquet")
    (market_dir / "metadata.json").write_text(
        json.dumps(
            {
                "slug": "btc-updown-5m-test",
                "condition_id": "condition-test",
                "up_token_id": "up-token",
                "down_token_id": "down-token",
                "market_end": "2030-01-01T00:05:00Z",
                "resolution_source": EXPECTED_SOURCE,
                "reconnects": 0,
                "resolution": {
                    "resolved": True,
                    "winning_side": "UP",
                    "source": "test",
                },
            }
        ),
        encoding="utf-8",
    )
    return market_dir


def _write_verified_btc(
    market_dir: Path,
    *,
    source: str = EXPECTED_SOURCE,
    reconnects: int = 0,
) -> None:
    start = 1_900_000_000_000_000_000
    samples = [
        {"ts_ns": start - 60_000_000_000, "price": 99_800.0},
        {"ts_ns": start - 30_000_000_000, "price": 99_900.0},
        {"ts_ns": start - 15_000_000_000, "price": 99_940.0},
        {"ts_ns": start - 5_000_000_000, "price": 99_950.0},
        {"ts_ns": start, "price": 100_000.0},
        {"ts_ns": start + 5_000_000_000, "price": 100_050.0},
        {"ts_ns": start + 10_000_000_000, "price": 100_120.0},
        {"ts_ns": start + 20_000_000_000, "price": 150_000.0},
    ]
    pq.write_table(pa.Table.from_pylist(samples), market_dir / "btc_samples.parquet")
    (market_dir / "btc_metadata.json").write_text(
        json.dumps(
            {
                "source": source,
                "reference_price": 100_000.0,
                "reference_verified": True,
                "reconnects": reconnects,
            }
        ),
        encoding="utf-8",
    )


def test_dataset_builds_causal_lags_and_labels(tmp_path):
    market_dir = _write_market(tmp_path)
    rows = build_market_dataset_rows(market_dir, sample_interval_ms=1000)
    assert len(rows) == 3

    first, second, third = rows
    assert first["up_probability_change_5s"] is None
    assert round(second["up_probability_change_5s"], 6) == 0.05
    assert round(third["up_probability_change_5s"], 6) == 0.05
    assert third["up_probability_change_15s"] is None

    assert third["recording_reconnects"] == 0
    assert third["resolved"] is True
    assert third["winning_side"] == "UP"
    assert third["label_up"] == 1
    assert third["selected_side"] == "UP"
    assert third["selected_won"] is True

    assert third["btc_current"] is None
    assert third["btc_impulse_z"] is None


def test_verified_chainlink_btc_samples_populate_dataset_without_future_leakage(tmp_path):
    market_dir = _write_market(tmp_path)
    _write_verified_btc(market_dir)

    rows = build_market_dataset_rows(market_dir, sample_interval_ms=1000)
    third = rows[-1]
    assert third["btc_reference"] == 100_000.0
    assert third["btc_current"] == 100_120.0
    assert third["btc_delta_usd"] == 120.0
    assert third["btc_momentum_15s"] == 170.0
    assert third["btc_momentum_30s"] == 220.0
    assert third["btc_momentum_60s"] == 320.0
    assert third["btc_realized_vol_30s"] is not None
    assert third["btc_realized_vol_60s"] is not None
    assert third["btc_impulse_z"] is not None
    assert third["btc_current"] != 150_000.0


def test_btc_source_mismatch_fails_closed(tmp_path):
    market_dir = _write_market(tmp_path)
    _write_verified_btc(market_dir, source="https://example.com/not-chainlink")

    try:
        build_market_dataset_rows(market_dir, sample_interval_ms=1000)
    except ValueError as exc:
        assert "btc_sample_source_mismatch" in str(exc)
    else:
        raise AssertionError("mismatched BTC source must fail closed")


def test_btc_reconnect_fails_closed(tmp_path):
    market_dir = _write_market(tmp_path)
    _write_verified_btc(market_dir, reconnects=1)

    try:
        build_market_dataset_rows(market_dir, sample_interval_ms=1000)
    except ValueError as exc:
        assert "btc_recording_reconnected:1" in str(exc)
    else:
        raise AssertionError("BTC recording reconnect must fail closed")


def test_dataset_write_roundtrip_and_summary(tmp_path):
    market_dir = _write_market(tmp_path)
    output = tmp_path / "features" / "btc5m_features.parquet"
    summary = write_dataset([market_dir], output, sample_interval_ms=1000)

    assert summary["rows"] == 3
    assert summary["resolved_rows"] == 3
    assert summary["markets"] == 1
    assert summary["btc_features_populated"] is False
    assert summary["recording_quality"] == "clob_and_present_btc_reconnects_required_zero"
    assert output.exists()

    table = pq.read_table(output)
    assert table.num_rows == 3
    row = table.to_pylist()[-1]
    assert row["label_up"] == 1
    assert row["recording_reconnects"] == 0
    assert round(row["up_mid_change_5s"], 6) == 0.05
    assert row["btc_reference"] is None

    sidecar = json.loads(output.with_suffix(".parquet.json").read_text(encoding="utf-8"))
    assert sidecar["rows"] == 3
