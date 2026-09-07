from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.readiness import assess_market_readiness, readiness_report

EXPECTED_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def _write_market(
    tmp_path,
    *,
    slug: str,
    clob_reconnects: int = 0,
    btc_reconnects: int = 0,
    seconds_left: float = 120.0,
):
    market_dir = tmp_path / "2030-01-01" / slug
    market_dir.mkdir(parents=True)
    (market_dir / "metadata.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "resolution_source": EXPECTED_SOURCE,
                "reconnects": clob_reconnects,
                "resolution": {"resolved": True, "winning_side": "UP"},
            }
        ),
        encoding="utf-8",
    )
    (market_dir / "clob_raw.jsonl").write_text("{}\n", encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "received_ts_ns": 1,
                    "seconds_left": float(seconds_left),
                    "up_ask": 0.60,
                    "down_ask": 0.40,
                }
            ]
        ),
        market_dir / "clob_snapshots_part-000000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"ts_ns": 1, "price": 100_000.0}]),
        market_dir / "btc_samples.parquet",
    )
    (market_dir / "btc_metadata.json").write_text(
        json.dumps(
            {
                "source": EXPECTED_SOURCE,
                "reconnects": btc_reconnects,
                "reference_verified": True,
                "reference_price": 100_000.0,
            }
        ),
        encoding="utf-8",
    )
    (market_dir / "fees.json").write_text(
        json.dumps({"fees_enabled": True, "rate": 0.07, "exponent": 1.0}),
        encoding="utf-8",
    )
    return market_dir


def test_ready_market_passes_all_preflight_checks(tmp_path):
    market_dir = _write_market(tmp_path, slug="btc-updown-5m-ready")
    result = assess_market_readiness(market_dir)

    assert result["v2_ready"] is True
    assert result["reasons"] == []
    assert result["clob_reconnects"] == 0
    assert result["decision_window_snapshot_present"] is True
    assert result["btc_reconnects"] == 0
    assert result["btc_reference_verified"] is True
    assert result["fee_ready"] is True


def test_reconnects_are_reported_as_blockers(tmp_path):
    market_dir = _write_market(
        tmp_path,
        slug="btc-updown-5m-bad",
        clob_reconnects=1,
        btc_reconnects=2,
    )
    result = assess_market_readiness(market_dir)

    assert result["v2_ready"] is False
    assert "clob_recording_reconnected" in result["reasons"]
    assert "btc_recording_reconnected" in result["reasons"]


def test_market_without_decision_window_snapshot_is_blocked(tmp_path):
    market_dir = _write_market(
        tmp_path,
        slug="btc-updown-5m-partial",
        seconds_left=240.0,
    )
    result = assess_market_readiness(market_dir)

    assert result["v2_ready"] is False
    assert result["decision_window_snapshot_present"] is False
    assert "missing_decision_window_snapshot" in result["reasons"]


def test_root_report_counts_ready_and_blocked_markets(tmp_path):
    _write_market(tmp_path, slug="btc-updown-5m-ready")
    _write_market(tmp_path, slug="btc-updown-5m-bad", btc_reconnects=1)
    _write_market(tmp_path, slug="btc-updown-5m-partial", seconds_left=240.0)

    report = readiness_report(tmp_path)
    assert report["markets_seen"] == 3
    assert report["v2_ready_markets"] == 1
    assert report["v2_ready_fraction"] == 1 / 3
    assert report["decision_window_markets"] == 2
    assert report["reason_counts"]["btc_recording_reconnected"] == 1
    assert report["reason_counts"]["missing_decision_window_snapshot"] == 1
