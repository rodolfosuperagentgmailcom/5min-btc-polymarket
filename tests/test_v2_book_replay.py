from __future__ import annotations

import json
import math

from btc5m_v2.execution.pricing import estimate_buy_fill
from btc5m_v2.research.book_replay import (
    reconstruct_and_align_books,
    reconstruct_book_snapshots,
    validate_snapshot_alignment,
)


def _write_raw(path):
    rows = [
        {
            "received_ts_ns": 1_000_000_000,
            "event_type": "book",
            "event": {
                "event_type": "book",
                "asset_id": "up-token",
                "timestamp": "1000",
                "bids": [{"price": "0.58", "size": "10"}],
                "asks": [
                    {"price": "0.60", "size": "5"},
                    {"price": "0.62", "size": "10"},
                ],
            },
        },
        {
            "received_ts_ns": 2_000_000_000,
            "event_type": "book",
            "event": {
                "event_type": "book",
                "asset_id": "down-token",
                "timestamp": "2000",
                "bids": [{"price": "0.38", "size": "10"}],
                "asks": [{"price": "0.40", "size": "20"}],
            },
        },
        {
            "received_ts_ns": 3_000_000_000,
            "event_type": "price_change",
            "event": {
                "event_type": "price_change",
                "timestamp": "3000",
                "price_changes": [
                    {
                        "asset_id": "up-token",
                        "side": "SELL",
                        "price": "0.60",
                        "size": "0",
                    }
                ],
            },
        },
        {
            "received_ts_ns": 3_000_000_000,
            "event_type": "price_change",
            "event": {
                "event_type": "price_change",
                "timestamp": "3001",
                "price_changes": [
                    {
                        "asset_id": "up-token",
                        "side": "SELL",
                        "price": "0.61",
                        "size": "8",
                    }
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_reconstruction_preserves_same_timestamp_event_order(tmp_path):
    raw = tmp_path / "clob_raw.jsonl"
    _write_raw(raw)

    snapshots = reconstruct_book_snapshots(
        raw,
        up_token_id="up-token",
        down_token_id="down-token",
    )

    assert len(snapshots) == 3
    assert snapshots[0].up_best_ask == 0.60
    assert snapshots[1].up_best_ask == 0.62
    assert snapshots[2].up_best_ask == 0.61
    assert snapshots[1].received_ts_ns == snapshots[2].received_ts_ns
    assert snapshots[1].event_exchange_ts_ms == 3_000_000
    assert snapshots[2].event_exchange_ts_ms == 3_001_000


def test_replayed_exact_asks_drive_real_vwap(tmp_path):
    raw = tmp_path / "clob_raw.jsonl"
    _write_raw(raw)

    snapshots = reconstruct_book_snapshots(
        raw,
        up_token_id="up-token",
        down_token_id="down-token",
    )
    first = snapshots[0]
    fill = estimate_buy_fill(first.asks("UP"), notional_usd=6.10)

    assert fill.fillable is True
    assert math.isclose(fill.average_price or 0.0, 0.61)
    assert fill.worst_price == 0.62


def test_alignment_matches_recorded_snapshot_sequence(tmp_path):
    raw = tmp_path / "clob_raw.jsonl"
    _write_raw(raw)
    recorded = [
        {
            "received_ts_ns": 2_000_000_000,
            "event_type": "book",
            "event_exchange_ts_ms": 2_000_000,
        },
        {
            "received_ts_ns": 3_000_000_000,
            "event_type": "price_change",
            "event_exchange_ts_ms": 3_000_000,
        },
        {
            "received_ts_ns": 3_000_000_000,
            "event_type": "price_change",
            "event_exchange_ts_ms": 3_001_000,
        },
    ]

    aligned = reconstruct_and_align_books(
        raw,
        recorded_rows=recorded,
        up_token_id="up-token",
        down_token_id="down-token",
    )
    assert len(aligned) == 3
    assert aligned[2].up_best_ask == 0.61


def test_alignment_fails_closed_on_sequence_mismatch(tmp_path):
    raw = tmp_path / "clob_raw.jsonl"
    _write_raw(raw)
    replayed = reconstruct_book_snapshots(
        raw,
        up_token_id="up-token",
        down_token_id="down-token",
    )
    recorded = [
        {
            "received_ts_ns": 2_000_000_000,
            "event_type": "book",
            "event_exchange_ts_ms": 2_000_000,
        },
        {
            "received_ts_ns": 3_000_000_000,
            "event_type": "price_change",
            "event_exchange_ts_ms": 9_999_000,
        },
        {
            "received_ts_ns": 3_000_000_000,
            "event_type": "price_change",
            "event_exchange_ts_ms": 3_001_000,
        },
    ]

    try:
        validate_snapshot_alignment(recorded, replayed)
    except ValueError as exc:
        assert "book_snapshot_alignment_mismatch" in str(exc)
    else:
        raise AssertionError("mismatched replay sequence must fail closed")
