from __future__ import annotations

import json
from dataclasses import dataclass

from btc5m_v2.research.study import (
    compact_readiness,
    study_duration_sec,
    study_summary,
    write_study_summary,
)


@dataclass
class FakeClob:
    raw_events: int
    snapshots: int
    reconnects: int = 0


@dataclass
class FakeBtc:
    samples: int
    reconnects: int = 0
    reference_price: float | None = None


def test_study_duration_covers_requested_market_count_plus_tail():
    assert study_duration_sec(1, tail_sec=1.5) == 301.5
    assert study_duration_sec(12, tail_sec=2.0) == 3602.0
    try:
        study_duration_sec(0)
    except ValueError as exc:
        assert "markets must be positive" in str(exc)
    else:
        raise AssertionError("zero markets must be rejected")


def test_compact_readiness_preserves_research_gate_counts():
    compact = compact_readiness(
        {
            "markets_seen": 10,
            "v2_ready_markets": 7,
            "v2_ready_fraction": 0.7,
            "resolved_markets": 9,
            "decision_window_markets": 8,
            "verified_btc_reference_markets": 7,
            "fee_ready_markets": 10,
            "reason_counts": {"btc_reference_unverified": 2},
            "markets": [{"large": "detail"}],
        }
    )
    assert compact["v2_ready_markets"] == 7
    assert compact["v2_ready_fraction"] == 0.7
    assert compact["reason_counts"] == {"btc_reference_unverified": 2}
    assert "markets" not in compact


def test_study_summary_is_explicitly_public_data_only(tmp_path):
    finalized = {
        "readiness": {"v2_ready_markets": 1},
        "fees": {"enriched": 1},
        "resolution": {"enriched": 1},
        "dataset": {"rows": 10},
        "dataset_error": None,
    }
    summary = study_summary(
        requested_markets=1,
        output_root=tmp_path,
        clob_completed=[FakeClob(raw_events=100, snapshots=80)],
        btc_completed=[FakeBtc(samples=50, reference_price=100_000.0)],
        finalized=finalized,
        started_ts=100.0,
        finished_ts=110.0,
    )
    assert summary["mode"] == "public-data-only"
    assert summary["credentials_loaded"] is False
    assert summary["order_path_used"] is False
    assert summary["raw_clob_events"] == 100
    assert summary["btc_samples"] == 50
    assert summary["btc_reference_verified_markets"] == 1
    assert summary["elapsed_sec"] == 10.0


def test_study_report_is_written_as_json(tmp_path):
    summary = {"status": "complete", "credentials_loaded": False, "order_path_used": False}
    path = write_study_summary(tmp_path, summary)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == summary
