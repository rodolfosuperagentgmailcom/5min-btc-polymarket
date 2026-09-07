from __future__ import annotations

import json
from pathlib import Path

import pytest

from btc5m_v2.research import ready_dataset


def test_ready_only_selects_qualified_markets_and_reports_exclusions(tmp_path, monkeypatch):
    ready_dir = tmp_path / "2030-01-01" / "btc-updown-5m-ready"
    blocked_dir = tmp_path / "2030-01-01" / "btc-updown-5m-blocked"
    ready_dir.mkdir(parents=True)
    blocked_dir.mkdir(parents=True)

    monkeypatch.setattr(
        ready_dataset,
        "readiness_report",
        lambda output_root: {
            "markets_seen": 2,
            "v2_ready_markets": 1,
            "reason_counts": {"btc_recording_reconnected": 1},
            "markets": [
                {"market_dir": str(ready_dir), "v2_ready": True, "reasons": []},
                {
                    "market_dir": str(blocked_dir),
                    "v2_ready": False,
                    "reasons": ["btc_recording_reconnected"],
                },
            ],
        },
    )

    captured: list[Path] = []

    def fake_write_dataset(market_dirs, output_path, *, sample_interval_ms=1000):
        captured.extend(Path(path) for path in market_dirs)
        return {
            "output": str(output_path),
            "rows": 10,
            "markets": len(captured),
            "sample_interval_ms": sample_interval_ms,
        }

    monkeypatch.setattr(ready_dataset, "write_dataset", fake_write_dataset)
    output = tmp_path / "research" / "features.parquet"
    summary = ready_dataset.write_ready_dataset(tmp_path, output, sample_interval_ms=500)

    assert captured == [ready_dir]
    assert summary["selection_mode"] == "v2_ready_only"
    assert summary["markets_seen"] == 2
    assert summary["markets_selected"] == 1
    assert summary["markets_excluded"] == 1
    assert summary["exclusion_reason_counts"] == {"btc_recording_reconnected": 1}
    assert summary["sample_interval_ms"] == 500

    sidecar = json.loads(output.with_suffix(".parquet.json").read_text(encoding="utf-8"))
    assert sidecar["markets_selected"] == 1
    assert sidecar["markets_excluded"] == 1


def test_ready_only_refuses_empty_qualified_set(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ready_dataset,
        "readiness_report",
        lambda output_root: {
            "markets_seen": 3,
            "v2_ready_markets": 0,
            "reason_counts": {"unresolved_market": 3},
            "markets": [
                {
                    "market_dir": str(tmp_path / f"bad-{index}"),
                    "v2_ready": False,
                    "reasons": ["unresolved_market"],
                }
                for index in range(3)
            ],
        },
    )

    with pytest.raises(ValueError, match="no_v2_ready_markets"):
        ready_dataset.write_ready_dataset(tmp_path, tmp_path / "features.parquet")
