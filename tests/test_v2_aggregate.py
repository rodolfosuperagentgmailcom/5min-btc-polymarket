from __future__ import annotations

import json
from pathlib import Path

from btc5m_v2.research import aggregate


def _make_market(root: Path, date: str, slug: str) -> Path:
    market = root / date / slug
    market.mkdir(parents=True)
    (market / "metadata.json").write_text(json.dumps({"slug": slug}), encoding="utf-8")
    return market


def test_aggregate_deduplicates_ready_market_and_excludes_unready(monkeypatch, tmp_path):
    artifact_a = tmp_path / "artifact-a" / "runtime" / "data"
    artifact_b = tmp_path / "artifact-b" / "runtime" / "data"

    first = _make_market(artifact_a, "2026-09-07", "btc-updown-5m-100")
    better_duplicate = _make_market(artifact_b, "2026-09-07", "btc-updown-5m-100")
    blocked = _make_market(artifact_b, "2026-09-07", "btc-updown-5m-200")

    def fake_assess(path):
        path = Path(path)
        if path == blocked:
            return {
                "market_dir": str(path),
                "slug": path.name,
                "v2_ready": False,
                "reasons": ["btc_reference_unverified"],
                "resolved": True,
                "decision_window_snapshot_present": True,
                "btc_reference_verified": False,
                "fee_ready": True,
                "clob_snapshot_rows": 500,
                "btc_sample_rows": 300,
            }
        fuller = path == better_duplicate
        return {
            "market_dir": str(path),
            "slug": path.name,
            "v2_ready": True,
            "reasons": [],
            "resolved": True,
            "decision_window_snapshot_present": True,
            "btc_reference_verified": True,
            "fee_ready": True,
            "clob_snapshot_rows": 1000 if fuller else 700,
            "btc_sample_rows": 400 if fuller else 300,
        }

    monkeypatch.setattr(aggregate, "assess_market_readiness", fake_assess)

    selected, manifest = aggregate.aggregate_ready_market_selection([tmp_path])

    assert selected == [better_duplicate.resolve()]
    assert manifest["recordings_seen"] == 3
    assert manifest["unique_market_slugs_seen"] == 2
    assert manifest["v2_ready_recordings"] == 2
    assert manifest["independent_v2_ready_markets"] == 1
    assert manifest["duplicate_ready_recordings_excluded"] == 1
    assert manifest["reason_counts"] == {"btc_reference_unverified": 1}


def test_aggregate_empty_or_unready_evidence_selects_nothing(monkeypatch, tmp_path):
    market = _make_market(tmp_path, "2026-09-07", "btc-updown-5m-300")

    monkeypatch.setattr(
        aggregate,
        "assess_market_readiness",
        lambda path: {
            "market_dir": str(path),
            "slug": market.name,
            "v2_ready": False,
            "reasons": ["unresolved_market"],
        },
    )

    selected, manifest = aggregate.aggregate_ready_market_selection([tmp_path])
    assert selected == []
    assert manifest["independent_v2_ready_markets"] == 0
    assert manifest["reason_counts"] == {"unresolved_market": 1}
