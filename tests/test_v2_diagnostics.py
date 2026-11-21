from __future__ import annotations

from btc5m_v2.research.diagnostics import rejection_diagnostics


def test_rejection_diagnostics_ranks_blockers_and_keeps_slugs():
    readiness = {
        "v2_ready_markets": 0,
        "v2_ready_fraction": 0.0,
        "markets": [
            {"slug": "m1", "reasons": ["missing_btc_samples", "unresolved_market"]},
            {"slug": "m2", "reasons": ["missing_btc_samples", "missing_fee_schedule"]},
            {"slug": "m3", "reasons": ["missing_btc_samples"]},
        ],
    }
    report = rejection_diagnostics(readiness)
    assert report["top_blocker"] == "missing_btc_samples"
    assert report["top_blocker_count"] == 3
    assert report["ranked_blockers"][0]["affected_slugs"] == ["m1", "m2", "m3"]
    assert "settlement-aligned BTC/TWAP" in report["next_collection_action"]


def test_rejection_diagnostics_reports_clean_collection():
    readiness = {
        "v2_ready_markets": 2,
        "v2_ready_fraction": 1.0,
        "markets": [
            {"slug": "m1", "reasons": []},
            {"slug": "m2", "reasons": []},
        ],
    }
    report = rejection_diagnostics(readiness)
    assert report["top_blocker"] is None
    assert report["blocking_reason_count"] == 0
    assert report["next_collection_action"] == "No readiness blockers detected."
