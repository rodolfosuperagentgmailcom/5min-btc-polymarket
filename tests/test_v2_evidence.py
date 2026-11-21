from __future__ import annotations

import json

from btc5m_v2.research.evidence import evidence_report, summarize_paper_evidence


def settled(slug: str, pnl: float, *, side: str = "UP", cost: float = 5.0, fee: float = 0.1):
    return {
        "status": "SETTLED",
        "mode": "paper",
        "slug": slug,
        "signal_ts_ns": int(slug.split("-")[-1]),
        "side": side,
        "model_version": "test-v1",
        "net_pnl_usd": pnl,
        "all_in_cost_usd": cost,
        "fee_usd": fee,
        "credentials_loaded": False,
        "order_path_used": False,
    }


def test_evidence_summary_counts_independent_settled_signals():
    rows = [
        settled("m-1", 2.0, side="UP"),
        settled("m-2", -1.0, side="DOWN"),
        {"status": "OPEN", "mode": "paper", "slug": "m-3", "credentials_loaded": False, "order_path_used": False},
    ]
    report = summarize_paper_evidence(rows, target_settled_signals=3)

    assert report["settled_independent_signals"] == 2
    assert report["remaining_to_target"] == 1
    assert report["evidence_target_reached"] is False
    assert report["wins"] == 1
    assert report["losses"] == 1
    assert report["win_rate"] == 0.5
    assert report["net_pnl_usd"] == 1.0
    assert report["mean_pnl_per_signal_usd"] == 0.5
    assert report["fees_paid_usd"] == 0.2
    assert report["open_positions"] == 1
    assert report["side_counts"] == {"UP": 1, "DOWN": 1}
    assert report["evidence_integrity_ok"] is True
    assert report["live_execution_approved"] is False


def test_evidence_drawdown_uses_chronological_realized_pnl():
    rows = [settled("m-1", 3.0), settled("m-2", -2.0), settled("m-3", -4.0)]
    report = summarize_paper_evidence(rows)
    assert report["max_drawdown_usd"] == 6.0


def test_duplicate_market_is_not_counted_twice_and_fails_integrity():
    rows = [settled("m-1", 1.0), settled("m-1", 2.0)]
    report = summarize_paper_evidence(rows)
    assert report["settled_independent_signals"] == 1
    assert report["duplicate_settled_slugs"] == ["m-1"]
    assert report["evidence_integrity_ok"] is False


def test_live_path_or_credentials_in_journal_fails_integrity():
    row = settled("m-1", 1.0)
    row["order_path_used"] = True
    report = summarize_paper_evidence([row])
    assert report["unsafe_journal_entries"] == 1
    assert report["evidence_integrity_ok"] is False


def test_evidence_report_reads_daily_journal_files(tmp_path):
    day = tmp_path / "2026-09-06"
    day.mkdir()
    (day / "m-1.json").write_text(json.dumps(settled("m-1", 1.25)), encoding="utf-8")
    report = evidence_report(tmp_path, target_settled_signals=1)
    assert report["journal_entries_seen"] == 1
    assert report["settled_independent_signals"] == 1
    assert report["evidence_target_reached"] is True
    assert report["journal_root"] == str(tmp_path)
