from __future__ import annotations

from collections import defaultdict
from typing import Any

RECOMMENDATIONS = {
    "missing_clob_raw": "Run a full public CLOB recording session; raw WebSocket evidence is required.",
    "missing_clob_snapshots": "Verify both UP/DOWN books initialize and normalized Parquet snapshots are written.",
    "missing_decision_window_snapshot": "Start collection before 150 seconds remain and keep recording through the 90–150 second decision window.",
    "clob_recording_reconnected": "Treat this market as non-contiguous evidence; collect another full market with zero CLOB reconnects.",
    "missing_btc_samples": "Record the settlement-aligned BTC/TWAP stream for the same market interval.",
    "missing_or_invalid_btc_metadata": "Write BTC source/reference/quality metadata alongside btc_samples.parquet.",
    "btc_recording_reconnected": "Collect another full market with zero BTC/TWAP reconnects.",
    "btc_source_mismatch": "Use the exact market-declared Chainlink BTC/USD 60-second TWAP source.",
    "btc_reference_unverified": "Capture and verify the market-start BTC reference before using this market for model research.",
    "missing_fee_schedule": "Enrich the market with its current CLOB fee schedule before paper P&L evaluation.",
    "unresolved_market": "Wait for settlement or rerun resolution enrichment after the market closes.",
    "invalid_clob_parquet": "Repair/recollect the CLOB snapshot artifact; unreadable Parquet is not accepted.",
    "invalid_btc_parquet": "Repair/recollect the BTC sample artifact; unreadable Parquet is not accepted.",
    "missing_or_invalid_clob_recording_quality": "Record explicit CLOB reconnect/quality metadata.",
    "missing_or_invalid_btc_recording_quality": "Record explicit BTC/TWAP reconnect/quality metadata.",
}


def rejection_diagnostics(readiness: dict[str, Any]) -> dict[str, Any]:
    markets = list(readiness.get("markets") or [])
    by_reason: dict[str, list[str]] = defaultdict(list)
    for market in markets:
        slug = str(market.get("slug") or market.get("market_dir") or "unknown")
        for reason in market.get("reasons") or []:
            by_reason[str(reason)].append(slug)

    ranked = sorted(
        (
            {
                "reason": reason,
                "count": len(slugs),
                "fraction_of_markets": None if not markets else len(slugs) / len(markets),
                "affected_slugs": sorted(set(slugs)),
                "recommendation": RECOMMENDATIONS.get(
                    reason,
                    "Inspect the affected market artifacts and preserve the fail-closed gate.",
                ),
            }
            for reason, slugs in by_reason.items()
        ),
        key=lambda item: (-int(item["count"]), str(item["reason"])),
    )

    top = ranked[0] if ranked else None
    return {
        "markets_seen": len(markets),
        "v2_ready_markets": int(readiness.get("v2_ready_markets") or 0),
        "v2_ready_fraction": readiness.get("v2_ready_fraction"),
        "blocking_reason_count": len(ranked),
        "top_blocker": None if top is None else top["reason"],
        "top_blocker_count": 0 if top is None else top["count"],
        "ranked_blockers": ranked,
        "next_collection_action": (
            "No readiness blockers detected."
            if top is None
            else top["recommendation"]
        ),
    }
