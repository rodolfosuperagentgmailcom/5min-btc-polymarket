from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from btc5m_v2.research.dataset import write_dataset
from btc5m_v2.research.readiness import readiness_report


def ready_market_selection(output_root: str | Path) -> tuple[list[Path], dict[str, Any]]:
    """Return only markets that pass the full V2 readiness contract."""

    report = readiness_report(output_root)
    selected = [
        Path(market["market_dir"])
        for market in report.get("markets", [])
        if market.get("v2_ready") is True
    ]
    return selected, report


def write_ready_dataset(
    output_root: str | Path,
    output_path: str | Path,
    *,
    sample_interval_ms: int = 1000,
) -> dict[str, Any]:
    selected, report = ready_market_selection(output_root)
    if not selected:
        raise ValueError("no_v2_ready_markets")

    summary = write_dataset(
        selected,
        output_path,
        sample_interval_ms=sample_interval_ms,
    )
    summary.update(
        {
            "selection_mode": "v2_ready_only",
            "markets_seen": int(report.get("markets_seen") or 0),
            "markets_selected": len(selected),
            "markets_excluded": int(report.get("markets_seen") or 0) - len(selected),
            "exclusion_reason_counts": dict(report.get("reason_counts") or {}),
        }
    )

    destination = Path(output_path)
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
