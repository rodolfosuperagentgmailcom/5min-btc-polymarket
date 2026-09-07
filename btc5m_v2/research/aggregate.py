from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from btc5m_v2.research.dataset import write_dataset
from btc5m_v2.research.readiness import assess_market_readiness


def _market_dirs(search_roots: Iterable[str | Path]) -> list[Path]:
    """Find recorded market directories beneath extracted evidence roots."""

    found: dict[str, Path] = {}
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        candidates = [root / "metadata.json"] if root.is_dir() else []
        if root.is_dir():
            candidates.extend(root.rglob("metadata.json"))
        for metadata in candidates:
            if metadata.exists() and metadata.is_file():
                market_dir = metadata.parent.resolve()
                found[str(market_dir)] = market_dir
    return sorted(found.values(), key=lambda value: str(value))


def _quality_score(assessment: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    """Prefer causally usable, fuller duplicate recordings deterministically."""

    return (
        int(assessment.get("v2_ready") is True),
        int(assessment.get("resolved") is True),
        int(assessment.get("decision_window_snapshot_present") is True),
        int(assessment.get("btc_reference_verified") is True),
        int(assessment.get("fee_ready") is True),
        int(assessment.get("clob_snapshot_rows") or 0),
        int(assessment.get("btc_sample_rows") or 0),
    )


def aggregate_ready_market_selection(
    search_roots: Iterable[str | Path],
) -> tuple[list[Path], dict[str, Any]]:
    """Select one best fully V2-ready recording per independent market slug.

    Raw artifact contents stay immutable. This function only returns paths to
    qualifying market directories, so downstream dataset construction can read
    them in-place from an extracted artifact workspace.
    """

    market_dirs = _market_dirs(search_roots)
    assessments: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    by_slug: dict[str, list[tuple[Path, dict[str, Any]]]] = {}

    for market_dir in market_dirs:
        assessment = assess_market_readiness(market_dir)
        assessment = dict(assessment)
        assessment["market_dir"] = str(market_dir)
        assessments.append(assessment)
        reason_counts.update(assessment.get("reasons") or [])
        slug = str(assessment.get("slug") or market_dir.name)
        by_slug.setdefault(slug, []).append((market_dir, assessment))

    selected: list[Path] = []
    selected_details: list[dict[str, Any]] = []
    duplicate_ready_recordings = 0

    for slug in sorted(by_slug):
        candidates = by_slug[slug]
        ready = [(path, item) for path, item in candidates if item.get("v2_ready") is True]
        if not ready:
            continue
        ready.sort(key=lambda pair: (_quality_score(pair[1]), str(pair[0])), reverse=True)
        chosen_path, chosen = ready[0]
        selected.append(chosen_path)
        selected_details.append(
            {
                "slug": slug,
                "market_dir": str(chosen_path),
                "quality_score": list(_quality_score(chosen)),
                "ready_recordings_for_slug": len(ready),
            }
        )
        duplicate_ready_recordings += max(0, len(ready) - 1)

    manifest = {
        "recordings_seen": len(assessments),
        "unique_market_slugs_seen": len(by_slug),
        "v2_ready_recordings": sum(1 for item in assessments if item.get("v2_ready") is True),
        "independent_v2_ready_markets": len(selected),
        "duplicate_ready_recordings_excluded": duplicate_ready_recordings,
        "reason_counts": dict(sorted(reason_counts.items())),
        "selected_markets": selected_details,
        "recordings": assessments,
    }
    return selected, manifest


def write_aggregate_ready_dataset(
    search_roots: Iterable[str | Path],
    output_path: str | Path,
    *,
    sample_interval_ms: int = 1000,
) -> dict[str, Any]:
    selected, manifest = aggregate_ready_market_selection(search_roots)
    if not selected:
        raise ValueError("no_v2_ready_markets_across_evidence")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary = write_dataset(
        selected,
        destination,
        sample_interval_ms=sample_interval_ms,
    )

    summary = {
        **manifest,
        "selection_mode": "aggregate_v2_ready_only_deduplicated_by_slug",
        "sample_interval_ms": int(sample_interval_ms),
        "dataset": dataset_summary,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".aggregate.json")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
