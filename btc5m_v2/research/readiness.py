from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from btc5m_v2.research.fee_enrich import read_fee_schedule
from btc5m_v2.research.replay import read_resolution


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_source(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _parquet_rows(paths: Iterable[Path]) -> int | None:
    total = 0
    seen = False
    try:
        for path in paths:
            seen = True
            total += int(pq.read_metadata(path).num_rows)
    except Exception:
        return None
    return total if seen else 0


def _int_field(payload: dict[str, Any] | None, name: str) -> int | None:
    if not isinstance(payload, dict) or name not in payload:
        return None
    try:
        value = int(payload[name])
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def assess_market_readiness(market_dir: str | Path) -> dict[str, Any]:
    directory = Path(market_dir)
    reasons: list[str] = []

    metadata = _json_object(directory / "metadata.json")
    if metadata is None:
        reasons.append("missing_or_invalid_market_metadata")
        expected_source = ""
        clob_reconnects = None
    else:
        expected_source = _normalize_source(metadata.get("resolution_source"))
        clob_reconnects = _int_field(metadata, "reconnects")
        if clob_reconnects is None:
            reasons.append("missing_or_invalid_clob_recording_quality")
        elif clob_reconnects != 0:
            reasons.append("clob_recording_reconnected")

    raw_path = directory / "clob_raw.jsonl"
    raw_present = raw_path.exists() and raw_path.stat().st_size > 0
    if not raw_present:
        reasons.append("missing_clob_raw")

    snapshot_paths = sorted(directory.glob("clob_snapshots_part-*.parquet"))
    clob_snapshot_rows = _parquet_rows(snapshot_paths)
    if clob_snapshot_rows is None:
        reasons.append("invalid_clob_parquet")
    elif clob_snapshot_rows <= 0:
        reasons.append("missing_clob_snapshots")

    resolved, winning_side = read_resolution(directory)
    resolved = bool(resolved and winning_side in {"UP", "DOWN"})
    if not resolved:
        reasons.append("unresolved_market")

    btc_samples_path = directory / "btc_samples.parquet"
    btc_rows = _parquet_rows([btc_samples_path]) if btc_samples_path.exists() else 0
    if btc_rows is None:
        reasons.append("invalid_btc_parquet")
    elif btc_rows <= 0:
        reasons.append("missing_btc_samples")

    btc_metadata = _json_object(directory / "btc_metadata.json")
    btc_reconnects = _int_field(btc_metadata, "reconnects")
    btc_source = _normalize_source((btc_metadata or {}).get("source"))
    reference_verified = bool((btc_metadata or {}).get("reference_verified") is True)
    reference_price = (btc_metadata or {}).get("reference_price")

    if btc_metadata is None:
        reasons.append("missing_or_invalid_btc_metadata")
    else:
        if btc_reconnects is None:
            reasons.append("missing_or_invalid_btc_recording_quality")
        elif btc_reconnects != 0:
            reasons.append("btc_recording_reconnected")
        if not expected_source or not btc_source or btc_source != expected_source:
            reasons.append("btc_source_mismatch")
        if not reference_verified or reference_price in (None, ""):
            reasons.append("btc_reference_unverified")

    fee_schedule = read_fee_schedule(directory)
    fee_ready = isinstance(fee_schedule, dict)
    if not fee_ready:
        reasons.append("missing_fee_schedule")

    unique_reasons = sorted(set(reasons))
    return {
        "market_dir": str(directory),
        "slug": str((metadata or {}).get("slug") or directory.name),
        "v2_ready": not unique_reasons,
        "reasons": unique_reasons,
        "resolution_source": expected_source or None,
        "resolved": resolved,
        "winning_side": winning_side if resolved else None,
        "clob_reconnects": clob_reconnects,
        "clob_snapshot_rows": clob_snapshot_rows,
        "clob_raw_present": raw_present,
        "btc_reconnects": btc_reconnects,
        "btc_sample_rows": btc_rows,
        "btc_source": btc_source or None,
        "btc_reference_verified": reference_verified,
        "fee_ready": fee_ready,
    }


def market_dirs_from_root(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    return sorted(path.parent for path in root.glob("*/*/metadata.json"))


def readiness_report(output_root: str | Path) -> dict[str, Any]:
    directories = market_dirs_from_root(output_root)
    markets = [assess_market_readiness(directory) for directory in directories]
    reason_counts: Counter[str] = Counter()
    for market in markets:
        reason_counts.update(market["reasons"])

    ready = [market for market in markets if market["v2_ready"]]
    return {
        "output_root": str(Path(output_root)),
        "markets_seen": len(markets),
        "v2_ready_markets": len(ready),
        "v2_ready_fraction": None if not markets else len(ready) / len(markets),
        "resolved_markets": sum(1 for market in markets if market["resolved"]),
        "clean_clob_markets": sum(1 for market in markets if market["clob_reconnects"] == 0),
        "clean_btc_markets": sum(1 for market in markets if market["btc_reconnects"] == 0),
        "verified_btc_reference_markets": sum(
            1 for market in markets if market["btc_reference_verified"]
        ),
        "fee_ready_markets": sum(1 for market in markets if market["fee_ready"]),
        "reason_counts": dict(sorted(reason_counts.items())),
        "markets": markets,
    }
