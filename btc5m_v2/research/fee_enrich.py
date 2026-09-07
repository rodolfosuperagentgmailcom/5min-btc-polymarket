from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from btc5m_v2.market_fees import (
    MarketFeeSchedule,
    fee_schedule_payload,
    fetch_market_fee_schedule,
)

FeeFetcher = Callable[[str], MarketFeeSchedule]


def enrich_market_fee_file(
    market_dir: str | Path,
    *,
    fetcher: FeeFetcher = fetch_market_fee_schedule,
    force: bool = False,
) -> dict[str, Any]:
    directory = Path(market_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    fee_path = directory / "fees.json"
    if fee_path.exists() and not force:
        existing = json.loads(fee_path.read_text(encoding="utf-8"))
        return {"updated": False, "fees": existing}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    condition_id = str(metadata.get("condition_id") or "").strip()
    if not condition_id:
        raise ValueError(f"missing condition_id in {metadata_path}")

    schedule = fetcher(condition_id)
    payload = fee_schedule_payload(schedule)
    if payload is None:
        raise ValueError("fee schedule fetcher returned no schedule")

    fee_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metadata["fee_schedule"] = payload
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"updated": True, "fees": payload}


def enrich_fee_schedules(
    output_root: str | Path,
    *,
    fetcher: FeeFetcher = fetch_market_fee_schedule,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    scanned = updated = failed = 0
    errors: list[dict[str, str]] = []

    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        scanned += 1
        market_dir = metadata_path.parent
        try:
            result = enrich_market_fee_file(market_dir, fetcher=fetcher, force=force)
            if result["updated"]:
                updated += 1
        except Exception as exc:
            failed += 1
            errors.append({"market_dir": str(market_dir), "error": str(exc)})

    return {
        "scanned": scanned,
        "updated": updated,
        "failed": failed,
        "errors": errors,
    }


def read_fee_schedule(market_dir: str | Path) -> dict[str, Any] | None:
    directory = Path(market_dir)
    fee_path = directory / "fees.json"
    if fee_path.exists():
        payload = json.loads(fee_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = metadata.get("fee_schedule")
    return payload if isinstance(payload, dict) else None
