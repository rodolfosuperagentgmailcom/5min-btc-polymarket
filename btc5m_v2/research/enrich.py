from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from btc5m_v2.market import MarketInfo
from btc5m_v2.research.resolution import fetch_gamma_resolution, resolution_payload


def market_from_metadata(payload: dict[str, Any]) -> MarketInfo:
    end_iso = str(payload["market_end"])
    end_ts = dt.datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp()
    return MarketInfo(
        slug=str(payload["slug"]),
        condition_id=str(payload["condition_id"]),
        up_token_id=str(payload["up_token_id"]),
        down_token_id=str(payload["down_token_id"]),
        end_iso=end_iso,
        end_ts=end_ts,
        resolution_source=str(payload["resolution_source"]),
    )


def enrich_metadata_file(path: str | Path, *, force: bool = False) -> bool:
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    existing = payload.get("resolution") or {}
    if not force and existing.get("resolved") is True:
        return False

    market = market_from_metadata(payload)
    if not force and time.time() < market.end_ts:
        return False

    result = fetch_gamma_resolution(market)
    if not result.resolved:
        return False

    resolved_payload = resolution_payload(result)
    payload["resolution"] = resolved_payload
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metadata_path.with_name("resolution.json").write_text(
        json.dumps(resolved_payload, indent=2), encoding="utf-8"
    )
    return True


def enrich_output_root(output_root: str | Path, *, force: bool = False) -> dict[str, int]:
    root = Path(output_root)
    scanned = 0
    enriched = 0
    unresolved = 0
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        scanned += 1
        if enrich_metadata_file(metadata_path, force=force):
            enriched += 1
        else:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not (payload.get("resolution") or {}).get("resolved"):
                unresolved += 1
    return {"scanned": scanned, "enriched": enriched, "unresolved": unresolved}
