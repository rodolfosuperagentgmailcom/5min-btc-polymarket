from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import V2Config
from .safety import resolution_source_reasons

UTC = dt.timezone.utc


@dataclass(frozen=True)
class MarketInfo:
    slug: str
    condition_id: str
    up_token_id: str
    down_token_id: str
    end_iso: str
    end_ts: float
    resolution_source: str

    def seconds_left(self, now_ts: float | None = None) -> float:
        now = time.time() if now_ts is None else float(now_ts)
        return max(0.0, self.end_ts - now)


def bucket_5m(ts: int | float) -> int:
    value = int(ts)
    return value - (value % 300)


def parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def fetch_event_by_slug(slug: str, session: requests.Session | None = None) -> dict[str, Any] | None:
    client = session or requests
    response = client.get(
        f"https://gamma-api.polymarket.com/events/slug/{slug}",
        timeout=12,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def _token_mapping(market: dict[str, Any]) -> tuple[str, str]:
    outcomes = parse_json_field(market.get("outcomes")) or []
    token_ids = parse_json_field(market.get("clobTokenIds")) or []
    if not isinstance(outcomes, list) or not isinstance(token_ids, list) or len(token_ids) < 2:
        raise ValueError("market is missing two CLOB outcome token ids")

    labels = [str(value).strip().lower() for value in outcomes]
    up_index = next((i for i, label in enumerate(labels) if label in {"up", "yes"}), None)
    down_index = next((i for i, label in enumerate(labels) if label in {"down", "no"}), None)

    if up_index is None or down_index is None:
        if len(token_ids) != 2:
            raise ValueError(f"cannot map outcomes to UP/DOWN: {outcomes!r}")
        up_index, down_index = 0, 1

    return str(token_ids[up_index]), str(token_ids[down_index])


def market_info_from_event(event: dict[str, Any], cfg: V2Config) -> MarketInfo:
    markets = event.get("markets") or []
    if not isinstance(markets, list) or not markets:
        raise ValueError("event has no markets")

    market = dict(markets[0])
    if cfg.raw.get("market", {}).get("require_active", True) and market.get("active") is False:
        raise ValueError("market is inactive")
    if cfg.raw.get("market", {}).get("require_not_closed", True) and market.get("closed") is True:
        raise ValueError("market is closed")

    up_token, down_token = _token_mapping(market)
    end_iso = str(market.get("endDate") or market.get("endDateIso") or event.get("endDate") or "").strip()
    if not end_iso:
        raise ValueError("market is missing end date")
    end_ts = dt.datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp()

    source = str(market.get("resolutionSource") or event.get("resolutionSource") or "").strip()
    source_reasons = resolution_source_reasons(cfg, source)
    if source_reasons:
        raise ValueError("resolution source blocked: " + ",".join(source_reasons))

    slug = str(market.get("slug") or event.get("slug") or "").strip()
    condition_id = str(market.get("conditionId") or market.get("condition_id") or "").strip()
    if not slug or not condition_id:
        raise ValueError("market is missing slug or condition id")

    return MarketInfo(
        slug=slug,
        condition_id=condition_id,
        up_token_id=up_token,
        down_token_id=down_token,
        end_iso=end_iso,
        end_ts=end_ts,
        resolution_source=source,
    )


def discover_current_market(cfg: V2Config, now_ts: float | None = None) -> MarketInfo | None:
    now = time.time() if now_ts is None else float(now_ts)
    prefix = str(cfg.raw.get("market", {}).get("slug_prefix", "btc-updown-5m-"))
    slug = f"{prefix}{bucket_5m(now)}"
    event = fetch_event_by_slug(slug)
    if not event:
        return None
    return market_info_from_event(event, cfg)
