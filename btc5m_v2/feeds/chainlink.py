from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

DISCOVERY_URL = "https://api.dataengine.chain.link/api/v1/discovery"
REST_BASE_URL = "https://api.dataengine.chain.link"
WS_BASE_URL = "wss://ws.dataengine.chain.link"


@dataclass(frozen=True)
class ChainlinkStream:
    feed_id: str
    name: str
    base_asset: str
    quote_asset: str
    schema_version: str
    status: str
    attribute_type: str | None = None
    service_level: str | None = None


@dataclass(frozen=True)
class ChainlinkCredentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "ChainlinkCredentials":
        key = os.environ.get("CHAINLINK_API_KEY", "").strip()
        secret = os.environ.get("CHAINLINK_API_SECRET", "").strip()
        if not key or not secret:
            raise RuntimeError("CHAINLINK_API_KEY and CHAINLINK_API_SECRET are required for Chainlink report access")
        return cls(key, secret)


def discover_streams(
    *,
    base_asset: str = "BTC",
    quote_asset: str = "USD",
    status: str = "live",
    session: requests.Session | None = None,
) -> list[ChainlinkStream]:
    client = session or requests
    response = client.get(
        DISCOVERY_URL,
        params={"base_asset": base_asset, "quote_asset": quote_asset, "status": status},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    feeds = payload.get("feeds") if isinstance(payload, dict) else None
    if not isinstance(feeds, list):
        raise ValueError("unexpected Chainlink discovery response")

    streams: list[ChainlinkStream] = []
    for item in feeds:
        if not isinstance(item, dict):
            continue
        feed_id = str(item.get("feedId") or "").strip()
        name = str(item.get("name") or "").strip()
        if not feed_id or not name:
            continue
        streams.append(
            ChainlinkStream(
                feed_id=feed_id,
                name=name,
                base_asset=str(item.get("baseAsset") or ""),
                quote_asset=str(item.get("quoteAsset") or ""),
                schema_version=str(item.get("schemaVersion") or ""),
                status=str(item.get("status") or ""),
                attribute_type=str(item.get("attributeType") or "") or None,
                service_level=str(item.get("serviceLevel") or "") or None,
            )
        )
    return streams


def twap_60s_candidates(streams: list[ChainlinkStream]) -> list[ChainlinkStream]:
    keywords = ("twap", "60s", "60-sec", "60sec", "60 second")
    return [stream for stream in streams if any(keyword in stream.name.lower() for keyword in keywords)]


def empty_body_hash() -> str:
    return hashlib.sha256(b"").hexdigest()


def auth_headers(
    *,
    method: str,
    full_path: str,
    credentials: ChainlinkCredentials,
    timestamp_ms: int | None = None,
) -> dict[str, str]:
    stamp = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    string_to_sign = f"{method.upper()} {full_path} {empty_body_hash()} {credentials.api_key} {stamp}"
    signature = hmac.new(
        credentials.api_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": credentials.api_key,
        "X-Authorization-Timestamp": str(stamp),
        "X-Authorization-Signature-SHA256": signature,
    }


def report_ws_path(feed_ids: list[str] | tuple[str, ...]) -> str:
    ids = [str(value).strip() for value in feed_ids if str(value).strip()]
    if not ids:
        raise ValueError("at least one Chainlink feed ID is required")
    return "/api/v1/ws?" + urlencode({"feedIDs": ",".join(ids)})


def discovery_summary(streams: list[ChainlinkStream]) -> list[dict[str, Any]]:
    return [
        {
            "feed_id": stream.feed_id,
            "name": stream.name,
            "base_asset": stream.base_asset,
            "quote_asset": stream.quote_asset,
            "schema_version": stream.schema_version,
            "status": stream.status,
            "attribute_type": stream.attribute_type,
            "service_level": stream.service_level,
        }
        for stream in streams
    ]
