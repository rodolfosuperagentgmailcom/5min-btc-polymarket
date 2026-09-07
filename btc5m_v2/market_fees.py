from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

CLOB_BASE_URL = "https://clob.polymarket.com"


@dataclass(frozen=True)
class MarketFeeSchedule:
    condition_id: str
    rate: float
    exponent: float
    taker_only: bool
    maker_base_fee_bps: int
    taker_base_fee_bps: int

    @property
    def fees_enabled(self) -> bool:
        return self.rate > 0.0


def parse_market_fee_schedule(
    condition_id: str,
    payload: dict[str, Any],
) -> MarketFeeSchedule:
    fd = payload.get("fd") or {}
    if not isinstance(fd, dict):
        raise ValueError("CLOB market fee details must be an object")

    rate = float(fd.get("r") or 0.0)
    exponent = float(fd.get("e") or 0.0)
    taker_only = bool(fd.get("to", True))
    maker_bps = int(payload.get("mbf") or 0)
    taker_bps = int(payload.get("tbf") or 0)

    if rate < 0.0:
        raise ValueError("market fee rate cannot be negative")
    if exponent < 0.0:
        raise ValueError("market fee exponent cannot be negative")
    if maker_bps < 0 or taker_bps < 0:
        raise ValueError("base fees cannot be negative")

    return MarketFeeSchedule(
        condition_id=str(condition_id),
        rate=rate,
        exponent=exponent,
        taker_only=taker_only,
        maker_base_fee_bps=maker_bps,
        taker_base_fee_bps=taker_bps,
    )


def fetch_market_fee_schedule(
    condition_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 12.0,
) -> MarketFeeSchedule:
    cid = str(condition_id).strip()
    if not cid:
        raise ValueError("condition_id is required")
    client = session or requests
    response = client.get(f"{CLOB_BASE_URL}/clob-markets/{cid}", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("unexpected CLOB market-info response")
    return parse_market_fee_schedule(cid, payload)


def fee_schedule_payload(schedule: MarketFeeSchedule | None) -> dict[str, Any] | None:
    if schedule is None:
        return None
    return {
        "condition_id": schedule.condition_id,
        "fees_enabled": schedule.fees_enabled,
        "rate": schedule.rate,
        "exponent": schedule.exponent,
        "taker_only": schedule.taker_only,
        "maker_base_fee_bps": schedule.maker_base_fee_bps,
        "taker_base_fee_bps": schedule.taker_base_fee_bps,
        "source": "clob_market_info",
    }
