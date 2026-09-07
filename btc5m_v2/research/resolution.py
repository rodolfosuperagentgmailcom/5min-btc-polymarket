from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from btc5m_v2.market import MarketInfo, fetch_event_by_slug, parse_json_field


@dataclass(frozen=True)
class ResolutionResult:
    resolved: bool
    winning_side: str | None = None
    winning_asset_id: str | None = None
    winning_outcome: str | None = None
    resolution_ts_ms: int | None = None
    source: str | None = None


def resolution_from_ws_event(event: dict[str, Any], market: MarketInfo) -> ResolutionResult | None:
    if str(event.get("event_type") or "").lower() != "market_resolved":
        return None

    winning_asset = str(event.get("winning_asset_id") or "").strip()
    if not winning_asset:
        return ResolutionResult(resolved=False, source="polymarket_ws")

    side = None
    if winning_asset == market.up_token_id:
        side = "UP"
    elif winning_asset == market.down_token_id:
        side = "DOWN"

    raw_ts = event.get("timestamp")
    ts_ms = None
    try:
        value = float(raw_ts)
        if value < 1e11:
            value *= 1000.0
        ts_ms = int(value)
    except Exception:
        pass

    return ResolutionResult(
        resolved=side is not None,
        winning_side=side,
        winning_asset_id=winning_asset,
        winning_outcome=str(event.get("winning_outcome") or "").strip() or None,
        resolution_ts_ms=ts_ms,
        source="polymarket_ws",
    )


def _market_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    markets = event.get("markets") or []
    if not isinstance(markets, list) or not markets:
        return None
    return dict(markets[0])


def resolution_from_gamma_payload(event: dict[str, Any], market: MarketInfo) -> ResolutionResult:
    gamma_market = _market_from_event(event)
    if gamma_market is None:
        return ResolutionResult(resolved=False, source="gamma")

    outcomes = parse_json_field(gamma_market.get("outcomes")) or []
    prices = parse_json_field(gamma_market.get("outcomePrices")) or []
    token_ids = parse_json_field(gamma_market.get("clobTokenIds")) or []

    if not isinstance(outcomes, list) or not isinstance(prices, list):
        return ResolutionResult(resolved=False, source="gamma")
    if len(outcomes) < 2 or len(prices) != len(outcomes):
        return ResolutionResult(resolved=False, source="gamma")

    parsed_prices: list[float] = []
    try:
        parsed_prices = [float(value) for value in prices]
    except Exception:
        return ResolutionResult(resolved=False, source="gamma")

    winner_index = max(range(len(parsed_prices)), key=lambda i: parsed_prices[i])
    winner_price = parsed_prices[winner_index]
    loser_prices = [value for i, value in enumerate(parsed_prices) if i != winner_index]

    closed = bool(gamma_market.get("closed") is True or event.get("closed") is True)
    resolution_status = str(gamma_market.get("umaResolutionStatus") or "").lower()
    resolution_like_status = resolution_status in {"resolved", "finalized", "complete", "completed"}

    # Binary markets should settle to 1/0. Requiring a near-certain terminal vector
    # avoids mistaking a pre-resolution market price for the winner.
    terminal_vector = winner_price >= 0.999 and all(value <= 0.001 for value in loser_prices)
    if not terminal_vector or not (closed or resolution_like_status):
        return ResolutionResult(resolved=False, source="gamma")

    winning_outcome = str(outcomes[winner_index])
    winning_asset = None
    if isinstance(token_ids, list) and winner_index < len(token_ids):
        winning_asset = str(token_ids[winner_index])

    side = None
    label = winning_outcome.strip().lower()
    if label in {"up", "yes"}:
        side = "UP"
    elif label in {"down", "no"}:
        side = "DOWN"
    elif winning_asset == market.up_token_id:
        side = "UP"
    elif winning_asset == market.down_token_id:
        side = "DOWN"

    return ResolutionResult(
        resolved=side is not None,
        winning_side=side,
        winning_asset_id=winning_asset,
        winning_outcome=winning_outcome,
        source="gamma",
    )


def fetch_gamma_resolution(
    market: MarketInfo,
    session: requests.Session | None = None,
) -> ResolutionResult:
    event = fetch_event_by_slug(market.slug, session=session)
    if not event:
        return ResolutionResult(resolved=False, source="gamma")
    return resolution_from_gamma_payload(event, market)


def resolution_payload(result: ResolutionResult) -> dict[str, Any]:
    return {
        "resolved": result.resolved,
        "winning_side": result.winning_side,
        "winning_asset_id": result.winning_asset_id,
        "winning_outcome": result.winning_outcome,
        "resolution_ts_ms": result.resolution_ts_ms,
        "source": result.source,
    }


def write_resolution_json(path: str, result: ResolutionResult) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(resolution_payload(result), indent=2), encoding="utf-8")
