from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import websockets

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
UTC = dt.timezone.utc


def iso_utc_from_ns(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


def market_subscription(token_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
    ids = [str(value) for value in token_ids if str(value)]
    if not ids:
        raise ValueError("at least one token id is required")
    return {
        "assets_ids": ids,
        "type": "market",
        "custom_feature_enabled": True,
    }


@dataclass(frozen=True)
class WsEnvelope:
    received_ts_ns: int
    received_iso: str
    wire_payload: str
    event: dict[str, Any]


@dataclass
class BookState:
    asset_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    exchange_ts_ms: int | None = None
    last_received_ts_ns: int | None = None

    @staticmethod
    def _levels(items: Any) -> dict[float, float]:
        levels: dict[float, float] = {}
        for item in items or []:
            try:
                price = float(item.get("price") if isinstance(item, dict) else getattr(item, "price"))
                size = float(item.get("size") if isinstance(item, dict) else getattr(item, "size"))
            except Exception:
                continue
            if price > 0 and size > 0:
                levels[price] = size
        return levels

    @staticmethod
    def _event_timestamp_ms(event: dict[str, Any]) -> int | None:
        raw = event.get("timestamp")
        if raw in (None, ""):
            return None
        try:
            value = float(raw)
            if value < 1e11:
                value *= 1000.0
            return int(value)
        except Exception:
            return None

    def apply(self, event: dict[str, Any], received_ts_ns: int) -> bool:
        event_type = str(event.get("event_type") or "").lower()
        changed = False

        if event_type == "book" and str(event.get("asset_id") or "") == self.asset_id:
            self.bids = self._levels(event.get("bids"))
            self.asks = self._levels(event.get("asks"))
            self.exchange_ts_ms = self._event_timestamp_ms(event)
            self.last_received_ts_ns = received_ts_ns
            return True

        if event_type == "price_change":
            for change in event.get("price_changes") or []:
                if str(change.get("asset_id") or "") != self.asset_id:
                    continue
                try:
                    price = float(change.get("price"))
                    size = float(change.get("size"))
                except Exception:
                    continue
                side = str(change.get("side") or "").upper()
                levels = self.bids if side in {"BUY", "BID"} else self.asks if side in {"SELL", "ASK"} else None
                if levels is None:
                    continue
                if size <= 0:
                    levels.pop(price, None)
                elif price > 0:
                    levels[price] = size
                changed = True

            if changed:
                self.exchange_ts_ms = self._event_timestamp_ms(event)
                self.last_received_ts_ns = received_ts_ns
            return changed

        if event_type == "best_bid_ask" and str(event.get("asset_id") or "") == self.asset_id:
            self.exchange_ts_ms = self._event_timestamp_ms(event)
            self.last_received_ts_ns = received_ts_ns
            return False

        return False

    @property
    def best_bid(self) -> float | None:
        return max(self.bids, default=None)

    @property
    def best_ask(self) -> float | None:
        return min(self.asks, default=None)

    @property
    def spread(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return max(0.0, ask - bid)

    def bid_notional(self, levels: int = 3) -> float:
        prices = sorted(self.bids, reverse=True)[:levels]
        return sum(price * self.bids[price] for price in prices)

    def ask_notional(self, levels: int = 3) -> float:
        prices = sorted(self.asks)[:levels]
        return sum(price * self.asks[price] for price in prices)

    def snapshot(self, side: str, received_ts_ns: int) -> dict[str, Any]:
        exchange_ts_ms = self.exchange_ts_ms
        receive_ts_ms = received_ts_ns / 1_000_000
        exchange_age_ms = None if exchange_ts_ms is None else max(0.0, receive_ts_ms - exchange_ts_ms)
        return {
            "side": side,
            "asset_id": self.asset_id,
            "bid": self.best_bid,
            "ask": self.best_ask,
            "spread": self.spread,
            "bid_depth_3_usd": self.bid_notional(3),
            "ask_depth_3_usd": self.ask_notional(3),
            "exchange_ts_ms": exchange_ts_ms,
            "receive_ts_ns": received_ts_ns,
            "exchange_age_ms": exchange_age_ms,
        }


async def _heartbeat(ws: Any, interval_sec: float) -> None:
    while True:
        await asyncio.sleep(interval_sec)
        await ws.send("PING")


async def market_events(
    token_ids: list[str] | tuple[str, ...],
    *,
    url: str = WS_MARKET_URL,
    heartbeat_sec: float = 10.0,
) -> AsyncIterator[WsEnvelope]:
    subscription = market_subscription(token_ids)
    async with websockets.connect(
        url,
        ping_interval=None,
        open_timeout=12,
        close_timeout=5,
        max_size=8 * 1024 * 1024,
    ) as ws:
        await ws.send(json.dumps(subscription, separators=(",", ":")))
        heartbeat = asyncio.create_task(_heartbeat(ws, heartbeat_sec))
        try:
            async for wire in ws:
                received_ns = time.time_ns()
                text = wire.decode("utf-8") if isinstance(wire, bytes) else str(wire)
                if text.strip().upper() == "PONG":
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                events = payload if isinstance(payload, list) else [payload]
                for event in events:
                    if isinstance(event, dict):
                        yield WsEnvelope(
                            received_ts_ns=received_ns,
                            received_iso=iso_utc_from_ns(received_ns),
                            wire_payload=text,
                            event=event,
                        )
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
