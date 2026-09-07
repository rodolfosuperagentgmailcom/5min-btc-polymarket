from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator

import websockets

RTDS_URL = "wss://ws-live-data.polymarket.com"
TWAP_60_TOPIC = "crypto_prices_twap_sixty"
TWAP_30_TOPIC = "crypto_prices_twap_thirty"
CHAINLINK_TWAP_RESOLUTION_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"
UTC = dt.timezone.utc


@dataclass(frozen=True)
class TwapObservation:
    symbol: str
    window_seconds: int
    value: Decimal
    full_accuracy_value: str | None
    chainlink_ts_ms: int
    publisher_ts_ms: int | None
    received_ts_ns: int
    raw_event: dict[str, Any]

    @property
    def received_iso(self) -> str:
        return dt.datetime.fromtimestamp(self.received_ts_ns / 1_000_000_000, tz=UTC).isoformat()

    @property
    def chainlink_age_ms(self) -> float:
        return max(0.0, self.received_ts_ns / 1_000_000 - self.chainlink_ts_ms)


def compact_symbol_filter(symbol: str) -> str:
    normalized = str(symbol).strip().lower()
    if not normalized or "/" not in normalized:
        raise ValueError("RTDS symbol must be lowercase slash-delimited, e.g. btc/usd")
    return json.dumps({"symbol": normalized}, separators=(",", ":"))


def twap_subscription(*, symbol: str = "btc/usd", window_seconds: int = 60) -> dict[str, Any]:
    window = int(window_seconds)
    if window not in {30, 60}:
        raise ValueError("window_seconds must be 30 or 60")
    topic = TWAP_30_TOPIC if window == 30 else TWAP_60_TOPIC
    return {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": topic,
                "type": "update",
                "filters": compact_symbol_filter(symbol),
            }
        ],
    }


def _timestamp_ms(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 1e11:
        number *= 1000.0
    return int(number)


def parse_twap_event(
    event: dict[str, Any],
    *,
    received_ts_ns: int,
    symbol: str = "btc/usd",
    window_seconds: int = 60,
) -> TwapObservation | None:
    expected_window = int(window_seconds)
    expected_topic = TWAP_30_TOPIC if expected_window == 30 else TWAP_60_TOPIC
    if str(event.get("topic") or "") != expected_topic:
        return None
    if str(event.get("type") or "") != "update":
        return None

    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    observed_symbol = str(payload.get("symbol") or "").strip().lower()
    if observed_symbol != str(symbol).strip().lower():
        return None

    raw_window = payload.get("window_s", payload.get("windowSeconds", payload.get("window_seconds")))
    try:
        observed_window = int(raw_window)
    except (TypeError, ValueError):
        observed_window = expected_window
    if observed_window != expected_window:
        return None

    raw_full = payload.get("full_accuracy_value")
    raw_value = payload.get("value")
    try:
        if raw_full not in (None, ""):
            # RTDS documents full_accuracy_value as signed E18. Prefer it over
            # the display-oriented numeric value to avoid float precision loss.
            value = Decimal(str(raw_full)) / Decimal(10**18)
            full_accuracy = str(raw_full)
        else:
            value = Decimal(str(raw_value))
            full_accuracy = None
    except (InvalidOperation, TypeError, ValueError):
        return None

    chainlink_ts_ms = _timestamp_ms(payload.get("timestamp"))
    if chainlink_ts_ms is None:
        return None

    return TwapObservation(
        symbol=observed_symbol,
        window_seconds=observed_window,
        value=value,
        full_accuracy_value=full_accuracy,
        chainlink_ts_ms=chainlink_ts_ms,
        publisher_ts_ms=_timestamp_ms(event.get("timestamp")),
        received_ts_ns=int(received_ts_ns),
        raw_event=dict(event),
    )


async def _heartbeat(websocket: Any, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=5.0)
        except TimeoutError:
            await websocket.send("PING")


async def twap_events(
    *,
    symbol: str = "btc/usd",
    window_seconds: int = 60,
    url: str = RTDS_URL,
) -> AsyncIterator[TwapObservation]:
    """Yield public Chainlink-computed TWAP observations relayed by Polymarket RTDS.

    No wallet, Polymarket trading credential, or Chainlink credential is used.
    Direct RTDS clients must reconnect/resubscribe after disconnect; callers may
    wrap this generator in their own reconnect loop.
    """

    subscription = twap_subscription(symbol=symbol, window_seconds=window_seconds)
    async with websockets.connect(url, ping_interval=None, close_timeout=5) as websocket:
        await websocket.send(json.dumps(subscription, separators=(",", ":")))
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(_heartbeat(websocket, stop))
        try:
            async for message in websocket:
                if not isinstance(message, str):
                    continue
                stripped = message.strip()
                if not stripped or stripped in {"PONG", "PING"}:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                events = payload if isinstance(payload, list) else [payload]
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    observation = parse_twap_event(
                        event,
                        received_ts_ns=time.time_ns(),
                        symbol=symbol,
                        window_seconds=window_seconds,
                    )
                    if observation is not None:
                        yield observation
        finally:
            stop.set()
            heartbeat.cancel()
            try:
                await heartbeat
            except (asyncio.CancelledError, Exception):
                pass
