from __future__ import annotations

import math
import time

MARKET_INTERVAL_SEC = 300.0


def next_market_start_ts(now_ts: float | None = None, *, interval_sec: float = MARKET_INTERVAL_SEC) -> float:
    """Return the next future 5-minute market boundary in Unix seconds."""

    interval = float(interval_sec)
    if interval <= 0.0:
        raise ValueError("interval_sec must be positive")
    now = time.time() if now_ts is None else float(now_ts)
    return (math.floor(now / interval) + 1) * interval


def aligned_start_ts(
    now_ts: float | None = None,
    *,
    offset_sec: float = 0.5,
    interval_sec: float = MARKET_INTERVAL_SEC,
) -> float:
    """Return a safe start just after the next market boundary.

    A small positive offset avoids racing the venue's market rollover while the
    BTC reference recorder still accepts the settlement-aligned observation
    within its configured source-timestamp tolerance.
    """

    offset = float(offset_sec)
    if offset < 0.0:
        raise ValueError("offset_sec cannot be negative")
    return next_market_start_ts(now_ts, interval_sec=interval_sec) + offset


def sleep_until_aligned_start(
    *,
    offset_sec: float = 0.5,
    interval_sec: float = MARKET_INTERVAL_SEC,
) -> float:
    """Sleep until the next aligned start and return the target timestamp."""

    target = aligned_start_ts(offset_sec=offset_sec, interval_sec=interval_sec)
    delay = max(0.0, target - time.time())
    if delay > 0.0:
        time.sleep(delay)
    return target
