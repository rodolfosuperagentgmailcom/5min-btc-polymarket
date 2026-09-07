from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PriceObservation:
    """One normalized BTC observation.

    ts_ns is the local receive/event ordering timestamp used by replay. source_ts_ms
    is preserved separately so latency can be studied without using future data.
    """

    ts_ns: int
    price: float
    source_ts_ms: int | None = None
    received_ts_ns: int | None = None

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("ts_ns must be positive")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("price must be a positive finite number")


@dataclass(frozen=True)
class BtcFeatures:
    current_price: float | None
    current_ts_ns: int | None
    current_source_ts_ms: int | None
    current_age_ms: float | None
    reference_price: float | None
    reference_ts_ns: int | None
    elapsed_from_reference_ms: float | None
    move_usd: float | None
    momentum_5s: float | None
    momentum_15s: float | None
    momentum_30s: float | None
    momentum_60s: float | None
    realized_vol_30s: float | None
    realized_vol_60s: float | None
    impulse_z: float | None


def _causal(observations: Iterable[PriceObservation], now_ns: int) -> list[PriceObservation]:
    """Return observations available at or before now_ns, sorted deterministically."""

    return sorted((obs for obs in observations if obs.ts_ns <= now_ns), key=lambda obs: obs.ts_ns)


def latest_at_or_before(
    observations: Sequence[PriceObservation],
    target_ns: int,
) -> PriceObservation | None:
    """Find the last observation at/before target_ns. Never looks forward."""

    best: PriceObservation | None = None
    for obs in observations:
        if obs.ts_ns > target_ns:
            break
        best = obs
    return best


def _anchor_price(
    observations: Sequence[PriceObservation],
    *,
    now_ns: int,
    horizon_sec: float,
    max_anchor_gap_sec: float | None,
) -> float | None:
    target_ns = now_ns - int(horizon_sec * 1_000_000_000)
    anchor = latest_at_or_before(observations, target_ns)
    if anchor is None:
        return None
    if max_anchor_gap_sec is not None:
        gap_sec = (target_ns - anchor.ts_ns) / 1_000_000_000
        if gap_sec > max_anchor_gap_sec:
            return None
    return anchor.price


def momentum_usd(
    observations: Iterable[PriceObservation],
    *,
    now_ns: int,
    horizon_sec: float,
    max_anchor_gap_sec: float | None = 5.0,
) -> float | None:
    causal = _causal(observations, now_ns)
    if not causal:
        return None
    current = causal[-1]
    anchor_price = _anchor_price(
        causal,
        now_ns=now_ns,
        horizon_sec=horizon_sec,
        max_anchor_gap_sec=max_anchor_gap_sec,
    )
    if anchor_price is None:
        return None
    return current.price - anchor_price


def realized_vol_usd_sqrt_sec(
    observations: Iterable[PriceObservation],
    *,
    now_ns: int,
    window_sec: float,
) -> float | None:
    """Quadratic-variation estimate in USD/sqrt(second), using only causal samples."""

    causal = _causal(observations, now_ns)
    if len(causal) < 2:
        return None
    start_ns = now_ns - int(window_sec * 1_000_000_000)
    window = [obs for obs in causal if obs.ts_ns >= start_ns]
    if len(window) < 2:
        anchor = latest_at_or_before(causal, start_ns)
        if anchor is not None and (not window or anchor.ts_ns < window[0].ts_ns):
            window.insert(0, anchor)
    if len(window) < 2:
        return None

    squared_moves = 0.0
    elapsed_sec = 0.0
    for left, right in zip(window, window[1:]):
        dt_sec = (right.ts_ns - left.ts_ns) / 1_000_000_000
        if dt_sec <= 0:
            continue
        move = right.price - left.price
        squared_moves += move * move
        elapsed_sec += dt_sec
    if elapsed_sec <= 0:
        return None
    return math.sqrt(squared_moves / elapsed_sec)


def orderbook_imbalance(bid_depth_usd: float | None, ask_depth_usd: float | None) -> float | None:
    if bid_depth_usd is None or ask_depth_usd is None:
        return None
    if bid_depth_usd < 0 or ask_depth_usd < 0:
        return None
    total = bid_depth_usd + ask_depth_usd
    if total <= 0:
        return None
    return (bid_depth_usd - ask_depth_usd) / total


def build_btc_features(
    observations: Iterable[PriceObservation],
    *,
    now_ns: int,
    reference_price: float | None,
    reference_ts_ns: int | None,
    max_anchor_gap_sec: float | None = 5.0,
) -> BtcFeatures:
    causal = _causal(observations, now_ns)
    current = causal[-1] if causal else None
    current_age_ms = None
    if current is not None:
        current_age_ms = (now_ns - current.ts_ns) / 1_000_000

    elapsed_from_reference_ms = None
    if reference_ts_ns is not None:
        elapsed_from_reference_ms = (now_ns - reference_ts_ns) / 1_000_000

    move_usd = None
    if current is not None and reference_price is not None:
        move_usd = current.price - reference_price

    vol30 = realized_vol_usd_sqrt_sec(causal, now_ns=now_ns, window_sec=30)
    vol60 = realized_vol_usd_sqrt_sec(causal, now_ns=now_ns, window_sec=60)
    impulse_z = None
    if (
        move_usd is not None
        and vol60 is not None
        and vol60 > 0
        and reference_ts_ns is not None
        and now_ns > reference_ts_ns
    ):
        elapsed_sec = (now_ns - reference_ts_ns) / 1_000_000_000
        scale = vol60 * math.sqrt(elapsed_sec)
        if scale > 0:
            impulse_z = abs(move_usd) / scale

    return BtcFeatures(
        current_price=None if current is None else current.price,
        current_ts_ns=None if current is None else current.ts_ns,
        current_source_ts_ms=None if current is None else current.source_ts_ms,
        current_age_ms=current_age_ms,
        reference_price=reference_price,
        reference_ts_ns=reference_ts_ns,
        elapsed_from_reference_ms=elapsed_from_reference_ms,
        move_usd=move_usd,
        momentum_5s=momentum_usd(
            causal, now_ns=now_ns, horizon_sec=5, max_anchor_gap_sec=max_anchor_gap_sec
        ),
        momentum_15s=momentum_usd(
            causal, now_ns=now_ns, horizon_sec=15, max_anchor_gap_sec=max_anchor_gap_sec
        ),
        momentum_30s=momentum_usd(
            causal, now_ns=now_ns, horizon_sec=30, max_anchor_gap_sec=max_anchor_gap_sec
        ),
        momentum_60s=momentum_usd(
            causal, now_ns=now_ns, horizon_sec=60, max_anchor_gap_sec=max_anchor_gap_sec
        ),
        realized_vol_30s=vol30,
        realized_vol_60s=vol60,
        impulse_z=impulse_z,
    )
