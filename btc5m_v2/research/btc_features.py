from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BTCSample:
    ts_ns: int
    price: float


def normalize_btc_samples(
    samples: Iterable[BTCSample | tuple[int, float]],
) -> list[BTCSample]:
    normalized: list[BTCSample] = []
    for sample in samples:
        if isinstance(sample, BTCSample):
            ts_ns = int(sample.ts_ns)
            price = float(sample.price)
        else:
            ts_ns = int(sample[0])
            price = float(sample[1])
        if ts_ns <= 0 or price <= 0 or not math.isfinite(price):
            continue
        normalized.append(BTCSample(ts_ns=ts_ns, price=price))
    normalized.sort(key=lambda item: item.ts_ns)
    return normalized


def _index_at_or_before(samples: Sequence[BTCSample], ts_ns: int) -> int:
    timestamps = [sample.ts_ns for sample in samples]
    return bisect.bisect_right(timestamps, int(ts_ns)) - 1


def _price_at_or_before(samples: Sequence[BTCSample], ts_ns: int) -> float | None:
    index = _index_at_or_before(samples, ts_ns)
    return None if index < 0 else samples[index].price


def _realized_log_move(
    samples: Sequence[BTCSample],
    *,
    start_ns: int,
    end_ns: int,
) -> float | None:
    window = [sample for sample in samples if start_ns <= sample.ts_ns <= end_ns]
    if len(window) < 2:
        return None

    sum_squared = 0.0
    observations = 0
    previous = window[0].price
    for sample in window[1:]:
        if previous > 0 and sample.price > 0:
            log_return = math.log(sample.price / previous)
            sum_squared += log_return * log_return
            observations += 1
        previous = sample.price

    if observations == 0:
        return None
    return math.sqrt(sum_squared)


def btc_features_at(
    samples: Iterable[BTCSample | tuple[int, float]],
    *,
    now_ns: int,
    reference_price: float | None = None,
    horizons_sec: Sequence[int] = (15, 30, 60),
    vol_window_sec: int = 60,
) -> dict[str, float | None]:
    """Calculate BTC features using samples observed at or before now_ns only."""

    ordered = normalize_btc_samples(samples)
    current_index = _index_at_or_before(ordered, now_ns)
    if current_index < 0:
        output: dict[str, float | None] = {
            "btc_price": None,
            "btc_feed_age_ms": None,
            "btc_reference_price": reference_price,
            "btc_move_from_reference": None,
            "btc_realized_log_move_60s": None,
            "btc_realized_move_usd_60s": None,
            "btc_impulse_z": None,
        }
        for horizon in horizons_sec:
            output[f"btc_move_{int(horizon)}s"] = None
        return output

    causal = ordered[: current_index + 1]
    current = causal[-1]
    output = {
        "btc_price": current.price,
        "btc_feed_age_ms": max(0.0, (int(now_ns) - current.ts_ns) / 1_000_000.0),
    }

    for horizon in horizons_sec:
        horizon_int = int(horizon)
        target_ns = int(now_ns) - horizon_int * 1_000_000_000
        historical = _price_at_or_before(causal, target_ns)
        output[f"btc_move_{horizon_int}s"] = (
            None if historical is None else current.price - historical
        )

    start_ns = int(now_ns) - int(vol_window_sec) * 1_000_000_000
    realized_log_move = _realized_log_move(causal, start_ns=start_ns, end_ns=int(now_ns))
    realized_move_usd = (
        None if realized_log_move is None else current.price * realized_log_move
    )

    ref = None if reference_price is None else float(reference_price)
    move_from_reference = None if ref is None else current.price - ref
    impulse_z = None
    if (
        move_from_reference is not None
        and realized_move_usd is not None
        and realized_move_usd > 0
    ):
        impulse_z = move_from_reference / realized_move_usd

    output.update(
        {
            "btc_reference_price": ref,
            "btc_move_from_reference": move_from_reference,
            "btc_realized_log_move_60s": realized_log_move,
            "btc_realized_move_usd_60s": realized_move_usd,
            "btc_impulse_z": impulse_z,
        }
    )
    return output


def attach_btc_features(
    rows: Iterable[dict],
    *,
    samples: Iterable[BTCSample | tuple[int, float]],
    reference_price: float | None = None,
) -> list[dict]:
    ordered_samples = normalize_btc_samples(samples)
    enriched: list[dict] = []
    for row in rows:
        payload = dict(row)
        payload.update(
            btc_features_at(
                ordered_samples,
                now_ns=int(payload["received_ts_ns"]),
                reference_price=reference_price,
            )
        )
        enriched.append(payload)
    return enriched
