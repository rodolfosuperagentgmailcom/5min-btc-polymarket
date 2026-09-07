from __future__ import annotations

from typing import Any

FEATURE_VERSION = "v2-microstructure-1"


def _safe_mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    if bid < 0 or ask < 0 or ask < bid:
        return None
    return (bid + ask) / 2.0


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def depth_imbalance(bid_depth_usd: float | None, ask_depth_usd: float | None) -> float | None:
    if bid_depth_usd is None or ask_depth_usd is None:
        return None
    total = bid_depth_usd + ask_depth_usd
    if total <= 0:
        return None
    return (bid_depth_usd - ask_depth_usd) / total


def legacy_threshold_side(row: dict[str, Any], threshold: float = 0.70) -> str | None:
    up_ask = row.get("up_ask")
    down_ask = row.get("down_ask")
    candidates: list[tuple[str, float]] = []
    if isinstance(up_ask, (int, float)) and float(up_ask) >= threshold:
        candidates.append(("UP", float(up_ask)))
    if isinstance(down_ask, (int, float)) and float(down_ask) >= threshold:
        candidates.append(("DOWN", float(down_ask)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def microstructure_features(row: dict[str, Any], *, legacy_threshold: float = 0.70) -> dict[str, Any]:
    up_bid = row.get("up_bid")
    up_ask = row.get("up_ask")
    down_bid = row.get("down_bid")
    down_ask = row.get("down_ask")

    up_mid = _safe_mid(up_bid, up_ask)
    down_mid = _safe_mid(down_bid, down_ask)
    mid_sum = None if up_mid is None or down_mid is None else up_mid + down_mid

    up_bid_depth = row.get("up_bid_depth_3_usd")
    up_ask_depth = row.get("up_ask_depth_3_usd")
    down_bid_depth = row.get("down_bid_depth_3_usd")
    down_ask_depth = row.get("down_ask_depth_3_usd")

    legacy_side = legacy_threshold_side(row, threshold=legacy_threshold)

    return {
        "feature_version": FEATURE_VERSION,
        "slug": row.get("slug"),
        "condition_id": row.get("condition_id"),
        "received_ts_ns": row.get("received_ts_ns"),
        "received_iso": row.get("received_iso"),
        "seconds_left": row.get("seconds_left"),
        "up_bid": up_bid,
        "up_ask": up_ask,
        "up_spread": row.get("up_spread"),
        "down_bid": down_bid,
        "down_ask": down_ask,
        "down_spread": row.get("down_spread"),
        "up_mid": up_mid,
        "down_mid": down_mid,
        "normalized_up_mid": _safe_ratio(up_mid, mid_sum),
        "price_skew_mid": None if up_mid is None or down_mid is None else up_mid - down_mid,
        "ask_complement_sum": None if up_ask is None or down_ask is None else float(up_ask) + float(down_ask),
        "bid_complement_sum": None if up_bid is None or down_bid is None else float(up_bid) + float(down_bid),
        "up_depth_imbalance_3": depth_imbalance(up_bid_depth, up_ask_depth),
        "down_depth_imbalance_3": depth_imbalance(down_bid_depth, down_ask_depth),
        "up_ask_depth_3_usd": up_ask_depth,
        "down_ask_depth_3_usd": down_ask_depth,
        "up_bid_depth_3_usd": up_bid_depth,
        "down_bid_depth_3_usd": down_bid_depth,
        "up_ask_to_bid_depth_ratio": _safe_ratio(up_ask_depth, up_bid_depth),
        "down_ask_to_bid_depth_ratio": _safe_ratio(down_ask_depth, down_bid_depth),
        "max_exchange_age_ms": max(float(row.get("up_exchange_age_ms") or 0.0), float(row.get("down_exchange_age_ms") or 0.0)),
        "legacy_threshold": float(legacy_threshold),
        "legacy_signal_side": legacy_side,
        "legacy_signal": legacy_side is not None,
    }
