from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FeatureRow:
    slug: str
    received_ts_ns: int
    seconds_left: float
    up_mid: float | None
    down_mid: float | None
    up_ask: float | None
    down_ask: float | None
    up_spread: float | None
    down_spread: float | None
    up_ask_depth_3_usd: float
    down_ask_depth_3_usd: float
    up_book_imbalance: float | None
    down_book_imbalance: float | None
    ask_sum: float | None
    up_market_probability: float | None
    down_market_probability: float | None
    selected_side: str | None
    selected_ask: float | None
    selected_spread: float | None
    selected_depth_3_usd: float | None
    selected_exchange_age_ms: float | None


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def midpoint(bid: Any, ask: Any) -> float | None:
    b = _f(bid)
    a = _f(ask)
    if b is None or a is None or b < 0 or a > 1 or b > a:
        return None
    return (b + a) / 2.0


def depth_imbalance(bid_depth_usd: Any, ask_depth_usd: Any) -> float | None:
    bid = _f(bid_depth_usd)
    ask = _f(ask_depth_usd)
    if bid is None or ask is None or bid < 0 or ask < 0:
        return None
    total = bid + ask
    if total <= 0:
        return None
    return (bid - ask) / total


def normalized_binary_probability(up_mid: float | None, down_mid: float | None) -> tuple[float | None, float | None]:
    if up_mid is None or down_mid is None:
        return None, None
    total = up_mid + down_mid
    if total <= 0:
        return None, None
    return up_mid / total, down_mid / total


def stronger_ask_side(up_ask: Any, down_ask: Any) -> str | None:
    up = _f(up_ask)
    down = _f(down_ask)
    if up is None and down is None:
        return None
    if up is None:
        return "DOWN"
    if down is None:
        return "UP"
    if up == down:
        return None
    return "UP" if up > down else "DOWN"


def build_feature_row(row: dict[str, Any]) -> FeatureRow:
    up_mid = midpoint(row.get("up_bid"), row.get("up_ask"))
    down_mid = midpoint(row.get("down_bid"), row.get("down_ask"))
    up_prob, down_prob = normalized_binary_probability(up_mid, down_mid)
    side = stronger_ask_side(row.get("up_ask"), row.get("down_ask"))

    selected_ask = selected_spread = selected_depth = selected_age = None
    if side == "UP":
        selected_ask = _f(row.get("up_ask"))
        selected_spread = _f(row.get("up_spread"))
        selected_depth = _f(row.get("up_ask_depth_3_usd"))
        selected_age = _f(row.get("up_exchange_age_ms"))
    elif side == "DOWN":
        selected_ask = _f(row.get("down_ask"))
        selected_spread = _f(row.get("down_spread"))
        selected_depth = _f(row.get("down_ask_depth_3_usd"))
        selected_age = _f(row.get("down_exchange_age_ms"))

    up_depth = _f(row.get("up_ask_depth_3_usd")) or 0.0
    down_depth = _f(row.get("down_ask_depth_3_usd")) or 0.0
    up_ask = _f(row.get("up_ask"))
    down_ask = _f(row.get("down_ask"))

    return FeatureRow(
        slug=str(row.get("slug") or ""),
        received_ts_ns=int(row.get("received_ts_ns") or 0),
        seconds_left=float(row.get("seconds_left") or 0.0),
        up_mid=up_mid,
        down_mid=down_mid,
        up_ask=up_ask,
        down_ask=down_ask,
        up_spread=_f(row.get("up_spread")),
        down_spread=_f(row.get("down_spread")),
        up_ask_depth_3_usd=up_depth,
        down_ask_depth_3_usd=down_depth,
        up_book_imbalance=depth_imbalance(row.get("up_bid_depth_3_usd"), row.get("up_ask_depth_3_usd")),
        down_book_imbalance=depth_imbalance(row.get("down_bid_depth_3_usd"), row.get("down_ask_depth_3_usd")),
        ask_sum=None if up_ask is None or down_ask is None else up_ask + down_ask,
        up_market_probability=up_prob,
        down_market_probability=down_prob,
        selected_side=side,
        selected_ask=selected_ask,
        selected_spread=selected_spread,
        selected_depth_3_usd=selected_depth,
        selected_exchange_age_ms=selected_age,
    )


def build_feature_rows(rows: Iterable[dict[str, Any]]) -> list[FeatureRow]:
    return [build_feature_row(row) for row in rows]


def feature_payload(feature: FeatureRow) -> dict[str, Any]:
    return feature.__dict__.copy()
