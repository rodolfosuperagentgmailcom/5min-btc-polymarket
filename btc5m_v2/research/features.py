from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Iterable

CAUSAL_LAGS_SECONDS = (5, 15, 30, 60)


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


def normalized_binary_probability(
    up_mid: float | None,
    down_mid: float | None,
) -> tuple[float | None, float | None]:
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


def stronger_mid_side(up_mid: Any, down_mid: Any) -> str | None:
    up = _f(up_mid)
    down = _f(down_mid)
    if up is None or down is None or up == down:
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
        up_book_imbalance=depth_imbalance(
            row.get("up_bid_depth_3_usd"), row.get("up_ask_depth_3_usd")
        ),
        down_book_imbalance=depth_imbalance(
            row.get("down_bid_depth_3_usd"), row.get("down_ask_depth_3_usd")
        ),
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


def _sum_pair(left: Any, right: Any) -> float | None:
    a = _f(left)
    b = _f(right)
    return None if a is None or b is None else a + b


def _lag_index(
    timestamps_ns: list[int],
    *,
    current_index: int,
    cutoff_ns: int,
) -> int | None:
    # Search only the prefix through current_index. Future observations are
    # structurally impossible to reach from this function.
    index = bisect.bisect_right(
        timestamps_ns,
        cutoff_ns,
        0,
        current_index + 1,
    ) - 1
    return index if index >= 0 else None


def _delta_and_velocity(
    current_value: float | None,
    previous_value: float | None,
    *,
    current_ts_ns: int,
    previous_ts_ns: int | None,
) -> tuple[float | None, float | None]:
    if current_value is None or previous_value is None or previous_ts_ns is None:
        return None, None
    elapsed_sec = (current_ts_ns - previous_ts_ns) / 1_000_000_000
    if elapsed_sec <= 0:
        return None, None
    delta = current_value - previous_value
    return delta, delta / elapsed_sec


def build_causal_features(
    rows: Iterable[dict[str, Any]],
    *,
    lags_seconds: tuple[int, ...] = CAUSAL_LAGS_SECONDS,
) -> list[dict[str, Any]]:
    """Build features using only information available at each row timestamp.

    For a lag L, the historical observation is the latest snapshot whose
    timestamp is <= (current_timestamp - L). Velocity divides by the actual
    elapsed time to that snapshot, not by the nominal lag.
    """

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: int(row.get("received_ts_ns") or 0),
    )
    if not ordered:
        return []

    timestamps = [int(row.get("received_ts_ns") or 0) for row in ordered]
    base_rows: list[dict[str, Any]] = []

    for row in ordered:
        feature = build_feature_row(row)
        payload = dict(row)
        payload.update(feature_payload(feature))
        payload.update(
            {
                "up_book_imbalance_3": feature.up_book_imbalance,
                "down_book_imbalance_3": feature.down_book_imbalance,
                "sum_best_asks": _sum_pair(row.get("up_ask"), row.get("down_ask")),
                "sum_best_bids": _sum_pair(row.get("up_bid"), row.get("down_bid")),
                "selected_side_by_mid": stronger_mid_side(feature.up_mid, feature.down_mid),
            }
        )
        base_rows.append(payload)

    output: list[dict[str, Any]] = []
    for index, payload in enumerate(base_rows):
        current_ts = timestamps[index]
        enriched = dict(payload)
        for lag in lags_seconds:
            cutoff = current_ts - int(lag * 1_000_000_000)
            previous_index = _lag_index(
                timestamps,
                current_index=index,
                cutoff_ns=cutoff,
            )
            previous = None if previous_index is None else base_rows[previous_index]
            previous_ts = None if previous_index is None else timestamps[previous_index]

            for side in ("up", "down"):
                current_mid = _f(payload.get(f"{side}_mid"))
                previous_mid = None if previous is None else _f(previous.get(f"{side}_mid"))
                delta, velocity = _delta_and_velocity(
                    current_mid,
                    previous_mid,
                    current_ts_ns=current_ts,
                    previous_ts_ns=previous_ts,
                )
                enriched[f"{side}_mid_delta_{lag}s"] = delta
                enriched[f"{side}_mid_velocity_{lag}s_per_sec"] = velocity

        output.append(enriched)
    return output


def attach_resolution_labels(
    features: Iterable[dict[str, Any]],
    *,
    winning_side: str,
) -> list[dict[str, Any]]:
    """Attach terminal labels after feature construction.

    This function is intentionally separate from build_causal_features so a
    signal/replay consumer cannot accidentally use the terminal result while
    constructing predictors.
    """

    winner = str(winning_side).strip().upper()
    if winner not in {"UP", "DOWN"}:
        raise ValueError("winning_side must be UP or DOWN")

    labeled: list[dict[str, Any]] = []
    for feature in features:
        row = dict(feature)
        selected = str(row.get("selected_side_by_mid") or "").upper() or None
        row["winning_side"] = winner
        row["target_up"] = 1 if winner == "UP" else 0
        row["selected_side_by_mid_won"] = (
            None if selected not in {"UP", "DOWN"} else selected == winner
        )
        labeled.append(row)
    return labeled
