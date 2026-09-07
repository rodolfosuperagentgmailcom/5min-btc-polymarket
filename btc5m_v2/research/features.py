from __future__ import annotations

from bisect import bisect_right
from typing import Any, Iterable


HORIZONS_SEC = (1, 5, 15)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mid(bid: Any, ask: Any) -> float | None:
    bid_f = _as_float(bid)
    ask_f = _as_float(ask)
    if bid_f is None or ask_f is None:
        return None
    return (bid_f + ask_f) / 2.0


def _book_imbalance(bid_depth: Any, ask_depth: Any) -> float | None:
    bid_f = _as_float(bid_depth)
    ask_f = _as_float(ask_depth)
    if bid_f is None or ask_f is None:
        return None
    total = bid_f + ask_f
    if total <= 0:
        return None
    return (bid_f - ask_f) / total


def _sum_if_present(left: Any, right: Any) -> float | None:
    left_f = _as_float(left)
    right_f = _as_float(right)
    if left_f is None or right_f is None:
        return None
    return left_f + right_f


def _difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _lag_index(timestamps_ns: list[int], current_index: int, horizon_sec: int) -> int | None:
    cutoff = timestamps_ns[current_index] - int(horizon_sec * 1_000_000_000)
    index = bisect_right(timestamps_ns, cutoff, 0, current_index + 1) - 1
    return index if index >= 0 else None


def _latency_ms(received_ts_ns: Any, exchange_ts_ms: Any) -> float | None:
    received = _as_float(received_ts_ns)
    exchange = _as_float(exchange_ts_ms)
    if received is None or exchange is None:
        return None
    return received / 1_000_000.0 - exchange


def _max_present(values: Iterable[Any]) -> float | None:
    parsed = [value for value in (_as_float(item) for item in values) if value is not None]
    return max(parsed) if parsed else None


def build_causal_features(snapshot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build CLOB-only features without using future snapshots or settlement labels.

    Rows are sorted by local receive timestamp. Lagged features use the latest snapshot
    at or before each historical cutoff, so replay and live feature construction can
    share the same semantics.
    """

    if not snapshot_rows:
        return []

    rows = sorted(snapshot_rows, key=lambda row: int(row["received_ts_ns"]))
    timestamps_ns = [int(row["received_ts_ns"]) for row in rows]
    up_mids = [_mid(row.get("up_bid"), row.get("up_ask")) for row in rows]
    down_mids = [_mid(row.get("down_bid"), row.get("down_ask")) for row in rows]

    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        up_mid = up_mids[index]
        down_mid = down_mids[index]
        mid_sum = None if up_mid is None or down_mid is None else up_mid + down_mid
        up_mid_share = None if mid_sum in (None, 0.0) else up_mid / mid_sum

        up_imbalance = _book_imbalance(
            row.get("up_bid_depth_3_usd"), row.get("up_ask_depth_3_usd")
        )
        down_imbalance = _book_imbalance(
            row.get("down_bid_depth_3_usd"), row.get("down_ask_depth_3_usd")
        )

        feature_row = dict(row)
        feature_row.update(
            {
                "snapshot_index": index,
                "up_mid": up_mid,
                "down_mid": down_mid,
                "up_mid_share": up_mid_share,
                "up_book_imbalance_3": up_imbalance,
                "down_book_imbalance_3": down_imbalance,
                "sum_best_asks": _sum_if_present(row.get("up_ask"), row.get("down_ask")),
                "sum_best_bids": _sum_if_present(row.get("up_bid"), row.get("down_bid")),
                "clob_age_max_ms": _max_present(
                    [row.get("up_exchange_age_ms"), row.get("down_exchange_age_ms")]
                ),
                "event_receive_latency_ms": _latency_ms(
                    row.get("received_ts_ns"), row.get("event_exchange_ts_ms")
                ),
            }
        )

        selected_side = None
        if up_mid is not None and down_mid is not None:
            selected_side = "UP" if up_mid >= down_mid else "DOWN"
        feature_row["selected_side_by_mid"] = selected_side
        if selected_side == "UP":
            feature_row["selected_bid"] = _as_float(row.get("up_bid"))
            feature_row["selected_ask"] = _as_float(row.get("up_ask"))
            feature_row["selected_spread"] = _as_float(row.get("up_spread"))
            feature_row["selected_ask_depth_3_usd"] = _as_float(row.get("up_ask_depth_3_usd"))
        elif selected_side == "DOWN":
            feature_row["selected_bid"] = _as_float(row.get("down_bid"))
            feature_row["selected_ask"] = _as_float(row.get("down_ask"))
            feature_row["selected_spread"] = _as_float(row.get("down_spread"))
            feature_row["selected_ask_depth_3_usd"] = _as_float(row.get("down_ask_depth_3_usd"))
        else:
            feature_row["selected_bid"] = None
            feature_row["selected_ask"] = None
            feature_row["selected_spread"] = None
            feature_row["selected_ask_depth_3_usd"] = None

        for horizon_sec in HORIZONS_SEC:
            lag_index = _lag_index(timestamps_ns, index, horizon_sec)
            suffix = f"{horizon_sec}s"
            if lag_index is None:
                feature_row[f"up_mid_delta_{suffix}"] = None
                feature_row[f"down_mid_delta_{suffix}"] = None
                feature_row[f"up_mid_velocity_{suffix}_per_sec"] = None
                feature_row[f"down_mid_velocity_{suffix}_per_sec"] = None
                continue

            elapsed_sec = (timestamps_ns[index] - timestamps_ns[lag_index]) / 1_000_000_000.0
            up_delta = _difference(up_mid, up_mids[lag_index])
            down_delta = _difference(down_mid, down_mids[lag_index])
            feature_row[f"up_mid_delta_{suffix}"] = up_delta
            feature_row[f"down_mid_delta_{suffix}"] = down_delta
            feature_row[f"up_mid_velocity_{suffix}_per_sec"] = (
                None if up_delta is None or elapsed_sec <= 0 else up_delta / elapsed_sec
            )
            feature_row[f"down_mid_velocity_{suffix}_per_sec"] = (
                None if down_delta is None or elapsed_sec <= 0 else down_delta / elapsed_sec
            )

        result.append(feature_row)

    return result


def attach_resolution_labels(
    feature_rows: list[dict[str, Any]],
    *,
    winning_side: str,
) -> list[dict[str, Any]]:
    """Attach terminal labels after feature generation.

    The label is intentionally added in a separate pass so no feature function can
    accidentally access the outcome during replay.
    """

    side = str(winning_side).strip().upper()
    if side not in {"UP", "DOWN"}:
        raise ValueError("winning_side must be UP or DOWN")

    labeled: list[dict[str, Any]] = []
    for row in feature_rows:
        item = dict(row)
        item["winning_side"] = side
        item["target_up"] = 1 if side == "UP" else 0
        selected = item.get("selected_side_by_mid")
        item["selected_side_by_mid_won"] = None if selected not in {"UP", "DOWN"} else selected == side
        labeled.append(item)
    return labeled
