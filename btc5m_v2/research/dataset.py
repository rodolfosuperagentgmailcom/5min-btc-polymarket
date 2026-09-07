from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.features import build_feature_row, feature_payload
from btc5m_v2.research.replay import iter_snapshot_rows, read_resolution

DEFAULT_LAGS_SECONDS = (5, 15, 30, 60)

DATASET_SCHEMA = pa.schema(
    [
        ("slug", pa.string()),
        ("received_ts_ns", pa.int64()),
        ("seconds_left", pa.float64()),
        ("up_mid", pa.float64()),
        ("down_mid", pa.float64()),
        ("up_ask", pa.float64()),
        ("down_ask", pa.float64()),
        ("up_spread", pa.float64()),
        ("down_spread", pa.float64()),
        ("up_ask_depth_3_usd", pa.float64()),
        ("down_ask_depth_3_usd", pa.float64()),
        ("up_book_imbalance", pa.float64()),
        ("down_book_imbalance", pa.float64()),
        ("ask_sum", pa.float64()),
        ("ask_overround", pa.float64()),
        ("up_market_probability", pa.float64()),
        ("down_market_probability", pa.float64()),
        ("market_skew_up", pa.float64()),
        ("selected_side", pa.string()),
        ("selected_ask", pa.float64()),
        ("selected_spread", pa.float64()),
        ("selected_depth_3_usd", pa.float64()),
        ("selected_exchange_age_ms", pa.float64()),
        ("up_probability_change_5s", pa.float64()),
        ("up_probability_change_15s", pa.float64()),
        ("up_probability_change_30s", pa.float64()),
        ("up_probability_change_60s", pa.float64()),
        ("up_mid_change_5s", pa.float64()),
        ("up_mid_change_15s", pa.float64()),
        ("up_mid_change_30s", pa.float64()),
        ("up_mid_change_60s", pa.float64()),
        # Reserved for the verified settlement-aligned Chainlink TWAP stream.
        # These remain null until that feed is ingested; no substitute BTC feed
        # is silently used for model training.
        ("btc_reference", pa.float64()),
        ("btc_current", pa.float64()),
        ("btc_delta_usd", pa.float64()),
        ("btc_momentum_15s", pa.float64()),
        ("btc_momentum_30s", pa.float64()),
        ("btc_momentum_60s", pa.float64()),
        ("btc_realized_vol_30s", pa.float64()),
        ("btc_realized_vol_60s", pa.float64()),
        ("btc_impulse_z", pa.float64()),
        # Labels are appended after causal feature construction. They must never
        # be passed to a signal function as predictors.
        ("resolved", pa.bool_()),
        ("winning_side", pa.string()),
        ("label_up", pa.int8()),
        ("selected_won", pa.bool_()),
    ]
)


def _downsample(rows: list[dict[str, Any]], interval_ms: int) -> list[dict[str, Any]]:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: int(row["received_ts_ns"]))
    selected: list[dict[str, Any]] = []
    min_gap_ns = interval_ms * 1_000_000
    last_ts: int | None = None
    for row in ordered:
        ts = int(row["received_ts_ns"])
        if last_ts is None or ts - last_ts >= min_gap_ns:
            selected.append(row)
            last_ts = ts
    return selected


def _asof_value(
    timestamps: list[int],
    values: list[float | None],
    *,
    current_index: int,
    target_ts_ns: int,
) -> float | None:
    # bisect only through the current row. This is the anti-leakage invariant:
    # no future snapshot can contribute to a feature at time t.
    idx = bisect.bisect_right(timestamps, target_ts_ns, hi=current_index + 1) - 1
    if idx < 0:
        return None
    return values[idx]


def _lag_change(
    timestamps: list[int],
    values: list[float | None],
    *,
    current_index: int,
    lag_seconds: int,
) -> float | None:
    current = values[current_index]
    if current is None:
        return None
    target = timestamps[current_index] - lag_seconds * 1_000_000_000
    previous = _asof_value(
        timestamps,
        values,
        current_index=current_index,
        target_ts_ns=target,
    )
    return None if previous is None else current - previous


def build_market_dataset_rows(
    market_dir: str | Path,
    *,
    sample_interval_ms: int = 1000,
    lag_seconds: tuple[int, ...] = DEFAULT_LAGS_SECONDS,
) -> list[dict[str, Any]]:
    raw_rows = list(iter_snapshot_rows(market_dir))
    sampled = _downsample(raw_rows, sample_interval_ms)
    features = [feature_payload(build_feature_row(row)) for row in sampled]
    timestamps = [int(row["received_ts_ns"]) for row in features]
    up_probabilities = [row.get("up_market_probability") for row in features]
    up_mids = [row.get("up_mid") for row in features]

    resolved, winning_side = read_resolution(market_dir)
    label_up = None if not resolved or winning_side not in {"UP", "DOWN"} else int(winning_side == "UP")

    output: list[dict[str, Any]] = []
    for idx, feature in enumerate(features):
        row = dict(feature)
        row["ask_overround"] = None if row.get("ask_sum") is None else float(row["ask_sum"]) - 1.0
        up_prob = row.get("up_market_probability")
        row["market_skew_up"] = None if up_prob is None else float(up_prob) - 0.5

        for lag in DEFAULT_LAGS_SECONDS:
            row[f"up_probability_change_{lag}s"] = None
            row[f"up_mid_change_{lag}s"] = None
        for lag in lag_seconds:
            if lag not in DEFAULT_LAGS_SECONDS:
                continue
            row[f"up_probability_change_{lag}s"] = _lag_change(
                timestamps, up_probabilities, current_index=idx, lag_seconds=lag
            )
            row[f"up_mid_change_{lag}s"] = _lag_change(
                timestamps, up_mids, current_index=idx, lag_seconds=lag
            )

        for name in (
            "btc_reference",
            "btc_current",
            "btc_delta_usd",
            "btc_momentum_15s",
            "btc_momentum_30s",
            "btc_momentum_60s",
            "btc_realized_vol_30s",
            "btc_realized_vol_60s",
            "btc_impulse_z",
        ):
            row.setdefault(name, None)

        row["resolved"] = resolved
        row["winning_side"] = winning_side
        row["label_up"] = label_up
        selected_side = row.get("selected_side")
        row["selected_won"] = (
            None
            if not resolved or selected_side not in {"UP", "DOWN"} or winning_side is None
            else selected_side == winning_side
        )
        output.append(row)
    return output


def market_dirs_from_output_root(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    return sorted(path.parent for path in root.glob("*/*/metadata.json"))


def build_dataset_rows(
    market_dirs: Iterable[str | Path],
    *,
    sample_interval_ms: int = 1000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market_dir in market_dirs:
        rows.extend(
            build_market_dataset_rows(
                market_dir,
                sample_interval_ms=sample_interval_ms,
            )
        )
    rows.sort(key=lambda row: int(row["received_ts_ns"]))
    return rows


def write_dataset(
    market_dirs: Iterable[str | Path],
    output_path: str | Path,
    *,
    sample_interval_ms: int = 1000,
) -> dict[str, Any]:
    rows = build_dataset_rows(market_dirs, sample_interval_ms=sample_interval_ms)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=DATASET_SCHEMA)
    pq.write_table(table, destination, compression="zstd")
    summary = {
        "output": str(destination),
        "rows": len(rows),
        "resolved_rows": sum(1 for row in rows if row["resolved"]),
        "markets": len({row["slug"] for row in rows}),
        "sample_interval_ms": sample_interval_ms,
        "btc_features_populated": any(row.get("btc_current") is not None for row in rows),
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
