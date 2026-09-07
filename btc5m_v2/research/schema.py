from __future__ import annotations

import hashlib
import json
from typing import Any

import pyarrow as pa

FEATURE_DATASET_VERSION = "btc5m-features-v1"

FEATURE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("snapshot_id", pa.string()),
        ("config_hash", pa.string()),
        ("slug", pa.string()),
        ("condition_id", pa.string()),
        ("up_token_id", pa.string()),
        ("down_token_id", pa.string()),
        ("resolution_source", pa.string()),
        ("market_end", pa.string()),
        ("received_ts_ns", pa.int64()),
        ("received_iso", pa.string()),
        ("event_type", pa.string()),
        ("event_exchange_ts_ms", pa.int64()),
        ("seconds_left", pa.float64()),
        ("up_bid", pa.float64()),
        ("up_ask", pa.float64()),
        ("up_spread", pa.float64()),
        ("up_bid_depth_3_usd", pa.float64()),
        ("up_ask_depth_3_usd", pa.float64()),
        ("up_obi_3", pa.float64()),
        ("up_exchange_ts_ms", pa.int64()),
        ("up_exchange_age_ms", pa.float64()),
        ("down_bid", pa.float64()),
        ("down_ask", pa.float64()),
        ("down_spread", pa.float64()),
        ("down_bid_depth_3_usd", pa.float64()),
        ("down_ask_depth_3_usd", pa.float64()),
        ("down_obi_3", pa.float64()),
        ("down_exchange_ts_ms", pa.int64()),
        ("down_exchange_age_ms", pa.float64()),
        ("btc_reference", pa.float64()),
        ("btc_reference_market_ts_ns", pa.int64()),
        ("btc_reference_source_ts_ms", pa.int64()),
        ("btc_reference_received_ts_ns", pa.int64()),
        ("btc_reference_source_gap_ms", pa.float64()),
        ("btc_elapsed_from_reference_ms", pa.float64()),
        ("btc_reference_method", pa.string()),
        ("btc_current", pa.float64()),
        ("btc_current_ts_ns", pa.int64()),
        ("btc_source_ts_ms", pa.int64()),
        ("btc_age_ms", pa.float64()),
        ("btc_move_usd", pa.float64()),
        ("momentum_5s", pa.float64()),
        ("momentum_15s", pa.float64()),
        ("momentum_30s", pa.float64()),
        ("momentum_60s", pa.float64()),
        ("realized_vol_30s_usd_sqrt_sec", pa.float64()),
        ("realized_vol_60s_usd_sqrt_sec", pa.float64()),
        ("impulse_z", pa.float64()),
        ("fees_enabled", pa.bool_()),
        ("taker_fee_rate", pa.float64()),
        ("feature_ready", pa.bool_()),
        ("skip_reason", pa.string()),
        ("model_version", pa.string()),
        ("model_probability_up", pa.float64()),
        ("up_breakeven", pa.float64()),
        ("down_breakeven", pa.float64()),
        ("edge_up", pa.float64()),
        ("edge_down", pa.float64()),
        ("signal", pa.string()),
        ("fill_price", pa.float64()),
        ("fee", pa.float64()),
        ("resolution_outcome", pa.string()),
        ("label_up", pa.bool_()),
        ("label_source", pa.string()),
        ("pnl", pa.float64()),
    ]
)


def config_hash(raw_config: dict[str, Any]) -> str:
    encoded = json.dumps(raw_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_snapshot_id(
    *,
    slug: str,
    condition_id: str,
    received_ts_ns: int,
    event_exchange_ts_ms: int | None,
) -> str:
    material = "|".join(
        [
            str(slug),
            str(condition_id),
            str(int(received_ts_ns)),
            "" if event_exchange_ts_ms is None else str(int(event_exchange_ts_ms)),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def empty_model_fields() -> dict[str, Any]:
    return {
        "model_version": None,
        "model_probability_up": None,
        "up_breakeven": None,
        "down_breakeven": None,
        "edge_up": None,
        "edge_down": None,
        "signal": None,
        "fill_price": None,
        "fee": None,
        "pnl": None,
    }


def empty_label_fields() -> dict[str, Any]:
    return {
        "resolution_outcome": None,
        "label_up": None,
        "label_source": None,
    }
