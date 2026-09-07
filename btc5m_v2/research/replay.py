from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.config import V2Config, load_config
from btc5m_v2.research.schema import (
    FEATURE_DATASET_VERSION,
    FEATURE_SCHEMA,
    config_hash,
    empty_label_fields,
    empty_model_fields,
    make_snapshot_id,
)
from btc5m_v2.strategy.features import (
    PriceObservation,
    build_btc_features,
    orderbook_imbalance,
)

UTC = dt.timezone.utc


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def market_start_ns(slug: str, market_end: str | None = None) -> int:
    """Use the 5m slug bucket timestamp; fall back to end minus five minutes."""

    suffix = str(slug).rsplit("-", 1)[-1]
    try:
        start_sec = int(suffix)
        if start_sec >= 1_000_000_000:
            return start_sec * 1_000_000_000
    except ValueError:
        pass

    if not market_end:
        raise ValueError("cannot determine market start without timestamp slug or market_end")
    end_ts = dt.datetime.fromisoformat(str(market_end).replace("Z", "+00:00")).timestamp()
    return int((end_ts - 300.0) * 1_000_000_000)


def load_price_observations_jsonl(path: str | Path) -> list[PriceObservation]:
    """Load normalized BTC observations.

    Expected JSONL fields:
      price: positive number
      ts_ns or received_ts_ns: local causal-availability timestamp
      source_ts_ms: optional source timestamp for latency/reference analysis
    """

    observations: list[PriceObservation] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        price = _float_or_none(payload.get("price"))
        ts_ns = _int_or_none(payload.get("ts_ns"))
        if ts_ns is None:
            ts_ns = _int_or_none(payload.get("received_ts_ns"))
        if price is None or ts_ns is None:
            raise ValueError(f"invalid BTC observation at line {line_number}")
        observations.append(
            PriceObservation(
                ts_ns=ts_ns,
                price=price,
                source_ts_ms=_int_or_none(payload.get("source_ts_ms")),
                received_ts_ns=_int_or_none(payload.get("received_ts_ns")),
            )
        )
    return sorted(observations, key=lambda obs: obs.ts_ns)


def load_clob_snapshots(market_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(market_dir)
    paths = sorted(root.glob("clob_snapshots_part-*.parquet"))
    if not paths:
        return []
    tables = [pq.read_table(path) for path in paths]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return sorted(table.to_pylist(), key=lambda row: int(row["received_ts_ns"]))


def load_metadata(market_dir: str | Path) -> dict[str, Any]:
    path = Path(market_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _observation_source_ts_ms(obs: PriceObservation) -> int:
    if obs.source_ts_ms is not None:
        return obs.source_ts_ms
    return obs.ts_ns // 1_000_000


def reference_observation_asof(
    observations: Iterable[PriceObservation],
    *,
    market_start_ts_ns: int,
    now_ns: int,
    max_source_gap_sec: float = 3.0,
) -> PriceObservation | None:
    """Find the latest start-reference observation that was actually available by now_ns."""

    start_ms = market_start_ts_ns // 1_000_000
    eligible = [
        obs
        for obs in observations
        if obs.ts_ns <= now_ns and _observation_source_ts_ms(obs) <= start_ms
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda obs: (_observation_source_ts_ms(obs), obs.ts_ns))
    source_gap_ms = start_ms - _observation_source_ts_ms(best)
    if source_gap_ms < 0 or source_gap_ms > max_source_gap_sec * 1000.0:
        return None
    return best


def _feature_readiness_reasons(
    *,
    snapshot: dict[str, Any],
    btc: Any,
    clob_max_age_sec: float,
    btc_max_age_sec: float,
) -> list[str]:
    reasons: list[str] = []
    if btc.current_price is None:
        reasons.append("missing_btc_current")
    elif btc.current_age_ms is None or btc.current_age_ms > btc_max_age_sec * 1000.0:
        reasons.append("stale_btc")

    if btc.reference_price is None:
        reasons.append("missing_btc_reference")
    if btc.momentum_15s is None:
        reasons.append("missing_momentum_15s")
    if btc.momentum_30s is None:
        reasons.append("missing_momentum_30s")
    if btc.momentum_60s is None:
        reasons.append("missing_momentum_60s")
    if btc.realized_vol_60s is None:
        reasons.append("missing_realized_vol_60s")
    if btc.impulse_z is None:
        reasons.append("missing_impulse_z")

    for side in ("up", "down"):
        age = _float_or_none(snapshot.get(f"{side}_exchange_age_ms"))
        if age is None:
            reasons.append(f"missing_{side}_book_age")
        elif age > clob_max_age_sec * 1000.0:
            reasons.append(f"stale_{side}_book")
    return reasons


def build_feature_row(
    snapshot: dict[str, Any],
    *,
    metadata: dict[str, Any],
    btc_observations: Iterable[PriceObservation],
    config: V2Config,
) -> dict[str, Any]:
    now_ns = int(snapshot["received_ts_ns"])
    observations = list(btc_observations)
    start_ns = market_start_ns(
        str(snapshot["slug"]),
        str(snapshot.get("market_end") or metadata.get("market_end") or ""),
    )
    btc_max_age_sec = float(config.raw.get("freshness", {}).get("btc_max_age_sec", 3.0))
    reference = reference_observation_asof(
        observations,
        market_start_ts_ns=start_ns,
        now_ns=now_ns,
        max_source_gap_sec=btc_max_age_sec,
    )
    btc = build_btc_features(
        observations,
        now_ns=now_ns,
        reference_price=None if reference is None else reference.price,
        reference_ts_ns=None if reference is None else start_ns,
    )
    reasons = _feature_readiness_reasons(
        snapshot=snapshot,
        btc=btc,
        clob_max_age_sec=config.clob_max_age_sec,
        btc_max_age_sec=btc_max_age_sec,
    )

    event_exchange_ts_ms = _int_or_none(snapshot.get("event_exchange_ts_ms"))
    row: dict[str, Any] = {
        "schema_version": FEATURE_DATASET_VERSION,
        "snapshot_id": make_snapshot_id(
            slug=str(snapshot["slug"]),
            condition_id=str(snapshot.get("condition_id") or metadata.get("condition_id") or ""),
            received_ts_ns=now_ns,
            event_exchange_ts_ms=event_exchange_ts_ms,
        ),
        "config_hash": config_hash(config.raw),
        "slug": str(snapshot["slug"]),
        "condition_id": str(snapshot.get("condition_id") or metadata.get("condition_id") or ""),
        "up_token_id": str(metadata.get("up_token_id") or ""),
        "down_token_id": str(metadata.get("down_token_id") or ""),
        "resolution_source": str(snapshot.get("resolution_source") or metadata.get("resolution_source") or ""),
        "market_end": str(snapshot.get("market_end") or metadata.get("market_end") or ""),
        "received_ts_ns": now_ns,
        "received_iso": str(snapshot.get("received_iso") or ""),
        "event_type": str(snapshot.get("event_type") or ""),
        "event_exchange_ts_ms": event_exchange_ts_ms,
        "seconds_left": _float_or_none(snapshot.get("seconds_left")),
        "up_bid": _float_or_none(snapshot.get("up_bid")),
        "up_ask": _float_or_none(snapshot.get("up_ask")),
        "up_spread": _float_or_none(snapshot.get("up_spread")),
        "up_bid_depth_3_usd": _float_or_none(snapshot.get("up_bid_depth_3_usd")),
        "up_ask_depth_3_usd": _float_or_none(snapshot.get("up_ask_depth_3_usd")),
        "up_obi_3": orderbook_imbalance(
            _float_or_none(snapshot.get("up_bid_depth_3_usd")),
            _float_or_none(snapshot.get("up_ask_depth_3_usd")),
        ),
        "up_exchange_ts_ms": _int_or_none(snapshot.get("up_exchange_ts_ms")),
        "up_exchange_age_ms": _float_or_none(snapshot.get("up_exchange_age_ms")),
        "down_bid": _float_or_none(snapshot.get("down_bid")),
        "down_ask": _float_or_none(snapshot.get("down_ask")),
        "down_spread": _float_or_none(snapshot.get("down_spread")),
        "down_bid_depth_3_usd": _float_or_none(snapshot.get("down_bid_depth_3_usd")),
        "down_ask_depth_3_usd": _float_or_none(snapshot.get("down_ask_depth_3_usd")),
        "down_obi_3": orderbook_imbalance(
            _float_or_none(snapshot.get("down_bid_depth_3_usd")),
            _float_or_none(snapshot.get("down_ask_depth_3_usd")),
        ),
        "down_exchange_ts_ms": _int_or_none(snapshot.get("down_exchange_ts_ms")),
        "down_exchange_age_ms": _float_or_none(snapshot.get("down_exchange_age_ms")),
        "btc_reference": btc.reference_price,
        "btc_reference_market_ts_ns": btc.reference_ts_ns,
        "btc_reference_source_ts_ms": None if reference is None else _observation_source_ts_ms(reference),
        "btc_reference_received_ts_ns": None if reference is None else reference.ts_ns,
        "btc_reference_source_gap_ms": (
            None
            if reference is None
            else (start_ns // 1_000_000) - _observation_source_ts_ms(reference)
        ),
        "btc_elapsed_from_reference_ms": btc.elapsed_from_reference_ms,
        "btc_reference_method": None if reference is None else "latest_source_at_or_before_market_start",
        "btc_current": btc.current_price,
        "btc_current_ts_ns": btc.current_ts_ns,
        "btc_source_ts_ms": btc.current_source_ts_ms,
        "btc_age_ms": btc.current_age_ms,
        "btc_move_usd": btc.move_usd,
        "momentum_5s": btc.momentum_5s,
        "momentum_15s": btc.momentum_15s,
        "momentum_30s": btc.momentum_30s,
        "momentum_60s": btc.momentum_60s,
        "realized_vol_30s_usd_sqrt_sec": btc.realized_vol_30s,
        "realized_vol_60s_usd_sqrt_sec": btc.realized_vol_60s,
        "impulse_z": btc.impulse_z,
        "fees_enabled": None,
        "taker_fee_rate": None,
        "feature_ready": not reasons,
        "skip_reason": None if not reasons else ",".join(sorted(set(reasons))),
    }
    row.update(empty_model_fields())
    row.update(empty_label_fields())
    return row


def attach_resolution_labels(
    rows: Iterable[dict[str, Any]],
    resolution_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach post-hoc labels without changing any causal feature or snapshot identifier."""

    winning_side = None if not resolution_payload else str(resolution_payload.get("winning_side") or "").upper()
    resolved = bool(
        resolution_payload
        and resolution_payload.get("resolved") is True
        and winning_side in {"UP", "DOWN"}
    )
    source = None if not resolution_payload else resolution_payload.get("source")

    labelled: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if resolved:
            row["resolution_outcome"] = winning_side
            row["label_up"] = winning_side == "UP"
            row["label_source"] = None if source is None else str(source)
        labelled.append(row)
    return labelled


def replay_market(
    market_dir: str | Path,
    *,
    btc_observations: Iterable[PriceObservation],
    config: V2Config | None = None,
    attach_labels: bool = False,
) -> list[dict[str, Any]]:
    cfg = config or load_config()
    root = Path(market_dir)
    metadata = load_metadata(root)
    observations = sorted(list(btc_observations), key=lambda obs: obs.ts_ns)
    rows = [
        build_feature_row(snapshot, metadata=metadata, btc_observations=observations, config=cfg)
        for snapshot in load_clob_snapshots(root)
    ]

    if attach_labels:
        resolution_path = root / "resolution.json"
        resolution = json.loads(resolution_path.read_text(encoding="utf-8")) if resolution_path.exists() else None
        rows = attach_resolution_labels(rows, resolution)
    return rows


def write_feature_parquet(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=FEATURE_SCHEMA)
    pq.write_table(table, output, compression="zstd")
    return output


def replay_to_parquet(
    market_dir: str | Path,
    *,
    btc_jsonl: str | Path,
    output_path: str | Path | None = None,
    config: V2Config | None = None,
    attach_labels: bool = False,
) -> Path:
    root = Path(market_dir)
    rows = replay_market(
        root,
        btc_observations=load_price_observations_jsonl(btc_jsonl),
        config=config,
        attach_labels=attach_labels,
    )
    destination = Path(output_path) if output_path is not None else root / "features.parquet"
    return write_feature_parquet(rows, destination)
