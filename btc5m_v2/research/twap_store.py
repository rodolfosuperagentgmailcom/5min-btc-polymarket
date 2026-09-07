from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.feeds.polymarket_twap import (
    CHAINLINK_TWAP_RESOLUTION_SOURCE,
    TwapObservation,
    twap_events,
)

TWAP_SCHEMA = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("price", pa.float64()),
        ("exact_value", pa.string()),
        ("full_accuracy_value", pa.string()),
        ("chainlink_ts_ms", pa.int64()),
        ("publisher_ts_ms", pa.int64()),
        ("received_ts_ns", pa.int64()),
        ("chainlink_age_ms", pa.float64()),
        ("symbol", pa.string()),
        ("window_seconds", pa.int32()),
    ]
)


def market_start_ms_from_slug(slug: str) -> int:
    prefix = "btc-updown-5m-"
    value = str(slug).strip()
    if not value.startswith(prefix):
        raise ValueError("unsupported_market_slug")
    raw = value[len(prefix) :]
    if not raw.isdigit():
        raise ValueError("market_slug_missing_unix_start")
    return int(raw) * 1000


def _observation_row(observation: TwapObservation) -> dict[str, Any]:
    return {
        "ts_ns": observation.chainlink_ts_ms * 1_000_000,
        "price": float(observation.value),
        "exact_value": format(observation.value, "f"),
        "full_accuracy_value": observation.full_accuracy_value,
        "chainlink_ts_ms": observation.chainlink_ts_ms,
        "publisher_ts_ms": observation.publisher_ts_ms,
        "received_ts_ns": observation.received_ts_ns,
        "chainlink_age_ms": observation.chainlink_age_ms,
        "symbol": observation.symbol,
        "window_seconds": observation.window_seconds,
    }


@dataclass
class TwapStore:
    output_root: Path
    batch_rows: int = 100

    def __post_init__(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.output_root / "btc_usd_twap_60s_raw.jsonl"
        self.metadata_path = self.output_root / "btc_usd_twap_60s_metadata.json"
        self.rows: list[dict[str, Any]] = []
        self.part_number = len(sorted(self.output_root.glob("btc_usd_twap_60s_part-*.parquet")))
        self.observations = 0
        self.reconnects = 0
        self.started_ts_ns = time.time_ns()
        self._write_metadata(status="recording")

    def _write_metadata(self, *, status: str) -> None:
        payload = {
            "status": status,
            "source": CHAINLINK_TWAP_RESOLUTION_SOURCE,
            "transport": "polymarket_rtds",
            "rtds_topic": "crypto_prices_twap_sixty",
            "symbol": "btc/usd",
            "window_seconds": 60,
            "credentials_required": False,
            "started_ts_ns": self.started_ts_ns,
            "observations": self.observations,
            "reconnects": self.reconnects,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append(self, observation: TwapObservation) -> None:
        raw = {
            "received_ts_ns": observation.received_ts_ns,
            "chainlink_ts_ms": observation.chainlink_ts_ms,
            "publisher_ts_ms": observation.publisher_ts_ms,
            "chainlink_age_ms": observation.chainlink_age_ms,
            "event": observation.raw_event,
        }
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()
        self.rows.append(_observation_row(observation))
        self.observations += 1
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> Path | None:
        if not self.rows:
            return None
        path = self.output_root / f"btc_usd_twap_60s_part-{self.part_number:06d}.parquet"
        table = pa.Table.from_pylist(self.rows, schema=TWAP_SCHEMA)
        pq.write_table(table, path, compression="zstd")
        self.rows.clear()
        self.part_number += 1
        self._write_metadata(status="recording")
        return path

    def close(self, *, status: str = "complete") -> None:
        self.flush()
        self._write_metadata(status=status)


async def record_twap(
    *,
    duration_sec: float,
    output_root: str | Path = "runtime/twap",
    batch_rows: int = 100,
) -> TwapStore:
    store = TwapStore(Path(output_root), batch_rows=batch_rows)
    stop_at = time.time() + max(1.0, float(duration_sec))
    status = "complete"
    try:
        while time.time() < stop_at:
            try:
                remaining = max(0.1, stop_at - time.time())
                async with asyncio.timeout(remaining):
                    async for observation in twap_events(symbol="btc/usd", window_seconds=60):
                        store.append(observation)
                        if time.time() >= stop_at:
                            break
                break
            except TimeoutError:
                break
            except Exception:
                store.reconnects += 1
                store._write_metadata(status="reconnecting")
                if time.time() >= stop_at:
                    break
                await asyncio.sleep(min(1.0, max(0.0, stop_at - time.time())))
    except Exception:
        status = "error"
        raise
    finally:
        store.close(status=status)
    return store


def load_global_twap_rows(output_root: str | Path = "runtime/twap") -> list[dict[str, Any]]:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("btc_usd_twap_60s_part-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["chainlink_ts_ms"]))
    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        deduped[int(row["chainlink_ts_ms"])] = row
    return [deduped[key] for key in sorted(deduped)]


def _market_end_ms(metadata: dict[str, Any], start_ms: int) -> int:
    # BTC 5m markets are defined by their 300-second slug bucket. Prefer that
    # deterministic boundary over parsing display strings.
    return start_ms + 300_000


def materialize_market_twap(
    market_dir: str | Path,
    global_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    root = Path(market_dir)
    market_metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    slug = str(market_metadata.get("slug") or root.name)
    expected_source = str(market_metadata.get("resolution_source") or "").strip().rstrip("/")
    actual_source = CHAINLINK_TWAP_RESOLUTION_SOURCE.rstrip("/")
    if expected_source != actual_source:
        raise ValueError("market_resolution_source_not_chainlink_twap_60s")

    start_ms = market_start_ms_from_slug(slug)
    end_ms = _market_end_ms(market_metadata, start_ms)
    rows = [
        dict(row)
        for row in global_rows
        if start_ms <= int(row["chainlink_ts_ms"]) <= end_ms
        and str(row.get("symbol") or "").lower() == "btc/usd"
        and int(row.get("window_seconds") or 0) == 60
    ]
    rows.sort(key=lambda row: int(row["chainlink_ts_ms"]))
    if not rows:
        return {
            "slug": slug,
            "samples": 0,
            "reference_price": None,
            "reference_status": "no_samples_in_market_window",
        }

    exact_reference = next(
        (row for row in rows if int(row["chainlink_ts_ms"]) == start_ms),
        None,
    )
    reference_price = None if exact_reference is None else float(exact_reference["price"])
    reference_status = (
        "exact_market_start_observation"
        if exact_reference is not None
        else "missing_exact_market_start_observation"
    )

    sample_table = pa.Table.from_pylist(
        [
            {
                "ts_ns": int(row["chainlink_ts_ms"]) * 1_000_000,
                "price": float(row["price"]),
                "exact_value": str(row.get("exact_value") or ""),
                "chainlink_ts_ms": int(row["chainlink_ts_ms"]),
                "received_ts_ns": int(row["received_ts_ns"]),
                "chainlink_age_ms": float(row["chainlink_age_ms"]),
            }
            for row in rows
        ]
    )
    pq.write_table(sample_table, root / "btc_samples.parquet", compression="zstd")

    btc_metadata = {
        "source": CHAINLINK_TWAP_RESOLUTION_SOURCE,
        "transport": "polymarket_rtds",
        "topic": "crypto_prices_twap_sixty",
        "symbol": "btc/usd",
        "window_seconds": 60,
        "verified": True,
        "market_start_ts_ms": start_ms,
        "market_end_ts_ms": end_ms,
        "samples": len(rows),
        "reference_price": reference_price,
        "reference_status": reference_status,
        "reference_chainlink_ts_ms": None if exact_reference is None else start_ms,
        "credentials_required": False,
    }
    (root / "btc_metadata.json").write_text(
        json.dumps(btc_metadata, indent=2), encoding="utf-8"
    )
    return {"slug": slug, **btc_metadata}


def materialize_output_root(
    data_root: str | Path = "runtime/data",
    twap_root: str | Path = "runtime/twap",
) -> dict[str, Any]:
    global_rows = load_global_twap_rows(twap_root)
    markets = sorted(Path(data_root).glob("*/*/metadata.json"))
    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for metadata_path in markets:
        try:
            results.append(materialize_market_twap(metadata_path.parent, global_rows))
        except Exception as exc:
            failed.append({"market": metadata_path.parent.name, "error": str(exc)})
    return {
        "global_samples": len(global_rows),
        "markets_scanned": len(markets),
        "markets_materialized": sum(1 for item in results if int(item.get("samples") or 0) > 0),
        "markets_with_exact_reference": sum(
            1 for item in results if item.get("reference_status") == "exact_market_start_observation"
        ),
        "results": results,
        "failed": failed,
    }
