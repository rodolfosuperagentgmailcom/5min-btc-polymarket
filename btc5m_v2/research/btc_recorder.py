from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.config import V2Config, load_config
from btc5m_v2.feeds.polymarket_twap import TwapObservation, twap_events
from btc5m_v2.market import MarketInfo, discover_current_market

BTC_SCHEMA = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("price", pa.float64()),
        ("chainlink_ts_ms", pa.int64()),
        ("publisher_ts_ms", pa.int64()),
        ("received_iso", pa.string()),
        ("chainlink_age_ms", pa.float64()),
        ("symbol", pa.string()),
        ("window_seconds", pa.int32()),
        ("full_accuracy_value", pa.string()),
    ]
)


def _existing_or_new_market_dir(output_root: Path, market: MarketInfo) -> Path:
    matches = sorted(output_root.glob(f"*/{market.slug}/metadata.json"))
    if matches:
        return matches[-1].parent
    date = time.strftime("%Y-%m-%d", time.gmtime())
    path = output_root / date / market.slug
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class BTCMarketRecorder:
    market: MarketInfo
    output_root: Path
    reference_tolerance_ms: int = 5_000
    checkpoint_rows: int = 25

    def __post_init__(self) -> None:
        self.market_dir = _existing_or_new_market_dir(self.output_root, self.market)
        self.market_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.market_dir / "btc_raw.jsonl"
        self.samples_path = self.market_dir / "btc_samples.parquet"
        self.metadata_path = self.market_dir / "btc_metadata.json"
        self.rows: list[dict[str, Any]] = []
        self.samples = 0
        self.reconnects = 0
        self.started_ts_ns = time.time_ns()
        self.reference_price: float | None = None
        self.reference_chainlink_ts_ms: int | None = None
        self.reference_offset_ms: int | None = None
        self._write_metadata(status="recording")

    @property
    def market_start_ts_ms(self) -> int:
        return int(round((self.market.end_ts - 300.0) * 1000.0))

    def _consider_reference(self, observation: TwapObservation) -> None:
        offset = int(observation.chainlink_ts_ms - self.market_start_ts_ms)
        if abs(offset) > int(self.reference_tolerance_ms):
            return
        if self.reference_offset_ms is None or abs(offset) < abs(self.reference_offset_ms):
            self.reference_price = float(observation.value)
            self.reference_chainlink_ts_ms = int(observation.chainlink_ts_ms)
            self.reference_offset_ms = offset

    def _write_metadata(self, *, status: str) -> None:
        payload = {
            "status": status,
            "slug": self.market.slug,
            "source": self.market.resolution_source,
            "symbol": "btc/usd",
            "window_seconds": 60,
            "market_start_ts_ms": self.market_start_ts_ms,
            "market_end_ts_ms": int(round(self.market.end_ts * 1000.0)),
            "reference_price": self.reference_price,
            "reference_chainlink_ts_ms": self.reference_chainlink_ts_ms,
            "reference_offset_ms": self.reference_offset_ms,
            "reference_tolerance_ms": int(self.reference_tolerance_ms),
            "reference_verified": self.reference_price is not None,
            "reference_method": "nearest_public_twap_observation_to_market_start_within_tolerance",
            "timestamp_semantics": {
                "ts_ns": "local_receive_timestamp_used_for_causal_asof_join",
                "chainlink_ts_ms": "source_timestamp_for_latency_and_reference_alignment",
            },
            "samples": self.samples,
            "reconnects": self.reconnects,
            "started_ts_ns": self.started_ts_ns,
            "credentials_loaded": False,
            "order_path_used": False,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append(self, observation: TwapObservation) -> None:
        raw = {
            "received_ts_ns": observation.received_ts_ns,
            "received_iso": observation.received_iso,
            "chainlink_ts_ms": observation.chainlink_ts_ms,
            "publisher_ts_ms": observation.publisher_ts_ms,
            "price": str(observation.value),
            "chainlink_age_ms": observation.chainlink_age_ms,
            "event": observation.raw_event,
        }
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()

        self._consider_reference(observation)
        self.rows.append(
            {
                # Use receive time for causal joins. A source timestamp can be
                # earlier than the moment our strategy actually knew the value.
                "ts_ns": int(observation.received_ts_ns),
                "price": float(observation.value),
                "chainlink_ts_ms": int(observation.chainlink_ts_ms),
                "publisher_ts_ms": observation.publisher_ts_ms,
                "received_iso": observation.received_iso,
                "chainlink_age_ms": float(observation.chainlink_age_ms),
                "symbol": observation.symbol,
                "window_seconds": int(observation.window_seconds),
                "full_accuracy_value": observation.full_accuracy_value,
            }
        )
        self.samples += 1
        if self.samples % max(1, int(self.checkpoint_rows)) == 0:
            self.checkpoint()

    def checkpoint(self) -> Path | None:
        if not self.rows:
            return None
        table = pa.Table.from_pylist(self.rows, schema=BTC_SCHEMA)
        temp_path = self.samples_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temp_path, compression="zstd")
        os.replace(temp_path, self.samples_path)
        self._write_metadata(status="recording")
        return self.samples_path

    def close(self, *, status: str = "complete") -> None:
        self.checkpoint()
        self._write_metadata(status=status)


async def record_btc_one_market(
    market: MarketInfo,
    *,
    output_root: Path,
    stop_ts: float,
    reference_tolerance_ms: int = 5_000,
) -> BTCMarketRecorder:
    recorder = BTCMarketRecorder(
        market,
        output_root=output_root,
        reference_tolerance_ms=reference_tolerance_ms,
    )
    final_ts = min(stop_ts, market.end_ts + 2.0)
    status = "complete"

    try:
        while time.time() < final_ts:
            remaining = max(0.1, final_ts - time.time())
            try:
                async with asyncio.timeout(remaining):
                    async for observation in twap_events(symbol="btc/usd", window_seconds=60):
                        recorder.append(observation)
                        if time.time() >= final_ts:
                            break
                break
            except TimeoutError:
                status = "timeout"
                break
            except Exception:
                recorder.reconnects += 1
                recorder._write_metadata(status="reconnecting")
                if time.time() >= final_ts:
                    break
                await asyncio.sleep(min(1.0, max(0.0, final_ts - time.time())))
    except Exception:
        status = "error"
        raise
    finally:
        recorder.close(status=status)
    return recorder


async def run_btc_recorder(
    *,
    duration_sec: float,
    config: V2Config | None = None,
    output_root: str | Path = "runtime/data",
    reference_tolerance_ms: int = 5_000,
) -> list[BTCMarketRecorder]:
    cfg = config or load_config()
    root = Path(output_root)
    stop_ts = time.time() + max(1.0, float(duration_sec))
    completed: list[BTCMarketRecorder] = []
    last_slug: str | None = None

    while time.time() < stop_ts:
        market = discover_current_market(cfg)
        if market is None or market.slug == last_slug:
            await asyncio.sleep(1.0)
            continue
        last_slug = market.slug
        recorder = await record_btc_one_market(
            market,
            output_root=root,
            stop_ts=stop_ts,
            reference_tolerance_ms=reference_tolerance_ms,
        )
        completed.append(recorder)

    return completed
