from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.config import V2Config, load_config
from btc5m_v2.feeds.polymarket_ws import BookState, WsEnvelope, market_events
from btc5m_v2.market import MarketInfo, discover_current_market

UTC = dt.timezone.utc


def utc_date_from_ns(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1_000_000_000, tz=UTC).date().isoformat()


def event_timestamp_ms(event: dict[str, Any]) -> int | None:
    raw = event.get("timestamp")
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
        if value < 1e11:
            value *= 1000.0
        return int(value)
    except Exception:
        return None


@dataclass
class MarketRecorder:
    market: MarketInfo
    output_root: Path
    parquet_batch_rows: int = 100

    def __post_init__(self) -> None:
        now_ns = time.time_ns()
        date = utc_date_from_ns(now_ns)
        self.market_dir = self.output_root / date / self.market.slug
        self.market_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.market_dir / "clob_raw.jsonl"
        self.metadata_path = self.market_dir / "metadata.json"
        self.up = BookState(self.market.up_token_id)
        self.down = BookState(self.market.down_token_id)
        self.rows: list[dict[str, Any]] = []
        existing = sorted(self.market_dir.glob("clob_snapshots_part-*.parquet"))
        self.part_number = len(existing)
        self.raw_events = 0
        self.snapshots = 0
        self.started_ts_ns = now_ns
        self._write_metadata(status="recording")

    def _write_metadata(self, *, status: str) -> None:
        payload = {
            "status": status,
            "slug": self.market.slug,
            "condition_id": self.market.condition_id,
            "up_token_id": self.market.up_token_id,
            "down_token_id": self.market.down_token_id,
            "market_end": self.market.end_iso,
            "resolution_source": self.market.resolution_source,
            "started_ts_ns": self.started_ts_ns,
            "raw_events": self.raw_events,
            "snapshots": self.snapshots,
            "credentials_loaded": False,
            "order_path_used": False,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_raw(self, envelope: WsEnvelope) -> None:
        row = {
            "received_ts_ns": envelope.received_ts_ns,
            "received_iso": envelope.received_iso,
            "exchange_ts_ms": event_timestamp_ms(envelope.event),
            "event_type": envelope.event.get("event_type"),
            "event": envelope.event,
        }
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()
        self.raw_events += 1

    def _snapshot_row(self, envelope: WsEnvelope) -> dict[str, Any] | None:
        if self.up.best_bid is None or self.up.best_ask is None:
            return None
        if self.down.best_bid is None or self.down.best_ask is None:
            return None

        up = self.up.snapshot("UP", envelope.received_ts_ns)
        down = self.down.snapshot("DOWN", envelope.received_ts_ns)
        return {
            "slug": self.market.slug,
            "condition_id": self.market.condition_id,
            "resolution_source": self.market.resolution_source,
            "market_end": self.market.end_iso,
            "received_ts_ns": envelope.received_ts_ns,
            "received_iso": envelope.received_iso,
            "event_type": str(envelope.event.get("event_type") or ""),
            "event_exchange_ts_ms": event_timestamp_ms(envelope.event),
            "seconds_left": self.market.seconds_left(envelope.received_ts_ns / 1_000_000_000),
            "up_bid": up["bid"],
            "up_ask": up["ask"],
            "up_spread": up["spread"],
            "up_bid_depth_3_usd": up["bid_depth_3_usd"],
            "up_ask_depth_3_usd": up["ask_depth_3_usd"],
            "up_exchange_ts_ms": up["exchange_ts_ms"],
            "up_exchange_age_ms": up["exchange_age_ms"],
            "down_bid": down["bid"],
            "down_ask": down["ask"],
            "down_spread": down["spread"],
            "down_bid_depth_3_usd": down["bid_depth_3_usd"],
            "down_ask_depth_3_usd": down["ask_depth_3_usd"],
            "down_exchange_ts_ms": down["exchange_ts_ms"],
            "down_exchange_age_ms": down["exchange_age_ms"],
        }

    def apply(self, envelope: WsEnvelope) -> bool:
        self.append_raw(envelope)
        up_changed = self.up.apply(envelope.event, envelope.received_ts_ns)
        down_changed = self.down.apply(envelope.event, envelope.received_ts_ns)
        if not (up_changed or down_changed):
            return False

        row = self._snapshot_row(envelope)
        if row is None:
            return False
        self.rows.append(row)
        self.snapshots += 1
        if len(self.rows) >= self.parquet_batch_rows:
            self.flush_parquet()
        return True

    def flush_parquet(self) -> Path | None:
        if not self.rows:
            return None
        path = self.market_dir / f"clob_snapshots_part-{self.part_number:06d}.parquet"
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, path, compression="zstd")
        self.rows.clear()
        self.part_number += 1
        self._write_metadata(status="recording")
        return path

    def close(self, *, status: str = "complete") -> None:
        self.flush_parquet()
        self._write_metadata(status=status)


async def record_one_market(
    market: MarketInfo,
    *,
    output_root: Path,
    stop_ts: float,
    parquet_batch_rows: int = 100,
) -> MarketRecorder:
    recorder = MarketRecorder(market, output_root=output_root, parquet_batch_rows=parquet_batch_rows)
    token_ids = [market.up_token_id, market.down_token_id]
    timeout_sec = max(0.1, min(stop_ts, market.end_ts + 2.0) - time.time())
    status = "complete"
    try:
        async with asyncio.timeout(timeout_sec):
            async for envelope in market_events(token_ids):
                recorder.apply(envelope)
                if time.time() >= stop_ts or time.time() >= market.end_ts + 1.0:
                    break
    except TimeoutError:
        status = "timeout"
    except Exception:
        status = "error"
        raise
    finally:
        recorder.close(status=status)
    return recorder


async def run_recorder(
    *,
    duration_sec: float,
    config: V2Config | None = None,
    output_root: str | Path = "runtime/data",
    parquet_batch_rows: int = 100,
) -> list[MarketRecorder]:
    cfg = config or load_config()
    root = Path(output_root)
    stop_ts = time.time() + max(1.0, float(duration_sec))
    completed: list[MarketRecorder] = []
    last_slug: str | None = None

    while time.time() < stop_ts:
        market = discover_current_market(cfg)
        if market is None or market.slug == last_slug:
            await asyncio.sleep(1.0)
            continue
        last_slug = market.slug
        recorder = await record_one_market(
            market,
            output_root=root,
            stop_ts=stop_ts,
            parquet_batch_rows=parquet_batch_rows,
        )
        completed.append(recorder)

    return completed
