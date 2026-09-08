from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from btc5m_v2.research.btc_recorder import run_btc_recorder
from btc5m_v2.research.enrich import enrich_output_root
from btc5m_v2.research.fee_enrich import enrich_fee_schedules
from btc5m_v2.research.readiness import readiness_report
from btc5m_v2.research.ready_dataset import write_ready_dataset
from btc5m_v2.research.recorder import run_recorder

MARKET_SECONDS = 300.0


def study_duration_sec(markets: int, *, tail_sec: float = 1.5) -> float:
    count = int(markets)
    if count <= 0:
        raise ValueError("markets must be positive")
    tail = max(0.0, float(tail_sec))
    return count * MARKET_SECONDS + tail


def compact_readiness(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "markets_seen": int(report.get("markets_seen") or 0),
        "v2_ready_markets": int(report.get("v2_ready_markets") or 0),
        "v2_ready_fraction": report.get("v2_ready_fraction"),
        "resolved_markets": int(report.get("resolved_markets") or 0),
        "decision_window_markets": int(report.get("decision_window_markets") or 0),
        "verified_btc_reference_markets": int(
            report.get("verified_btc_reference_markets") or 0
        ),
        "fee_ready_markets": int(report.get("fee_ready_markets") or 0),
        "reason_counts": dict(report.get("reason_counts") or {}),
    }


async def record_study(
    *,
    markets: int,
    output_root: str | Path,
    parquet_batch_rows: int = 100,
    reference_tolerance_ms: int = 5_000,
    tail_sec: float = 1.5,
) -> tuple[list[Any], list[Any]]:
    duration = study_duration_sec(markets, tail_sec=tail_sec)
    root = Path(output_root)
    clob_task = asyncio.create_task(
        run_recorder(
            duration_sec=duration,
            output_root=root,
            parquet_batch_rows=max(1, int(parquet_batch_rows)),
        )
    )
    btc_task = asyncio.create_task(
        run_btc_recorder(
            duration_sec=duration,
            output_root=root,
            reference_tolerance_ms=max(0, int(reference_tolerance_ms)),
        )
    )
    return await asyncio.gather(clob_task, btc_task)


def finalize_study(
    *,
    output_root: str | Path,
    dataset_path: str | Path | None = None,
    sample_interval_ms: int = 1_000,
) -> dict[str, Any]:
    root = Path(output_root)
    fee_summary = enrich_fee_schedules(root)
    resolution_summary = enrich_output_root(root)
    readiness = readiness_report(root)

    dataset_summary: dict[str, Any] | None = None
    dataset_error: str | None = None
    if dataset_path is not None and int(readiness.get("v2_ready_markets") or 0) > 0:
        try:
            dataset_summary = write_ready_dataset(
                root,
                dataset_path,
                sample_interval_ms=max(1, int(sample_interval_ms)),
            )
        except Exception as exc:
            dataset_error = f"{type(exc).__name__}:{exc}"

    return {
        "fees": fee_summary,
        "resolution": resolution_summary,
        "readiness": compact_readiness(readiness),
        "dataset": dataset_summary,
        "dataset_error": dataset_error,
    }


def study_summary(
    *,
    requested_markets: int,
    output_root: str | Path,
    clob_completed: list[Any],
    btc_completed: list[Any],
    finalized: dict[str, Any],
    started_ts: float,
    finished_ts: float | None = None,
) -> dict[str, Any]:
    finished = time.time() if finished_ts is None else float(finished_ts)
    return {
        "status": "complete",
        "mode": "public-data-only",
        "requested_markets": int(requested_markets),
        "elapsed_sec": max(0.0, finished - float(started_ts)),
        "output_root": str(Path(output_root)),
        "clob_markets_recorded": len(clob_completed),
        "raw_clob_events": sum(int(getattr(item, "raw_events", 0)) for item in clob_completed),
        "clob_snapshots": sum(int(getattr(item, "snapshots", 0)) for item in clob_completed),
        "clob_reconnects": sum(int(getattr(item, "reconnects", 0)) for item in clob_completed),
        "btc_markets_recorded": len(btc_completed),
        "btc_samples": sum(int(getattr(item, "samples", 0)) for item in btc_completed),
        "btc_reconnects": sum(int(getattr(item, "reconnects", 0)) for item in btc_completed),
        "btc_reference_verified_markets": sum(
            1 for item in btc_completed if getattr(item, "reference_price", None) is not None
        ),
        "finalize": finalized,
        "credentials_loaded": False,
        "order_path_used": False,
    }


def write_study_summary(output_root: str | Path, summary: dict[str, Any]) -> Path:
    root = Path(output_root)
    reports = root / "_study_reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = reports / f"study_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
