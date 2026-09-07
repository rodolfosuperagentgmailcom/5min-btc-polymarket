#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from btc5m_v2.research.btc_recorder import run_btc_recorder
from btc5m_v2.research.recorder import run_recorder


async def _record_all(
    *,
    duration_sec: float,
    output_root: Path,
    parquet_batch_rows: int,
    reference_tolerance_ms: int,
):
    clob_task = asyncio.create_task(
        run_recorder(
            duration_sec=duration_sec,
            output_root=output_root,
            parquet_batch_rows=parquet_batch_rows,
        )
    )
    btc_task = asyncio.create_task(
        run_btc_recorder(
            duration_sec=duration_sec,
            output_root=output_root,
            reference_tolerance_ms=reference_tolerance_ms,
        )
    )
    return await asyncio.gather(clob_task, btc_task)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BTC5M V2 public CLOB + Chainlink-TWAP data recorder"
    )
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--output-root", default="runtime/data")
    parser.add_argument("--parquet-batch-rows", type=int, default=100)
    parser.add_argument(
        "--reference-tolerance-ms",
        type=int,
        default=5000,
        help=(
            "Maximum absolute source-timestamp distance from the market's 5-minute "
            "start boundary for accepting a BTC reference price."
        ),
    )
    args = parser.parse_args()

    clob_completed, btc_completed = asyncio.run(
        _record_all(
            duration_sec=args.duration_sec,
            output_root=Path(args.output_root),
            parquet_batch_rows=max(1, args.parquet_batch_rows),
            reference_tolerance_ms=max(0, args.reference_tolerance_ms),
        )
    )

    summary = {
        "status": "complete",
        "mode": "public-data-only",
        "markets_recorded": len(clob_completed),
        "raw_clob_events": sum(item.raw_events for item in clob_completed),
        "clob_snapshots": sum(item.snapshots for item in clob_completed),
        "clob_reconnects": sum(item.reconnects for item in clob_completed),
        "btc_markets_recorded": len(btc_completed),
        "btc_samples": sum(item.samples for item in btc_completed),
        "btc_reconnects": sum(item.reconnects for item in btc_completed),
        "btc_reference_verified_markets": sum(
            1 for item in btc_completed if item.reference_price is not None
        ),
        "output_root": str(Path(args.output_root)),
        "credentials_loaded": False,
        "order_path_used": False,
    }
    print(json.dumps(summary, indent=2))
    return 0 if clob_completed and btc_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
