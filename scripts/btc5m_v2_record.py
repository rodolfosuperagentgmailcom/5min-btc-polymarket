#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from btc5m_v2.research.recorder import run_recorder


def main() -> int:
    parser = argparse.ArgumentParser(description="BTC5M V2 public market-data recorder")
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--output-root", default="runtime/data")
    parser.add_argument("--parquet-batch-rows", type=int, default=100)
    args = parser.parse_args()

    completed = asyncio.run(
        run_recorder(
            duration_sec=args.duration_sec,
            output_root=Path(args.output_root),
            parquet_batch_rows=max(1, args.parquet_batch_rows),
        )
    )

    summary = {
        "status": "complete",
        "mode": "public-data-only",
        "markets_recorded": len(completed),
        "raw_events": sum(item.raw_events for item in completed),
        "snapshots": sum(item.snapshots for item in completed),
        "reconnects": sum(item.reconnects for item in completed),
        "output_root": str(Path(args.output_root)),
        "credentials_loaded": False,
        "order_path_used": False,
    }
    print(json.dumps(summary, indent=2))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
