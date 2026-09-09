#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from btc5m_v2.research.alignment import sleep_until_aligned_start
from btc5m_v2.research.study import (
    finalize_study,
    record_study,
    study_summary,
    write_study_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect aligned BTC5M V2 research data, enrich public fee/resolution metadata, "
            "run readiness checks, and optionally rebuild the ready-only dataset."
        )
    )
    parser.add_argument("--markets", type=int, default=1, help="Consecutive 5-minute markets to observe")
    parser.add_argument("--output-root", default="runtime/data")
    parser.add_argument("--dataset-path", default="runtime/research/v2_ready.parquet")
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--parquet-batch-rows", type=int, default=100)
    parser.add_argument("--reference-tolerance-ms", type=int, default=5000)
    parser.add_argument("--tail-sec", type=float, default=1.5)
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Start immediately instead of waiting for the next 5-minute boundary (not recommended for research).",
    )
    parser.add_argument("--alignment-offset-sec", type=float, default=0.5)
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Skip rebuilding the ready-only aggregate Parquet dataset after collection.",
    )
    args = parser.parse_args()

    if args.markets <= 0:
        parser.error("--markets must be positive")

    if not args.no_align:
        target = sleep_until_aligned_start(offset_sec=max(0.0, float(args.alignment_offset_sec)))
        print(json.dumps({"status": "aligned", "target_ts": target}, indent=2), flush=True)

    started = time.time()
    clob_completed, btc_completed = asyncio.run(
        record_study(
            markets=args.markets,
            output_root=Path(args.output_root),
            parquet_batch_rows=max(1, args.parquet_batch_rows),
            reference_tolerance_ms=max(0, args.reference_tolerance_ms),
            tail_sec=max(0.0, args.tail_sec),
        )
    )

    dataset_path = None if args.no_dataset else args.dataset_path
    finalized = finalize_study(
        output_root=args.output_root,
        dataset_path=dataset_path,
        sample_interval_ms=max(1, args.sample_interval_ms),
    )
    summary = study_summary(
        requested_markets=args.markets,
        output_root=args.output_root,
        clob_completed=clob_completed,
        btc_completed=btc_completed,
        finalized=finalized,
        started_ts=started,
    )
    report_path = write_study_summary(args.output_root, summary)
    summary["study_report"] = str(report_path)
    print(json.dumps(summary, indent=2))

    ready = int((finalized.get("readiness") or {}).get("v2_ready_markets") or 0)
    return 0 if clob_completed and btc_completed and ready > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
