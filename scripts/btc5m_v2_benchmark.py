#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from btc5m_v2.research.benchmark import benchmark_legacy_threshold, trade_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the legacy BTC5M stronger-ask threshold strategy")
    parser.add_argument(
        "--dataset",
        default="runtime/research/btc5m_features.parquet",
        help="Unified V2 research dataset Parquet",
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-seconds-left", type=float, default=60.0)
    parser.add_argument(
        "--output",
        default="runtime/research/legacy_threshold_benchmark.json",
        help="JSON report output",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    rows = pq.read_table(dataset_path).to_pylist()
    trades, summary = benchmark_legacy_threshold(
        rows,
        threshold=args.threshold,
        min_seconds_left=args.min_seconds_left,
    )
    payload = {
        "summary": summary,
        "trades": [trade_payload(trade) for trade in trades],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
