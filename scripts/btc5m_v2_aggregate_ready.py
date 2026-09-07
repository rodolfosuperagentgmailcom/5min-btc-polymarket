#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from btc5m_v2.research.aggregate import write_aggregate_ready_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one deduplicated causal dataset from extracted BTC5M V2 evidence bundles."
    )
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Directory to scan recursively for extracted evidence market folders. Repeat as needed.",
    )
    parser.add_argument(
        "--output",
        default="runtime/research/btc5m_aggregate_ready_features.parquet",
        help="Output Parquet dataset path.",
    )
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_aggregate_ready_dataset(
        args.input_root,
        args.output,
        sample_interval_ms=args.sample_interval_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
