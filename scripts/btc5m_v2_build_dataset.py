#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc5m_v2.research.dataset import market_dirs_from_output_root, write_dataset
from btc5m_v2.research.ready_dataset import write_ready_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a causal BTC5M V2 research Parquet from recorder output"
    )
    parser.add_argument(
        "--input-root",
        default="runtime/data",
        help="Recorder output root containing date/market directories",
    )
    parser.add_argument(
        "--output",
        default="runtime/research/btc5m_features.parquet",
        help="Unified Parquet output path",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=int,
        default=1000,
        help="Minimum interval between causal observations per market",
    )
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help=(
            "Build only from markets that pass the full V2 readiness preflight. "
            "Without this flag the core builder remains strict and any invalid market can fail the build."
        ),
    )
    args = parser.parse_args()

    if args.ready_only:
        summary = write_ready_dataset(
            args.input_root,
            Path(args.output),
            sample_interval_ms=args.sample_interval_ms,
        )
    else:
        market_dirs = market_dirs_from_output_root(args.input_root)
        if not market_dirs:
            raise SystemExit(f"no recorded markets found under {args.input_root}")
        summary = write_dataset(
            market_dirs,
            Path(args.output),
            sample_interval_ms=args.sample_interval_ms,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
