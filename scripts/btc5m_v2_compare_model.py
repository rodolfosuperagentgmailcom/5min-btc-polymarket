#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc5m_v2.research.model_paper import compare_from_parquet
from btc5m_v2.research.train_model import (
    DEFAULT_FEATURE_COLUMNS,
    MICROSTRUCTURE_FEATURE_COLUMNS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the V2 probability model with the original 0.70 threshold on the same "
            "chronological held-out markets using realistic paper execution"
        )
    )
    parser.add_argument(
        "--dataset",
        default="runtime/research/btc5m_features.parquet",
        help="Unified V2 feature dataset",
    )
    parser.add_argument(
        "--data-root",
        default="runtime/data",
        help="Recorder output root containing raw books, metadata, resolution, and fees",
    )
    parser.add_argument(
        "--output",
        default="runtime/research/model_vs_legacy.json",
        help="Comparison summary JSON",
    )
    parser.add_argument(
        "--feature-set",
        choices=("full", "microstructure"),
        default="full",
        help="full requires verified settlement-aligned BTC samples; microstructure is a control",
    )
    parser.add_argument("--target-seconds-left", type=float, default=120.0)
    parser.add_argument("--seconds-left-min", type=float, default=90.0)
    parser.add_argument("--seconds-left-max", type=float, default=150.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-markets", type=int, default=100)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--notional-usd", type=float, default=5.0)
    parser.add_argument("--latency-ms", type=float, default=100.0)
    parser.add_argument("--max-participation-pct", type=float, default=20.0)
    parser.add_argument("--legacy-threshold", type=float, default=0.70)
    args = parser.parse_args()

    features = (
        DEFAULT_FEATURE_COLUMNS
        if args.feature_set == "full"
        else MICROSTRUCTURE_FEATURE_COLUMNS
    )
    result = compare_from_parquet(
        args.dataset,
        args.data_root,
        feature_columns=features,
        target_seconds_left=args.target_seconds_left,
        seconds_left_min=args.seconds_left_min,
        seconds_left_max=args.seconds_left_max,
        train_fraction=args.train_fraction,
        minimum_markets=args.minimum_markets,
        regularization_c=args.regularization_c,
        notional_usd=args.notional_usd,
        latency_ms=args.latency_ms,
        max_participation_pct=args.max_participation_pct,
        legacy_threshold=args.legacy_threshold,
    )
    result["feature_set"] = args.feature_set

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
