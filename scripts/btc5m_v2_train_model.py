#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc5m_v2.research.train_model import (
    DEFAULT_FEATURE_COLUMNS,
    MICROSTRUCTURE_FEATURE_COLUMNS,
    save_training_result,
    train_from_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a chronological market-level BTC5M V2 logistic probability baseline"
    )
    parser.add_argument(
        "--dataset",
        default="runtime/research/btc5m_features.parquet",
        help="Unified V2 feature Parquet created by btc5m_v2_build_dataset.py",
    )
    parser.add_argument(
        "--output",
        default="runtime/models/btc5m_logit_v1.json",
        help="Output path for lightweight JSON inference model",
    )
    parser.add_argument(
        "--feature-set",
        choices=("full", "microstructure"),
        default="full",
        help="full requires verified settlement-aligned BTC features; microstructure is a research control",
    )
    parser.add_argument("--target-seconds-left", type=float, default=120.0)
    parser.add_argument("--seconds-left-min", type=float, default=90.0)
    parser.add_argument("--seconds-left-max", type=float, default=150.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-markets", type=int, default=100)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    args = parser.parse_args()

    feature_columns = (
        DEFAULT_FEATURE_COLUMNS
        if args.feature_set == "full"
        else MICROSTRUCTURE_FEATURE_COLUMNS
    )
    result = train_from_parquet(
        args.dataset,
        feature_columns=feature_columns,
        target_seconds_left=args.target_seconds_left,
        seconds_left_min=args.seconds_left_min,
        seconds_left_max=args.seconds_left_max,
        train_fraction=args.train_fraction,
        minimum_markets=args.minimum_markets,
        regularization_c=args.regularization_c,
    )
    model_path, metrics_path = save_training_result(
        result,
        model_path=Path(args.output),
    )
    payload = dict(result.summary)
    payload["model_path"] = str(model_path)
    payload["metrics_path"] = str(metrics_path)
    payload["feature_set"] = args.feature_set
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
