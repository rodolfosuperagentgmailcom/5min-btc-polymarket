#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc5m_v2.research.train_model import load_dataset_rows
from btc5m_v2.research.walk_forward_paper import walk_forward_paper_compare


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BTC5M V2 expanding-window executable paper P&L evaluation"
    )
    parser.add_argument("dataset", help="Path to dataset.parquet")
    parser.add_argument("--output-root", default="runtime/data")
    parser.add_argument("--minimum-train-markets", type=int, default=100)
    parser.add_argument("--test-block-markets", type=int, default=50)
    parser.add_argument("--step-markets", type=int, default=None)
    parser.add_argument("--notional-usd", type=float, default=5.0)
    parser.add_argument("--latency-ms", type=float, default=250.0)
    parser.add_argument("--max-participation-pct", type=float, default=20.0)
    parser.add_argument("--legacy-threshold", type=float, default=0.70)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    rows = load_dataset_rows(args.dataset)
    result = walk_forward_paper_compare(
        rows,
        Path(args.output_root),
        minimum_train_markets=args.minimum_train_markets,
        test_block_markets=args.test_block_markets,
        step_markets=args.step_markets,
        notional_usd=args.notional_usd,
        latency_ms=args.latency_ms,
        max_participation_pct=args.max_participation_pct,
        legacy_threshold=args.legacy_threshold,
    )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        destination = Path(args.json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
