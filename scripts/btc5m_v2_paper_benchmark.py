from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc5m_v2.research.paper_benchmark import latency_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the legacy BTC5M threshold strategy with book depth, fees, and latency.")
    parser.add_argument("--input-root", default="runtime/data")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-seconds-left", type=float, default=60.0)
    parser.add_argument("--notional-usd", type=float, default=5.0)
    parser.add_argument("--max-participation-pct", type=float, default=20.0)
    parser.add_argument(
        "--latency-ms",
        type=float,
        nargs="+",
        default=[0.0, 100.0, 250.0, 500.0, 1000.0],
        help="One or more modeled signal-to-book latencies in milliseconds.",
    )
    parser.add_argument(
        "--allow-missing-fees",
        action="store_true",
        help="Treat missing fee metadata as zero fee. Not recommended for net-performance claims.",
    )
    parser.add_argument("--output", default="runtime/research/legacy_paper_latency_sweep.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = latency_sweep(
        args.input_root,
        args.latency_ms,
        threshold=args.threshold,
        min_seconds_left=args.min_seconds_left,
        notional_usd=args.notional_usd,
        max_participation_pct=args.max_participation_pct,
        require_fee_schedule=not args.allow_missing_fees,
    )
    payload = {
        "input_root": args.input_root,
        "latencies_ms": [float(value) for value in args.latency_ms],
        "results": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
