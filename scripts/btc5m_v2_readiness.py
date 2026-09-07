#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from btc5m_v2.research.readiness import readiness_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report whether recorded BTC5M markets are ready for V2 model/paper research"
    )
    parser.add_argument(
        "--input-root",
        default="runtime/data",
        help="Recorder output root containing date/market directories",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-market detail from printed JSON",
    )
    args = parser.parse_args()

    report = readiness_report(args.input_root)
    if args.summary_only:
        report = {key: value for key, value in report.items() if key != "markets"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
