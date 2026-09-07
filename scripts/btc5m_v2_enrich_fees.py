#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from btc5m_v2.research.fee_enrich import enrich_fee_schedules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach public CLOB V2 fee schedules to recorded BTC5M markets"
    )
    parser.add_argument("--output-root", default="runtime/data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = enrich_fee_schedules(args.output_root, force=args.force)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
