#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from btc5m_v2.research.diagnostics import rejection_diagnostics
from btc5m_v2.research.readiness import readiness_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain why recorded BTC5M markets are not V2-ready"
    )
    parser.add_argument("--input-root", default="runtime/data")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of ranked blockers to print",
    )
    args = parser.parse_args()

    readiness = readiness_report(args.input_root)
    report = rejection_diagnostics(readiness)
    report["ranked_blockers"] = report["ranked_blockers"][: max(0, args.top)]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
