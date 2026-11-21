#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc5m_v2.research.evidence import evidence_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize BTC5M V2 forward-paper evidence. No wallet or order path is used."
    )
    parser.add_argument(
        "--journal-root",
        default=str(ROOT / "runtime" / "v2" / "paper"),
        help="Root containing YYYY-MM-DD/<slug>.json paper journals",
    )
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Exit non-zero until the independent settled-signal target is reached.",
    )
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be positive")

    report = evidence_report(args.journal_root, target_settled_signals=args.target)
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report["evidence_integrity_ok"]:
        return 3
    if args.require_target and not report["evidence_target_reached"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
