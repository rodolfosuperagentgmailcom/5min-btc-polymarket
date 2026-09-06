#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from btc5m_v2.research.enrich import enrich_output_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach final Gamma resolutions to recorded BTC5M markets")
    parser.add_argument("--output-root", default="runtime/data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = enrich_output_root(args.output_root, force=args.force)
    print(json.dumps({"status": "complete", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
