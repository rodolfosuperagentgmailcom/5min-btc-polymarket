#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from btc5m_v2.research.twap_store import materialize_output_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach recorded public Chainlink 60s TWAP samples to BTC5M market directories"
    )
    parser.add_argument("--data-root", default="runtime/data")
    parser.add_argument("--twap-root", default="runtime/twap")
    args = parser.parse_args()

    summary = materialize_output_root(args.data_root, args.twap_root)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
