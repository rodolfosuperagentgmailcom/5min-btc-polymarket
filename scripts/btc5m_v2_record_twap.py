#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from btc5m_v2.research.twap_store import record_twap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record public Polymarket RTDS Chainlink BTC/USD 60-second TWAP updates"
    )
    parser.add_argument("--duration-sec", type=float, default=1800.0)
    parser.add_argument("--output-root", default="runtime/twap")
    parser.add_argument("--batch-rows", type=int, default=100)
    args = parser.parse_args()

    store = asyncio.run(
        record_twap(
            duration_sec=args.duration_sec,
            output_root=args.output_root,
            batch_rows=args.batch_rows,
        )
    )
    print(
        json.dumps(
            {
                "output_root": str(store.output_root),
                "observations": store.observations,
                "reconnects": store.reconnects,
                "source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
                "transport": "polymarket_rtds",
                "credentials_required": False,
            },
            indent=2,
        )
    )
    return 0 if store.observations > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
