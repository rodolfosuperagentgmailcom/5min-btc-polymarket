#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json

from btc5m_v2.config import load_config
from btc5m_v2.feeds.polymarket_ws import BookState, market_events
from btc5m_v2.market import discover_current_market


async def run() -> int:
    cfg = load_config()
    market = discover_current_market(cfg)
    if market is None:
        print(json.dumps({"status": "no_current_market"}))
        return 2

    up = BookState(market.up_token_id)
    down = BookState(market.down_token_id)

    try:
        async with asyncio.timeout(15):
            async for envelope in market_events([market.up_token_id, market.down_token_id]):
                up.apply(envelope.event, envelope.received_ts_ns)
                down.apply(envelope.event, envelope.received_ts_ns)
                if up.best_bid is None or up.best_ask is None or down.best_bid is None or down.best_ask is None:
                    continue

                payload = {
                    "status": "ws_smoke_ok",
                    "slug": market.slug,
                    "resolution_source": market.resolution_source,
                    "seconds_left": round(market.seconds_left(), 3),
                    "up": up.snapshot("UP", envelope.received_ts_ns),
                    "down": down.snapshot("DOWN", envelope.received_ts_ns),
                    "credentials_loaded": False,
                    "order_path_used": False,
                }
                print(json.dumps(payload, indent=2))
                return 0
    except TimeoutError:
        print(json.dumps({"status": "ws_smoke_timeout", "slug": market.slug}))
        return 3

    return 4


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
