#!/usr/bin/env python3
from __future__ import annotations

import json

from btc5m_v2.config import load_config
from btc5m_v2.market import discover_current_market
from btc5m_v2.market_fees import fee_schedule_payload, fetch_market_fee_schedule


def main() -> None:
    market = discover_current_market(load_config())
    if market is None:
        raise SystemExit("current BTC5M market not found")
    schedule = fetch_market_fee_schedule(market.condition_id)
    payload = {
        "slug": market.slug,
        "condition_id": market.condition_id,
        "resolution_source": market.resolution_source,
        "fee_schedule": fee_schedule_payload(schedule),
        "credentials_loaded": False,
        "order_path_used": False,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
