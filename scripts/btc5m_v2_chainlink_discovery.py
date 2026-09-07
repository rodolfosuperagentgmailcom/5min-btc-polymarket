#!/usr/bin/env python3
from __future__ import annotations

import json

from btc5m_v2.feeds.chainlink import discover_streams, discovery_summary, twap_60s_candidates


def main() -> int:
    streams = discover_streams(base_asset="BTC", quote_asset="USD", status="live")
    candidates = twap_60s_candidates(streams)
    print(json.dumps({
        "status": "ok",
        "btc_usd_live_streams": len(streams),
        "twap_60s_candidates": discovery_summary(candidates),
        "all_btc_usd_streams": discovery_summary(streams),
        "authentication_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
