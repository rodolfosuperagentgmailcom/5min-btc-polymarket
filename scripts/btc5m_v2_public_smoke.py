#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

from btc5m_v2.config import load_config
from btc5m_v2_runner import (
    metrics_from_book,
    parse_market,
    resolution_source,
    resolve_current_market,
    snapshot_dict,
    ts_utc,
)


def main() -> int:
    cfg = load_config()
    slug_prefix = str(cfg.raw["market"].get("slug_prefix", "btc-updown-5m-"))
    market = resolve_current_market(slug_prefix)
    if not market:
        print(json.dumps({"ts": ts_utc(), "status": "smoke_no_current_market"}))
        return 2

    up_token, down_token, slug, end_iso, seconds_left = parse_market(market)
    clob = ClobClient(host="https://clob.polymarket.com", chain_id=POLYGON)
    up = metrics_from_book(clob.get_order_book(up_token))
    down = metrics_from_book(clob.get_order_book(down_token))

    payload = {
        "ts": ts_utc(),
        "status": "smoke_ok",
        "slug": slug,
        "market_end": end_iso,
        "seconds_left": round(seconds_left, 3),
        "resolution_source": resolution_source(market),
        "up": snapshot_dict(up),
        "down": snapshot_dict(down),
        "credentials_loaded": False,
        "order_path_used": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if up.best_bid is None or up.best_ask is None or down.best_bid is None or down.best_ask is None:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
