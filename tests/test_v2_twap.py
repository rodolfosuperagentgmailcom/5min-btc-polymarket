from __future__ import annotations

import json
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.feeds.polymarket_twap import (
    CHAINLINK_TWAP_RESOLUTION_SOURCE,
    parse_twap_event,
    twap_subscription,
)
from btc5m_v2.research.twap_store import (
    market_start_ms_from_slug,
    materialize_market_twap,
)


def test_twap_subscription_matches_official_rtds_shape():
    payload = twap_subscription(symbol="btc/usd", window_seconds=60)
    assert payload == {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_twap_sixty",
                "type": "update",
                "filters": '{"symbol":"btc/usd"}',
            }
        ],
    }


def test_twap_parser_prefers_exact_e18_value():
    event = {
        "topic": "crypto_prices_twap_sixty",
        "type": "update",
        "timestamp": 1785178800123,
        "payload": {
            "symbol": "btc/usd",
            "value": 65000.5,
            "full_accuracy_value": "65000500000000000000000",
            "timestamp": 1785178800000,
            "window_s": 60,
        },
    }
    observation = parse_twap_event(event, received_ts_ns=1785178800200 * 1_000_000)
    assert observation is not None
    assert observation.value == Decimal("65000.5")
    assert observation.chainlink_ts_ms == 1785178800000
    assert observation.window_seconds == 60
    assert observation.chainlink_age_ms == 200.0


def test_market_start_comes_from_slug_unix_bucket():
    assert market_start_ms_from_slug("btc-updown-5m-1785178800") == 1785178800000


def test_materializer_requires_exact_boundary_for_reference(tmp_path):
    slug = "btc-updown-5m-1785178800"
    market_dir = tmp_path / slug
    market_dir.mkdir()
    (market_dir / "metadata.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "resolution_source": CHAINLINK_TWAP_RESOLUTION_SOURCE,
            }
        ),
        encoding="utf-8",
    )
    start = 1785178800000
    rows = [
        {
            "chainlink_ts_ms": start,
            "price": 65000.5,
            "exact_value": "65000.5",
            "received_ts_ns": (start + 50) * 1_000_000,
            "chainlink_age_ms": 50.0,
            "symbol": "btc/usd",
            "window_seconds": 60,
        },
        {
            "chainlink_ts_ms": start + 1000,
            "price": 65001.0,
            "exact_value": "65001",
            "received_ts_ns": (start + 1050) * 1_000_000,
            "chainlink_age_ms": 50.0,
            "symbol": "btc/usd",
            "window_seconds": 60,
        },
    ]
    result = materialize_market_twap(market_dir, rows)
    assert result["reference_price"] == 65000.5
    assert result["reference_status"] == "exact_market_start_observation"

    table = pq.read_table(market_dir / "btc_samples.parquet")
    assert table.num_rows == 2
    metadata = json.loads((market_dir / "btc_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"] == CHAINLINK_TWAP_RESOLUTION_SOURCE
    assert metadata["transport"] == "polymarket_rtds"
    assert metadata["credentials_required"] is False


def test_materializer_does_not_approximate_missing_reference(tmp_path):
    slug = "btc-updown-5m-1785178800"
    market_dir = tmp_path / slug
    market_dir.mkdir()
    (market_dir / "metadata.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "resolution_source": CHAINLINK_TWAP_RESOLUTION_SOURCE,
            }
        ),
        encoding="utf-8",
    )
    start = 1785178800000
    rows = [
        {
            "chainlink_ts_ms": start + 1000,
            "price": 65001.0,
            "exact_value": "65001",
            "received_ts_ns": (start + 1050) * 1_000_000,
            "chainlink_age_ms": 50.0,
            "symbol": "btc/usd",
            "window_seconds": 60,
        }
    ]
    result = materialize_market_twap(market_dir, rows)
    assert result["reference_price"] is None
    assert result["reference_status"] == "missing_exact_market_start_observation"


def test_materializer_rejects_nonmatching_resolution_source(tmp_path):
    slug = "btc-updown-5m-1785178800"
    market_dir = tmp_path / slug
    market_dir.mkdir()
    (market_dir / "metadata.json").write_text(
        json.dumps({"slug": slug, "resolution_source": "https://example.com/wrong"}),
        encoding="utf-8",
    )
    try:
        materialize_market_twap(market_dir, [])
    except ValueError as exc:
        assert "market_resolution_source_not_chainlink_twap_60s" in str(exc)
    else:
        raise AssertionError("wrong market source must fail closed")
