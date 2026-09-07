from __future__ import annotations

import hashlib
import hmac

from btc5m_v2.feeds.chainlink import (
    ChainlinkCredentials,
    ChainlinkStream,
    auth_headers,
    empty_body_hash,
    report_ws_path,
    twap_60s_candidates,
)
from btc5m_v2.market import MarketInfo
from btc5m_v2.research.resolution import resolution_from_gamma_payload, resolution_from_ws_event


def market() -> MarketInfo:
    return MarketInfo(
        slug="btc-updown-5m-test",
        condition_id="condition-1",
        up_token_id="up-token",
        down_token_id="down-token",
        end_iso="2030-01-01T00:05:00Z",
        end_ts=1_893_456_300.0,
        resolution_source="https://data.chain.link/streams/btc-usd-twap-60s-streams",
    )


def test_ws_market_resolution_maps_winning_asset_to_side():
    result = resolution_from_ws_event(
        {
            "event_type": "market_resolved",
            "winning_asset_id": "down-token",
            "winning_outcome": "Down",
            "timestamp": "1788738900123",
        },
        market(),
    )
    assert result is not None
    assert result.resolved
    assert result.winning_side == "DOWN"
    assert result.winning_asset_id == "down-token"
    assert result.resolution_ts_ms == 1788738900123
    assert result.source == "polymarket_ws"


def test_gamma_resolution_requires_terminal_closed_vector():
    payload = {
        "closed": True,
        "markets": [
            {
                "closed": True,
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["1", "0"]',
                "clobTokenIds": '["up-token", "down-token"]',
            }
        ],
    }
    result = resolution_from_gamma_payload(payload, market())
    assert result.resolved
    assert result.winning_side == "UP"
    assert result.winning_asset_id == "up-token"
    assert result.source == "gamma"

    unresolved = resolution_from_gamma_payload(
        {
            "closed": False,
            "markets": [
                {
                    "closed": False,
                    "outcomes": '["Up", "Down"]',
                    "outcomePrices": '["0.71", "0.29"]',
                    "clobTokenIds": '["up-token", "down-token"]',
                }
            ],
        },
        market(),
    )
    assert not unresolved.resolved


def test_chainlink_hmac_headers_follow_documented_string_to_sign():
    creds = ChainlinkCredentials(api_key="test-key", api_secret="test-secret")
    path = "/api/v1/ws?feedIDs=0xabc"
    stamp = 1_788_738_900_123
    expected_string = f"GET {path} {empty_body_hash()} test-key {stamp}"
    expected_sig = hmac.new(b"test-secret", expected_string.encode(), hashlib.sha256).hexdigest()

    headers = auth_headers(
        method="GET",
        full_path=path,
        credentials=creds,
        timestamp_ms=stamp,
    )
    assert headers["Authorization"] == "test-key"
    assert headers["X-Authorization-Timestamp"] == str(stamp)
    assert headers["X-Authorization-Signature-SHA256"] == expected_sig


def test_chainlink_ws_path_and_twap_candidate_filter():
    assert report_ws_path(["0xabc", "0xdef"]) == "/api/v1/ws?feedIDs=0xabc%2C0xdef"
    streams = [
        ChainlinkStream("0x1", "BTC/USD-Streams-CexPrice", "BTC", "USD", "V3", "live"),
        ChainlinkStream("0x2", "BTC/USD-TWAP-60s", "BTC", "USD", "V3", "live"),
    ]
    candidates = twap_60s_candidates(streams)
    assert [item.feed_id for item in candidates] == ["0x2"]
