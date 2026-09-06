from __future__ import annotations

import time

import pyarrow.parquet as pq

from btc5m_v2.config import V2Config
from btc5m_v2.feeds.polymarket_ws import BookState, WsEnvelope, market_subscription
from btc5m_v2.market import MarketInfo, market_info_from_event
from btc5m_v2.research.recorder import MarketRecorder

EXPECTED_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def cfg() -> V2Config:
    return V2Config(
        {
            "market": {"require_active": True, "require_not_closed": True},
            "resolution": {
                "expected_source": EXPECTED_SOURCE,
                "fail_if_missing": True,
                "fail_if_unexpected": True,
            },
            "entry": {"seconds_left_min": 90, "seconds_left_max": 150},
            "market_quality": {
                "max_selected_side_spread": 0.02,
                "min_top_3_ask_notional_usd": 30,
            },
            "freshness": {"clob_max_age_sec": 2, "max_consecutive_data_errors": 3},
        }
    )


def test_market_subscription_is_public_market_shape():
    payload = market_subscription(["up-token", "down-token"])
    assert payload == {
        "assets_ids": ["up-token", "down-token"],
        "type": "market",
        "custom_feature_enabled": True,
    }


def test_market_info_maps_up_and_down_tokens():
    event = {
        "slug": "btc-updown-5m-123",
        "resolutionSource": EXPECTED_SOURCE,
        "markets": [
            {
                "slug": "btc-updown-5m-123",
                "active": True,
                "closed": False,
                "conditionId": "condition-1",
                "outcomes": '["Down", "Up"]',
                "clobTokenIds": '["down-token", "up-token"]',
                "endDate": "2030-01-01T00:05:00Z",
            }
        ],
    }
    market = market_info_from_event(event, cfg())
    assert market.up_token_id == "up-token"
    assert market.down_token_id == "down-token"
    assert market.resolution_source == EXPECTED_SOURCE


def test_market_info_blocks_changed_resolution_source():
    event = {
        "slug": "btc-updown-5m-123",
        "resolutionSource": "https://example.com/other-feed",
        "markets": [
            {
                "active": True,
                "closed": False,
                "conditionId": "condition-1",
                "outcomes": ["Up", "Down"],
                "clobTokenIds": ["up-token", "down-token"],
                "endDate": "2030-01-01T00:05:00Z",
            }
        ],
    }
    try:
        market_info_from_event(event, cfg())
    except ValueError as exc:
        assert "unexpected_resolution_source" in str(exc)
    else:
        raise AssertionError("changed resolution source must fail closed")


def test_book_state_reconstructs_book_and_price_changes():
    state = BookState("up-token")
    received_ns = time.time_ns()
    book_event = {
        "event_type": "book",
        "asset_id": "up-token",
        "timestamp": str(received_ns // 1_000_000),
        "bids": [{"price": "0.60", "size": "10"}, {"price": "0.61", "size": "20"}],
        "asks": [{"price": "0.63", "size": "30"}, {"price": "0.64", "size": "40"}],
    }
    assert state.apply(book_event, received_ns)
    assert state.best_bid == 0.61
    assert state.best_ask == 0.63
    assert round(state.spread or 0, 6) == 0.02
    assert state.ask_notional(1) == 0.63 * 30

    change = {
        "event_type": "price_change",
        "timestamp": str(received_ns // 1_000_000 + 10),
        "price_changes": [
            {"asset_id": "up-token", "side": "BUY", "price": "0.62", "size": "25"},
            {"asset_id": "up-token", "side": "SELL", "price": "0.63", "size": "0"},
        ],
    }
    assert state.apply(change, received_ns + 10_000_000)
    assert state.best_bid == 0.62
    assert state.best_ask == 0.64


def test_recorder_writes_raw_jsonl_and_fixed_schema_parquet(tmp_path):
    now = time.time()
    market = MarketInfo(
        slug="btc-updown-5m-test",
        condition_id="condition-test",
        up_token_id="up-token",
        down_token_id="down-token",
        end_iso="2030-01-01T00:05:00Z",
        end_ts=now + 300,
        resolution_source=EXPECTED_SOURCE,
    )
    recorder = MarketRecorder(market, output_root=tmp_path, parquet_batch_rows=1)
    received_ns = time.time_ns()

    up_event = {
        "event_type": "book",
        "asset_id": "up-token",
        "timestamp": str(received_ns // 1_000_000),
        "bids": [{"price": "0.60", "size": "100"}],
        "asks": [{"price": "0.61", "size": "100"}],
    }
    down_event = {
        "event_type": "book",
        "asset_id": "down-token",
        "timestamp": str(received_ns // 1_000_000),
        "bids": [{"price": "0.39", "size": "100"}],
        "asks": [{"price": "0.40", "size": "100"}],
    }

    first = WsEnvelope(received_ns, "2026-09-06T00:00:00Z", "{}", up_event)
    second = WsEnvelope(received_ns + 1_000_000, "2026-09-06T00:00:00.001Z", "{}", down_event)

    assert not recorder.apply(first)
    assert recorder.apply(second)
    recorder.close()

    assert recorder.raw_events == 2
    assert recorder.snapshots == 1
    assert recorder.raw_path.exists()
    assert len(recorder.raw_path.read_text(encoding="utf-8").splitlines()) == 2

    parquet_files = sorted(recorder.market_dir.glob("clob_snapshots_part-*.parquet"))
    assert len(parquet_files) == 1
    table = pq.read_table(parquet_files[0])
    row = table.to_pylist()[0]
    assert row["up_bid"] == 0.60
    assert row["up_ask"] == 0.61
    assert row["down_bid"] == 0.39
    assert row["down_ask"] == 0.40
    assert row["resolution_source"] == EXPECTED_SOURCE
