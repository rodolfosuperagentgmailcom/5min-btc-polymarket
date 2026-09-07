from __future__ import annotations

import json
from decimal import Decimal

import pyarrow.parquet as pq

from btc5m_v2.feeds.polymarket_twap import (
    CHAINLINK_TWAP_RESOLUTION_SOURCE,
    TwapObservation,
)
from btc5m_v2.market import MarketInfo
from btc5m_v2.research.btc_recorder import BTCMarketRecorder


def _market(end_ts: float = 2_000_000_300.0) -> MarketInfo:
    return MarketInfo(
        slug="btc-updown-5m-test",
        condition_id="condition-test",
        up_token_id="up-token",
        down_token_id="down-token",
        end_iso="2033-05-18T03:38:20Z",
        end_ts=end_ts,
        resolution_source=CHAINLINK_TWAP_RESOLUTION_SOURCE,
    )


def _observation(*, chainlink_ts_ms: int, received_ts_ns: int, price: str) -> TwapObservation:
    return TwapObservation(
        symbol="btc/usd",
        window_seconds=60,
        value=Decimal(price),
        full_accuracy_value=None,
        chainlink_ts_ms=chainlink_ts_ms,
        publisher_ts_ms=chainlink_ts_ms + 25,
        received_ts_ns=received_ts_ns,
        raw_event={"topic": "crypto_prices_twap_sixty"},
    )


def test_btc_recorder_writes_causal_parquet_and_verified_reference(tmp_path):
    market = _market()
    start_ms = int((market.end_ts - 300.0) * 1000)
    recorder = BTCMarketRecorder(
        market,
        output_root=tmp_path,
        reference_tolerance_ms=2_000,
        checkpoint_rows=1,
    )

    received_ns = (start_ms + 1_500) * 1_000_000 + 123_000
    recorder.append(
        _observation(
            chainlink_ts_ms=start_ms + 1_000,
            received_ts_ns=received_ns,
            price="60000.25",
        )
    )
    recorder.close()

    table = pq.read_table(recorder.samples_path)
    row = table.to_pylist()[0]
    # Dataset joins must use when the process received the observation, not the
    # earlier source timestamp.
    assert row["ts_ns"] == received_ns
    assert row["chainlink_ts_ms"] == start_ms + 1_000
    assert row["price"] == 60000.25

    metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"] == CHAINLINK_TWAP_RESOLUTION_SOURCE
    assert metadata["reference_verified"] is True
    assert metadata["reference_price"] == 60000.25
    assert metadata["reference_offset_ms"] == 1_000
    assert metadata["credentials_loaded"] is False
    assert metadata["order_path_used"] is False


def test_btc_recorder_does_not_invent_reference_when_started_too_late(tmp_path):
    market = _market()
    start_ms = int((market.end_ts - 300.0) * 1000)
    recorder = BTCMarketRecorder(
        market,
        output_root=tmp_path,
        reference_tolerance_ms=2_000,
        checkpoint_rows=10,
    )

    recorder.append(
        _observation(
            chainlink_ts_ms=start_ms + 10_000,
            received_ts_ns=(start_ms + 10_100) * 1_000_000,
            price="60100.00",
        )
    )
    recorder.close()

    metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
    assert metadata["reference_verified"] is False
    assert metadata["reference_price"] is None
    assert metadata["reference_chainlink_ts_ms"] is None
    assert metadata["reference_offset_ms"] is None
