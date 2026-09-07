from __future__ import annotations

import json
import time

import pytest

from btc5m_v2.execution.paper import simulate_taker_buy_to_settlement
from btc5m_v2.market import MarketInfo
from btc5m_v2.research.book_replay import ReplayedBookSnapshot
from btc5m_v2.research.paper_benchmark import benchmark_market_paper
from btc5m_v2.research.recorder import MarketRecorder
from btc5m_v2.feeds.polymarket_ws import WsEnvelope

EXPECTED_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def _book(ts_ns: int, up_ask: float, up_size: float = 100.0) -> ReplayedBookSnapshot:
    return ReplayedBookSnapshot(
        ordinal=0,
        received_ts_ns=ts_ns,
        event_type="book",
        event_exchange_ts_ms=ts_ns // 1_000_000,
        up_bids=((max(0.01, up_ask - 0.01), 100.0),),
        up_asks=((up_ask, up_size),),
        down_bids=((0.25, 100.0),),
        down_asks=((0.26, 100.0),),
    )


def test_paper_fill_uses_first_book_after_latency():
    t0 = 1_000_000_000
    books = [_book(t0, 0.70), _book(t0 + 200_000_000, 0.74)]
    result = simulate_taker_buy_to_settlement(
        books,
        signal_ts_ns=t0,
        side="UP",
        notional_usd=7.40,
        winning_side="UP",
        latency_ms=150,
        max_participation_pct=20,
    )
    assert result.filled
    assert result.execution_ts_ns == t0 + 200_000_000
    assert result.average_price == pytest.approx(0.74)
    assert result.shares == pytest.approx(10.0)
    assert result.gross_pnl_usd == pytest.approx(2.60)
    assert result.net_pnl_usd == pytest.approx(2.60)


def test_paper_fill_rejects_excessive_visible_book_participation():
    t0 = 1_000_000_000
    result = simulate_taker_buy_to_settlement(
        [_book(t0, 0.50, up_size=10.0)],
        signal_ts_ns=t0,
        side="UP",
        notional_usd=2.50,
        winning_side="UP",
        latency_ms=0,
        max_participation_pct=20,
    )
    assert not result.filled
    assert result.skip_reason == "book_participation_limit"
    assert result.participation_pct == pytest.approx(50.0)


def test_taker_fee_reduces_paper_net_pnl():
    t0 = 1_000_000_000
    result = simulate_taker_buy_to_settlement(
        [_book(t0, 0.74)],
        signal_ts_ns=t0,
        side="UP",
        notional_usd=7.40,
        winning_side="UP",
        fees_enabled=True,
        fee_rate=0.07,
        fee_exponent=1.0,
    )
    assert result.filled
    assert result.fee_usd == pytest.approx(10 * 0.07 * 0.74 * 0.26)
    assert result.net_pnl_usd < result.gross_pnl_usd
    assert result.all_in_cost_per_share > result.average_price


def _write_synthetic_recording(tmp_path):
    now = time.time()
    market = MarketInfo(
        slug="btc-updown-5m-paper-test",
        condition_id="condition-paper-test",
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
        "bids": [{"price": "0.69", "size": "100"}],
        "asks": [{"price": "0.70", "size": "100"}],
    }
    down_event = {
        "event_type": "book",
        "asset_id": "down-token",
        "timestamp": str(received_ns // 1_000_000 + 1),
        "bids": [{"price": "0.30", "size": "100"}],
        "asks": [{"price": "0.31", "size": "100"}],
    }
    recorder.apply(WsEnvelope(received_ns, "2026-09-06T00:00:00Z", "{}", up_event))
    recorder.apply(
        WsEnvelope(received_ns + 1_000_000, "2026-09-06T00:00:00.001Z", "{}", down_event)
    )
    recorder.close()
    (recorder.market_dir / "resolution.json").write_text(
        json.dumps({"resolved": True, "winning_side": "UP"}), encoding="utf-8"
    )
    return recorder.market_dir


def test_paper_benchmark_requires_fee_schedule_by_default(tmp_path):
    market_dir = _write_synthetic_recording(tmp_path)
    trade, reason = benchmark_market_paper(market_dir, notional_usd=5.0)
    assert trade is None
    assert reason == "missing_fee_schedule"


def test_paper_benchmark_replays_raw_book_and_settles(tmp_path):
    market_dir = _write_synthetic_recording(tmp_path)
    (market_dir / "fees.json").write_text(
        json.dumps({"fees_enabled": False, "rate": 0.0, "exponent": 1.0}),
        encoding="utf-8",
    )
    trade, reason = benchmark_market_paper(
        market_dir,
        threshold=0.70,
        notional_usd=5.0,
        latency_ms=0,
        max_participation_pct=20,
    )
    assert reason is None
    assert trade is not None
    assert trade.side == "UP"
    assert trade.signal_ask == pytest.approx(0.70)
    assert trade.execution_average_price == pytest.approx(0.70)
    assert trade.entry_slippage == pytest.approx(0.0)
    assert trade.won is True
    assert trade.shares == pytest.approx(5.0 / 0.70)
    assert trade.net_pnl_usd == pytest.approx((5.0 / 0.70) - 5.0)
