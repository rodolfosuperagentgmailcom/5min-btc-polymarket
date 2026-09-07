from __future__ import annotations

from btc5m_v2.research.benchmark import benchmark_legacy_threshold, first_legacy_trade_for_market


def row(slug: str, ts: int, *, seconds_left: float, up_ask: float, down_ask: float, winner: str) -> dict:
    return {
        "slug": slug,
        "received_ts_ns": ts,
        "seconds_left": seconds_left,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "resolved": True,
        "winning_side": winner,
    }


def test_legacy_benchmark_takes_first_qualifying_threshold_signal() -> None:
    rows = [
        row("m1", 1, seconds_left=200, up_ask=0.65, down_ask=0.35, winner="UP"),
        row("m1", 2, seconds_left=190, up_ask=0.71, down_ask=0.29, winner="UP"),
        row("m1", 3, seconds_left=180, up_ask=0.80, down_ask=0.20, winner="UP"),
    ]
    trade = first_legacy_trade_for_market(rows)
    assert trade is not None
    assert trade.received_ts_ns == 2
    assert trade.side == "UP"
    assert trade.entry_ask == 0.71
    assert trade.won is True
    assert round(trade.gross_pnl_per_share, 6) == 0.29


def test_legacy_benchmark_chooses_stronger_ask_if_both_qualify() -> None:
    rows = [row("m1", 1, seconds_left=120, up_ask=0.72, down_ask=0.75, winner="DOWN")]
    trade = first_legacy_trade_for_market(rows)
    assert trade is not None
    assert trade.side == "DOWN"
    assert trade.entry_ask == 0.75
    assert trade.won is True


def test_legacy_benchmark_only_enforces_minimum_time_guard() -> None:
    # The public legacy runner can enter far earlier than the advertised 120s
    # target because it only checks a minimum seconds-left threshold.
    rows = [row("m1", 1, seconds_left=290, up_ask=0.72, down_ask=0.28, winner="UP")]
    trade = first_legacy_trade_for_market(rows, min_seconds_left=60)
    assert trade is not None
    assert trade.seconds_left == 290


def test_legacy_benchmark_summary_is_explicitly_gross_only() -> None:
    rows = [
        row("win", 1, seconds_left=120, up_ask=0.70, down_ask=0.30, winner="UP"),
        row("loss", 2, seconds_left=120, up_ask=0.30, down_ask=0.70, winner="UP"),
    ]
    trades, summary = benchmark_legacy_threshold(rows)
    assert len(trades) == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["fees_included"] is False
    assert summary["slippage_included"] is False
    assert summary["fill_model_included"] is False
    assert summary["upper_entry_window_enforced"] is False
