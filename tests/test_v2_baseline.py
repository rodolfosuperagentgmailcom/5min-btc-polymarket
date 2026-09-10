from __future__ import annotations

import pytest

from btc5m_v2.research.baseline import (
    backtest_legacy_threshold,
    legacy_threshold_signal,
    polymarket_fee_per_share,
)


def row(*, slug, ts, seconds_left, up_ask, down_ask, winner, resolved=True):
    return {
        "slug": slug,
        "received_ts_ns": ts,
        "seconds_left": seconds_left,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "winning_side": winner,
        "resolved": resolved,
    }


def test_legacy_signal_chooses_stronger_side_above_threshold():
    assert legacy_threshold_signal({"up_ask": 0.72, "down_ask": 0.20}) == ("UP", 0.72)
    assert legacy_threshold_signal({"up_ask": 0.71, "down_ask": 0.76}) == ("DOWN", 0.76)
    assert legacy_threshold_signal({"up_ask": 0.69, "down_ask": 0.68}) is None


def test_legacy_signal_skips_exact_tie():
    assert legacy_threshold_signal({"up_ask": 0.75, "down_ask": 0.75}) is None


def test_fee_formula_can_be_disabled_or_enabled():
    assert polymarket_fee_per_share(0.70, fee_rate=0.07, enabled=False) == 0.0
    assert polymarket_fee_per_share(0.70, fee_rate=0.07, enabled=True) == pytest.approx(0.0147)


def test_backtest_uses_one_trade_per_market_and_holds_to_resolution():
    rows = [
        row(slug="m1", ts=1, seconds_left=140, up_ask=0.72, down_ask=0.29, winner="UP"),
        row(slug="m1", ts=2, seconds_left=130, up_ask=0.80, down_ask=0.21, winner="UP"),
        row(slug="m2", ts=3, seconds_left=120, up_ask=0.25, down_ask=0.74, winner="UP"),
        row(slug="m3", ts=4, seconds_left=160, up_ask=0.79, down_ask=0.22, winner="UP"),
    ]
    trades, report = backtest_legacy_threshold(rows, fee_rate=0.07, fees_enabled=True)

    assert [trade.slug for trade in trades] == ["m1", "m2"]
    assert trades[0].won is True
    assert trades[0].pnl_per_share == pytest.approx(1.0 - 0.72 - (0.07 * 0.72 * 0.28))
    assert trades[1].won is False
    assert report.trades == 2
    assert report.wins == 1
    assert report.losses == 1
    assert report.win_rate == 0.5
    assert report.fees_per_share > 0
    assert report.max_drawdown_per_share > 0


def test_backtest_skips_unresolved_rows_and_rows_outside_window():
    rows = [
        row(slug="m1", ts=1, seconds_left=120, up_ask=0.80, down_ask=0.20, winner="UP", resolved=False),
        row(slug="m2", ts=2, seconds_left=89, up_ask=0.80, down_ask=0.20, winner="UP"),
        row(slug="m3", ts=3, seconds_left=151, up_ask=0.80, down_ask=0.20, winner="UP"),
    ]
    trades, report = backtest_legacy_threshold(rows)
    assert trades == []
    assert report.trades == 0
    assert report.win_rate is None
