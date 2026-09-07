from __future__ import annotations

from btc5m_v2.research.net_benchmark import benchmark_legacy_threshold_net


def row(slug: str, ts: int, up_ask: float, down_ask: float, winner: str) -> dict:
    return {
        "slug": slug,
        "received_ts_ns": ts,
        "seconds_left": 120.0,
        "up_ask": up_ask,
        "down_ask": down_ask,
        "resolved": True,
        "winning_side": winner,
    }


def test_net_benchmark_reproduces_first_threshold_trade_per_market():
    rows = [
        row("m1", 1, 0.69, 0.30, "UP"),
        row("m1", 2, 0.71, 0.29, "UP"),
        row("m1", 3, 0.80, 0.20, "UP"),
        row("m2", 4, 0.25, 0.74, "UP"),
    ]
    trades, summary = benchmark_legacy_threshold_net(rows)
    assert len(trades) == 2
    assert trades[0].slug == "m1"
    assert trades[0].quoted_entry_ask == 0.71
    assert trades[0].won is True
    assert trades[1].side == "DOWN"
    assert trades[1].won is False
    assert summary["trades"] == 2
    assert summary["wins"] == 1


def test_fees_and_slippage_reduce_net_pnl():
    rows = [row("m1", 1, 0.70, 0.30, "UP")]
    grossish, grossish_summary = benchmark_legacy_threshold_net(rows)
    net, net_summary = benchmark_legacy_threshold_net(
        rows,
        fee_enabled=True,
        fee_rate=0.07,
        fee_exponent=1.0,
        entry_slippage_abs=0.01,
    )

    assert net[0].modeled_entry_price == 0.71
    assert net[0].entry_fee_per_share > 0
    assert net[0].net_pnl_per_share < grossish[0].net_pnl_per_share
    assert net_summary["net_pnl_per_share_total"] < grossish_summary["net_pnl_per_share_total"]
    assert net_summary["total_modeled_slippage_per_share"] > 0


def test_fee_schedule_is_not_assumed_when_disabled():
    rows = [row("m1", 1, 0.72, 0.28, "UP")]
    trades, summary = benchmark_legacy_threshold_net(
        rows,
        fee_enabled=False,
        fee_rate=999.0,
    )
    assert trades[0].entry_fee_per_share == 0.0
    assert trades[0].net_pnl_per_share == 0.28
    assert summary["fee_enabled"] is False


def test_summary_reports_chronological_drawdown_and_profit_factor():
    rows = [
        # Intentionally use slugs whose alphabetical order differs from time order.
        row("z-first", 1, 0.70, 0.30, "UP"),   # +0.30
        row("a-second", 2, 0.70, 0.30, "DOWN"), # -0.70
        row("m-third", 3, 0.70, 0.30, "UP"),    # +0.30
    ]
    trades, summary = benchmark_legacy_threshold_net(rows)

    assert [trade.slug for trade in trades] == ["z-first", "a-second", "m-third"]
    assert summary["trade_order"] == "received_ts_ns_ascending"
    assert round(summary["max_drawdown_per_share"], 10) == 0.70
    assert round(summary["profit_factor"], 10) == round(0.60 / 0.70, 10)


def test_profit_factor_is_none_when_there_are_no_losses():
    rows = [
        row("m1", 1, 0.70, 0.30, "UP"),
        row("m2", 2, 0.71, 0.29, "UP"),
    ]
    _, summary = benchmark_legacy_threshold_net(rows)
    assert summary["profit_factor"] is None
    assert summary["max_drawdown_per_share"] == 0.0
