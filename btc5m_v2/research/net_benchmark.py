from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from btc5m_v2.research.benchmark import LegacyTrade, benchmark_legacy_threshold
from btc5m_v2.research.economics import fee_per_share


@dataclass(frozen=True)
class NetLegacyTrade:
    slug: str
    side: str
    winning_side: str
    won: bool
    quoted_entry_ask: float
    modeled_entry_price: float
    entry_fee_per_share: float
    net_pnl_per_share: float
    net_return_on_cost: float
    received_ts_ns: int
    seconds_left: float


def _net_trade(
    trade: LegacyTrade,
    *,
    fee_rate: float,
    fee_enabled: bool,
    fee_exponent: float,
    entry_slippage_abs: float,
) -> NetLegacyTrade:
    slippage = max(0.0, float(entry_slippage_abs))
    modeled_entry = min(1.0, trade.entry_ask + slippage)
    fee = fee_per_share(
        modeled_entry,
        fee_rate=fee_rate,
        enabled=fee_enabled,
        fee_exponent=fee_exponent,
    )
    payout = 1.0 if trade.won else 0.0
    total_cost = modeled_entry + fee
    pnl = payout - total_cost
    return NetLegacyTrade(
        slug=trade.slug,
        side=trade.side,
        winning_side=trade.winning_side,
        won=trade.won,
        quoted_entry_ask=trade.entry_ask,
        modeled_entry_price=modeled_entry,
        entry_fee_per_share=fee,
        net_pnl_per_share=pnl,
        net_return_on_cost=pnl / total_cost if total_cost > 0 else 0.0,
        received_ts_ns=trade.received_ts_ns,
        seconds_left=trade.seconds_left,
    )


def _max_drawdown_per_share(trades: Iterable[NetLegacyTrade]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        equity += trade.net_pnl_per_share
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown


def benchmark_legacy_threshold_net(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
    fee_rate: float = 0.0,
    fee_enabled: bool = False,
    fee_exponent: float = 1.0,
    entry_slippage_abs: float = 0.0,
) -> tuple[list[NetLegacyTrade], dict[str, Any]]:
    """Run the public legacy threshold benchmark with explicit cost assumptions.

    This deliberately keeps fee/slippage inputs configurable. It does not assume
    a market fee schedule that has not been captured with the historical event.
    Settlement is modeled as a hold-to-resolution payout of 1 or 0.
    """

    gross_trades, gross_summary = benchmark_legacy_threshold(
        rows,
        threshold=threshold,
        min_seconds_left=min_seconds_left,
    )
    trades = [
        _net_trade(
            trade,
            fee_rate=fee_rate,
            fee_enabled=fee_enabled,
            fee_exponent=fee_exponent,
            entry_slippage_abs=entry_slippage_abs,
        )
        for trade in gross_trades
    ]
    trades.sort(key=lambda trade: (trade.received_ts_ns, trade.slug))

    wins = sum(1 for trade in trades if trade.won)
    net_total = sum(trade.net_pnl_per_share for trade in trades)
    total_fees = sum(trade.entry_fee_per_share for trade in trades)
    total_slippage = sum(
        trade.modeled_entry_price - trade.quoted_entry_ask for trade in trades
    )
    gross_profit = sum(max(0.0, trade.net_pnl_per_share) for trade in trades)
    gross_loss = sum(max(0.0, -trade.net_pnl_per_share) for trade in trades)

    summary = {
        "strategy": "legacy_stronger_ask_threshold_net",
        "threshold": float(threshold),
        "min_seconds_left": float(min_seconds_left),
        "upper_entry_window_enforced": False,
        "markets_seen": gross_summary["markets_seen"],
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": None if not trades else wins / len(trades),
        "fee_enabled": bool(fee_enabled),
        "fee_rate": float(fee_rate),
        "fee_exponent": float(fee_exponent),
        "entry_slippage_abs": max(0.0, float(entry_slippage_abs)),
        "total_entry_fees_per_share": total_fees,
        "total_modeled_slippage_per_share": total_slippage,
        "net_pnl_per_share_total": net_total,
        "net_pnl_per_share_average": None if not trades else net_total / len(trades),
        "net_return_on_cost_average": None
        if not trades
        else sum(trade.net_return_on_cost for trade in trades) / len(trades),
        "gross_pnl_per_share_total": gross_summary["gross_pnl_per_share_total"],
        "profit_factor": None if gross_loss <= 0 else gross_profit / gross_loss,
        "max_drawdown_per_share": _max_drawdown_per_share(trades),
        "trade_order": "received_ts_ns_ascending",
        "warning": (
            "Cost-aware control benchmark only. Fill probability and market-specific "
            "historical fee metadata must be added before interpreting as tradable P&L."
        ),
    }
    return trades, summary


def trade_payload(trade: NetLegacyTrade) -> dict[str, Any]:
    return trade.__dict__.copy()
