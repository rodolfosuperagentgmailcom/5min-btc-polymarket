from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from btc5m_v2.research.fees import taker_fee_usdc


@dataclass(frozen=True)
class LegacyTrade:
    slug: str
    received_ts_ns: int
    seconds_left: float
    side: str
    entry_ask: float
    winning_side: str
    won: bool
    gross_pnl_per_share: float
    gross_return_on_cost: float


@dataclass(frozen=True)
class LegacyNetTrade:
    slug: str
    received_ts_ns: int
    seconds_left: float
    side: str
    observed_ask: float
    executed_price: float
    winning_side: str
    won: bool
    fee_rate: float
    entry_fee_per_share: float
    slippage_per_share: float
    cash_cost_per_share: float
    gross_pnl_per_share: float
    net_pnl_per_share: float
    net_return_on_cash_cost: float


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_legacy_trade_for_market(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
) -> LegacyTrade | None:
    """Reproduce the current public runner's first threshold entry.

    The current runner has a minimum seconds-left guard but no configured upper
    entry-window bound. It chooses the stronger ask if UP/DOWN both qualify.
    This benchmark intentionally mirrors that behavior rather than the V2
    90-150 second research gate.

    Fees, slippage, and fill probability are intentionally excluded here. The
    result is a GROSS control benchmark, not evidence of tradable profitability.
    """

    ordered = sorted(rows, key=lambda row: int(row.get("received_ts_ns") or 0))
    for row in ordered:
        if row.get("resolved") is not True:
            continue
        winner = str(row.get("winning_side") or "").upper()
        if winner not in {"UP", "DOWN"}:
            continue

        seconds_left = _f(row.get("seconds_left"))
        up_ask = _f(row.get("up_ask"))
        down_ask = _f(row.get("down_ask"))
        if seconds_left is None or seconds_left < min_seconds_left:
            continue

        candidates: list[tuple[str, float]] = []
        if up_ask is not None and up_ask >= threshold:
            candidates.append(("UP", up_ask))
        if down_ask is not None and down_ask >= threshold:
            candidates.append(("DOWN", down_ask))
        if not candidates:
            continue

        side, entry = max(candidates, key=lambda item: item[1])
        won = side == winner
        settlement_value = 1.0 if won else 0.0
        pnl = settlement_value - entry
        return LegacyTrade(
            slug=str(row.get("slug") or ""),
            received_ts_ns=int(row.get("received_ts_ns") or 0),
            seconds_left=seconds_left,
            side=side,
            entry_ask=entry,
            winning_side=winner,
            won=won,
            gross_pnl_per_share=pnl,
            gross_return_on_cost=pnl / entry,
        )
    return None


def fee_adjust_legacy_trade(
    trade: LegacyTrade,
    *,
    fee_rate: float = 0.07,
    fees_enabled: bool = True,
    fee_exponent: float = 1.0,
    entry_slippage: float = 0.0,
) -> LegacyNetTrade | None:
    """Apply entry execution costs without changing the legacy signal rule.

    Slippage is expressed as an absolute probability-price increment, e.g.
    0.005 = half a cent worse than the observed ask.
    """

    slip = max(0.0, float(entry_slippage))
    executed = float(trade.entry_ask) + slip
    if not 0 < executed < 1:
        return None

    fee_per_share = taker_fee_usdc(
        1.0,
        executed,
        fee_rate,
        fee_exponent=fee_exponent,
        fees_enabled=fees_enabled,
    )
    cash_cost = executed + fee_per_share
    settlement_value = 1.0 if trade.won else 0.0
    gross_pnl = settlement_value - executed
    net_pnl = settlement_value - cash_cost

    return LegacyNetTrade(
        slug=trade.slug,
        received_ts_ns=trade.received_ts_ns,
        seconds_left=trade.seconds_left,
        side=trade.side,
        observed_ask=trade.entry_ask,
        executed_price=executed,
        winning_side=trade.winning_side,
        won=trade.won,
        fee_rate=float(fee_rate),
        entry_fee_per_share=fee_per_share,
        slippage_per_share=slip,
        cash_cost_per_share=cash_cost,
        gross_pnl_per_share=gross_pnl,
        net_pnl_per_share=net_pnl,
        net_return_on_cash_cost=net_pnl / cash_cost,
    )


def benchmark_legacy_threshold(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
) -> tuple[list[LegacyTrade], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        slug = str(row.get("slug") or "")
        if not slug:
            continue
        grouped.setdefault(slug, []).append(dict(row))

    trades: list[LegacyTrade] = []
    for slug in sorted(grouped):
        trade = first_legacy_trade_for_market(
            grouped[slug],
            threshold=threshold,
            min_seconds_left=min_seconds_left,
        )
        if trade is not None:
            trades.append(trade)

    wins = sum(1 for trade in trades if trade.won)
    losses = len(trades) - wins
    summary = {
        "strategy": "legacy_stronger_ask_threshold",
        "threshold": threshold,
        "min_seconds_left": min_seconds_left,
        "upper_entry_window_enforced": False,
        "markets_seen": len(grouped),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": None if not trades else wins / len(trades),
        "average_entry_ask": None
        if not trades
        else sum(trade.entry_ask for trade in trades) / len(trades),
        "gross_pnl_per_share_total": sum(trade.gross_pnl_per_share for trade in trades),
        "gross_pnl_per_share_average": None
        if not trades
        else sum(trade.gross_pnl_per_share for trade in trades) / len(trades),
        "gross_return_on_cost_average": None
        if not trades
        else sum(trade.gross_return_on_cost for trade in trades) / len(trades),
        "fees_included": False,
        "slippage_included": False,
        "fill_model_included": False,
        "warning": "Gross benchmark only; do not interpret as net tradable edge.",
    }
    return trades, summary


def benchmark_legacy_threshold_net(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
    fee_rate: float = 0.07,
    fees_enabled: bool = True,
    fee_exponent: float = 1.0,
    entry_slippage: float = 0.0,
) -> tuple[list[LegacyNetTrade], dict[str, Any]]:
    """Run the same legacy trigger with explicit entry execution costs."""

    gross_trades, gross_summary = benchmark_legacy_threshold(
        rows,
        threshold=threshold,
        min_seconds_left=min_seconds_left,
    )

    net_trades: list[LegacyNetTrade] = []
    for trade in gross_trades:
        adjusted = fee_adjust_legacy_trade(
            trade,
            fee_rate=fee_rate,
            fees_enabled=fees_enabled,
            fee_exponent=fee_exponent,
            entry_slippage=entry_slippage,
        )
        if adjusted is not None:
            net_trades.append(adjusted)

    total_fees = sum(trade.entry_fee_per_share for trade in net_trades)
    total_slippage = sum(trade.slippage_per_share for trade in net_trades)
    total_net = sum(trade.net_pnl_per_share for trade in net_trades)
    wins = sum(1 for trade in net_trades if trade.won)

    summary = dict(gross_summary)
    summary.update(
        {
            "strategy": "legacy_stronger_ask_threshold_fee_aware",
            "trades": len(net_trades),
            "wins": wins,
            "losses": len(net_trades) - wins,
            "win_rate": None if not net_trades else wins / len(net_trades),
            "fee_rate": float(fee_rate),
            "fee_exponent": float(fee_exponent),
            "fees_enabled": bool(fees_enabled),
            "entry_slippage": float(entry_slippage),
            "fees_included": bool(fees_enabled and fee_rate > 0),
            "slippage_included": float(entry_slippage) > 0,
            "fill_model_included": False,
            "entry_fee_per_share_total": total_fees,
            "entry_slippage_per_share_total": total_slippage,
            "net_pnl_per_share_total": total_net,
            "net_pnl_per_share_average": None
            if not net_trades
            else total_net / len(net_trades),
            "net_return_on_cash_cost_average": None
            if not net_trades
            else sum(trade.net_return_on_cash_cost for trade in net_trades)
            / len(net_trades),
            "warning": (
                "Fee/slippage-aware benchmark only; no queue/fill-probability model yet."
            ),
        }
    )
    return net_trades, summary


def trade_payload(trade: LegacyTrade | LegacyNetTrade) -> dict[str, Any]:
    return trade.__dict__.copy()
