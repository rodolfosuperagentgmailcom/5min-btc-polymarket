from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


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
class FeeAwareLegacyTrade:
    slug: str
    received_ts_ns: int
    seconds_left: float
    side: str
    entry_ask: float
    winning_side: str
    won: bool
    fee_per_share: float
    all_in_cost_per_share: float
    breakeven_probability: float
    gross_pnl_per_share: float
    net_pnl_per_share: float
    net_return_on_cost: float


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fee_per_share(
    price: float,
    *,
    fee_rate: float,
    fees_enabled: bool = True,
) -> float:
    """Return taker fee expressed per share.

    The caller supplies the market-specific fee rate. This research module does
    not hard-code a global Polymarket fee assumption.
    """

    p = float(price)
    rate = float(fee_rate)
    if not 0.0 <= p <= 1.0:
        raise ValueError("price must be in [0, 1]")
    if rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not fees_enabled:
        return 0.0
    return rate * p * (1.0 - p)


def breakeven_probability(
    entry_price: float,
    *,
    fee_rate: float,
    fees_enabled: bool = True,
) -> float:
    return float(entry_price) + fee_per_share(
        entry_price,
        fee_rate=fee_rate,
        fees_enabled=fees_enabled,
    )


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


def first_fee_aware_legacy_trade_for_market(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
    fee_rate: float = 0.0,
    fees_enabled: bool = False,
) -> FeeAwareLegacyTrade | None:
    gross_trade = first_legacy_trade_for_market(
        rows,
        threshold=threshold,
        min_seconds_left=min_seconds_left,
    )
    if gross_trade is None:
        return None

    fee = fee_per_share(
        gross_trade.entry_ask,
        fee_rate=fee_rate,
        fees_enabled=fees_enabled,
    )
    all_in = gross_trade.entry_ask + fee
    settlement_value = 1.0 if gross_trade.won else 0.0
    net_pnl = settlement_value - all_in

    return FeeAwareLegacyTrade(
        slug=gross_trade.slug,
        received_ts_ns=gross_trade.received_ts_ns,
        seconds_left=gross_trade.seconds_left,
        side=gross_trade.side,
        entry_ask=gross_trade.entry_ask,
        winning_side=gross_trade.winning_side,
        won=gross_trade.won,
        fee_per_share=fee,
        all_in_cost_per_share=all_in,
        breakeven_probability=all_in,
        gross_pnl_per_share=gross_trade.gross_pnl_per_share,
        net_pnl_per_share=net_pnl,
        net_return_on_cost=net_pnl / all_in,
    )


def _max_drawdown(pnls: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += float(pnl)
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


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


def benchmark_legacy_threshold_fee_aware(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
    fee_rate: float = 0.0,
    fees_enabled: bool = False,
) -> tuple[list[FeeAwareLegacyTrade], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        slug = str(row.get("slug") or "")
        if not slug:
            continue
        grouped.setdefault(slug, []).append(dict(row))

    trades: list[FeeAwareLegacyTrade] = []
    for slug in sorted(grouped):
        trade = first_fee_aware_legacy_trade_for_market(
            grouped[slug],
            threshold=threshold,
            min_seconds_left=min_seconds_left,
            fee_rate=fee_rate,
            fees_enabled=fees_enabled,
        )
        if trade is not None:
            trades.append(trade)

    wins = sum(1 for trade in trades if trade.won)
    losses = len(trades) - wins
    positive = sum(max(0.0, trade.net_pnl_per_share) for trade in trades)
    negative = sum(max(0.0, -trade.net_pnl_per_share) for trade in trades)
    net_total = sum(trade.net_pnl_per_share for trade in trades)
    fee_total = sum(trade.fee_per_share for trade in trades)

    summary = {
        "strategy": "legacy_stronger_ask_threshold_fee_aware",
        "threshold": threshold,
        "min_seconds_left": min_seconds_left,
        "upper_entry_window_enforced": False,
        "markets_seen": len(grouped),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": None if not trades else wins / len(trades),
        "fee_rate": float(fee_rate),
        "fees_enabled": bool(fees_enabled),
        "fees_per_share_total": fee_total,
        "gross_pnl_per_share_total": sum(trade.gross_pnl_per_share for trade in trades),
        "net_pnl_per_share_total": net_total,
        "net_pnl_per_share_average": None if not trades else net_total / len(trades),
        "average_entry_ask": None
        if not trades
        else sum(trade.entry_ask for trade in trades) / len(trades),
        "average_breakeven_probability": None
        if not trades
        else sum(trade.breakeven_probability for trade in trades) / len(trades),
        "profit_factor": None if negative <= 0 else positive / negative,
        "max_drawdown_per_share": _max_drawdown(
            trade.net_pnl_per_share for trade in trades
        ),
        "slippage_included": False,
        "fill_model_included": False,
        "warning": "Fee-aware only; still excludes slippage and fill probability.",
    }
    return trades, summary


def trade_payload(trade: LegacyTrade | FeeAwareLegacyTrade) -> dict[str, Any]:
    return trade.__dict__.copy()
