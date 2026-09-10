from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BaselineTrade:
    slug: str
    received_ts_ns: int
    seconds_left: float
    side: str
    entry_price: float
    fee_per_share: float
    all_in_cost_per_share: float
    winning_side: str
    won: bool
    pnl_per_share: float


@dataclass(frozen=True)
class BaselineReport:
    trades: int
    wins: int
    losses: int
    win_rate: float | None
    gross_pnl_per_share: float
    fees_per_share: float
    net_pnl_per_share: float
    average_entry_price: float | None
    average_net_pnl_per_trade: float | None
    profit_factor: float | None
    max_drawdown_per_share: float


def polymarket_fee_per_share(price: float, *, fee_rate: float, enabled: bool) -> float:
    p = float(price)
    if not enabled:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError("price must be between 0 and 1")
    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    return float(fee_rate) * p * (1.0 - p)


def legacy_threshold_signal(
    row: dict[str, Any],
    *,
    threshold: float = 0.70,
) -> tuple[str, float] | None:
    """Reproduce the original stronger-side ask>=threshold rule.

    If both sides clear the threshold, choose the higher ask. Ties are skipped.
    """

    up = _as_price(row.get("up_ask"))
    down = _as_price(row.get("down_ask"))
    candidates: list[tuple[str, float]] = []
    if up is not None and up >= threshold:
        candidates.append(("UP", up))
    if down is not None and down >= threshold:
        candidates.append(("DOWN", down))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
        return None
    return candidates[0]


def backtest_legacy_threshold(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.70,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
    fee_rate: float = 0.0,
    fees_enabled: bool = False,
    one_trade_per_market: bool = True,
) -> tuple[list[BaselineTrade], BaselineReport]:
    """Backtest the legacy threshold rule on chronologically ordered replay rows.

    The terminal label is used only after the entry signal is formed. P&L assumes
    holding one share to resolution: winner pays 1, loser pays 0.
    """

    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row.get("received_ts_ns") or 0))
    seen_markets: set[str] = set()
    trades: list[BaselineTrade] = []

    for row in ordered:
        if row.get("resolved") is not True:
            continue
        winner = str(row.get("winning_side") or "").upper()
        if winner not in {"UP", "DOWN"}:
            continue

        seconds_left = _as_float(row.get("seconds_left"))
        if seconds_left is None or not (seconds_left_min <= seconds_left <= seconds_left_max):
            continue

        slug = str(row.get("slug") or "")
        if one_trade_per_market and slug in seen_markets:
            continue

        signal = legacy_threshold_signal(row, threshold=threshold)
        if signal is None:
            continue

        side, entry = signal
        fee = polymarket_fee_per_share(entry, fee_rate=fee_rate, enabled=fees_enabled)
        all_in = entry + fee
        won = side == winner
        settlement = 1.0 if won else 0.0
        pnl = settlement - all_in
        trades.append(
            BaselineTrade(
                slug=slug,
                received_ts_ns=int(row.get("received_ts_ns") or 0),
                seconds_left=seconds_left,
                side=side,
                entry_price=entry,
                fee_per_share=fee,
                all_in_cost_per_share=all_in,
                winning_side=winner,
                won=won,
                pnl_per_share=pnl,
            )
        )
        if one_trade_per_market:
            seen_markets.add(slug)

    return trades, summarize_baseline(trades)


def summarize_baseline(trades: Iterable[BaselineTrade]) -> BaselineReport:
    items = list(trades)
    wins = sum(1 for trade in items if trade.won)
    losses = len(items) - wins
    gross = sum((1.0 if trade.won else 0.0) - trade.entry_price for trade in items)
    fees = sum(trade.fee_per_share for trade in items)
    net = sum(trade.pnl_per_share for trade in items)
    average_entry = None if not items else sum(t.entry_price for t in items) / len(items)
    avg_net = None if not items else net / len(items)

    positive = sum(max(0.0, trade.pnl_per_share) for trade in items)
    negative = -sum(min(0.0, trade.pnl_per_share) for trade in items)
    profit_factor = None if negative <= 0.0 else positive / negative

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade in items:
        equity += trade.pnl_per_share
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return BaselineReport(
        trades=len(items),
        wins=wins,
        losses=losses,
        win_rate=None if not items else wins / len(items),
        gross_pnl_per_share=gross,
        fees_per_share=fees,
        net_pnl_per_share=net,
        average_entry_price=average_entry,
        average_net_pnl_per_trade=avg_net,
        profit_factor=profit_factor,
        max_drawdown_per_share=max_dd,
    )


def report_payload(report: BaselineReport) -> dict[str, Any]:
    return report.__dict__.copy()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_price(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is None or not 0.0 <= parsed <= 1.0:
        return None
    return parsed
