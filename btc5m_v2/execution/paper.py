from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from btc5m_v2.execution.pricing import estimate_buy_fill
from btc5m_v2.fees import all_in_cost_per_share, fee_for_fills
from btc5m_v2.research.book_replay import ReplayedBookSnapshot


@dataclass(frozen=True)
class PaperBuyResult:
    filled: bool
    skip_reason: str | None
    side: str
    signal_ts_ns: int
    execution_ts_ns: int | None
    latency_ms: float
    requested_notional_usd: float
    spent_usd: float
    shares: float
    average_price: float | None
    worst_price: float | None
    participation_pct: float | None
    fee_usd: float
    all_in_cost_usd: float
    all_in_cost_per_share: float | None
    settlement_value_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    return_on_cost: float | None
    winning_side: str
    won: bool | None


def _execution_book(
    books: Sequence[ReplayedBookSnapshot],
    *,
    signal_ts_ns: int,
    latency_ms: float,
) -> ReplayedBookSnapshot | None:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    target_ns = int(signal_ts_ns) + int(float(latency_ms) * 1_000_000)
    for book in books:
        if int(book.received_ts_ns) >= target_ns:
            return book
    return None


def _skip(
    *,
    reason: str,
    side: str,
    signal_ts_ns: int,
    latency_ms: float,
    requested_notional_usd: float,
    winning_side: str,
    execution_ts_ns: int | None = None,
    participation_pct: float | None = None,
) -> PaperBuyResult:
    return PaperBuyResult(
        filled=False,
        skip_reason=reason,
        side=side,
        signal_ts_ns=int(signal_ts_ns),
        execution_ts_ns=execution_ts_ns,
        latency_ms=float(latency_ms),
        requested_notional_usd=float(requested_notional_usd),
        spent_usd=0.0,
        shares=0.0,
        average_price=None,
        worst_price=None,
        participation_pct=participation_pct,
        fee_usd=0.0,
        all_in_cost_usd=0.0,
        all_in_cost_per_share=None,
        settlement_value_usd=0.0,
        gross_pnl_usd=0.0,
        net_pnl_usd=0.0,
        return_on_cost=None,
        winning_side=winning_side,
        won=None,
    )


def simulate_taker_buy_to_settlement(
    books: Sequence[ReplayedBookSnapshot],
    *,
    signal_ts_ns: int,
    side: str,
    notional_usd: float,
    winning_side: str,
    latency_ms: float = 0.0,
    max_participation_pct: float | None = None,
    fees_enabled: bool = False,
    fee_rate: float = 0.0,
    fee_exponent: float = 1.0,
) -> PaperBuyResult:
    """Simulate a taker buy using the first visible book after signal+latency.

    This is intentionally conservative in one respect: if the full requested
    notional is not visible, the paper order is treated as not filled rather than
    crediting a partial FAK fill. That avoids manufacturing P&L from an exposure
    size the strategy did not request.
    """

    label = str(side).strip().upper()
    winner = str(winning_side).strip().upper()
    if label not in {"UP", "DOWN"}:
        raise ValueError("side must be UP or DOWN")
    if winner not in {"UP", "DOWN"}:
        raise ValueError("winning_side must be UP or DOWN")
    requested = float(notional_usd)
    if requested <= 0:
        raise ValueError("notional_usd must be positive")
    if max_participation_pct is not None and float(max_participation_pct) <= 0:
        raise ValueError("max_participation_pct must be positive when set")

    book = _execution_book(books, signal_ts_ns=signal_ts_ns, latency_ms=latency_ms)
    if book is None:
        return _skip(
            reason="no_book_after_latency",
            side=label,
            signal_ts_ns=signal_ts_ns,
            latency_ms=latency_ms,
            requested_notional_usd=requested,
            winning_side=winner,
        )

    fill = estimate_buy_fill(book.asks(label), notional_usd=requested)
    if not fill.fillable:
        return _skip(
            reason="insufficient_visible_ask_liquidity",
            side=label,
            signal_ts_ns=signal_ts_ns,
            latency_ms=latency_ms,
            requested_notional_usd=requested,
            winning_side=winner,
            execution_ts_ns=book.received_ts_ns,
            participation_pct=fill.participation_pct,
        )

    if (
        max_participation_pct is not None
        and fill.participation_pct is not None
        and fill.participation_pct > float(max_participation_pct)
    ):
        return _skip(
            reason="book_participation_limit",
            side=label,
            signal_ts_ns=signal_ts_ns,
            latency_ms=latency_ms,
            requested_notional_usd=requested,
            winning_side=winner,
            execution_ts_ns=book.received_ts_ns,
            participation_pct=fill.participation_pct,
        )

    fee_usd = fee_for_fills(
        fill.fills,
        fee_rate=float(fee_rate),
        fee_exponent=float(fee_exponent),
        fees_enabled=bool(fees_enabled),
        is_taker=True,
    )
    total_cost = fill.spent_usd + fee_usd
    won = label == winner
    settlement_value = fill.shares if won else 0.0
    gross_pnl = settlement_value - fill.spent_usd
    net_pnl = settlement_value - total_cost
    return_pct = net_pnl / total_cost if total_cost > 0 else None

    return PaperBuyResult(
        filled=True,
        skip_reason=None,
        side=label,
        signal_ts_ns=int(signal_ts_ns),
        execution_ts_ns=int(book.received_ts_ns),
        latency_ms=float(latency_ms),
        requested_notional_usd=requested,
        spent_usd=fill.spent_usd,
        shares=fill.shares,
        average_price=fill.average_price,
        worst_price=fill.worst_price,
        participation_pct=fill.participation_pct,
        fee_usd=fee_usd,
        all_in_cost_usd=total_cost,
        all_in_cost_per_share=all_in_cost_per_share(
            spent_usd=fill.spent_usd,
            shares=fill.shares,
            fee_usd=fee_usd,
        ),
        settlement_value_usd=settlement_value,
        gross_pnl_usd=gross_pnl,
        net_pnl_usd=net_pnl,
        return_on_cost=return_pct,
        winning_side=winner,
        won=won,
    )
