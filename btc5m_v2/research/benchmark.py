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


def trade_payload(trade: LegacyTrade) -> dict[str, Any]:
    return trade.__dict__.copy()
