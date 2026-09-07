from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from btc5m_v2.execution.paper import PaperBuyResult, simulate_taker_buy_to_settlement
from btc5m_v2.research.book_replay import reconstruct_and_align_books
from btc5m_v2.research.fee_enrich import read_fee_schedule
from btc5m_v2.research.quality import require_contiguous_recording
from btc5m_v2.research.replay import iter_snapshot_rows, read_resolution


@dataclass(frozen=True)
class PaperBenchmarkTrade:
    slug: str
    signal_ts_ns: int
    execution_ts_ns: int
    seconds_left: float
    latency_ms: float
    side: str
    signal_ask: float
    execution_average_price: float
    entry_slippage: float
    requested_notional_usd: float
    shares: float
    fee_usd: float
    all_in_cost_usd: float
    all_in_cost_per_share: float
    participation_pct: float | None
    winning_side: str
    won: bool
    gross_pnl_usd: float
    net_pnl_usd: float
    return_on_cost: float


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _metadata(market_dir: str | Path) -> dict[str, Any]:
    path = Path(market_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid metadata object: {path}")
    return payload


def _first_signal(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    min_seconds_left: float,
) -> tuple[int, dict[str, Any], str, float] | None:
    for index, row in enumerate(rows):
        seconds_left = _f(row.get("seconds_left"))
        if seconds_left is None or seconds_left < float(min_seconds_left):
            continue
        up_ask = _f(row.get("up_ask"))
        down_ask = _f(row.get("down_ask"))
        candidates: list[tuple[str, float]] = []
        if up_ask is not None and up_ask >= threshold:
            candidates.append(("UP", up_ask))
        if down_ask is not None and down_ask >= threshold:
            candidates.append(("DOWN", down_ask))
        if not candidates:
            continue
        side, ask = max(candidates, key=lambda item: item[1])
        return index, row, side, ask
    return None


def _fee_values(schedule: dict[str, Any] | None) -> tuple[bool, float, float]:
    if not schedule:
        raise ValueError("missing_fee_schedule")
    return (
        bool(schedule.get("fees_enabled", False)),
        float(schedule.get("rate") or 0.0),
        float(schedule.get("exponent") or 1.0),
    )


def benchmark_market_paper(
    market_dir: str | Path,
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
    notional_usd: float = 5.0,
    latency_ms: float = 0.0,
    max_participation_pct: float = 20.0,
    require_fee_schedule: bool = True,
) -> tuple[PaperBenchmarkTrade | None, str | None]:
    directory = Path(market_dir)
    try:
        require_contiguous_recording(directory)
    except ValueError as exc:
        return None, str(exc)

    resolved, winning_side = read_resolution(directory)
    if not resolved or winning_side not in {"UP", "DOWN"}:
        return None, "unresolved_market"

    rows = list(iter_snapshot_rows(directory))
    if not rows:
        return None, "missing_snapshots"
    signal = _first_signal(
        rows,
        threshold=float(threshold),
        min_seconds_left=float(min_seconds_left),
    )
    if signal is None:
        return None, "no_signal"

    metadata = _metadata(directory)
    up_token = str(metadata.get("up_token_id") or "").strip()
    down_token = str(metadata.get("down_token_id") or "").strip()
    if not up_token or not down_token:
        return None, "missing_token_ids"
    raw_path = directory / "clob_raw.jsonl"
    if not raw_path.exists():
        return None, "missing_raw_book"

    try:
        books = reconstruct_and_align_books(
            raw_path,
            recorded_rows=rows,
            up_token_id=up_token,
            down_token_id=down_token,
        )
    except Exception as exc:
        return None, f"book_replay_error:{type(exc).__name__}"

    fee_schedule = read_fee_schedule(directory)
    if require_fee_schedule and fee_schedule is None:
        return None, "missing_fee_schedule"
    try:
        fees_enabled, fee_rate, fee_exponent = _fee_values(fee_schedule or {
            "fees_enabled": False,
            "rate": 0.0,
            "exponent": 1.0,
        })
    except Exception:
        return None, "invalid_fee_schedule"

    _, row, side, signal_ask = signal
    signal_ts_ns = int(row.get("received_ts_ns") or 0)
    result: PaperBuyResult = simulate_taker_buy_to_settlement(
        books,
        signal_ts_ns=signal_ts_ns,
        side=side,
        notional_usd=float(notional_usd),
        winning_side=winning_side,
        latency_ms=float(latency_ms),
        max_participation_pct=float(max_participation_pct),
        fees_enabled=fees_enabled,
        fee_rate=fee_rate,
        fee_exponent=fee_exponent,
    )
    if not result.filled:
        return None, result.skip_reason or "paper_fill_rejected"

    assert result.execution_ts_ns is not None
    assert result.average_price is not None
    assert result.all_in_cost_per_share is not None
    assert result.won is not None
    assert result.return_on_cost is not None
    seconds_left = float(row.get("seconds_left") or 0.0)
    return (
        PaperBenchmarkTrade(
            slug=str(row.get("slug") or directory.name),
            signal_ts_ns=signal_ts_ns,
            execution_ts_ns=result.execution_ts_ns,
            seconds_left=seconds_left,
            latency_ms=float(latency_ms),
            side=side,
            signal_ask=signal_ask,
            execution_average_price=result.average_price,
            entry_slippage=result.average_price - signal_ask,
            requested_notional_usd=float(notional_usd),
            shares=result.shares,
            fee_usd=result.fee_usd,
            all_in_cost_usd=result.all_in_cost_usd,
            all_in_cost_per_share=result.all_in_cost_per_share,
            participation_pct=result.participation_pct,
            winning_side=winning_side,
            won=result.won,
            gross_pnl_usd=result.gross_pnl_usd,
            net_pnl_usd=result.net_pnl_usd,
            return_on_cost=result.return_on_cost,
        ),
        None,
    )


def market_dirs(output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    return sorted(path.parent for path in root.glob("*/*/metadata.json"))


def benchmark_paper_root(
    output_root: str | Path,
    *,
    threshold: float = 0.70,
    min_seconds_left: float = 60.0,
    notional_usd: float = 5.0,
    latency_ms: float = 0.0,
    max_participation_pct: float = 20.0,
    require_fee_schedule: bool = True,
) -> tuple[list[PaperBenchmarkTrade], dict[str, Any]]:
    trades: list[PaperBenchmarkTrade] = []
    skipped: Counter[str] = Counter()
    directories = market_dirs(output_root)

    for directory in directories:
        trade, reason = benchmark_market_paper(
            directory,
            threshold=threshold,
            min_seconds_left=min_seconds_left,
            notional_usd=notional_usd,
            latency_ms=latency_ms,
            max_participation_pct=max_participation_pct,
            require_fee_schedule=require_fee_schedule,
        )
        if trade is not None:
            trades.append(trade)
        else:
            skipped[str(reason or "unknown")] += 1

    wins = sum(1 for trade in trades if trade.won)
    total_cost = sum(trade.all_in_cost_usd for trade in trades)
    net_pnl = sum(trade.net_pnl_usd for trade in trades)
    summary = {
        "strategy": "legacy_stronger_ask_threshold_paper",
        "threshold": float(threshold),
        "min_seconds_left": float(min_seconds_left),
        "upper_entry_window_enforced": False,
        "notional_usd": float(notional_usd),
        "latency_ms": float(latency_ms),
        "max_participation_pct": float(max_participation_pct),
        "fee_schedule_required": bool(require_fee_schedule),
        "recording_quality_required": "reconnects == 0",
        "markets_seen": len(directories),
        "executed_trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": None if not trades else wins / len(trades),
        "gross_pnl_usd": sum(trade.gross_pnl_usd for trade in trades),
        "fees_usd": sum(trade.fee_usd for trade in trades),
        "net_pnl_usd": net_pnl,
        "return_on_cost": None if total_cost <= 0 else net_pnl / total_cost,
        "average_entry_slippage": None
        if not trades
        else sum(trade.entry_slippage for trade in trades) / len(trades),
        "average_participation_pct": None
        if not trades
        else sum((trade.participation_pct or 0.0) for trade in trades) / len(trades),
        "skipped": dict(sorted(skipped.items())),
        "warning": "Paper replay only. Latency is a modeled assumption, not measured personal execution latency.",
    }
    return trades, summary


def trade_payload(trade: PaperBenchmarkTrade) -> dict[str, Any]:
    return trade.__dict__.copy()


def latency_sweep(
    output_root: str | Path,
    latencies_ms: Iterable[float],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for latency in latencies_ms:
        _, summary = benchmark_paper_root(
            output_root,
            latency_ms=float(latency),
            **kwargs,
        )
        summaries.append(summary)
    return summaries
