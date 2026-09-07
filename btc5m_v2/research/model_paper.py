from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from btc5m_v2.config import V2Config, load_config
from btc5m_v2.execution.paper import simulate_taker_buy_to_settlement
from btc5m_v2.research.book_replay import ReplayedBookSnapshot, reconstruct_and_align_books
from btc5m_v2.research.fee_enrich import read_fee_schedule
from btc5m_v2.research.paper_benchmark import PaperBenchmarkTrade, benchmark_market_paper
from btc5m_v2.research.quality import require_contiguous_recording
from btc5m_v2.research.replay import iter_snapshot_rows, read_resolution
from btc5m_v2.research.train_model import (
    DEFAULT_FEATURE_COLUMNS,
    TrainingResult,
    eligible_rows,
    fit_logistic_baseline,
    load_dataset_rows,
    select_market_decision_rows,
)
from btc5m_v2.strategy.edge import evaluate_buy_edge
from btc5m_v2.strategy.signal import decide_snapshot_with_model


@dataclass(frozen=True)
class ModelPaperTrade:
    slug: str
    signal_ts_ns: int
    execution_ts_ns: int
    seconds_left: float
    latency_ms: float
    side: str
    q_up: float
    model_probability_side: float
    signal_best_ask: float
    executable_edge_per_share: float
    execution_average_price: float
    requested_notional_usd: float
    shares: float
    fee_usd: float
    all_in_cost_usd: float
    all_in_cost_per_share: float
    participation_pct: float | None
    winning_side: str
    won: bool
    net_pnl_usd: float
    return_on_cost: float


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fee_values(schedule: dict[str, Any] | None) -> tuple[bool, float, float]:
    if not schedule:
        raise ValueError("missing_fee_schedule")
    enabled = bool(schedule.get("fees_enabled", False))
    rate = float(schedule.get("rate") or 0.0)
    exponent = float(schedule.get("exponent") or 1.0)
    if rate < 0 or exponent < 0:
        raise ValueError("invalid_fee_schedule")
    return enabled, rate, exponent


def _market_dirs_by_slug(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    mapping: dict[str, Path] = {}
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = str(payload.get("slug") or metadata_path.parent.name).strip()
        if slug:
            mapping[slug] = metadata_path.parent
    return mapping


def chronological_test_rows(
    rows: Iterable[dict[str, Any]],
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_seconds_left: float = 120.0,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
    train_fraction: float = 0.70,
) -> list[dict[str, Any]]:
    """Reproduce the trainer's market-level chronological held-out partition."""

    selected = select_market_decision_rows(
        rows,
        target_seconds_left=target_seconds_left,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
    )
    clean, _ = eligible_rows(selected, feature_columns=feature_columns)
    if len(clean) < 4:
        return []
    split_index = int(len(clean) * float(train_fraction))
    split_index = max(2, min(split_index, len(clean) - 2))
    return clean[split_index:]


def _same_snapshot(recorded: dict[str, Any], feature: dict[str, Any]) -> bool:
    if int(recorded.get("received_ts_ns") or 0) != int(feature.get("received_ts_ns") or 0):
        return False
    fields = (
        "up_ask",
        "down_ask",
        "up_spread",
        "down_spread",
        "up_ask_depth_3_usd",
        "down_ask_depth_3_usd",
    )
    for name in fields:
        left = _finite(recorded.get(name))
        right = _finite(feature.get(name))
        if left is None or right is None:
            if left != right:
                return False
            continue
        if abs(left - right) > 1e-9:
            return False
    return True


def _signal_snapshot(
    recorded_rows: list[dict[str, Any]],
    books: list[ReplayedBookSnapshot],
    feature_row: dict[str, Any],
) -> tuple[dict[str, Any], ReplayedBookSnapshot] | None:
    for index, recorded in enumerate(recorded_rows):
        if _same_snapshot(recorded, feature_row):
            payload = dict(recorded)
            payload.update(feature_row)
            return payload, books[index]
    return None


def _metadata(market_dir: Path) -> dict[str, Any]:
    path = market_dir / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_market_metadata")
    return payload


def benchmark_model_market(
    market_dir: str | Path,
    feature_row: dict[str, Any],
    *,
    training: TrainingResult,
    cfg: V2Config,
    notional_usd: float = 5.0,
    latency_ms: float = 0.0,
    max_participation_pct: float = 20.0,
) -> tuple[ModelPaperTrade | None, str | None]:
    directory = Path(market_dir)
    try:
        require_contiguous_recording(directory)
    except ValueError as exc:
        return None, str(exc)

    resolved, winning_side = read_resolution(directory)
    if not resolved or winning_side not in {"UP", "DOWN"}:
        return None, "unresolved_market"

    recorded_rows = list(iter_snapshot_rows(directory))
    if not recorded_rows:
        return None, "missing_snapshots"
    raw_path = directory / "clob_raw.jsonl"
    if not raw_path.exists():
        return None, "missing_raw_book"

    try:
        metadata = _metadata(directory)
        up_token = str(metadata.get("up_token_id") or "").strip()
        down_token = str(metadata.get("down_token_id") or "").strip()
        if not up_token or not down_token:
            return None, "missing_token_ids"
        books = reconstruct_and_align_books(
            raw_path,
            recorded_rows=recorded_rows,
            up_token_id=up_token,
            down_token_id=down_token,
        )
    except Exception as exc:
        return None, f"book_replay_error:{type(exc).__name__}"

    matched = _signal_snapshot(recorded_rows, books, feature_row)
    if matched is None:
        return None, "decision_snapshot_not_found"
    signal_row, signal_book = matched

    try:
        fees_enabled, fee_rate, fee_exponent = _fee_values(read_fee_schedule(directory))
    except ValueError as exc:
        return None, str(exc)

    decision = decide_snapshot_with_model(
        signal_row,
        model=training.model,
        cfg=cfg,
        fees_enabled=fees_enabled,
        taker_fee_rate=fee_rate,
        fee_exponent=fee_exponent,
    )
    if not decision.trade or decision.side not in {"UP", "DOWN"}:
        reason = decision.reasons[0] if decision.reasons else "model_no_trade"
        return None, f"model_gate:{reason}"
    if decision.model_probability is None:
        return None, "model_probability_missing"

    opportunity = evaluate_buy_edge(
        side=decision.side,
        model_probability=decision.model_probability,
        asks=signal_book.asks(decision.side),
        notional_usd=float(notional_usd),
        fees_enabled=fees_enabled,
        fee_rate=fee_rate,
        fee_exponent=fee_exponent,
        is_taker=True,
        max_book_participation_pct=float(max_participation_pct),
    )
    minimum_edge = float(cfg.raw["signal"]["minimum_edge_after_fees_and_slippage"])
    if not opportunity.tradable:
        reason = opportunity.reasons[0] if opportunity.reasons else "unfillable"
        return None, f"executable_edge:{reason}"
    if opportunity.edge_per_share is None or opportunity.edge_per_share < minimum_edge:
        return None, "executable_edge:edge_below_minimum"
    if opportunity.expected_value_usd is None or opportunity.expected_value_usd <= 0:
        return None, "executable_edge:non_positive_expected_value"

    signal_ts_ns = int(signal_row.get("received_ts_ns") or 0)
    result = simulate_taker_buy_to_settlement(
        books,
        signal_ts_ns=signal_ts_ns,
        side=decision.side,
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
    assert opportunity.edge_per_share is not None
    q_up = training.model.predict_up(feature_row)
    assert q_up is not None

    return (
        ModelPaperTrade(
            slug=str(feature_row.get("slug") or directory.name),
            signal_ts_ns=signal_ts_ns,
            execution_ts_ns=result.execution_ts_ns,
            seconds_left=float(feature_row.get("seconds_left") or 0.0),
            latency_ms=float(latency_ms),
            side=decision.side,
            q_up=q_up,
            model_probability_side=decision.model_probability,
            signal_best_ask=float(decision.entry_price or 0.0),
            executable_edge_per_share=opportunity.edge_per_share,
            execution_average_price=result.average_price,
            requested_notional_usd=float(notional_usd),
            shares=result.shares,
            fee_usd=result.fee_usd,
            all_in_cost_usd=result.all_in_cost_usd,
            all_in_cost_per_share=result.all_in_cost_per_share,
            participation_pct=result.participation_pct,
            winning_side=winning_side,
            won=result.won,
            net_pnl_usd=result.net_pnl_usd,
            return_on_cost=result.return_on_cost,
        ),
        None,
    )


def _trade_summary(
    trades: Sequence[ModelPaperTrade | PaperBenchmarkTrade],
    *,
    skipped: Counter[str],
) -> dict[str, Any]:
    wins = sum(1 for trade in trades if trade.won)
    total_cost = sum(float(trade.all_in_cost_usd) for trade in trades)
    net_pnl = sum(float(trade.net_pnl_usd) for trade in trades)
    return {
        "executed_trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": None if not trades else wins / len(trades),
        "fees_usd": sum(float(trade.fee_usd) for trade in trades),
        "net_pnl_usd": net_pnl,
        "return_on_cost": None if total_cost <= 0 else net_pnl / total_cost,
        "skipped": dict(sorted(skipped.items())),
    }


def compare_model_vs_legacy_out_of_sample(
    dataset_rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    cfg: V2Config | None = None,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_seconds_left: float = 120.0,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
    train_fraction: float = 0.70,
    minimum_markets: int = 100,
    regularization_c: float = 1.0,
    notional_usd: float = 5.0,
    latency_ms: float = 0.0,
    max_participation_pct: float = 20.0,
    legacy_threshold: float = 0.70,
) -> dict[str, Any]:
    rows = [dict(row) for row in dataset_rows]
    configuration = cfg or load_config()
    training = fit_logistic_baseline(
        rows,
        feature_columns=feature_columns,
        target_seconds_left=target_seconds_left,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
        train_fraction=train_fraction,
        minimum_markets=minimum_markets,
        regularization_c=regularization_c,
    )
    test_rows = chronological_test_rows(
        rows,
        feature_columns=feature_columns,
        target_seconds_left=target_seconds_left,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
        train_fraction=train_fraction,
    )
    directories = _market_dirs_by_slug(output_root)

    model_trades: list[ModelPaperTrade] = []
    legacy_trades: list[PaperBenchmarkTrade] = []
    model_skipped: Counter[str] = Counter()
    legacy_skipped: Counter[str] = Counter()

    for feature_row in test_rows:
        slug = str(feature_row.get("slug") or "")
        directory = directories.get(slug)
        if directory is None:
            model_skipped["missing_market_directory"] += 1
            legacy_skipped["missing_market_directory"] += 1
            continue

        model_trade, model_reason = benchmark_model_market(
            directory,
            feature_row,
            training=training,
            cfg=configuration,
            notional_usd=notional_usd,
            latency_ms=latency_ms,
            max_participation_pct=max_participation_pct,
        )
        if model_trade is None:
            model_skipped[str(model_reason or "unknown")] += 1
        else:
            model_trades.append(model_trade)

        legacy_trade, legacy_reason = benchmark_market_paper(
            directory,
            threshold=float(legacy_threshold),
            min_seconds_left=60.0,
            notional_usd=float(notional_usd),
            latency_ms=float(latency_ms),
            max_participation_pct=float(max_participation_pct),
            require_fee_schedule=True,
        )
        if legacy_trade is None:
            legacy_skipped[str(legacy_reason or "unknown")] += 1
        else:
            legacy_trades.append(legacy_trade)

    model_summary = _trade_summary(model_trades, skipped=model_skipped)
    legacy_summary = _trade_summary(legacy_trades, skipped=legacy_skipped)
    model_return = model_summary["return_on_cost"]
    legacy_return = legacy_summary["return_on_cost"]

    return {
        "evaluation": "chronological_out_of_sample_same_market_set",
        "held_out_markets": len(test_rows),
        "target_seconds_left": float(target_seconds_left),
        "decision_window_seconds_left": [float(seconds_left_min), float(seconds_left_max)],
        "notional_usd": float(notional_usd),
        "latency_ms": float(latency_ms),
        "max_participation_pct": float(max_participation_pct),
        "model_training_metrics": training.summary,
        "model_paper": model_summary,
        "legacy_threshold": float(legacy_threshold),
        "legacy_paper": legacy_summary,
        "net_pnl_delta_model_minus_legacy": (
            float(model_summary["net_pnl_usd"]) - float(legacy_summary["net_pnl_usd"])
        ),
        "return_on_cost_delta_model_minus_legacy": (
            None
            if model_return is None or legacy_return is None
            else float(model_return) - float(legacy_return)
        ),
        "warning": (
            "Research paper replay only. The model is trained on earlier markets and evaluated on later markets; "
            "one chronological split is not sufficient evidence for live deployment."
        ),
    }


def compare_from_parquet(
    dataset_path: str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return compare_model_vs_legacy_out_of_sample(
        load_dataset_rows(dataset_path),
        output_root,
        **kwargs,
    )
