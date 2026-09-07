from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from btc5m_v2.config import V2Config, load_config
from btc5m_v2.research.model_paper import ModelPaperTrade, benchmark_model_market
from btc5m_v2.research.paper_benchmark import PaperBenchmarkTrade, benchmark_market_paper
from btc5m_v2.research.train_model import (
    DEFAULT_FEATURE_COLUMNS,
    TrainingResult,
    eligible_rows,
    select_market_decision_rows,
)
from btc5m_v2.strategy.model import LogisticProbabilityModel, ensure_no_label_features


def _market_dirs_by_slug(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    result: dict[str, Path] = {}
    for metadata in sorted(root.glob("*/*/metadata.json")):
        result[metadata.parent.name] = metadata.parent
    return result


def _fit_past_only(
    rows: Sequence[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
    regularization_c: float,
) -> TrainingResult:
    columns = ensure_no_label_features(feature_columns)
    if len(rows) < 4:
        raise ValueError("insufficient_training_markets")

    y = np.asarray([int(row["label_up"]) for row in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("training_partition_has_single_class")
    x = np.asarray([[float(row[name]) for name in columns] for row in rows], dtype=float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    estimator = LogisticRegression(
        C=float(regularization_c),
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    estimator.fit(x_scaled, y)

    model = LogisticProbabilityModel(
        feature_columns=tuple(columns),
        means=tuple(float(value) for value in scaler.mean_),
        scales=tuple(float(value) if float(value) > 0 else 1.0 for value in scaler.scale_),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
    )
    return TrainingResult(
        model=model,
        summary={
            "split": "walk_forward_past_only_full_training_window",
            "train_markets": len(rows),
            "train_start_ts_ns": int(rows[0].get("received_ts_ns") or 0),
            "train_end_ts_ns": int(rows[-1].get("received_ts_ns") or 0),
            "feature_columns": list(columns),
        },
    )


def _trade_summary(trades: Sequence[ModelPaperTrade | PaperBenchmarkTrade]) -> dict[str, Any]:
    wins = sum(1 for trade in trades if trade.won)
    gross_cost = sum(float(trade.all_in_cost_usd) for trade in trades)
    net_pnl = sum(float(trade.net_pnl_usd) for trade in trades)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        equity += float(trade.net_pnl_usd)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": None if not trades else wins / len(trades),
        "fees_usd": sum(float(trade.fee_usd) for trade in trades),
        "all_in_cost_usd": gross_cost,
        "net_pnl_usd": net_pnl,
        "return_on_cost": None if gross_cost <= 0 else net_pnl / gross_cost,
        "max_drawdown_usd": max_drawdown,
        "mean_pnl_per_trade_usd": None if not trades else net_pnl / len(trades),
    }


def walk_forward_paper_compare(
    dataset_rows: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    cfg: V2Config | None = None,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_seconds_left: float = 120.0,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
    minimum_train_markets: int = 100,
    test_block_markets: int = 50,
    step_markets: int | None = None,
    regularization_c: float = 1.0,
    notional_usd: float = 5.0,
    latency_ms: float = 250.0,
    max_participation_pct: float = 20.0,
    legacy_threshold: float = 0.70,
) -> dict[str, Any]:
    """Expanding-window model-vs-legacy paper replay with realistic execution.

    Each fold trains the probability model on all eligible markets strictly before
    the test block. Test markets are never used to fit the scaler or model. Both
    strategies are evaluated only on the same chronological held-out market set.
    """
    columns = ensure_no_label_features(feature_columns)
    step = int(test_block_markets if step_markets is None else step_markets)
    if minimum_train_markets < 4 or test_block_markets < 1 or step < 1:
        raise ValueError("invalid_walk_forward_window")

    selected = select_market_decision_rows(
        dataset_rows,
        target_seconds_left=target_seconds_left,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
    )
    clean, dropped = eligible_rows(selected, feature_columns=columns)
    clean.sort(key=lambda row: int(row.get("received_ts_ns") or 0))
    required = minimum_train_markets + test_block_markets
    if len(clean) < required:
        raise ValueError(f"insufficient_markets_for_walk_forward_paper:{len(clean)}<{required}")

    configuration = cfg or load_config()
    directories = _market_dirs_by_slug(output_root)
    folds: list[dict[str, Any]] = []
    all_model_trades: list[ModelPaperTrade] = []
    all_legacy_trades: list[PaperBenchmarkTrade] = []
    test_start = int(minimum_train_markets)
    fold_number = 1

    while test_start < len(clean):
        test_end = min(test_start + int(test_block_markets), len(clean))
        test_rows = clean[test_start:test_end]
        if len(test_rows) < int(test_block_markets) and folds:
            break
        train_rows = clean[:test_start]

        try:
            training = _fit_past_only(
                train_rows,
                feature_columns=columns,
                regularization_c=regularization_c,
            )
        except ValueError as exc:
            if str(exc) == "training_partition_has_single_class":
                test_start += step
                continue
            raise

        model_trades: list[ModelPaperTrade] = []
        legacy_trades: list[PaperBenchmarkTrade] = []
        model_skips: Counter[str] = Counter()
        legacy_skips: Counter[str] = Counter()

        for row in test_rows:
            slug = str(row.get("slug") or "")
            directory = directories.get(slug)
            if directory is None:
                model_skips["missing_market_directory"] += 1
                legacy_skips["missing_market_directory"] += 1
                continue

            model_trade, model_reason = benchmark_model_market(
                directory,
                row,
                training=training,
                cfg=configuration,
                notional_usd=notional_usd,
                latency_ms=latency_ms,
                max_participation_pct=max_participation_pct,
            )
            if model_trade is None:
                model_skips[str(model_reason or "unknown")] += 1
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
                legacy_skips[str(legacy_reason or "unknown")] += 1
            else:
                legacy_trades.append(legacy_trade)

        model_summary = _trade_summary(model_trades)
        legacy_summary = _trade_summary(legacy_trades)
        fold = {
            "fold": fold_number,
            "train_markets": len(train_rows),
            "test_markets": len(test_rows),
            "train_start_ts_ns": int(train_rows[0].get("received_ts_ns") or 0),
            "train_end_ts_ns": int(train_rows[-1].get("received_ts_ns") or 0),
            "test_start_ts_ns": int(test_rows[0].get("received_ts_ns") or 0),
            "test_end_ts_ns": int(test_rows[-1].get("received_ts_ns") or 0),
            "model": model_summary,
            "legacy": legacy_summary,
            "model_skips": dict(sorted(model_skips.items())),
            "legacy_skips": dict(sorted(legacy_skips.items())),
            "net_pnl_delta_model_minus_legacy": (
                model_summary["net_pnl_usd"] - legacy_summary["net_pnl_usd"]
            ),
        }
        folds.append(fold)
        all_model_trades.extend(model_trades)
        all_legacy_trades.extend(legacy_trades)
        fold_number += 1
        test_start += step

    if not folds:
        raise ValueError("no_valid_walk_forward_paper_folds")

    model_total = _trade_summary(all_model_trades)
    legacy_total = _trade_summary(all_legacy_trades)
    profitable_model_folds = sum(1 for fold in folds if fold["model"]["net_pnl_usd"] > 0)
    model_beats_legacy_folds = sum(
        1 for fold in folds if fold["net_pnl_delta_model_minus_legacy"] > 0
    )

    return {
        "evaluation": "expanding_window_walk_forward_executable_paper_replay",
        "feature_columns": list(columns),
        "markets_selected": len(selected),
        "markets_eligible": len(clean),
        "markets_dropped_missing_features": dropped,
        "minimum_train_markets": int(minimum_train_markets),
        "test_block_markets": int(test_block_markets),
        "step_markets": step,
        "notional_usd": float(notional_usd),
        "latency_ms": float(latency_ms),
        "max_participation_pct": float(max_participation_pct),
        "legacy_threshold": float(legacy_threshold),
        "fold_count": len(folds),
        "folds": folds,
        "model_total": model_total,
        "legacy_total": legacy_total,
        "net_pnl_delta_model_minus_legacy": (
            model_total["net_pnl_usd"] - legacy_total["net_pnl_usd"]
        ),
        "profitable_model_folds": profitable_model_folds,
        "fraction_profitable_model_folds": profitable_model_folds / len(folds),
        "model_beats_legacy_folds": model_beats_legacy_folds,
        "fraction_model_beats_legacy_folds": model_beats_legacy_folds / len(folds),
        "same_market_cross_split_possible": False,
        "live_trading_approved": False,
        "warning": (
            "Research paper replay only. Positive walk-forward P&L is necessary but not sufficient for live deployment; "
            "forward paper trading and independent-market sample targets still apply."
        ),
    }
