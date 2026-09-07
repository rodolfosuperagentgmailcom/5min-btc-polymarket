from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from btc5m_v2.research.quality import validate_recording_quality_rows
from btc5m_v2.research.train_model import (
    DEFAULT_FEATURE_COLUMNS,
    eligible_rows,
    select_market_decision_rows,
)
from btc5m_v2.strategy.model import ensure_no_label_features


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start_ts_ns: int
    train_end_ts_ns: int
    test_start_ts_ns: int
    test_end_ts_ns: int
    train_markets: int
    test_markets: int
    brier_score: float
    market_brier_score: float | None
    brier_improvement_vs_market: float | None
    log_loss: float
    roc_auc: float | None
    test_up_rate: float


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _market_brier(rows: Sequence[dict[str, Any]], labels: np.ndarray) -> float | None:
    probabilities: list[float] = []
    targets: list[int] = []
    for row, label in zip(rows, labels.tolist(), strict=True):
        probability = _finite(row.get("up_market_probability"))
        if probability is None or not 0.0 <= probability <= 1.0:
            continue
        probabilities.append(probability)
        targets.append(int(label))
    if not probabilities:
        return None
    return float(brier_score_loss(targets, probabilities))


def _fit_fold(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
    regularization_c: float,
    fold_number: int,
) -> WalkForwardFold:
    columns = ensure_no_label_features(feature_columns)
    if not train_rows or not test_rows:
        raise ValueError("walk_forward_fold_requires_train_and_test_rows")

    y_train = np.asarray([int(row["label_up"]) for row in train_rows], dtype=int)
    y_test = np.asarray([int(row["label_up"]) for row in test_rows], dtype=int)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("training_partition_has_single_class")

    x_train = np.asarray(
        [[float(row[name]) for name in columns] for row in train_rows],
        dtype=float,
    )
    x_test = np.asarray(
        [[float(row[name]) for name in columns] for row in test_rows],
        dtype=float,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    estimator = LogisticRegression(
        C=float(regularization_c),
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    estimator.fit(x_train_scaled, y_train)
    probabilities = estimator.predict_proba(x_test_scaled)[:, 1]

    model_brier = float(brier_score_loss(y_test, probabilities))
    market_brier = _market_brier(test_rows, y_test)
    auc = None
    if len(set(y_test.tolist())) > 1:
        auc = float(roc_auc_score(y_test, probabilities))

    return WalkForwardFold(
        fold=int(fold_number),
        train_start_ts_ns=int(train_rows[0].get("received_ts_ns") or 0),
        train_end_ts_ns=int(train_rows[-1].get("received_ts_ns") or 0),
        test_start_ts_ns=int(test_rows[0].get("received_ts_ns") or 0),
        test_end_ts_ns=int(test_rows[-1].get("received_ts_ns") or 0),
        train_markets=len(train_rows),
        test_markets=len(test_rows),
        brier_score=model_brier,
        market_brier_score=market_brier,
        brier_improvement_vs_market=(
            None if market_brier is None else market_brier - model_brier
        ),
        log_loss=float(log_loss(y_test, probabilities, labels=[0, 1])),
        roc_auc=auc,
        test_up_rate=float(np.mean(y_test)),
    )


def expanding_walk_forward(
    rows: Iterable[dict[str, Any]],
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_seconds_left: float = 120.0,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
    minimum_train_markets: int = 100,
    test_block_markets: int = 50,
    step_markets: int | None = None,
    regularization_c: float = 1.0,
) -> dict[str, Any]:
    """Expanding-window chronological validation at the market level.

    Each market contributes at most one decision row. For every fold, model and
    scaler are fit only on markets strictly earlier than that fold's test block.
    """

    columns = ensure_no_label_features(feature_columns)
    if minimum_train_markets < 4:
        raise ValueError("minimum_train_markets must be at least 4")
    if test_block_markets < 1:
        raise ValueError("test_block_markets must be positive")
    step = test_block_markets if step_markets is None else int(step_markets)
    if step < 1:
        raise ValueError("step_markets must be positive")
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")

    input_rows = [dict(row) for row in rows]
    validate_recording_quality_rows(input_rows)
    selected = select_market_decision_rows(
        input_rows,
        target_seconds_left=target_seconds_left,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
    )
    clean, dropped = eligible_rows(selected, feature_columns=columns)
    clean.sort(key=lambda row: int(row.get("received_ts_ns") or 0))

    required = minimum_train_markets + test_block_markets
    if len(clean) < required:
        raise ValueError(f"insufficient_markets_for_walk_forward:{len(clean)}<{required}")

    folds: list[WalkForwardFold] = []
    skipped_single_class = 0
    test_start = minimum_train_markets
    fold_number = 1
    while test_start < len(clean):
        test_end = min(test_start + test_block_markets, len(clean))
        if test_end <= test_start:
            break
        train_rows = clean[:test_start]
        test_rows = clean[test_start:test_end]
        if len(test_rows) < test_block_markets and folds:
            break
        try:
            fold = _fit_fold(
                train_rows,
                test_rows,
                feature_columns=columns,
                regularization_c=regularization_c,
                fold_number=fold_number,
            )
        except ValueError as exc:
            if str(exc) != "training_partition_has_single_class":
                raise
            skipped_single_class += 1
        else:
            folds.append(fold)
            fold_number += 1
        test_start += step

    if not folds:
        raise ValueError("no_valid_walk_forward_folds")

    total_test = sum(fold.test_markets for fold in folds)
    weighted_brier = sum(fold.brier_score * fold.test_markets for fold in folds) / total_test
    weighted_log_loss = sum(fold.log_loss * fold.test_markets for fold in folds) / total_test

    market_weight = sum(
        fold.test_markets for fold in folds if fold.market_brier_score is not None
    )
    weighted_market_brier = None
    if market_weight:
        weighted_market_brier = sum(
            float(fold.market_brier_score) * fold.test_markets
            for fold in folds
            if fold.market_brier_score is not None
        ) / market_weight

    positive_improvement_folds = sum(
        1
        for fold in folds
        if fold.brier_improvement_vs_market is not None
        and fold.brier_improvement_vs_market > 0
    )

    return {
        "evaluation": "expanding_window_walk_forward_market_level",
        "feature_columns": list(columns),
        "recording_quality": "reconnects_required_zero",
        "markets_selected": len(selected),
        "markets_eligible": len(clean),
        "markets_dropped_missing_features": dropped,
        "minimum_train_markets": int(minimum_train_markets),
        "test_block_markets": int(test_block_markets),
        "step_markets": int(step),
        "folds": [fold.__dict__.copy() for fold in folds],
        "fold_count": len(folds),
        "skipped_single_class_folds": skipped_single_class,
        "total_test_market_observations": total_test,
        "weighted_brier_score": weighted_brier,
        "weighted_log_loss": weighted_log_loss,
        "weighted_market_brier_score": weighted_market_brier,
        "weighted_brier_improvement_vs_market": (
            None
            if weighted_market_brier is None
            else weighted_market_brier - weighted_brier
        ),
        "folds_beating_market_brier": positive_improvement_folds,
        "fraction_folds_beating_market_brier": positive_improvement_folds / len(folds),
        "same_market_cross_split_possible": False,
        "warning": (
            "Research validation only. Walk-forward predictive improvement does not establish "
            "tradable edge until executable fee/slippage paper replay is also positive."
        ),
    }
