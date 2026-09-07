from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from btc5m_v2.strategy.model import LogisticProbabilityModel, ensure_no_label_features

MICROSTRUCTURE_FEATURE_COLUMNS = (
    "seconds_left",
    "market_skew_up",
    "up_book_imbalance",
    "down_book_imbalance",
    "up_spread",
    "down_spread",
    "up_mid_change_5s",
    "up_mid_change_15s",
    "up_mid_change_30s",
    "up_mid_change_60s",
)

BTC_FEATURE_COLUMNS = (
    "btc_delta_usd",
    "btc_momentum_15s",
    "btc_momentum_30s",
    "btc_momentum_60s",
    "btc_realized_vol_30s",
    "btc_realized_vol_60s",
    "btc_impulse_z",
)

DEFAULT_FEATURE_COLUMNS = MICROSTRUCTURE_FEATURE_COLUMNS + BTC_FEATURE_COLUMNS


@dataclass(frozen=True)
class TrainingResult:
    model: LogisticProbabilityModel
    summary: dict[str, Any]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_recording_quality_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("recording_reconnects") is None:
            raise ValueError("dataset_missing_recording_quality")
        try:
            reconnects = int(row["recording_reconnects"])
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset_invalid_recording_quality") from exc
        if reconnects != 0:
            raise ValueError("dataset_contains_reconnected_recording")


def select_market_decision_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_seconds_left: float = 120.0,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
) -> list[dict[str, Any]]:
    """Select one causal decision snapshot per independent 5-minute market."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        slug = str(row.get("slug") or "").strip()
        seconds_left = _finite(row.get("seconds_left"))
        label = row.get("label_up")
        if not slug or seconds_left is None or label not in {0, 1, False, True}:
            continue
        if seconds_left < float(seconds_left_min) or seconds_left > float(seconds_left_max):
            continue
        grouped.setdefault(slug, []).append(row)

    selected: list[dict[str, Any]] = []
    target = float(target_seconds_left)
    for slug, candidates in grouped.items():
        chosen = min(
            candidates,
            key=lambda row: (
                abs(float(row["seconds_left"]) - target),
                int(row.get("received_ts_ns") or 0),
            ),
        )
        selected.append(chosen)

    selected.sort(key=lambda row: int(row.get("received_ts_ns") or 0))
    return selected


def eligible_rows(
    rows: Iterable[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    columns = ensure_no_label_features(feature_columns)
    eligible: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if row.get("label_up") not in {0, 1, False, True}:
            dropped += 1
            continue
        values = [_finite(row.get(name)) for name in columns]
        if any(value is None for value in values):
            dropped += 1
            continue
        eligible.append(dict(row))
    return eligible, dropped


def calibration_bins(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bin_count: int = 5,
) -> list[dict[str, Any]]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count
        members = [
            (float(probability), int(label))
            for probability, label in zip(probabilities, labels, strict=True)
            if (low <= probability < high) or (index == bin_count - 1 and probability == 1.0)
        ]
        if not members:
            continue
        bins.append(
            {
                "low": low,
                "high": high,
                "count": len(members),
                "mean_predicted": sum(value[0] for value in members) / len(members),
                "observed_up_rate": sum(value[1] for value in members) / len(members),
            }
        )
    return bins


def _market_baseline_brier(rows: Sequence[dict[str, Any]], labels: Sequence[int]) -> float | None:
    probabilities: list[float] = []
    targets: list[int] = []
    for row, label in zip(rows, labels, strict=True):
        probability = _finite(row.get("up_market_probability"))
        if probability is None or not 0.0 <= probability <= 1.0:
            continue
        probabilities.append(probability)
        targets.append(int(label))
    if not probabilities:
        return None
    return float(brier_score_loss(targets, probabilities))


def fit_logistic_baseline(
    rows: Iterable[dict[str, Any]],
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_seconds_left: float = 120.0,
    seconds_left_min: float = 90.0,
    seconds_left_max: float = 150.0,
    train_fraction: float = 0.70,
    minimum_markets: int = 100,
    regularization_c: float = 1.0,
) -> TrainingResult:
    columns = ensure_no_label_features(feature_columns)
    if not 0.5 <= float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if int(minimum_markets) < 4:
        raise ValueError("minimum_markets must be at least 4")
    if float(regularization_c) <= 0:
        raise ValueError("regularization_c must be positive")

    input_rows = [dict(row) for row in rows]
    _validate_recording_quality_rows(input_rows)
    decision_rows = select_market_decision_rows(
        input_rows,
        target_seconds_left=target_seconds_left,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
    )
    clean_rows, dropped = eligible_rows(decision_rows, feature_columns=columns)
    if len(clean_rows) < int(minimum_markets):
        raise ValueError(
            f"insufficient_markets_for_training:{len(clean_rows)}<{int(minimum_markets)}"
        )

    split_index = int(len(clean_rows) * float(train_fraction))
    split_index = max(2, min(split_index, len(clean_rows) - 2))
    train_rows = clean_rows[:split_index]
    test_rows = clean_rows[split_index:]

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
    predictions = (probabilities >= 0.5).astype(int)

    model = LogisticProbabilityModel(
        feature_columns=tuple(columns),
        means=tuple(float(value) for value in scaler.mean_),
        scales=tuple(float(value) if float(value) > 0 else 1.0 for value in scaler.scale_),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
    )

    exported_probabilities = np.asarray(
        [float(model.predict_up(row)) for row in test_rows],
        dtype=float,
    )
    max_export_difference = float(np.max(np.abs(exported_probabilities - probabilities)))
    if max_export_difference > 1e-10:
        raise RuntimeError("exported_model_probability_mismatch")

    model_brier = float(brier_score_loss(y_test, probabilities))
    market_brier = _market_baseline_brier(test_rows, y_test.tolist())
    auc = None
    if len(set(y_test.tolist())) > 1:
        auc = float(roc_auc_score(y_test, probabilities))

    summary: dict[str, Any] = {
        "model_version": model.model_version,
        "feature_columns": list(columns),
        "decision_target_seconds_left": float(target_seconds_left),
        "decision_window_seconds_left": [float(seconds_left_min), float(seconds_left_max)],
        "recording_quality": "reconnects_required_zero",
        "markets_selected": len(decision_rows),
        "markets_eligible": len(clean_rows),
        "markets_dropped_missing_features": dropped,
        "train_markets": len(train_rows),
        "test_markets": len(test_rows),
        "train_start_ts_ns": int(train_rows[0].get("received_ts_ns") or 0),
        "train_end_ts_ns": int(train_rows[-1].get("received_ts_ns") or 0),
        "test_start_ts_ns": int(test_rows[0].get("received_ts_ns") or 0),
        "test_end_ts_ns": int(test_rows[-1].get("received_ts_ns") or 0),
        "split": "chronological_market_level",
        "same_market_cross_split_possible": False,
        "brier_score": model_brier,
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": auc,
        "market_implied_brier_score": market_brier,
        "brier_improvement_vs_market": None if market_brier is None else market_brier - model_brier,
        "positive_brier_improvement_vs_market": None if market_brier is None else model_brier < market_brier,
        "test_up_rate": float(np.mean(y_test)),
        "calibration_bins": calibration_bins(probabilities.tolist(), y_test.tolist()),
        "regularization_c": float(regularization_c),
        "max_export_probability_difference": max_export_difference,
        "warning": "Research baseline only. Positive in-sample or single-split metrics do not establish tradable edge.",
    }
    return TrainingResult(model=model, summary=summary)


def load_dataset_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    _validate_recording_quality_rows(rows)
    return rows


def train_from_parquet(
    dataset_path: str | Path,
    **kwargs: Any,
) -> TrainingResult:
    return fit_logistic_baseline(load_dataset_rows(dataset_path), **kwargs)


def save_training_result(
    result: TrainingResult,
    *,
    model_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path]:
    model_destination = result.model.save_json(model_path)
    summary_destination = (
        Path(summary_path)
        if summary_path is not None
        else model_destination.with_suffix(".metrics.json")
    )
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    return model_destination, summary_destination
