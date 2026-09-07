from __future__ import annotations

import math

from btc5m_v2.research.train_model import (
    MICROSTRUCTURE_FEATURE_COLUMNS,
    fit_logistic_baseline,
    select_market_decision_rows,
)
from btc5m_v2.strategy.model import LogisticProbabilityModel, ensure_no_label_features


def _row(market_index: int, *, seconds_left: float = 120.0, offset_ns: int = 0) -> dict:
    label = 1 if market_index % 2 else 0
    direction = 1.0 if label else -1.0
    probability = 0.58 if label else 0.42
    return {
        "slug": f"btc-updown-5m-{market_index:04d}",
        "received_ts_ns": 1_900_000_000_000_000_000 + market_index * 300_000_000_000 + offset_ns,
        "seconds_left": seconds_left,
        "market_skew_up": probability - 0.5,
        "up_market_probability": probability,
        "up_book_imbalance": 0.25 * direction,
        "down_book_imbalance": -0.20 * direction,
        "up_spread": 0.01 + (market_index % 3) * 0.001,
        "down_spread": 0.011 + (market_index % 2) * 0.001,
        "up_mid_change_5s": 0.01 * direction,
        "up_mid_change_15s": 0.02 * direction,
        "up_mid_change_30s": 0.03 * direction,
        "up_mid_change_60s": 0.04 * direction,
        "resolved": True,
        "winning_side": "UP" if label else "DOWN",
        "label_up": label,
    }


def test_label_columns_are_forbidden_as_model_features():
    assert ensure_no_label_features(["seconds_left", "market_skew_up"]) == (
        "seconds_left",
        "market_skew_up",
    )
    try:
        ensure_no_label_features(["seconds_left", "label_up"])
    except ValueError as exc:
        assert "label leakage" in str(exc)
    else:
        raise AssertionError("label feature must be rejected")


def test_decision_selection_uses_one_snapshot_per_market_nearest_target():
    rows = [
        _row(0, seconds_left=145.0, offset_ns=0),
        _row(0, seconds_left=121.0, offset_ns=1_000_000_000),
        _row(0, seconds_left=100.0, offset_ns=2_000_000_000),
        _row(1, seconds_left=119.0, offset_ns=0),
        _row(1, seconds_left=80.0, offset_ns=1_000_000_000),
    ]
    selected = select_market_decision_rows(rows, target_seconds_left=120.0)
    assert len(selected) == 2
    by_slug = {row["slug"]: row for row in selected}
    assert by_slug["btc-updown-5m-0000"]["seconds_left"] == 121.0
    assert by_slug["btc-updown-5m-0001"]["seconds_left"] == 119.0


def test_logistic_training_is_chronological_market_level_and_exportable(tmp_path):
    rows = [_row(index) for index in range(24)]
    result = fit_logistic_baseline(
        rows,
        feature_columns=MICROSTRUCTURE_FEATURE_COLUMNS,
        minimum_markets=12,
        train_fraction=0.70,
    )

    summary = result.summary
    assert summary["markets_selected"] == 24
    assert summary["markets_eligible"] == 24
    assert summary["split"] == "chronological_market_level"
    assert summary["same_market_cross_split_possible"] is False
    assert summary["train_markets"] + summary["test_markets"] == 24
    assert summary["train_end_ts_ns"] < summary["test_start_ts_ns"]
    assert 0.0 <= summary["brier_score"] <= 1.0
    assert summary["max_export_probability_difference"] <= 1e-10

    prediction = result.model.predict_up(rows[-1])
    assert prediction is not None
    assert 0.0 <= prediction <= 1.0
    assert math.isfinite(prediction)

    path = result.model.save_json(tmp_path / "model.json")
    loaded = LogisticProbabilityModel.load_json(path)
    loaded_prediction = loaded.predict_up(rows[-1])
    assert loaded_prediction is not None
    assert abs(loaded_prediction - prediction) < 1e-12


def test_training_refuses_too_few_independent_markets():
    rows = [_row(index) for index in range(6)]
    try:
        fit_logistic_baseline(
            rows,
            feature_columns=MICROSTRUCTURE_FEATURE_COLUMNS,
            minimum_markets=10,
        )
    except ValueError as exc:
        assert "insufficient_markets_for_training" in str(exc)
    else:
        raise AssertionError("trainer must enforce minimum independent markets")
