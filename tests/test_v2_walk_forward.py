from __future__ import annotations

from btc5m_v2.research.walk_forward import expanding_walk_forward


def _rows(count: int = 12) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        label = index % 2
        rows.append(
            {
                "slug": f"btc-updown-5m-{index}",
                "received_ts_ns": (index + 1) * 1_000_000_000,
                "seconds_left": 120.0,
                "signal_x": 1.0 if label else -1.0,
                "up_market_probability": 0.75 if label else 0.25,
                "label_up": label,
            }
        )
    return rows


def test_expanding_walk_forward_uses_strict_chronological_blocks():
    result = expanding_walk_forward(
        _rows(),
        feature_columns=("signal_x",),
        minimum_train_markets=6,
        test_block_markets=2,
        step_markets=2,
    )

    assert result["fold_count"] == 3
    folds = result["folds"]

    assert folds[0]["train_markets"] == 6
    assert folds[0]["test_markets"] == 2
    assert folds[0]["train_end_ts_ns"] < folds[0]["test_start_ts_ns"]

    assert folds[1]["train_markets"] == 8
    assert folds[1]["train_end_ts_ns"] < folds[1]["test_start_ts_ns"]

    assert folds[2]["train_markets"] == 10
    assert folds[2]["train_end_ts_ns"] < folds[2]["test_start_ts_ns"]
    assert result["same_market_cross_split_possible"] is False


def test_future_market_mutation_cannot_change_first_fold():
    original = _rows()
    baseline = expanding_walk_forward(
        original,
        feature_columns=("signal_x",),
        minimum_train_markets=6,
        test_block_markets=2,
        step_markets=2,
    )

    mutated = _rows()
    # Fold 1 trains on markets 0..5 and tests on 6..7. Market 11 is future
    # information relative to that fold and must not affect its metrics.
    mutated[11]["signal_x"] = 9999.0
    mutated[11]["label_up"] = 1 - int(mutated[11]["label_up"])

    after = expanding_walk_forward(
        mutated,
        feature_columns=("signal_x",),
        minimum_train_markets=6,
        test_block_markets=2,
        step_markets=2,
    )

    first_before = baseline["folds"][0]
    first_after = after["folds"][0]
    assert first_before["brier_score"] == first_after["brier_score"]
    assert first_before["log_loss"] == first_after["log_loss"]
    assert first_before["test_start_ts_ns"] == first_after["test_start_ts_ns"]
    assert first_before["test_end_ts_ns"] == first_after["test_end_ts_ns"]


def test_duplicate_snapshots_still_select_one_market_decision_row():
    rows = _rows()
    duplicate = dict(rows[0])
    duplicate["received_ts_ns"] += 100_000_000
    duplicate["seconds_left"] = 121.0
    rows.append(duplicate)

    result = expanding_walk_forward(
        rows,
        feature_columns=("signal_x",),
        minimum_train_markets=6,
        test_block_markets=2,
        step_markets=2,
    )

    assert result["markets_selected"] == 12
    assert result["markets_eligible"] == 12
