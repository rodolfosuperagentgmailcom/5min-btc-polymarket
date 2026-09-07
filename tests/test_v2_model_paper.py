from __future__ import annotations

from pathlib import Path

from btc5m_v2.research import model_paper
from btc5m_v2.research.train_model import MICROSTRUCTURE_FEATURE_COLUMNS, TrainingResult
from btc5m_v2.strategy.model import LogisticProbabilityModel


def _row(index: int) -> dict:
    label = index % 2
    direction = 1.0 if label else -1.0
    probability = 0.58 if label else 0.42
    return {
        "slug": f"btc-updown-5m-{index:04d}",
        "received_ts_ns": 1_900_000_000_000_000_000 + index * 300_000_000_000,
        "seconds_left": 120.0,
        "market_skew_up": probability - 0.5,
        "up_market_probability": probability,
        "up_book_imbalance": 0.20 * direction,
        "down_book_imbalance": -0.20 * direction,
        "up_spread": 0.01,
        "down_spread": 0.01,
        "up_mid_change_5s": 0.01 * direction,
        "up_mid_change_15s": 0.02 * direction,
        "up_mid_change_30s": 0.03 * direction,
        "up_mid_change_60s": 0.04 * direction,
        "label_up": label,
    }


def _dummy_training() -> TrainingResult:
    model = LogisticProbabilityModel(
        feature_columns=("market_skew_up",),
        means=(0.0,),
        scales=(1.0,),
        coefficients=(1.0,),
        intercept=0.0,
        model_version="dummy",
    )
    return TrainingResult(model=model, summary={"dummy": True})


def test_chronological_test_rows_are_only_later_independent_markets():
    rows = [_row(index) for index in range(10)]
    held_out = model_paper.chronological_test_rows(
        rows,
        feature_columns=MICROSTRUCTURE_FEATURE_COLUMNS,
        train_fraction=0.60,
    )
    assert [row["slug"] for row in held_out] == [
        "btc-updown-5m-0006",
        "btc-updown-5m-0007",
        "btc-updown-5m-0008",
        "btc-updown-5m-0009",
    ]


def test_snapshot_match_requires_timestamp_and_book_state():
    feature = {
        "received_ts_ns": 123,
        "up_ask": 0.61,
        "down_ask": 0.41,
        "up_spread": 0.01,
        "down_spread": 0.01,
        "up_ask_depth_3_usd": 100.0,
        "down_ask_depth_3_usd": 90.0,
    }
    assert model_paper._same_snapshot(dict(feature), feature) is True
    changed = dict(feature)
    changed["up_ask_depth_3_usd"] = 99.0
    assert model_paper._same_snapshot(changed, feature) is False


def test_comparison_calls_both_strategies_only_on_same_held_out_market_set(monkeypatch):
    rows = [_row(index) for index in range(10)]
    held_out_slugs = [f"btc-updown-5m-{index:04d}" for index in range(6, 10)]
    directories = {slug: Path("/tmp") / slug for slug in held_out_slugs}
    model_calls: list[str] = []
    legacy_calls: list[str] = []

    monkeypatch.setattr(model_paper, "fit_logistic_baseline", lambda *args, **kwargs: _dummy_training())
    monkeypatch.setattr(model_paper, "_market_dirs_by_slug", lambda output_root: directories)

    def fake_model(directory, feature_row, **kwargs):
        model_calls.append(Path(directory).name)
        return None, "model_no_trade"

    def fake_legacy(directory, **kwargs):
        legacy_calls.append(Path(directory).name)
        return None, "no_signal"

    monkeypatch.setattr(model_paper, "benchmark_model_market", fake_model)
    monkeypatch.setattr(model_paper, "benchmark_market_paper", fake_legacy)

    result = model_paper.compare_model_vs_legacy_out_of_sample(
        rows,
        "/unused",
        feature_columns=MICROSTRUCTURE_FEATURE_COLUMNS,
        train_fraction=0.60,
        minimum_markets=4,
    )

    assert result["held_out_markets"] == 4
    assert model_calls == held_out_slugs
    assert legacy_calls == held_out_slugs
    assert result["model_paper"]["executed_trades"] == 0
    assert result["legacy_paper"]["executed_trades"] == 0
