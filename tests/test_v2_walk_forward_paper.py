from __future__ import annotations

import json

from btc5m_v2.config import V2Config
from btc5m_v2.research import walk_forward_paper as wfp


def _cfg() -> V2Config:
    return V2Config(
        {
            "resolution": {
                "expected_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
                "fail_if_missing": True,
                "fail_if_unexpected": True,
            },
            "entry": {"seconds_left_min": 90, "seconds_left_max": 150},
            "market_quality": {
                "max_selected_side_spread": 0.02,
                "min_top_3_ask_notional_usd": 30,
            },
            "freshness": {"clob_max_age_sec": 2, "max_consecutive_data_errors": 3},
        }
    )


def _rows(count: int = 12) -> list[dict]:
    rows = []
    for index in range(count):
        label = index % 2
        rows.append(
            {
                "slug": f"btc-updown-5m-{index}",
                "received_ts_ns": (index + 1) * 1_000_000_000,
                "seconds_left": 120.0,
                "recording_reconnects": 0,
                "signal_x": 1.0 if label else -1.0,
                "label_up": label,
            }
        )
    return rows


def test_walk_forward_paper_uses_past_only_training_windows(tmp_path, monkeypatch):
    for row in _rows():
        market_dir = tmp_path / "2026-09-07" / row["slug"]
        market_dir.mkdir(parents=True)
        (market_dir / "metadata.json").write_text(
            json.dumps({"slug": row["slug"], "reconnects": 0}), encoding="utf-8"
        )

    seen_train_counts: list[int] = []
    seen_train_end: list[int] = []

    def fake_model(directory, feature_row, *, training, **kwargs):
        seen_train_counts.append(int(training.summary["train_markets"]))
        seen_train_end.append(int(training.summary["train_end_ts_ns"]))
        assert int(training.summary["train_end_ts_ns"]) < int(feature_row["received_ts_ns"])
        return None, "mock_no_trade"

    def fake_legacy(directory, **kwargs):
        return None, "mock_no_trade"

    monkeypatch.setattr(wfp, "benchmark_model_market", fake_model)
    monkeypatch.setattr(wfp, "benchmark_market_paper", fake_legacy)

    result = wfp.walk_forward_paper_compare(
        _rows(),
        tmp_path,
        cfg=_cfg(),
        feature_columns=("signal_x",),
        minimum_train_markets=6,
        test_block_markets=2,
        step_markets=2,
    )

    assert result["fold_count"] == 3
    assert result["same_market_cross_split_possible"] is False
    assert result["live_trading_approved"] is False
    assert seen_train_counts == [6, 6, 8, 8, 10, 10]
    assert seen_train_end == [6_000_000_000, 6_000_000_000, 8_000_000_000, 8_000_000_000, 10_000_000_000, 10_000_000_000]


def test_walk_forward_paper_rejects_too_few_independent_markets(tmp_path):
    try:
        wfp.walk_forward_paper_compare(
            _rows(7),
            tmp_path,
            cfg=_cfg(),
            feature_columns=("signal_x",),
            minimum_train_markets=6,
            test_block_markets=2,
        )
    except ValueError as exc:
        assert "insufficient_markets_for_walk_forward_paper" in str(exc)
    else:
        raise AssertionError("insufficient independent markets must fail closed")
