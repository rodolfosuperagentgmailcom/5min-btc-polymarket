from __future__ import annotations

import json

from btc5m_v2.research.paper_benchmark import benchmark_market_paper


def test_paper_benchmark_rejects_reconnected_recording(tmp_path):
    market_dir = tmp_path / "2030-01-01" / "btc-updown-5m-reconnected-paper"
    market_dir.mkdir(parents=True)
    (market_dir / "metadata.json").write_text(
        json.dumps({"reconnects": 2}),
        encoding="utf-8",
    )

    trade, reason = benchmark_market_paper(market_dir)
    assert trade is None
    assert reason == "recording_reconnected:2"


def test_paper_benchmark_rejects_missing_recording_quality(tmp_path):
    market_dir = tmp_path / "2030-01-01" / "btc-updown-5m-unknown-paper"
    market_dir.mkdir(parents=True)
    (market_dir / "metadata.json").write_text("{}", encoding="utf-8")

    trade, reason = benchmark_market_paper(market_dir)
    assert trade is None
    assert reason == "missing_recording_quality"
