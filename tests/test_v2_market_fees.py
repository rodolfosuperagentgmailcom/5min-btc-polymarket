from __future__ import annotations

import json

from btc5m_v2.market_fees import MarketFeeSchedule, parse_market_fee_schedule
from btc5m_v2.research.fee_enrich import enrich_market_fee_file, read_fee_schedule
from btc5m_v2.research.fees import taker_fee_usdc


def test_parse_clob_market_fee_details() -> None:
    schedule = parse_market_fee_schedule(
        "condition-1",
        {
            "mbf": 0,
            "tbf": 0,
            "fd": {"r": 0.25, "e": 2, "to": True},
        },
    )
    assert schedule.condition_id == "condition-1"
    assert schedule.rate == 0.25
    assert schedule.exponent == 2.0
    assert schedule.taker_only is True
    assert schedule.fees_enabled is True


def test_fee_exponent_changes_the_curve() -> None:
    e1 = taker_fee_usdc(100, 0.50, 0.25, fee_exponent=1)
    e2 = taker_fee_usdc(100, 0.50, 0.25, fee_exponent=2)
    assert round(e1, 8) == 6.25
    assert round(e2, 8) == 1.5625


def test_fee_enrichment_writes_sidecar_and_metadata(tmp_path) -> None:
    market_dir = tmp_path / "2026-09-07" / "btc-updown-5m-test"
    market_dir.mkdir(parents=True)
    metadata_path = market_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"condition_id": "condition-1", "slug": "btc-updown-5m-test"}),
        encoding="utf-8",
    )

    def fake_fetcher(condition_id: str) -> MarketFeeSchedule:
        assert condition_id == "condition-1"
        return MarketFeeSchedule(condition_id, 0.07, 1.0, True, 0, 0)

    result = enrich_market_fee_file(market_dir, fetcher=fake_fetcher)
    assert result["updated"] is True
    stored = read_fee_schedule(market_dir)
    assert stored is not None
    assert stored["fees_enabled"] is True
    assert stored["rate"] == 0.07
    assert stored["exponent"] == 1.0
    assert stored["source"] == "clob_market_info"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["fee_schedule"]["rate"] == 0.07
