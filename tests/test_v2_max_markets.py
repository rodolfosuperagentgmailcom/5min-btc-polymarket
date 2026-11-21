from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from btc5m_v2.research import btc_recorder as btc_module
from btc5m_v2.research import recorder as clob_module


def fake_market(slug: str):
    return SimpleNamespace(
        slug=slug,
        up_token_id="up",
        down_token_id="down",
        end_ts=time.time() + 300,
    )


def test_clob_run_recorder_honors_max_markets(monkeypatch, tmp_path):
    calls = {"discover": 0, "record": 0}

    def discover(_cfg):
        calls["discover"] += 1
        return fake_market(f"market-{calls['discover']}")

    async def record(market, **_kwargs):
        calls["record"] += 1
        return SimpleNamespace(market=market)

    monkeypatch.setattr(clob_module, "discover_current_market", discover)
    monkeypatch.setattr(clob_module, "record_one_market", record)

    rows = asyncio.run(
        clob_module.run_recorder(
            duration_sec=30,
            config=object(),
            output_root=tmp_path,
            max_markets=1,
        )
    )

    assert len(rows) == 1
    assert calls["record"] == 1
    assert calls["discover"] == 1


def test_btc_run_recorder_honors_max_markets(monkeypatch, tmp_path):
    calls = {"discover": 0, "record": 0}

    def discover(_cfg):
        calls["discover"] += 1
        return fake_market(f"market-{calls['discover']}")

    async def record(market, **_kwargs):
        calls["record"] += 1
        return SimpleNamespace(market=market)

    monkeypatch.setattr(btc_module, "discover_current_market", discover)
    monkeypatch.setattr(btc_module, "record_btc_one_market", record)

    rows = asyncio.run(
        btc_module.run_btc_recorder(
            duration_sec=30,
            config=object(),
            output_root=tmp_path,
            max_markets=1,
        )
    )

    assert len(rows) == 1
    assert calls["record"] == 1
    assert calls["discover"] == 1
