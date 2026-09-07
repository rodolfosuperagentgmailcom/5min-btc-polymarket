from __future__ import annotations

import pytest

from btc5m_v2.research.alignment import aligned_start_ts, next_market_start_ts


def test_next_market_start_uses_strict_future_five_minute_boundary():
    assert next_market_start_ts(1_000.0) == 1_200.0
    assert next_market_start_ts(1_200.0) == 1_500.0


def test_aligned_start_applies_small_positive_rollover_offset():
    assert aligned_start_ts(1_000.0, offset_sec=0.5) == 1_200.5


def test_alignment_rejects_invalid_interval_and_negative_offset():
    with pytest.raises(ValueError, match="interval_sec"):
        next_market_start_ts(1_000.0, interval_sec=0.0)
    with pytest.raises(ValueError, match="offset_sec"):
        aligned_start_ts(1_000.0, offset_sec=-0.1)
