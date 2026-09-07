from __future__ import annotations

from btc5m_v2.research.features import build_causal_features


def test_empty_input_returns_empty_list() -> None:
    assert build_causal_features([]) == []
