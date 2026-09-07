from __future__ import annotations

from btc5m_v2.research.fair_value import MarketImpliedBaseline, best_edge_decision


def test_market_implied_baseline_normalizes_midpoints():
    model = MarketImpliedBaseline()
    q = model.probability_up({"up_mid": 0.60, "down_mid": 0.40})
    assert q == 0.60


def test_market_implied_baseline_does_not_create_fake_edge_against_same_market():
    model = MarketImpliedBaseline()
    decision = best_edge_decision(
        {"up_mid": 0.60, "down_mid": 0.40, "up_ask": 0.61, "down_ask": 0.41},
        model=model,
        fee_rate=0.07,
        fee_enabled=True,
        minimum_model_probability=0.50,
        minimum_edge=0.0,
    )
    assert decision.trade is False
    assert decision.reason == "edge_below_minimum"
    assert decision.edge is not None
    assert decision.edge < 0


class FixedModel:
    name = "fixed_test_model"

    def __init__(self, q_up: float):
        self.q_up = q_up

    def probability_up(self, row):
        return self.q_up


def test_positive_edge_model_selects_up():
    decision = best_edge_decision(
        {"up_ask": 0.70, "down_ask": 0.31},
        model=FixedModel(0.80),
        fee_rate=0.0,
        fee_enabled=False,
        minimum_model_probability=0.58,
        minimum_edge=0.04,
    )
    assert decision.trade is True
    assert decision.side == "UP"
    assert round(decision.edge or 0.0, 6) == 0.10


def test_fee_can_remove_apparent_edge():
    no_fee = best_edge_decision(
        {"up_ask": 0.70, "down_ask": 0.31},
        model=FixedModel(0.73),
        fee_rate=0.0,
        fee_enabled=False,
        minimum_model_probability=0.58,
        minimum_edge=0.02,
    )
    with_fee = best_edge_decision(
        {"up_ask": 0.70, "down_ask": 0.31},
        model=FixedModel(0.73),
        fee_rate=0.07,
        fee_enabled=True,
        minimum_model_probability=0.58,
        minimum_edge=0.02,
    )
    assert no_fee.trade is True
    assert with_fee.trade is False
    assert with_fee.edge is not None
    assert no_fee.edge is not None
    assert with_fee.edge < no_fee.edge


def test_missing_probability_fails_closed():
    decision = best_edge_decision(
        {"up_ask": 0.70, "down_ask": 0.30},
        model=MarketImpliedBaseline(),
        fee_rate=0.0,
        fee_enabled=False,
    )
    assert decision.trade is False
    assert decision.reason == "missing_probability"
