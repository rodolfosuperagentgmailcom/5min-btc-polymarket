from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from btc5m_v2.research.economics import breakeven_probability


class ProbabilityModel(Protocol):
    name: str

    def probability_up(self, row: dict[str, Any]) -> float | None:
        ...


@dataclass(frozen=True)
class MarketImpliedBaseline:
    """No-alpha baseline using normalized contemporaneous market midpoints."""

    name: str = "market_implied_midpoint"

    def probability_up(self, row: dict[str, Any]) -> float | None:
        value = row.get("up_market_probability")
        if value is None:
            up_mid = row.get("up_mid")
            down_mid = row.get("down_mid")
            try:
                up = float(up_mid)
                down = float(down_mid)
            except (TypeError, ValueError):
                return None
            total = up + down
            if total <= 0:
                return None
            value = up / total
        try:
            q = float(value)
        except (TypeError, ValueError):
            return None
        return q if 0.0 <= q <= 1.0 else None


@dataclass(frozen=True)
class EdgeDecision:
    trade: bool
    side: str | None
    model_probability: float | None
    entry_price: float | None
    breakeven_probability: float | None
    edge: float | None
    reason: str
    model_name: str


def _side_probability(q_up: float, side: str) -> float:
    return q_up if side == "UP" else 1.0 - q_up


def best_edge_decision(
    row: dict[str, Any],
    *,
    model: ProbabilityModel,
    fee_rate: float,
    fee_enabled: bool,
    fee_exponent: float = 1.0,
    minimum_model_probability: float = 0.58,
    minimum_edge: float = 0.04,
    entry_slippage_abs: float = 0.0,
) -> EdgeDecision:
    """Choose UP/DOWN only when modeled probability clears all-in break-even.

    This is a pure research/paper decision function. It performs no order I/O.
    Fee/slippage inputs are explicit so historical runs do not silently assume
    today's market economics.
    """

    q_up = model.probability_up(row)
    if q_up is None:
        return EdgeDecision(False, None, None, None, None, None, "missing_probability", model.name)

    candidates: list[tuple[float, str, float, float, float]] = []
    slip = max(0.0, float(entry_slippage_abs))
    for side, ask_key in (("UP", "up_ask"), ("DOWN", "down_ask")):
        try:
            ask = float(row.get(ask_key))
        except (TypeError, ValueError):
            continue
        if ask <= 0.0 or ask > 1.0:
            continue
        entry = min(1.0, ask + slip)
        q_side = _side_probability(q_up, side)
        break_even = breakeven_probability(
            entry,
            fee_rate=fee_rate,
            enabled=fee_enabled,
            fee_exponent=fee_exponent,
        )
        edge = q_side - break_even
        candidates.append((edge, side, q_side, entry, break_even))

    if not candidates:
        return EdgeDecision(False, None, q_up, None, None, None, "missing_executable_ask", model.name)

    edge, side, q_side, entry, break_even = max(candidates, key=lambda item: item[0])
    if q_side < float(minimum_model_probability):
        return EdgeDecision(False, side, q_side, entry, break_even, edge, "probability_below_minimum", model.name)
    if edge < float(minimum_edge):
        return EdgeDecision(False, side, q_side, entry, break_even, edge, "edge_below_minimum", model.name)

    return EdgeDecision(True, side, q_side, entry, break_even, edge, "positive_edge", model.name)
