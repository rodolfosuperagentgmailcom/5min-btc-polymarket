from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from btc5m_v2.execution.pricing import BuyFillEstimate, estimate_buy_fill
from btc5m_v2.fees import all_in_cost_per_share, fee_for_fills


@dataclass(frozen=True)
class SideEdge:
    side: str
    model_probability: float
    fill: BuyFillEstimate
    fee_usd: float
    all_in_cost_per_share: float | None
    breakeven_probability: float | None
    edge_per_share: float | None
    expected_value_usd: float | None
    is_taker: bool
    reasons: tuple[str, ...]

    @property
    def tradable(self) -> bool:
        return self.fill.fillable and self.edge_per_share is not None and not self.reasons


@dataclass(frozen=True)
class EdgeDecision:
    action: str
    side: str | None
    selected: SideEdge | None
    reasons: tuple[str, ...]


def evaluate_buy_edge(
    *,
    side: str,
    model_probability: float,
    asks: Iterable[tuple[float, float]],
    notional_usd: float,
    fees_enabled: bool,
    fee_rate: float,
    fee_exponent: float = 1.0,
    is_taker: bool = True,
    max_book_participation_pct: float | None = None,
) -> SideEdge:
    label = str(side).strip().upper()
    if label not in {"UP", "DOWN"}:
        raise ValueError("side must be UP or DOWN")

    probability = float(model_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("model_probability must be in [0, 1]")

    fill = estimate_buy_fill(asks, notional_usd=notional_usd)
    reasons: list[str] = []
    if not fill.fillable:
        reasons.append("insufficient_visible_ask_depth")

    if (
        max_book_participation_pct is not None
        and fill.participation_pct is not None
        and fill.participation_pct > float(max_book_participation_pct)
    ):
        reasons.append("book_participation_too_high")

    fee = fee_for_fills(
        fill.fills,
        fee_rate=fee_rate,
        fee_exponent=fee_exponent,
        fees_enabled=fees_enabled,
        is_taker=is_taker,
    )
    cost_per_share = all_in_cost_per_share(
        spent_usd=fill.spent_usd,
        shares=fill.shares,
        fee_usd=fee,
    )
    edge_per_share = None if cost_per_share is None else probability - cost_per_share
    expected_value = None
    if fill.shares > 0.0:
        expected_value = probability * fill.shares - (fill.spent_usd + fee)

    return SideEdge(
        side=label,
        model_probability=probability,
        fill=fill,
        fee_usd=fee,
        all_in_cost_per_share=cost_per_share,
        breakeven_probability=cost_per_share,
        edge_per_share=edge_per_share,
        expected_value_usd=expected_value,
        is_taker=is_taker,
        reasons=tuple(reasons),
    )


def choose_best_edge(
    up: SideEdge,
    down: SideEdge,
    *,
    minimum_model_probability: float,
    minimum_edge_per_share: float,
) -> EdgeDecision:
    min_probability = float(minimum_model_probability)
    min_edge = float(minimum_edge_per_share)
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("minimum_model_probability must be in [0, 1]")
    if min_edge < 0.0:
        raise ValueError("minimum_edge_per_share cannot be negative")

    candidates: list[SideEdge] = []
    rejection_reasons: list[str] = []

    for opportunity in (up, down):
        prefix = opportunity.side.lower()
        if opportunity.reasons:
            rejection_reasons.extend(f"{prefix}:{reason}" for reason in opportunity.reasons)
            continue
        if not opportunity.fill.fillable:
            rejection_reasons.append(f"{prefix}:unfillable")
            continue
        if opportunity.model_probability < min_probability:
            rejection_reasons.append(f"{prefix}:probability_below_minimum")
            continue
        if opportunity.edge_per_share is None or opportunity.edge_per_share < min_edge:
            rejection_reasons.append(f"{prefix}:edge_below_minimum")
            continue
        if opportunity.expected_value_usd is None or opportunity.expected_value_usd <= 0.0:
            rejection_reasons.append(f"{prefix}:non_positive_expected_value")
            continue
        candidates.append(opportunity)

    if not candidates:
        return EdgeDecision(
            action="NO_TRADE",
            side=None,
            selected=None,
            reasons=tuple(rejection_reasons) or ("no_positive_edge",),
        )

    selected = max(
        candidates,
        key=lambda item: (
            float(item.edge_per_share or 0.0),
            float(item.expected_value_usd or 0.0),
            item.model_probability,
            item.side == "UP",
        ),
    )
    return EdgeDecision(action="BUY", side=selected.side, selected=selected, reasons=())
