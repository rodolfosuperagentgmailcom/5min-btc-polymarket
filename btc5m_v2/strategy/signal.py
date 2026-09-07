from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from btc5m_v2.config import V2Config
from btc5m_v2.research.economics import breakeven_probability
from btc5m_v2.safety import BookMetrics, evaluate_market_gate


@dataclass(frozen=True)
class SideEvaluation:
    side: str
    model_probability: float
    entry_price: float | None
    breakeven_probability: float | None
    edge: float | None
    gate_ok: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SignalDecision:
    trade: bool
    side: str | None
    model_probability: float | None
    entry_price: float | None
    breakeven_probability: float | None
    edge: float | None
    reasons: tuple[str, ...]
    up: SideEvaluation
    down: SideEvaluation


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minimum_model_probability(cfg: V2Config) -> float:
    return float(cfg.raw["signal"]["minimum_model_probability"])


def _minimum_edge(cfg: V2Config) -> float:
    return float(cfg.raw["signal"]["minimum_edge_after_fees_and_slippage"])


def _book_for_side(row: dict[str, Any], side: str) -> BookMetrics:
    prefix = "up" if side == "UP" else "down"
    age_ms = _f(row.get(f"{prefix}_exchange_age_ms"))
    return BookMetrics(
        best_bid=_f(row.get(f"{prefix}_bid")),
        best_ask=_f(row.get(f"{prefix}_ask")),
        spread=_f(row.get(f"{prefix}_spread")),
        top3_ask_notional_usd=_f(row.get(f"{prefix}_ask_depth_3_usd")) or 0.0,
        quote_age_sec=None if age_ms is None else age_ms / 1000.0,
    )


def _evaluate_side(
    row: dict[str, Any],
    *,
    side: str,
    model_probability: float,
    cfg: V2Config,
    resolution_source: str | None,
    fees_enabled: bool,
    taker_fee_rate: float,
    consecutive_data_errors: int,
) -> SideEvaluation:
    book = _book_for_side(row, side)
    gate = evaluate_market_gate(
        cfg,
        seconds_left=float(row.get("seconds_left") or 0.0),
        book=book,
        resolution_source=resolution_source,
        consecutive_data_errors=consecutive_data_errors,
    )
    reasons = list(gate.reasons)
    entry = book.best_ask
    breakeven = None
    edge = None

    if model_probability < _minimum_model_probability(cfg):
        reasons.append("model_probability_too_low")

    if entry is None:
        reasons.append("missing_entry_price")
    else:
        breakeven = breakeven_probability(entry, taker_fee_rate, fees_enabled)
        edge = model_probability - breakeven
        if edge < _minimum_edge(cfg):
            reasons.append("edge_too_small")

    return SideEvaluation(
        side=side,
        model_probability=model_probability,
        entry_price=entry,
        breakeven_probability=breakeven,
        edge=edge,
        gate_ok=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def decide_snapshot(
    row: dict[str, Any],
    *,
    q_up: float,
    cfg: V2Config,
    resolution_source: str | None = None,
    fees_enabled: bool = True,
    taker_fee_rate: float = 0.07,
    consecutive_data_errors: int = 0,
) -> SignalDecision:
    """Pure side-by-side V2 decision using only data available at this snapshot.

    q_up is injected by the model/research layer. This function does not invent
    predictive coefficients. It applies market-quality, freshness, settlement-source,
    probability and fee-adjusted edge gates symmetrically to UP and DOWN.
    """
    q = float(q_up)
    if q < 0 or q > 1:
        raise ValueError("q_up must be between 0 and 1")
    if taker_fee_rate < 0:
        raise ValueError("taker_fee_rate must be non-negative")

    source = resolution_source
    if source is None:
        source = str(row.get("resolution_source") or "") or None

    up = _evaluate_side(
        row,
        side="UP",
        model_probability=q,
        cfg=cfg,
        resolution_source=source,
        fees_enabled=fees_enabled,
        taker_fee_rate=taker_fee_rate,
        consecutive_data_errors=consecutive_data_errors,
    )
    down = _evaluate_side(
        row,
        side="DOWN",
        model_probability=1.0 - q,
        cfg=cfg,
        resolution_source=source,
        fees_enabled=fees_enabled,
        taker_fee_rate=taker_fee_rate,
        consecutive_data_errors=consecutive_data_errors,
    )

    candidates = [evaluation for evaluation in (up, down) if evaluation.gate_ok and evaluation.edge is not None]
    if not candidates:
        reasons = tuple(dict.fromkeys(up.reasons + down.reasons))
        return SignalDecision(False, None, None, None, None, None, reasons, up, down)

    selected = max(candidates, key=lambda evaluation: float(evaluation.edge or float("-inf")))
    return SignalDecision(
        True,
        selected.side,
        selected.model_probability,
        selected.entry_price,
        selected.breakeven_probability,
        selected.edge,
        (),
        up,
        down,
    )
