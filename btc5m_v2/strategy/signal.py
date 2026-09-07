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


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


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
    fees_enabled: bool | None,
    taker_fee_rate: float | None,
    fee_exponent: float | None,
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

    if fees_enabled is None:
        reasons.append("missing_fee_metadata")
    elif fees_enabled and (taker_fee_rate is None or fee_exponent is None):
        reasons.append("missing_fee_metadata")
    elif taker_fee_rate is not None and taker_fee_rate < 0:
        reasons.append("invalid_fee_rate")
    elif fee_exponent is not None and fee_exponent < 0:
        reasons.append("invalid_fee_exponent")

    if entry is None:
        reasons.append("missing_entry_price")
    elif "missing_fee_metadata" not in reasons and "invalid_fee_rate" not in reasons and "invalid_fee_exponent" not in reasons:
        rate = 0.0 if not fees_enabled else float(taker_fee_rate or 0.0)
        exponent = 1.0 if not fees_enabled else float(fee_exponent or 0.0)
        breakeven = breakeven_probability(entry, rate, bool(fees_enabled), exponent)
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
    fees_enabled: bool | None = None,
    taker_fee_rate: float | None = None,
    fee_exponent: float | None = None,
    consecutive_data_errors: int = 0,
) -> SignalDecision:
    """Pure V2 decision using only data available at this snapshot.

    ``q_up`` is injected by the model/research layer. Fee inputs must come from
    the market's CLOB V2 fee schedule (or explicit test inputs); missing fee
    metadata blocks trading rather than silently assuming a category rate.
    """

    q = float(q_up)
    if q < 0 or q > 1:
        raise ValueError("q_up must be between 0 and 1")

    source = resolution_source
    if source is None:
        source = str(row.get("resolution_source") or "") or None

    enabled = fees_enabled
    if enabled is None:
        enabled = _optional_bool(row.get("fees_enabled"))
    rate = taker_fee_rate if taker_fee_rate is not None else _f(row.get("fee_rate"))
    exponent = fee_exponent if fee_exponent is not None else _f(row.get("fee_exponent"))

    up = _evaluate_side(
        row,
        side="UP",
        model_probability=q,
        cfg=cfg,
        resolution_source=source,
        fees_enabled=enabled,
        taker_fee_rate=rate,
        fee_exponent=exponent,
        consecutive_data_errors=consecutive_data_errors,
    )
    down = _evaluate_side(
        row,
        side="DOWN",
        model_probability=1.0 - q,
        cfg=cfg,
        resolution_source=source,
        fees_enabled=enabled,
        taker_fee_rate=rate,
        fee_exponent=exponent,
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
