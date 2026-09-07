from __future__ import annotations

import bisect
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from btc5m_v2.config import V2Config
from btc5m_v2.fees import all_in_cost_per_share
from btc5m_v2.market import MarketInfo
from btc5m_v2.market_fees import MarketFeeSchedule
from btc5m_v2.research.btc_features import BTCSample, btc_features_at, normalize_btc_samples
from btc5m_v2.research.features import build_feature_row, feature_payload
from btc5m_v2.strategy.edge import SideEdge, evaluate_buy_edge
from btc5m_v2.strategy.model import LogisticProbabilityModel
from btc5m_v2.strategy.signal import SignalDecision, decide_snapshot_with_model

MODEL_LAGS_SECONDS = (5, 15, 30, 60)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _asof_value(
    timestamps_ns: list[int],
    values: list[float | None],
    *,
    target_ns: int,
) -> float | None:
    index = bisect.bisect_right(timestamps_ns, int(target_ns)) - 1
    if index < 0:
        return None
    return values[index]


def build_online_model_row(
    current_snapshot: dict[str, Any],
    snapshot_history: Iterable[dict[str, Any]],
    *,
    btc_samples: Iterable[BTCSample | tuple[int, float]],
    reference_price: float | None,
) -> dict[str, Any]:
    """Build the same causal feature names used by the offline training dataset.

    `snapshot_history` may contain the current row. Historical values are selected
    strictly by receive timestamp at-or-before each lag cutoff; no future row is
    reachable. Final resolution labels are intentionally absent.
    """

    current = dict(current_snapshot)
    now_ns = int(current.get("received_ts_ns") or 0)
    if now_ns <= 0:
        raise ValueError("current snapshot requires received_ts_ns")

    history = sorted(
        (dict(row) for row in snapshot_history if int(row.get("received_ts_ns") or 0) <= now_ns),
        key=lambda row: int(row.get("received_ts_ns") or 0),
    )
    if not history or int(history[-1].get("received_ts_ns") or 0) != now_ns:
        history.append(current)

    history_features = [feature_payload(build_feature_row(row)) for row in history]
    timestamps = [int(row["received_ts_ns"]) for row in history_features]
    up_mids = [_finite(row.get("up_mid")) for row in history_features]

    base = feature_payload(build_feature_row(current))
    row = dict(current)
    row.update(base)
    row["ask_overround"] = None if row.get("ask_sum") is None else float(row["ask_sum"]) - 1.0
    up_probability = _finite(row.get("up_market_probability"))
    row["market_skew_up"] = None if up_probability is None else up_probability - 0.5

    current_up_mid = _finite(row.get("up_mid"))
    for lag in MODEL_LAGS_SECONDS:
        previous = _asof_value(
            timestamps,
            up_mids,
            target_ns=now_ns - lag * 1_000_000_000,
        )
        row[f"up_mid_change_{lag}s"] = (
            None if current_up_mid is None or previous is None else current_up_mid - previous
        )

    samples = normalize_btc_samples(btc_samples)
    btc60 = btc_features_at(
        samples,
        now_ns=now_ns,
        reference_price=reference_price,
        vol_window_sec=60,
    )
    btc30 = btc_features_at(
        samples,
        now_ns=now_ns,
        reference_price=reference_price,
        vol_window_sec=30,
    )
    row.update(
        {
            "btc_reference": btc60.get("btc_reference_price"),
            "btc_current": btc60.get("btc_price"),
            "btc_feed_age_ms": btc60.get("btc_feed_age_ms"),
            "btc_delta_usd": btc60.get("btc_move_from_reference"),
            "btc_momentum_15s": btc60.get("btc_move_15s"),
            "btc_momentum_30s": btc60.get("btc_move_30s"),
            "btc_momentum_60s": btc60.get("btc_move_60s"),
            "btc_realized_vol_30s": btc30.get("btc_realized_move_usd"),
            "btc_realized_vol_60s": btc60.get("btc_realized_move_usd"),
            "btc_impulse_z": btc60.get("btc_impulse_z"),
        }
    )
    return row


def configured_signal_filter_reasons(
    row: dict[str, Any],
    cfg: V2Config,
    *,
    side: str | None = None,
) -> tuple[str, ...]:
    """Apply V2 BTC-quality/impulse gates that are configured outside the model."""

    signal_cfg = cfg.raw.get("signal", {})
    freshness_cfg = cfg.raw.get("freshness", {})
    reasons: list[str] = []

    feed_age_ms = _finite(row.get("btc_feed_age_ms"))
    max_age_ms = float(freshness_cfg.get("btc_max_age_sec", 3.0)) * 1000.0
    if feed_age_ms is None:
        reasons.append("missing_btc_feed")
    elif feed_age_ms > max_age_ms:
        reasons.append("stale_btc_feed")

    delta = _finite(row.get("btc_delta_usd"))
    min_move = float(signal_cfg.get("minimum_absolute_btc_move_usd", 0.0))
    if delta is None:
        reasons.append("missing_btc_reference")
    elif abs(delta) < min_move:
        reasons.append("btc_move_below_minimum")

    impulse = _finite(row.get("btc_impulse_z"))
    min_impulse = float(signal_cfg.get("minimum_impulse_z", 0.0))
    if impulse is None:
        reasons.append("missing_btc_impulse_z")
    elif abs(impulse) < min_impulse:
        reasons.append("btc_impulse_below_minimum")

    selected = str(side or "").strip().upper()
    if selected in {"UP", "DOWN"}:
        direction = 1.0 if selected == "UP" else -1.0
        if bool(signal_cfg.get("require_30s_direction_agreement", False)):
            momentum = _finite(row.get("btc_momentum_30s"))
            if momentum is None:
                reasons.append("missing_btc_momentum_30s")
            elif direction * momentum <= 0:
                reasons.append("btc_momentum_30s_disagrees")
        if bool(signal_cfg.get("require_15s_direction_agreement", False)):
            momentum = _finite(row.get("btc_momentum_15s"))
            if momentum is None:
                reasons.append("missing_btc_momentum_15s")
            elif direction * momentum <= 0:
                reasons.append("btc_momentum_15s_disagrees")

    return tuple(dict.fromkeys(reasons))


def evaluate_online_candidate(
    row: dict[str, Any],
    *,
    model: LogisticProbabilityModel,
    cfg: V2Config,
    fee_schedule: MarketFeeSchedule,
    up_asks: Iterable[tuple[float, float]],
    down_asks: Iterable[tuple[float, float]],
    notional_usd: float,
) -> tuple[SignalDecision, SideEdge | None, tuple[str, ...]]:
    """Run model + safety + fee + full-book executable-edge gates for one snapshot."""

    pre_reasons = configured_signal_filter_reasons(row, cfg)
    if pre_reasons:
        decision = decide_snapshot_with_model(
            row,
            model=model,
            cfg=cfg,
            fees_enabled=fee_schedule.fees_enabled,
            taker_fee_rate=fee_schedule.rate,
            fee_exponent=fee_schedule.exponent,
        )
        return decision, None, pre_reasons

    decision = decide_snapshot_with_model(
        row,
        model=model,
        cfg=cfg,
        fees_enabled=fee_schedule.fees_enabled,
        taker_fee_rate=fee_schedule.rate,
        fee_exponent=fee_schedule.exponent,
    )
    if not decision.trade or decision.side not in {"UP", "DOWN"}:
        return decision, None, decision.reasons or ("model_no_trade",)

    direction_reasons = configured_signal_filter_reasons(row, cfg, side=decision.side)
    if direction_reasons:
        return decision, None, direction_reasons

    asks = up_asks if decision.side == "UP" else down_asks
    probability = float(decision.model_probability or 0.0)
    opportunity = evaluate_buy_edge(
        side=decision.side,
        model_probability=probability,
        asks=asks,
        notional_usd=float(notional_usd),
        fees_enabled=fee_schedule.fees_enabled,
        fee_rate=fee_schedule.rate,
        fee_exponent=fee_schedule.exponent,
        is_taker=True,
        max_book_participation_pct=float(
            cfg.raw.get("market_quality", {}).get("max_order_book_participation_pct", 20.0)
        ),
    )
    min_edge = float(cfg.raw.get("signal", {}).get("minimum_edge_after_fees_and_slippage", 0.0))
    reasons = list(opportunity.reasons)
    if opportunity.edge_per_share is None or opportunity.edge_per_share < min_edge:
        reasons.append("executable_edge_below_minimum")
    if opportunity.expected_value_usd is None or opportunity.expected_value_usd <= 0:
        reasons.append("non_positive_expected_value")
    if reasons:
        return decision, opportunity, tuple(dict.fromkeys(reasons))
    return decision, opportunity, ()


@dataclass
class SessionRiskState:
    starting_equity_usd: float
    trades_today: int = 0
    realized_pnl_usd: float = 0.0
    consecutive_losses: int = 0

    def gate_reasons(self, cfg: V2Config) -> tuple[str, ...]:
        risk_cfg = cfg.raw.get("risk", {})
        reasons: list[str] = []
        if self.trades_today >= int(risk_cfg.get("max_trades_per_day", 12)):
            reasons.append("max_trades_per_day")
        max_daily_loss = self.starting_equity_usd * float(
            risk_cfg.get("max_daily_loss_pct_equity", 3.0)
        ) / 100.0
        if self.realized_pnl_usd <= -max_daily_loss:
            reasons.append("daily_loss_limit")
        if self.consecutive_losses >= int(risk_cfg.get("max_consecutive_losses", 3)):
            reasons.append("max_consecutive_losses")
        return tuple(reasons)

    def register_trade(self) -> None:
        self.trades_today += 1

    def settle(self, pnl_usd: float) -> None:
        pnl = float(pnl_usd)
        self.realized_pnl_usd += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0


@dataclass(frozen=True)
class PaperPosition:
    slug: str
    condition_id: str
    side: str
    signal_ts_ns: int
    seconds_left: float
    model_version: str
    q_up: float
    model_probability_side: float
    requested_notional_usd: float
    spent_usd: float
    shares: float
    average_price: float
    worst_price: float
    fee_usd: float
    all_in_cost_usd: float
    all_in_cost_per_share: float
    edge_per_share: float
    expected_value_usd: float
    participation_pct: float | None
    market_end_ts: float
    resolution_source: str
    feature_snapshot: dict[str, Any] = field(repr=False)

    @classmethod
    def from_opportunity(
        cls,
        *,
        market: MarketInfo,
        row: dict[str, Any],
        model: LogisticProbabilityModel,
        decision: SignalDecision,
        opportunity: SideEdge,
    ) -> "PaperPosition":
        fill = opportunity.fill
        if not fill.fillable or fill.average_price is None or fill.worst_price is None:
            raise ValueError("paper position requires a complete executable fill")
        if opportunity.all_in_cost_per_share is None or opportunity.edge_per_share is None:
            raise ValueError("paper position requires all-in cost and edge")
        q_up = model.predict_up(row)
        if q_up is None or decision.side not in {"UP", "DOWN"}:
            raise ValueError("paper position requires model probability and side")
        return cls(
            slug=market.slug,
            condition_id=market.condition_id,
            side=decision.side,
            signal_ts_ns=int(row["received_ts_ns"]),
            seconds_left=float(row["seconds_left"]),
            model_version=model.model_version,
            q_up=float(q_up),
            model_probability_side=float(decision.model_probability or 0.0),
            requested_notional_usd=float(fill.requested_notional_usd),
            spent_usd=float(fill.spent_usd),
            shares=float(fill.shares),
            average_price=float(fill.average_price),
            worst_price=float(fill.worst_price),
            fee_usd=float(opportunity.fee_usd),
            all_in_cost_usd=float(fill.spent_usd + opportunity.fee_usd),
            all_in_cost_per_share=float(opportunity.all_in_cost_per_share),
            edge_per_share=float(opportunity.edge_per_share),
            expected_value_usd=float(opportunity.expected_value_usd or 0.0),
            participation_pct=fill.participation_pct,
            market_end_ts=float(market.end_ts),
            resolution_source=market.resolution_source,
            feature_snapshot={name: row.get(name) for name in model.feature_columns},
        )

    def payload(self) -> dict[str, Any]:
        return {
            "status": "OPEN",
            "mode": "paper",
            "credentials_loaded": False,
            "order_path_used": False,
            **self.__dict__,
        }


def settle_position_payload(position: PaperPosition, *, winning_side: str) -> dict[str, Any]:
    winner = str(winning_side).strip().upper()
    if winner not in {"UP", "DOWN"}:
        raise ValueError("winning_side must be UP or DOWN")
    won = position.side == winner
    settlement_value = position.shares if won else 0.0
    net_pnl = settlement_value - position.all_in_cost_usd
    payload = position.payload()
    payload.update(
        {
            "status": "SETTLED",
            "winning_side": winner,
            "won": won,
            "settlement_value_usd": settlement_value,
            "net_pnl_usd": net_pnl,
            "return_on_cost": (
                None if position.all_in_cost_usd <= 0 else net_pnl / position.all_in_cost_usd
            ),
            "settled_ts_ns": time.time_ns(),
        }
    )
    return payload


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, destination)
    return destination
