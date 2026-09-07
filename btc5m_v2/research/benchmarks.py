from __future__ import annotations

from dataclasses import dataclass

from btc5m_v2.research.replay import ReplayRow


@dataclass(frozen=True)
class BenchmarkDecision:
    trade: bool
    side: str | None
    entry_price: float | None
    reason: str


def legacy_threshold_decision(
    replay: ReplayRow,
    *,
    threshold: float = 0.70,
    seconds_left_min: float = 60.0,
    seconds_left_max: float | None = None,
) -> BenchmarkDecision:
    feature = replay.feature
    if feature.seconds_left < seconds_left_min:
        return BenchmarkDecision(False, None, None, "too_late")
    if seconds_left_max is not None and feature.seconds_left > seconds_left_max:
        return BenchmarkDecision(False, None, None, "too_early")

    candidates: list[tuple[str, float]] = []
    if feature.up_ask is not None and feature.up_ask >= threshold:
        candidates.append(("UP", feature.up_ask))
    if feature.down_ask is not None and feature.down_ask >= threshold:
        candidates.append(("DOWN", feature.down_ask))
    if not candidates:
        return BenchmarkDecision(False, None, None, "below_threshold")

    candidates.sort(key=lambda item: item[1], reverse=True)
    side, price = candidates[0]
    return BenchmarkDecision(True, side, price, "legacy_threshold")


def benchmark_settlement_pnl(
    decision: BenchmarkDecision,
    *,
    winning_side: str | None,
    fee_per_share_paid: float = 0.0,
) -> float | None:
    if not decision.trade or decision.side is None or decision.entry_price is None:
        return 0.0
    if winning_side not in {"UP", "DOWN"}:
        return None
    payout = 1.0 if decision.side == winning_side else 0.0
    return payout - decision.entry_price - float(fee_per_share_paid)
