#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc5m_v2.config import load_config
from btc5m_v2.feeds.polymarket_twap import TwapObservation, twap_events
from btc5m_v2.feeds.polymarket_ws import BookState, WsEnvelope, market_events
from btc5m_v2.market import MarketInfo, discover_current_market
from btc5m_v2.market_fees import MarketFeeSchedule, fetch_market_fee_schedule
from btc5m_v2.online_paper import (
    PaperPosition,
    SessionRiskState,
    build_online_model_row,
    evaluate_online_candidate,
    settle_position_payload,
    write_json_atomic,
)
from btc5m_v2.research.btc_features import BTCSample
from btc5m_v2.research.resolution import fetch_gamma_resolution
from btc5m_v2.strategy.model import LogisticProbabilityModel

UTC = dt.timezone.utc


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps({"ts": dt.datetime.now(UTC).isoformat(), **payload}, default=str), flush=True)


def _snapshot_row(market: MarketInfo, up: BookState, down: BookState, received_ts_ns: int) -> dict[str, Any] | None:
    if up.best_bid is None or up.best_ask is None or down.best_bid is None or down.best_ask is None:
        return None
    up_snap = up.snapshot("UP", received_ts_ns)
    down_snap = down.snapshot("DOWN", received_ts_ns)
    return {
        "slug": market.slug,
        "condition_id": market.condition_id,
        "resolution_source": market.resolution_source,
        "market_end": market.end_iso,
        "received_ts_ns": int(received_ts_ns),
        "seconds_left": market.seconds_left(received_ts_ns / 1_000_000_000),
        "up_bid": up_snap["bid"],
        "up_ask": up_snap["ask"],
        "up_spread": up_snap["spread"],
        "up_bid_depth_3_usd": up_snap["bid_depth_3_usd"],
        "up_ask_depth_3_usd": up_snap["ask_depth_3_usd"],
        "up_exchange_age_ms": up_snap["exchange_age_ms"],
        "down_bid": down_snap["bid"],
        "down_ask": down_snap["ask"],
        "down_spread": down_snap["spread"],
        "down_bid_depth_3_usd": down_snap["bid_depth_3_usd"],
        "down_ask_depth_3_usd": down_snap["ask_depth_3_usd"],
        "down_exchange_age_ms": down_snap["exchange_age_ms"],
    }


async def _pump_books(market: MarketInfo, queue: asyncio.Queue[tuple[str, Any]]) -> None:
    async for envelope in market_events([market.up_token_id, market.down_token_id]):
        await queue.put(("book", envelope))


async def _pump_twap(queue: asyncio.Queue[tuple[str, Any]]) -> None:
    async for observation in twap_events(symbol="btc/usd", window_seconds=60):
        await queue.put(("btc", observation))


def _paper_path(root: Path, slug: str) -> Path:
    day = dt.datetime.now(UTC).date().isoformat()
    return root / day / f"{slug}.json"


async def _resolve_position(
    position: PaperPosition,
    market: MarketInfo,
    *,
    risk: SessionRiskState,
    journal_root: Path,
    timeout_sec: float = 240.0,
) -> None:
    if time.time() < market.end_ts:
        await asyncio.sleep(max(0.0, market.end_ts - time.time()))
    deadline = time.time() + max(30.0, float(timeout_sec))
    while time.time() < deadline:
        try:
            result = await asyncio.to_thread(fetch_gamma_resolution, market)
        except Exception as exc:
            emit({"status": "resolution_poll_error", "slug": market.slug, "error": str(exc)})
            await asyncio.sleep(5.0)
            continue
        if result.resolved and result.winning_side in {"UP", "DOWN"}:
            payload = settle_position_payload(position, winning_side=result.winning_side)
            payload["resolution_method"] = result.source
            write_json_atomic(_paper_path(journal_root, market.slug), payload)
            risk.settle(float(payload["net_pnl_usd"]))
            emit(
                {
                    "status": "paper_settled",
                    "slug": market.slug,
                    "side": position.side,
                    "winning_side": result.winning_side,
                    "net_pnl_usd": round(float(payload["net_pnl_usd"]), 6),
                    "session_realized_pnl_usd": round(risk.realized_pnl_usd, 6),
                    "consecutive_losses": risk.consecutive_losses,
                }
            )
            return
        await asyncio.sleep(5.0)
    emit({"status": "paper_resolution_pending", "slug": market.slug})


async def _run_market(
    market: MarketInfo,
    *,
    model: LogisticProbabilityModel,
    cfg: Any,
    stake_usd: float,
    target_seconds_left: float,
    risk: SessionRiskState,
    journal_root: Path,
    reference_tolerance_ms: int,
) -> PaperPosition | None:
    try:
        fee_schedule: MarketFeeSchedule = await asyncio.to_thread(
            fetch_market_fee_schedule, market.condition_id
        )
    except Exception as exc:
        emit({"status": "market_blocked", "slug": market.slug, "reasons": ["fee_schedule_unavailable"], "error": str(exc)})
        fee_schedule = None  # type: ignore[assignment]

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=50000)
    book_task = asyncio.create_task(_pump_books(market, queue))
    btc_task = asyncio.create_task(_pump_twap(queue))
    up = BookState(market.up_token_id)
    down = BookState(market.down_token_id)
    btc_samples: list[BTCSample] = []
    snapshots: list[dict[str, Any]] = []
    reference_price: float | None = None
    reference_offset_ms: int | None = None
    market_start_ms = int(round((market.end_ts - 300.0) * 1000.0))
    decision_done = False
    position: PaperPosition | None = None

    emit(
        {
            "status": "market_started",
            "slug": market.slug,
            "seconds_left": round(market.seconds_left(), 3),
            "model_version": model.model_version,
            "target_seconds_left": target_seconds_left,
            "mode": "paper",
        }
    )

    try:
        while time.time() < market.end_ts + 1.5:
            if book_task.done():
                exc = book_task.exception()
                if exc is not None:
                    raise RuntimeError(f"clob_websocket_stopped:{exc}")
            if btc_task.done():
                exc = btc_task.exception()
                if exc is not None:
                    raise RuntimeError(f"twap_websocket_stopped:{exc}")

            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            if kind == "btc":
                observation: TwapObservation = payload
                btc_samples.append(BTCSample(int(observation.received_ts_ns), float(observation.value)))
                # Keep enough history for the model plus a small safety margin.
                cutoff = observation.received_ts_ns - 90 * 1_000_000_000
                if len(btc_samples) > 500:
                    btc_samples = [sample for sample in btc_samples if sample.ts_ns >= cutoff]
                offset = int(observation.chainlink_ts_ms - market_start_ms)
                if abs(offset) <= int(reference_tolerance_ms):
                    if reference_offset_ms is None or abs(offset) < abs(reference_offset_ms):
                        reference_offset_ms = offset
                        reference_price = float(observation.value)
                        emit(
                            {
                                "status": "btc_reference_captured",
                                "slug": market.slug,
                                "reference_price": reference_price,
                                "reference_offset_ms": reference_offset_ms,
                            }
                        )
                continue

            envelope: WsEnvelope = payload
            changed = up.apply(envelope.event, envelope.received_ts_ns)
            changed = down.apply(envelope.event, envelope.received_ts_ns) or changed
            if not changed:
                continue
            raw = _snapshot_row(market, up, down, envelope.received_ts_ns)
            if raw is None:
                continue
            snapshots.append(raw)
            history_cutoff = envelope.received_ts_ns - 75 * 1_000_000_000
            if len(snapshots) > 10000:
                snapshots = [row for row in snapshots if int(row["received_ts_ns"]) >= history_cutoff]

            seconds_left = float(raw["seconds_left"])
            if decision_done or seconds_left > float(target_seconds_left):
                continue
            if seconds_left < cfg.entry_min:
                decision_done = True
                emit({"status": "paper_no_trade", "slug": market.slug, "reasons": ["missed_decision_window"]})
                continue

            # Match training semantics: first observable snapshot after crossing
            # the 120-second target inside the configured 90-150 second window.
            decision_done = True
            row = build_online_model_row(
                raw,
                snapshots,
                btc_samples=btc_samples,
                reference_price=reference_price,
            )
            risk_reasons = risk.gate_reasons(cfg)
            if risk_reasons:
                emit({"status": "paper_no_trade", "slug": market.slug, "reasons": list(risk_reasons)})
                continue
            if fee_schedule is None:
                emit({"status": "paper_no_trade", "slug": market.slug, "reasons": ["missing_fee_schedule"]})
                continue

            decision, opportunity, reasons = evaluate_online_candidate(
                row,
                model=model,
                cfg=cfg,
                fee_schedule=fee_schedule,
                up_asks=list(up.asks.items()),
                down_asks=list(down.asks.items()),
                notional_usd=stake_usd,
            )
            q_up = model.predict_up(row)
            if reasons or opportunity is None or not decision.trade:
                emit(
                    {
                        "status": "paper_no_trade",
                        "slug": market.slug,
                        "seconds_left": round(seconds_left, 3),
                        "q_up": q_up,
                        "reasons": list(reasons),
                    }
                )
                continue

            position = PaperPosition.from_opportunity(
                market=market,
                row=row,
                model=model,
                decision=decision,
                opportunity=opportunity,
            )
            risk.register_trade()
            write_json_atomic(_paper_path(journal_root, market.slug), position.payload())
            emit(
                {
                    "status": "paper_buy",
                    "slug": market.slug,
                    "side": position.side,
                    "seconds_left": round(position.seconds_left, 3),
                    "q_up": round(position.q_up, 6),
                    "model_probability_side": round(position.model_probability_side, 6),
                    "average_price": round(position.average_price, 6),
                    "worst_price": round(position.worst_price, 6),
                    "fee_usd": round(position.fee_usd, 6),
                    "edge_per_share": round(position.edge_per_share, 6),
                    "expected_value_usd": round(position.expected_value_usd, 6),
                    "participation_pct": position.participation_pct,
                    "live_order_placed": False,
                }
            )
    finally:
        for task in (book_task, btc_task):
            task.cancel()
        await asyncio.gather(book_task, btc_task, return_exceptions=True)
    return position


async def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model_path = Path(args.model)
    if not model_path.exists():
        emit(
            {
                "status": "startup_blocked",
                "reason": "trained_model_missing",
                "model": str(model_path),
                "hint": "Build a ready dataset and run scripts/btc5m_v2_train_model.py first.",
            }
        )
        return 2
    model = LogisticProbabilityModel.load_json(model_path)
    risk = SessionRiskState(starting_equity_usd=float(args.paper_equity_usd))
    journal_root = Path(args.journal_root)
    settle_tasks: set[asyncio.Task[None]] = set()
    started = time.time()
    last_slug: str | None = None

    emit(
        {
            "status": "v2_online_paper_started",
            "model": str(model_path),
            "model_version": model.model_version,
            "feature_columns": list(model.feature_columns),
            "stake_usd": float(args.stake_usd),
            "paper_equity_usd": float(args.paper_equity_usd),
            "live_execution": False,
            "wallet_credentials_loaded": False,
        }
    )

    try:
        while args.session_minutes <= 0 or time.time() - started < args.session_minutes * 60:
            try:
                market = await asyncio.to_thread(discover_current_market, cfg)
            except Exception as exc:
                emit({"status": "market_discovery_error", "error": str(exc)})
                await asyncio.sleep(1.0)
                continue
            if market is None or market.slug == last_slug:
                await asyncio.sleep(0.5)
                continue
            last_slug = market.slug
            try:
                position = await _run_market(
                    market,
                    model=model,
                    cfg=cfg,
                    stake_usd=float(args.stake_usd),
                    target_seconds_left=float(args.target_seconds_left),
                    risk=risk,
                    journal_root=journal_root,
                    reference_tolerance_ms=int(args.reference_tolerance_ms),
                )
            except Exception as exc:
                emit({"status": "market_session_failed_closed", "slug": market.slug, "error": str(exc)})
                continue
            if position is not None:
                task = asyncio.create_task(
                    _resolve_position(
                        position,
                        market,
                        risk=risk,
                        journal_root=journal_root,
                        timeout_sec=float(args.resolution_timeout_sec),
                    )
                )
                settle_tasks.add(task)
                task.add_done_callback(settle_tasks.discard)
    except KeyboardInterrupt:
        emit({"status": "stopped_by_user"})
    finally:
        if settle_tasks:
            await asyncio.gather(*settle_tasks, return_exceptions=True)

    emit(
        {
            "status": "v2_online_paper_complete",
            "trades": risk.trades_today,
            "realized_pnl_usd": round(risk.realized_pnl_usd, 6),
            "consecutive_losses": risk.consecutive_losses,
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BTC5M V2 model-driven online PAPER bot (no wallet/order path)"
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "btc_5m_v2.yaml"))
    parser.add_argument("--model", default=str(ROOT / "runtime" / "models" / "btc5m_logit_v1.json"))
    parser.add_argument("--stake-usd", type=float, default=5.0)
    parser.add_argument("--paper-equity-usd", type=float, default=1000.0)
    parser.add_argument("--target-seconds-left", type=float, default=120.0)
    parser.add_argument("--reference-tolerance-ms", type=int, default=5000)
    parser.add_argument("--session-minutes", type=float, default=0.0, help="0 means run until stopped")
    parser.add_argument("--resolution-timeout-sec", type=float, default=240.0)
    parser.add_argument("--journal-root", default=str(ROOT / "runtime" / "v2" / "paper"))
    args = parser.parse_args()
    if args.stake_usd <= 0 or args.paper_equity_usd <= 0:
        parser.error("stake and paper equity must be positive")
    if args.target_seconds_left < 0:
        parser.error("target seconds left cannot be negative")
    # No --execute option exists by design. Unknown live flags fail argparse.
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
