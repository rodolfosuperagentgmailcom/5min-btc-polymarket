#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc5m_v2.config import load_config
from btc5m_v2.safety import BookMetrics, evaluate_market_gate, top_n_ask_notional

UTC = dt.timezone.utc


def ts_utc() -> str:
    return dt.datetime.now(UTC).isoformat().replace("+00:00", "Z")


def bucket_5m(ts: int) -> int:
    return ts - (ts % 300)


def parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def fetch_event(slug: str) -> dict[str, Any] | None:
    response = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"slug": slug},
        timeout=12,
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else None


def resolve_current_market(slug_prefix: str) -> dict[str, Any] | None:
    slug = f"{slug_prefix}{bucket_5m(int(time.time()))}"
    event = fetch_event(slug)
    if not event:
        return None
    markets = event.get("markets") or []
    if not markets:
        return None
    market = dict(markets[0])
    if market.get("closed") is True or market.get("active") is False:
        return None
    market["_event_slug"] = slug
    market["_event_resolution_source"] = event.get("resolutionSource")
    market["_event_description"] = event.get("description")
    return market


def resolution_source(market: dict[str, Any]) -> str:
    """Return the most specific Gamma resolution source available."""
    return str(
        market.get("resolutionSource")
        or market.get("_event_resolution_source")
        or ""
    ).strip()


def parse_market(market: dict[str, Any]) -> tuple[str, str, str, str, float]:
    outcomes = parse_json_field(market.get("outcomes")) or []
    token_ids = parse_json_field(market.get("clobTokenIds")) or []
    if len(token_ids) < 2:
        raise RuntimeError("missing clobTokenIds")

    labels = [str(x).lower() for x in outcomes[:2]] if isinstance(outcomes, list) else []
    up_i, down_i = 0, 1
    if len(labels) >= 2 and ("up" in labels[1] or "yes" in labels[1]):
        up_i, down_i = 1, 0

    end_iso = str(market.get("endDate") or market.get("endDateIso") or "")
    if not end_iso:
        raise RuntimeError("missing market end")
    end_ts = dt.datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp()

    return (
        str(token_ids[up_i]),
        str(token_ids[down_i]),
        str(market.get("slug") or market.get("_event_slug") or ""),
        end_iso,
        max(0.0, end_ts - time.time()),
    )


def level_pairs(levels: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for level in levels or []:
        try:
            out.append((float(getattr(level, "price")), float(getattr(level, "size"))))
        except Exception:
            continue
    return out


def quote_age_sec(book: Any) -> float | None:
    raw = getattr(book, "timestamp", None)
    if raw in (None, ""):
        return None
    try:
        stamp = float(raw)
        if stamp > 1e12:
            stamp /= 1000.0
        age = time.time() - stamp
        return max(0.0, age)
    except Exception:
        return None


def metrics_from_book(book: Any) -> BookMetrics:
    bids = level_pairs(getattr(book, "bids", []) or [])
    asks = level_pairs(getattr(book, "asks", []) or [])
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min((p for p, _ in asks), default=None)
    spread = None
    if best_bid is not None and best_ask is not None:
        spread = max(0.0, best_ask - best_bid)
    return BookMetrics(
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        top3_ask_notional_usd=top_n_ask_notional(asks, 3),
        quote_age_sec=quote_age_sec(book),
    )


def snapshot_dict(book: BookMetrics) -> dict[str, float | None]:
    return {
        "bid": book.best_bid,
        "ask": book.best_ask,
        "spread": book.spread,
        "top3_ask_notional_usd": round(book.top3_ask_notional_usd, 4),
        "quote_age_sec": None if book.quote_age_sec is None else round(book.quote_age_sec, 3),
    }


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="BTC5M V2 fail-closed paper safety runner")
    parser.add_argument("--config", default=str(ROOT / "config" / "btc_5m_v2.yaml"))
    parser.add_argument("--threshold", type=float, default=0.70, help="Benchmark threshold only; not the future V2 alpha model")
    parser.add_argument("--entry-timeout-min", type=int, default=35)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--profile", default="conservative")
    parser.add_argument("--stake-usd", type=float, default=5.0)
    parser.add_argument("--close-retry-max", type=int, default=30)
    parser.add_argument("--close-retry-delay-sec", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.execute:
        emit({
            "ts": ts_utc(),
            "status": "live_execution_blocked",
            "reason": "V2 safety runner is paper-only until safety tests and recorder validation are complete",
        })
        return 2

    cfg = load_config(args.config)
    slug_prefix = str(cfg.raw["market"].get("slug_prefix", "btc-updown-5m-"))
    clob = ClobClient(host="https://clob.polymarket.com", chain_id=POLYGON)
    deadline = time.time() + max(1, args.entry_timeout_min) * 60
    consecutive_errors = 0

    emit({
        "ts": ts_utc(),
        "status": "v2_started",
        "mode": "paper",
        "entry_window_sec": [cfg.entry_min, cfg.entry_max],
        "expected_resolution_source": cfg.expected_resolution_source,
        "max_selected_side_spread": cfg.max_spread,
        "min_top3_ask_notional_usd": cfg.min_depth,
        "live_orders": False,
    })

    while time.time() < deadline:
        try:
            market = resolve_current_market(slug_prefix)
            if not market:
                consecutive_errors += 1
                emit({"ts": ts_utc(), "status": "no_current_market", "consecutive_errors": consecutive_errors})
                time.sleep(args.poll_sec)
                continue

            up_token, down_token, slug, end_iso, seconds_left = parse_market(market)
            source = resolution_source(market)
            up_book = metrics_from_book(clob.get_order_book(up_token))
            down_book = metrics_from_book(clob.get_order_book(down_token))
            consecutive_errors = 0

            books = {"UP": up_book, "DOWN": down_book}
            candidates = [
                (side, book.best_ask)
                for side, book in books.items()
                if book.best_ask is not None and book.best_ask >= args.threshold
            ]

            base = {
                "ts": ts_utc(),
                "slug": slug,
                "market_end": end_iso,
                "seconds_left": round(seconds_left, 3),
                "resolution_source": source,
                "up": snapshot_dict(up_book),
                "down": snapshot_dict(down_book),
                "benchmark_threshold": args.threshold,
            }

            if not candidates:
                source_gate = evaluate_market_gate(
                    cfg,
                    seconds_left=seconds_left,
                    book=up_book,
                    resolution_source=source,
                    consecutive_data_errors=consecutive_errors,
                )
                source_reasons = [
                    reason
                    for reason in source_gate.reasons
                    if reason in {"missing_resolution_source", "unexpected_resolution_source"}
                ]
                if source_reasons:
                    emit({**base, "status": "market_blocked", "gate_reasons": source_reasons})
                else:
                    emit({**base, "status": "observe_no_threshold_candidate"})
                time.sleep(args.poll_sec)
                continue

            side, ask = max(candidates, key=lambda item: float(item[1]))
            selected = books[side]
            gate = evaluate_market_gate(
                cfg,
                seconds_left=seconds_left,
                book=selected,
                resolution_source=source,
                consecutive_data_errors=consecutive_errors,
            )

            if not gate.ok:
                emit({
                    **base,
                    "status": "paper_candidate_blocked",
                    "selected_side": side,
                    "selected_ask": ask,
                    "gate_reasons": list(gate.reasons),
                })
            else:
                emit({
                    **base,
                    "status": "paper_signal",
                    "selected_side": side,
                    "selected_ask": ask,
                    "stake_usd": args.stake_usd,
                    "note": "Benchmark threshold candidate passed V2 market safety gates; no live order placed.",
                })

            time.sleep(args.poll_sec)
        except KeyboardInterrupt:
            emit({"ts": ts_utc(), "status": "stopped_by_user"})
            return 0
        except Exception as exc:
            consecutive_errors += 1
            emit({
                "ts": ts_utc(),
                "status": "data_error",
                "error": str(exc),
                "consecutive_errors": consecutive_errors,
            })
            if consecutive_errors >= cfg.max_consecutive_data_errors:
                emit({
                    "ts": ts_utc(),
                    "status": "circuit_breaker",
                    "reason": "max_consecutive_data_errors",
                })
                return 3
            time.sleep(args.poll_sec)

    emit({"ts": ts_utc(), "status": "paper_session_complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
