from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from .config import V2Config


@dataclass(frozen=True)
class BookMetrics:
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    top3_ask_notional_usd: float
    quote_age_sec: float | None


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: tuple[str, ...]


def normalize_resolution_source(source: str | None) -> str:
    """Normalize a resolution-source URL for strict, stable comparison."""
    raw = str(source or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except Exception:
        return raw.lower().rstrip("/")
    if not parts.scheme or not parts.netloc:
        return raw.lower().rstrip("/")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def resolution_source_reasons(cfg: V2Config, source: str | None) -> tuple[str, ...]:
    """Fail closed when Gamma omits or changes the configured settlement source."""
    actual = normalize_resolution_source(source)
    expected = normalize_resolution_source(cfg.expected_resolution_source)
    reasons: list[str] = []

    if not actual:
        if cfg.fail_if_resolution_source_missing:
            reasons.append("missing_resolution_source")
        return tuple(reasons)

    if actual != expected and cfg.fail_if_resolution_source_unexpected:
        reasons.append("unexpected_resolution_source")

    return tuple(reasons)


def evaluate_market_gate(
    cfg: V2Config,
    *,
    seconds_left: float,
    book: BookMetrics,
    resolution_source: str | None,
    consecutive_data_errors: int = 0,
) -> GateResult:
    reasons: list[str] = list(resolution_source_reasons(cfg, resolution_source))

    if seconds_left < cfg.entry_min:
        reasons.append("too_late")
    if seconds_left > cfg.entry_max:
        reasons.append("too_early")

    if consecutive_data_errors >= cfg.max_consecutive_data_errors:
        reasons.append("data_error_circuit_breaker")

    if book.best_bid is None or book.best_ask is None:
        reasons.append("missing_bid_or_ask")
    if book.spread is None:
        reasons.append("missing_spread")
    elif book.spread > cfg.max_spread:
        reasons.append("spread_too_wide")

    if book.top3_ask_notional_usd < cfg.min_depth:
        reasons.append("insufficient_ask_depth")

    if book.quote_age_sec is None:
        reasons.append("missing_quote_timestamp")
    elif book.quote_age_sec > cfg.clob_max_age_sec:
        reasons.append("stale_clob_quote")

    return GateResult(ok=not reasons, reasons=tuple(reasons))


def position_mark_price(book: BookMetrics) -> float | None:
    """Mark long prediction-token positions at executable best bid, not Gamma mid/outcome price."""
    return book.best_bid


def top_n_ask_notional(levels: Iterable[tuple[float, float]], n: int = 3) -> float:
    """Return USD notional available across the lowest n ask levels."""
    ordered = sorted((float(p), float(s)) for p, s in levels if float(p) > 0 and float(s) > 0)
    return sum(price * size for price, size in ordered[:n])
