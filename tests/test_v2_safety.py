from btc5m_v2.config import V2Config
from btc5m_v2.safety import (
    BookMetrics,
    evaluate_market_gate,
    normalize_resolution_source,
    position_mark_price,
    resolution_source_reasons,
    top_n_ask_notional,
)

EXPECTED_SOURCE = "https://data.chain.link/streams/btc-usd-twap-60s-streams"


def cfg() -> V2Config:
    return V2Config(
        {
            "resolution": {
                "expected_source": EXPECTED_SOURCE,
                "fail_if_missing": True,
                "fail_if_unexpected": True,
            },
            "entry": {"seconds_left_min": 90, "seconds_left_max": 150},
            "market_quality": {
                "max_selected_side_spread": 0.02,
                "min_top_3_ask_notional_usd": 30,
            },
            "freshness": {
                "clob_max_age_sec": 2.0,
                "max_consecutive_data_errors": 3,
            },
        }
    )


def good_book() -> BookMetrics:
    return BookMetrics(
        best_bid=0.70,
        best_ask=0.71,
        spread=0.01,
        top3_ask_notional_usd=60.0,
        quote_age_sec=0.5,
    )


def gate(*, seconds_left: float = 120, book: BookMetrics | None = None, source: str | None = EXPECTED_SOURCE, errors: int = 0):
    return evaluate_market_gate(
        cfg(),
        seconds_left=seconds_left,
        book=book or good_book(),
        resolution_source=source,
        consecutive_data_errors=errors,
    )


def test_good_candidate_passes():
    result = gate()
    assert result.ok
    assert result.reasons == ()


def test_resolution_source_exact_match_passes():
    assert resolution_source_reasons(cfg(), EXPECTED_SOURCE) == ()


def test_resolution_source_trailing_slash_normalizes():
    assert normalize_resolution_source(EXPECTED_SOURCE + "/") == normalize_resolution_source(EXPECTED_SOURCE)
    assert resolution_source_reasons(cfg(), EXPECTED_SOURCE + "/") == ()


def test_missing_resolution_source_fails_closed():
    result = gate(source="")
    assert not result.ok
    assert "missing_resolution_source" in result.reasons


def test_unexpected_resolution_source_fails_closed():
    result = gate(source="https://example.com/btc-usd")
    assert not result.ok
    assert "unexpected_resolution_source" in result.reasons


def test_entry_window_fails_closed():
    early = gate(seconds_left=151)
    late = gate(seconds_left=89)
    assert "too_early" in early.reasons
    assert "too_late" in late.reasons


def test_selected_side_spread_is_enforced():
    book = BookMetrics(0.65, 0.69, 0.04, 100.0, 0.5)
    result = gate(book=book)
    assert not result.ok
    assert "spread_too_wide" in result.reasons


def test_selected_side_depth_is_enforced():
    book = BookMetrics(0.70, 0.71, 0.01, 10.0, 0.5)
    result = gate(book=book)
    assert "insufficient_ask_depth" in result.reasons


def test_stale_or_missing_timestamp_is_blocked():
    stale = BookMetrics(0.70, 0.71, 0.01, 60.0, 3.0)
    missing = BookMetrics(0.70, 0.71, 0.01, 60.0, None)
    assert "stale_clob_quote" in gate(book=stale).reasons
    assert "missing_quote_timestamp" in gate(book=missing).reasons


def test_data_error_circuit_breaker_blocks():
    result = gate(errors=3)
    assert "data_error_circuit_breaker" in result.reasons


def test_position_mark_uses_executable_bid():
    book = BookMetrics(0.62, 0.68, 0.06, 60.0, 0.5)
    assert position_mark_price(book) == 0.62


def test_top_three_ask_notional_uses_lowest_asks():
    levels = [(0.75, 10), (0.71, 10), (0.72, 10), (0.73, 10)]
    assert top_n_ask_notional(levels, 3) == (0.71 * 10 + 0.72 * 10 + 0.73 * 10)
