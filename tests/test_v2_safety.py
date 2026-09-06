from btc5m_v2.config import V2Config
from btc5m_v2.safety import BookMetrics, evaluate_market_gate, position_mark_price, top_n_ask_notional


def cfg() -> V2Config:
    return V2Config(
        {
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


def test_good_candidate_passes():
    result = evaluate_market_gate(cfg(), seconds_left=120, book=good_book())
    assert result.ok
    assert result.reasons == ()


def test_entry_window_fails_closed():
    early = evaluate_market_gate(cfg(), seconds_left=151, book=good_book())
    late = evaluate_market_gate(cfg(), seconds_left=89, book=good_book())
    assert "too_early" in early.reasons
    assert "too_late" in late.reasons


def test_selected_side_spread_is_enforced():
    book = BookMetrics(0.65, 0.69, 0.04, 100.0, 0.5)
    result = evaluate_market_gate(cfg(), seconds_left=120, book=book)
    assert not result.ok
    assert "spread_too_wide" in result.reasons


def test_selected_side_depth_is_enforced():
    book = BookMetrics(0.70, 0.71, 0.01, 10.0, 0.5)
    result = evaluate_market_gate(cfg(), seconds_left=120, book=book)
    assert "insufficient_ask_depth" in result.reasons


def test_stale_or_missing_timestamp_is_blocked():
    stale = BookMetrics(0.70, 0.71, 0.01, 60.0, 3.0)
    missing = BookMetrics(0.70, 0.71, 0.01, 60.0, None)
    assert "stale_clob_quote" in evaluate_market_gate(cfg(), seconds_left=120, book=stale).reasons
    assert "missing_quote_timestamp" in evaluate_market_gate(cfg(), seconds_left=120, book=missing).reasons


def test_data_error_circuit_breaker_blocks():
    result = evaluate_market_gate(cfg(), seconds_left=120, book=good_book(), consecutive_data_errors=3)
    assert "data_error_circuit_breaker" in result.reasons


def test_position_mark_uses_executable_bid():
    book = BookMetrics(0.62, 0.68, 0.06, 60.0, 0.5)
    assert position_mark_price(book) == 0.62


def test_top_three_ask_notional_uses_lowest_asks():
    levels = [(0.75, 10), (0.71, 10), (0.72, 10), (0.73, 10)]
    assert top_n_ask_notional(levels, 3) == (0.71 * 10 + 0.72 * 10 + 0.73 * 10)
