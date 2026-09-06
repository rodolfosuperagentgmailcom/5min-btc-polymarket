# BTC5M V2 Roadmap

This fork develops a safety-first, research-driven successor to the original 5-minute BTC Polymarket runner.

## Operating principle

The V2 signal should not buy simply because one side is already expensive. It should estimate fair settlement probability, compare that estimate with the executable all-in market price, subtract fees/slippage, and trade only when the remaining edge clears a configured threshold.

## Milestone 1 — SAFE (P0)

- [ ] Load risk/profile configuration from YAML.
- [ ] Enforce a bounded entry window (research default: 90–150 seconds remaining).
- [ ] Validate spread for the selected token, not the minimum spread across both outcomes.
- [ ] Enforce selected-side spread and depth limits before entry.
- [ ] Remove permissive execution defaults (`PM_MAX_SPREAD=1`, `PM_MIN_TOP_ASK_NOTIONAL_USD=0`).
- [ ] Use executable CLOB bids for position marks and stop logic.
- [ ] Add quote-staleness and consecutive-data-error circuit breakers.
- [ ] Add session limits: max daily loss, max trades/day, consecutive-loss stop.
- [ ] Disable automatic hedge by default.
- [ ] Make dry-run/paper mode the controller default; live execution must be explicit.
- [ ] Add tests for each safety gate.

## Milestone 2 — DATA (P0)

- [ ] Polymarket market WebSocket feed.
- [ ] Settlement-aligned BTC source adapter.
- [ ] Record exchange timestamp and local receive timestamp.
- [ ] Record UP/DOWN bid, ask, spread, and depth.
- [ ] Record BTC reference, current value, delta, 15s/30s/60s momentum, and realized volatility.
- [ ] Store raw/replayable data in Parquet.
- [ ] Attach final market resolution to each 5-minute event.

## Milestone 3 — EDGE (P1)

- [ ] Reproduce the original `ask >= 0.70` strategy as a benchmark.
- [ ] Add volatility-normalized BTC impulse.
- [ ] Add momentum consistency/acceleration features.
- [ ] Add order-book imbalance as a model feature.
- [ ] Build an interpretable fair-probability baseline.
- [ ] Calculate fee-adjusted break-even probability.
- [ ] Estimate executable VWAP/slippage for intended order size.
- [ ] Trade only when model probability minus all-in break-even exceeds configured minimum edge.

## Milestone 4 — PAPER (P1)

- [ ] Pure signal function shared by replay, paper, and live adapters.
- [ ] Paper executor with realistic fills, fees, slippage, and latency.
- [ ] Walk-forward/out-of-sample evaluation.
- [ ] Performance report: trades, win rate, gross/net P&L, fees, slippage, profit factor, max drawdown, calibration.
- [ ] Minimum 500 independent forward-paper signals before any live-sizing discussion.

## Live execution policy

Live execution is intentionally disabled by default. No wallet private key, API secret, or `.env` file should be committed to this repository. Any eventual live adapter must require an explicit execution flag and pass every risk/data-quality gate.
