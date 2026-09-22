# BTC5M V2 Roadmap

This fork develops a safety-first, research-driven successor to the original 5-minute BTC Polymarket runner.

## Operating principle

V2 does not buy simply because one side is already expensive. It estimates a settlement probability from causal data, compares that estimate with the executable all-in market cost, accounts for the market-specific fee schedule and realistic execution constraints, and permits a paper signal only when the remaining edge clears configured gates.

**Status convention**
- `[x]` = implemented and covered by the research/safety code path.
- Evidence requirements are listed separately. Implemented does **not** mean proven profitable or approved for live trading.

## Milestone 1 — SAFE (P0)

- [x] Load V2 risk/profile configuration from YAML.
- [x] Enforce a bounded entry window (research default: 90–150 seconds remaining).
- [x] Validate spread for the selected token, not the minimum spread across both outcomes.
- [x] Enforce selected-side spread and depth limits before a paper signal can pass.
- [ ] Remove permissive execution defaults (`PM_MAX_SPREAD=1`, `PM_MIN_TOP_ASK_NOTIONAL_USD=0`) from the legacy live execution path before any future live adapter is enabled.
- [x] Define V2 position marks from executable CLOB best bid rather than Gamma outcome price.
- [x] Add quote-staleness and consecutive-data-error circuit breakers.
- [x] Enforce paper-session limits: max daily loss, max trades/day, and consecutive-loss stop.
- [x] Disable automatic hedge by default.
- [x] Make V2 dry-run/paper mode the controller default and reject V2 `--execute` during research.
- [x] Add tests for current safety gates.
- [x] Fail closed if the BTC 5-minute market resolution source is missing or differs from the configured Chainlink BTC/USD 60-second TWAP source.

## Milestone 2 — DATA (P0)

- [x] Polymarket public market WebSocket feed.
- [x] Public Polymarket RTDS ingestion of Chainlink-computed BTC/USD TWAP observations.
- [x] Prefer exact signed-E18 `full_accuracy_value` when available.
- [x] Record exchange/source timestamps and local receive timestamps for causal joins.
- [x] Record UP/DOWN bid, ask, spread, and top-3 depth.
- [x] Record BTC reference, current value, delta, 15s/30s/60s momentum, realized volatility, and impulse-Z.
- [x] Verify BTC sample provenance against each market's declared resolution source and fail closed on mismatch.
- [x] Store raw WebSocket events in append-only JSONL and normalized snapshots in Zstd Parquet.
- [x] Attach final market resolution through the market WebSocket, with post-close Gamma enrichment for unresolved recordings.
- [x] Fetch and persist the market-specific CLOB fee schedule.
- [x] Require clean/no-reconnect recordings, a verified market-start BTC reference, resolution, fee metadata, and decision-window coverage before a market is V2-ready.
- [x] Build a ready-only aggregate Parquet dataset.
- [x] Add public WebSocket, TWAP, fee, recorder, and end-to-end evidence workflows with no wallet/order path.

## Milestone 3 — EDGE (P1)

- [x] Reproduce the original `ask >= 0.70` stronger-side strategy as the control benchmark.
- [x] Add fee-aware legacy control metrics without changing the legacy signal rule.
- [x] Add executable L2-book paper replay with VWAP/slippage, latency, and visible-liquidity participation limits.
- [x] Add volatility-normalized BTC impulse.
- [x] Add causal 5s/15s/30s/60s market-price and BTC momentum features.
- [x] Add order-book imbalance and microstructure features.
- [x] Build an interpretable probability-model path with explicit anti-label-leakage checks.
- [x] Calculate fee-adjusted break-even probability using captured market-specific fee rate/exponent.
- [x] Estimate executable VWAP/slippage for intended order size.
- [x] Require model probability minus executable all-in cost to exceed the configured minimum edge.

## Milestone 4 — PAPER / OUT-OF-SAMPLE (P1)

- [x] Shared causal feature/signal logic for replay and online paper paths.
- [x] Paper executor with executable book fills, fees, slippage, latency assumptions, and participation limits.
- [x] Expanding-window chronological walk-forward evaluation with past-only scaler/model fitting.
- [x] Compare V2 model and legacy strategy on the same held-out market blocks.
- [x] Report trades, win rate, fees, all-in cost, net P&L, return on cost, mean P&L/trade, and max drawdown.
- [ ] Accumulate at least 500 independent forward-paper signals before any live-sizing discussion.
- [ ] Demonstrate positive fee/slippage-inclusive out-of-sample performance across multiple volatility/time regimes.
- [ ] Review confidence intervals / robustness after sufficient independent markets are collected.

## Evidence gate — CURRENT BOTTLENECK

The primary remaining blocker is **evidence volume**, not missing architecture.

Before any live review:
1. Collect enough fully V2-ready markets with clean CLOB + settlement-aligned BTC TWAP + exact reference + fee metadata + final resolution.
2. Keep chronological market boundaries intact; never random-split rows from the same 5-minute market across train/test.
3. Run the executable walk-forward paper comparison across multiple regimes.
4. Reach at least 500 independent forward-paper signals.
5. Require net results after fees, spread/slippage, modeled latency, and fill/participation constraints.
6. Keep `live_trading_approved = false` unless the evidence gate is explicitly reviewed later.

## Verification record

- Safety CI: passing.
- Research CI: passing through commit `362fa23e21ea4a5467750c2fe410b518083047a5` on 2026-09-22.
- Public CLOB / recorder smoke workflows: passing.
- Public RTDS TWAP feed path implemented with causal receive timestamps and exact-value parsing.
- Polymarket BTC 5-minute market pages re-verified on **2026-09-22**: resolution source remains `https://data.chain.link/streams/btc-usd-twap-60s-streams`.
- Market-specific fee metadata is persisted and now carried into V2 dataset rows as `fees_enabled`, `fee_rate`, and `fee_exponent`.
- Legacy threshold benchmark now has explicit gross and fee-aware controls; executable paper replay remains the stronger realism benchmark.
- No wallet credentials are required for V2 research collection and no live-order path is approved.

## Live execution policy

Live execution remains intentionally disabled. No wallet private key, API secret, or `.env` file should be committed to this repository. A future live adapter must be a separate reviewed step, explicitly enabled, legally accessible to the operator, and gated by the complete safety/data/evidence contract above.
