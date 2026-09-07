# BTC5M V2 Roadmap

This fork develops a safety-first, research-driven successor to the original 5-minute BTC Polymarket runner.

## Operating principle

The V2 signal should not buy simply because one side is already expensive. It should estimate fair settlement probability, compare that estimate with the executable all-in market price, subtract fees/slippage, and trade only when the remaining edge clears a configured threshold.

## Milestone 1 — SAFE (P0)

- [x] Load V2 risk/profile configuration from YAML.
- [x] Enforce a bounded entry window (research default: 90–150 seconds remaining).
- [x] Validate spread for the selected token, not the minimum spread across both outcomes.
- [x] Enforce selected-side spread and depth limits before a paper signal can pass.
- [ ] Remove permissive execution defaults (`PM_MAX_SPREAD=1`, `PM_MIN_TOP_ASK_NOTIONAL_USD=0`) from the legacy live execution path before any future live adapter is enabled.
- [x] Define V2 position marks from executable CLOB best bid rather than Gamma outcome price.
- [x] Add quote-staleness and consecutive-data-error circuit breakers.
- [ ] Enforce session limits in an executor: max daily loss, max trades/day, consecutive-loss stop. Values are configured now; execution remains disabled.
- [x] Disable automatic hedge by default.
- [x] Make V2 dry-run/paper mode the controller default and block V2 `--execute` entirely during research.
- [x] Add tests for current safety gates.
- [x] Fail closed if the BTC 5-minute market resolution source is missing or differs from the configured Chainlink BTC/USD 60-second TWAP source.

## Milestone 2 — DATA (P0)

- [x] Polymarket public market WebSocket feed.
- [x] Add Chainlink Data Streams discovery/authentication adapter primitives without hard-pinning an unverified feed ID.
- [ ] Ingest the verified settlement-aligned BTC/USD 60-second TWAP report stream. The public discovery catalog currently exposes a BTC/USD CEX-price stream, not the settlement TWAP feed.
- [x] Record exchange timestamps and local receive timestamps for CLOB events.
- [x] Record UP/DOWN bid, ask, spread, and top-3 depth.
- [ ] Record BTC reference, current value, delta, 15s/30s/60s momentum, and realized volatility from the verified settlement-aligned feed.
- [x] Store raw WebSocket events in append-only JSONL and normalized snapshots in fixed-schema Zstd Parquet parts.
- [x] Attach final market resolution to each 5-minute event when observed through the Polymarket market WebSocket, with post-close Gamma enrichment available for unresolved recordings.
- [x] Add live public WebSocket and end-to-end recorder smoke workflows with no credentials or order path.
- [x] Add a public Chainlink discovery smoke workflow that fails safely without substituting a non-TWAP BTC feed.

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

- [ ] Pure signal function shared by replay, paper, and any future live adapter.
- [ ] Paper executor with realistic fills, fees, slippage, and latency.
- [ ] Walk-forward/out-of-sample evaluation.
- [ ] Performance report: trades, win rate, gross/net P&L, fees, slippage, profit factor, max drawdown, calibration.
- [ ] Minimum 500 independent forward-paper signals before any live-sizing discussion.

## Verification record

- Safety CI: passing.
- Research CI: passing.
- Public Gamma/CLOB safety smoke: passing.
- Public CLOB WebSocket smoke: passing; observed exchange-to-receive age around 48 ms in the 2026-09-06 smoke run.
- End-to-end recorder smoke: passing; a 12-second public-data run recorded 2,718 raw events and 2,598 normalized snapshots and successfully reopened the generated Parquet.
- Chainlink public discovery smoke on 2026-09-06: passing. It discovered one live BTC/USD stream, `BTC/USD-Streams-CexPrice` (`0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8`), and zero 60-second TWAP candidates. That CEX-price feed is intentionally not used as the settlement feed.
- The exact Polymarket BTC 5-minute resolution source observed in the live market matched `https://data.chain.link/streams/btc-usd-twap-60s-streams`.
- No wallet credentials were loaded and no order path was used by V2 smoke/recorder workflows.

## Live execution policy

Live execution is intentionally disabled by default. No wallet private key, API secret, or `.env` file should be committed to this repository. Any eventual live adapter must require an explicit execution flag and pass every risk/data-quality gate.
