# BTC5M V2 Quickstart — Research / Paper Only

This V2 branch is intentionally isolated from wallet credentials and live execution.

## 1. Update your local copy

```bash
cd ~/trading/5min-btc-polymarket
git fetch origin
git checkout btc5m-v2-research
git pull --ff-only origin btc5m-v2-research
```

If this is the first setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-v2.txt
chmod +x scripts/btc5m_v2_ctl.sh
```

If the virtual environment already exists:

```bash
source .venv/bin/activate
```

## 2. Verify the code before collecting data

```bash
PYTHONPATH=. pytest -q tests/test_v2_*.py
```

Do not continue if the V2 tests fail.

## 3. Collect one fully aligned 5-minute research market

```bash
scripts/btc5m_v2_ctl.sh collect --markets 1
```

`collect` waits for the next 5-minute boundary and then records public data only:

- Polymarket UP/DOWN CLOB WebSocket events and reconstructed book snapshots;
- Polymarket RTDS BTC/USD Chainlink-computed 60-second TWAP observations;
- local receive timestamps and source timestamps;
- market-start BTC reference when it can be verified within tolerance;
- final resolution enrichment;
- market-specific fee enrichment;
- V2 readiness checks;
- ready-only aggregate Parquet dataset when at least one market passes every gate.

No wallet `.env` is loaded. No private key is required. There is no order-submission path in `collect`, and `--execute` is rejected by the V2 controller.

## 4. Check collection status and logs

```bash
scripts/btc5m_v2_ctl.sh status
scripts/btc5m_v2_ctl.sh logs
```

To stop a collection:

```bash
scripts/btc5m_v2_ctl.sh stop
```

## 5. Review readiness

After the collection finishes:

```bash
PYTHONPATH=. python scripts/btc5m_v2_readiness.py \
  --input-root runtime/data \
  --summary-only
```

A market is `v2_ready` only when all required research-integrity checks pass, including:

- uninterrupted CLOB capture (`reconnects == 0`);
- a CLOB snapshot inside the 90–150 second decision window;
- terminal UP/DOWN resolution;
- uninterrupted BTC/TWAP capture;
- BTC source matching the market's Chainlink resolution source;
- verified market-start BTC reference;
- market-specific fee schedule.

If a market is not ready, omit `--summary-only` to see the exact blocker.

## 6. Scale the collection

Once a one-market run produces clean artifacts, collect twelve consecutive 5-minute markets:

```bash
scripts/btc5m_v2_ctl.sh collect --markets 12
```

The ready-only aggregate dataset is written by default to:

```text
runtime/research/v2_ready.parquet
```

Study reports are written under:

```text
runtime/data/_study_reports/
```

## 7. Benchmark the original strategy

The original public runner's control logic is preserved as a benchmark: first UP/DOWN ask at or above `0.70`, choosing the stronger ask, with only the original minimum-time guard.

For execution-sensitive conclusions use the realistic paper benchmark, which reconstructs L2 books and applies visible liquidity, participation limits, latency assumptions, and recorded fee schedules:

```bash
PYTHONPATH=. python scripts/btc5m_v2_paper_benchmark.py \
  --input-root runtime/data \
  --threshold 0.70 \
  --notional-usd 5 \
  --max-participation-pct 20 \
  --latency-ms 0 100 250 500 1000
```

## 8. Train/evaluate only after enough ready markets exist

The branch already contains causal feature construction, model training, chronological walk-forward evaluation, fee-aware edge pricing, and paper simulation. Do not interpret a fitted model as proof of edge.

Before any live-review discussion, require a substantial independent sample and the project target of at least 500 forward-paper signals, with positive out-of-sample P&L after fees/latency/slippage and acceptable drawdown/calibration.

If those conditions are not met, the correct V2 action is **NO TRADE**.
