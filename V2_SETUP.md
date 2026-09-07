# BTC5M V2 Local Setup (Paper / Research Only)

This setup is intentionally isolated from wallet credentials and live execution.

## 1. Clone your fork and use the research branch

```bash
git clone https://github.com/rodolfosuperagentgmailcom/5min-btc-polymarket.git
cd 5min-btc-polymarket
git checkout btc5m-v2-research
```

If the repo already exists locally:

```bash
git fetch origin
git checkout btc5m-v2-research
git pull --ff-only origin btc5m-v2-research
```

## 2. Create a local virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-v2.txt
```

## 3. Run all V2 tests

```bash
PYTHONPATH=. pytest -q tests/test_v2_*.py
```

Do not continue if the V2 tests fail.

## 4. Optional paper observation

```bash
chmod +x scripts/btc5m_v2_ctl.sh
scripts/btc5m_v2_ctl.sh start --entry-timeout-min 35 --poll-sec 2
```

The V2 controller never loads wallet `.env` credentials, does not submit orders, and rejects `--execute`.

```bash
scripts/btc5m_v2_ctl.sh logs
scripts/btc5m_v2_ctl.sh stop
```

## 5. Record public CLOB + settlement-aligned BTC/TWAP data

```bash
PYTHONPATH=. python scripts/btc5m_v2_record.py \
  --duration-sec 1800 \
  --output-root runtime/data
```

Current V2 research capture combines:
- public Polymarket CLOB market WebSocket data for UP/DOWN books;
- the public Polymarket RTDS BTC/USD Chainlink-computed 60-second TWAP relay used by the research BTC adapter;
- local receive timestamps plus source timestamps for causal joins and latency analysis.

Recorder layout:

```text
runtime/data/
  YYYY-MM-DD/
    btc-updown-5m-.../
      clob_raw.jsonl
      clob_snapshots_part-000000.parquet
      metadata.json
      resolution.json              # once observed/enriched
      fees.json                    # after fee enrichment
      btc_raw.jsonl
      btc_samples.parquet
      btc_metadata.json
```

No trading keys are required. The research recorder does not contain an order path.

## 6. Enrich completed markets with final results

```bash
PYTHONPATH=. python scripts/btc5m_v2_resolve.py --output-root runtime/data
```

This fills missing terminal UP/DOWN labels for recordings that ended before the WebSocket delivered `market_resolved`.

## 7. Enrich each market with its actual fee schedule

```bash
PYTHONPATH=. python scripts/btc5m_v2_enrich_fees.py --output-root runtime/data
```

Fee-aware paper evaluation fails closed when fee metadata is missing.

## 8. Data-integrity contract

The configured settlement source is:

```text
https://data.chain.link/streams/btc-usd-twap-60s-streams
```

For CLOB data, `metadata.json` must explicitly show:

```json
{
  "reconnects": 0
}
```

For BTC/TWAP data, `btc_metadata.json` must identify the exact source and prove uninterrupted capture:

```json
{
  "source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
  "window_seconds": 60,
  "reference_price": 100000.0,
  "reference_verified": true,
  "reconnects": 0,
  "credentials_loaded": false,
  "order_path_used": false
}
```

V2 fails closed when:
- the CLOB recording reconnect count is missing, malformed, or nonzero;
- BTC samples exist but BTC recording continuity metadata is missing, malformed, or nonzero;
- the BTC source does not exactly match the recorded market settlement source;
- required fee metadata is missing for fee-aware paper replay;
- the market is unresolved when a terminal label is required.

A market-start BTC reference is accepted only when the recorder observes a source value within the configured boundary tolerance. If it cannot verify the reference, it remains null rather than being fabricated.

**Do not substitute Binance/Coinbase spot prices and label them as the Chainlink TWAP settlement source.**

## 9. Run the data-readiness preflight

Before building/training anything:

```bash
PYTHONPATH=. python scripts/btc5m_v2_readiness.py \
  --input-root runtime/data \
  --summary-only
```

For per-market blockers, omit `--summary-only`:

```bash
PYTHONPATH=. python scripts/btc5m_v2_readiness.py --input-root runtime/data
```

The report shows:
- markets seen;
- fully V2-ready markets;
- resolved markets;
- clean CLOB recordings;
- clean BTC recordings;
- verified BTC start references;
- fee-ready markets;
- blocker counts such as reconnects, unresolved markets, missing BTC samples, unverified references, source mismatch, or missing fees.

A market is `v2_ready` only when every required integrity/provenance check passes.

## 10. Build the unified causal dataset

```bash
PYTHONPATH=. python scripts/btc5m_v2_build_dataset.py \
  --input-root runtime/data \
  --output runtime/research/btc5m_features.parquet \
  --sample-interval-ms 1000
```

Dataset construction rejects reconnected CLOB recordings. If BTC samples are present, it also rejects BTC/TWAP recordings with reconnects.

The dataset includes:
- CLOB midpoint / normalized market probability;
- UP/DOWN spread and top-book depth;
- order-book imbalance, overround, and market skew;
- causal 5s/15s/30s/60s market-probability and midpoint changes;
- verified BTC reference/current/delta, 15s/30s/60s momentum, realized movement, and impulse-Z;
- explicit recording-continuity provenance;
- final resolution labels attached only after feature construction.

Future CLOB or BTC observations are not used to create past feature rows.

## 11. Run the original `.70` control benchmark

Gross-only control:

```bash
PYTHONPATH=. python scripts/btc5m_v2_benchmark.py \
  --dataset runtime/research/btc5m_features.parquet \
  --threshold 0.70 \
  --min-seconds-left 60 \
  --output runtime/research/legacy_threshold_benchmark.json
```

Realistic paper replay with raw book depth, market-specific fees, participation limits, recording-continuity checks, and multiple latency assumptions:

```bash
PYTHONPATH=. python scripts/btc5m_v2_paper_benchmark.py \
  --input-root runtime/data \
  --threshold 0.70 \
  --notional-usd 5 \
  --max-participation-pct 20 \
  --latency-ms 0 100 250 500 1000
```

Use the realistic paper result—not the gross benchmark—for execution-sensitive conclusions.

## 12. Train the probability baseline

### Full model

```bash
PYTHONPATH=. python scripts/btc5m_v2_train_model.py \
  --dataset runtime/research/btc5m_features.parquet \
  --feature-set full \
  --minimum-markets 100 \
  --output runtime/models/btc5m_logit_v1.json
```

### Microstructure control model

```bash
PYTHONPATH=. python scripts/btc5m_v2_train_model.py \
  --dataset runtime/research/btc5m_features.parquet \
  --feature-set microstructure \
  --minimum-markets 100 \
  --output runtime/models/btc5m_microstructure_logit_v1.json
```

Training rules:
- verified recording continuity is mandatory;
- one decision row per independent 5-minute market, nearest 120 seconds remaining;
- only rows in the 90–150 second research window;
- chronological train/test split, never random row splitting;
- scaler fit on training markets only;
- terminal-label columns forbidden as model inputs;
- Brier score compared directly with Polymarket's implied probability.

A trained model is a research artifact, not evidence of tradable edge.

## 13. Run expanding walk-forward paper evaluation

This is preferred over relying on one chronological split:

```bash
PYTHONPATH=. python scripts/btc5m_v2_walk_forward_paper.py \
  --dataset runtime/research/btc5m_features.parquet \
  --data-root runtime/data \
  --feature-set full \
  --minimum-train-markets 100 \
  --test-block-markets 50 \
  --notional-usd 5 \
  --latency-ms 100 \
  --max-participation-pct 20 \
  --output runtime/research/walk_forward_paper.json
```

This process:
- trains only on earlier markets;
- tests on later chronological blocks;
- refits the scaler/model inside each past-only training window;
- evaluates V2 and the legacy `.70` control on later markets;
- reconstructs recorded L2 books;
- models signal-to-fill latency;
- rejects unverified/reconnected recordings;
- enforces visible liquidity and participation limits;
- applies actual recorded fee schedules;
- reports settlement P&L after taker fees.

## 14. Optional single-split V2 vs legacy comparison

```bash
PYTHONPATH=. python scripts/btc5m_v2_compare_model.py \
  --dataset runtime/research/btc5m_features.parquet \
  --data-root runtime/data \
  --feature-set full \
  --minimum-markets 100 \
  --notional-usd 5 \
  --latency-ms 100 \
  --max-participation-pct 20 \
  --output runtime/research/model_vs_legacy.json
```

The walk-forward paper result should carry more weight than this single split.

## 15. Evidence threshold before any live review

Do not add wallet keys or trading API secrets for this phase.

At minimum, require:
- a substantial set of independent resolved, fully V2-ready 5-minute markets;
- the project target of 500+ forward-paper signals before live review;
- positive out-of-sample net P&L after fees, latency, and slippage;
- probability calibration competitive with or better than the market;
- stable performance across time and volatility regimes;
- acceptable drawdown and no single-period concentration of profits.

If those conditions are not met, the correct V2 decision is **NO TRADE**.
