# BTC5M V2 Local Setup (Paper / Research Only)

This setup is intentionally isolated from wallet credentials and live execution.

## 1. Clone your fork and use the research branch

```bash
git clone https://github.com/rodolfosuperagentgmailcom/5min-btc-polymarket.git
cd 5min-btc-polymarket
git checkout btc5m-v2-research
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

## 5. Record public Polymarket CLOB data

```bash
PYTHONPATH=. python scripts/btc5m_v2_record.py \
  --duration-sec 1800 \
  --output-root runtime/data
```

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
      btc_samples.parquet          # only from verified settlement-aligned source
      btc_metadata.json            # provenance/reference metadata for BTC samples
```

The recorder uses public market data only. No trading keys are required.

## 6. Enrich completed markets with final results

```bash
PYTHONPATH=. python scripts/btc5m_v2_resolve.py --output-root runtime/data
```

This fills missing terminal UP/DOWN labels for recordings that ended before the WebSocket delivered `market_resolved`.

## 7. Enrich each market with its actual fee schedule

```bash
PYTHONPATH=. python scripts/btc5m_v2_enrich_fees.py --output-root runtime/data
```

The fee-aware paper tests fail closed when fee metadata is missing unless a research command explicitly allows missing fees.

## 8. Settlement-aligned BTC sample contract

The current market resolution source is configured as:

```text
https://data.chain.link/streams/btc-usd-twap-60s-streams
```

V2 will populate BTC features only if a market directory contains:

```text
btc_samples.parquet
btc_metadata.json
```

`btc_samples.parquet` must contain:

```text
ts_ns   int64 nanosecond timestamp
price   BTC/USD settlement-aligned observed value
```

`btc_metadata.json` must identify the exact source and may include the market reference price:

```json
{
  "source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
  "reference_price": 100000.0,
  "verified": true
}
```

The dataset builder compares `btc_metadata.source` with the recorded market's `resolution_source`. A mismatch fails closed.

**Do not manually relabel Binance, Coinbase, or the public Chainlink CEX-price stream as this source.** The public Chainlink discovery catalog did not expose a verified 60-second TWAP feed ID, so authenticated settlement-aligned report ingestion remains a separate pending integration.

Without verified BTC samples, the microstructure research/control pipeline still works, while BTC feature columns remain null.

## 9. Build the unified causal dataset

```bash
PYTHONPATH=. python scripts/btc5m_v2_build_dataset.py \
  --input-root runtime/data \
  --output runtime/research/btc5m_features.parquet \
  --sample-interval-ms 1000
```

The dataset includes:
- CLOB midpoint / normalized market probability;
- UP/DOWN spread and top-book depth;
- order-book imbalance, overround, and market skew;
- causal 5s/15s/30s/60s market-probability and midpoint changes;
- verified BTC reference/current/delta, 15s/30s/60s momentum, realized movement, and impulse-Z when approved BTC samples are present;
- final resolution labels attached only after feature construction.

Future CLOB or BTC samples are not used to create past feature rows.

## 10. Run the original `.70` control benchmark

Gross-only control:

```bash
PYTHONPATH=. python scripts/btc5m_v2_benchmark.py \
  --dataset runtime/research/btc5m_features.parquet \
  --threshold 0.70 \
  --min-seconds-left 60 \
  --output runtime/research/legacy_threshold_benchmark.json
```

Realistic paper replay with raw book depth, fee schedules, participation limits, and multiple latency assumptions:

```bash
PYTHONPATH=. python scripts/btc5m_v2_paper_benchmark.py \
  --input-root runtime/data \
  --threshold 0.70 \
  --notional-usd 5 \
  --max-participation-pct 20 \
  --latency-ms 0 100 250 500 1000
```

Use the realistic paper result—not the gross benchmark—for execution-sensitive conclusions.

## 11. Train the probability baseline

### Full model

Requires verified settlement-aligned BTC features:

```bash
PYTHONPATH=. python scripts/btc5m_v2_train_model.py \
  --dataset runtime/research/btc5m_features.parquet \
  --feature-set full \
  --minimum-markets 100 \
  --output runtime/models/btc5m_logit_v1.json
```

### Microstructure control model

Can be used before the verified Chainlink TWAP sample stream is available:

```bash
PYTHONPATH=. python scripts/btc5m_v2_train_model.py \
  --dataset runtime/research/btc5m_features.parquet \
  --feature-set microstructure \
  --minimum-markets 100 \
  --output runtime/models/btc5m_microstructure_logit_v1.json
```

Training rules:
- one decision row per independent 5-minute market, nearest 120 seconds remaining;
- only rows in the 90–150 second research window;
- chronological train/test split, never random row splitting;
- scaler fit on training markets only;
- terminal-label columns forbidden as model inputs;
- Brier score compared directly with Polymarket's own implied probability.

A trained model is a research artifact, not evidence of tradable edge.

## 12. Compare V2 with the legacy strategy on the same held-out markets

Full feature comparison once verified BTC features exist:

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

Until then, the microstructure control can be tested with:

```bash
PYTHONPATH=. python scripts/btc5m_v2_compare_model.py \
  --dataset runtime/research/btc5m_features.parquet \
  --data-root runtime/data \
  --feature-set microstructure \
  --minimum-markets 100 \
  --notional-usd 5 \
  --latency-ms 100
```

The comparison:
- trains V2 on earlier markets only;
- evaluates on later held-out markets only;
- evaluates V2 and legacy `.70` on the same held-out market set;
- applies actual recorded fee schedules;
- reconstructs the raw order book;
- enforces visible-liquidity and participation limits;
- includes modeled signal-to-fill latency;
- reports net settlement P&L after taker fees.

The model also must clear the V2 safety gates and a multi-level executable ask-stack edge check. A top-of-book theoretical edge is not enough.

## 13. Evidence threshold before any live review

Do not add wallet keys or Polymarket trading API secrets for this phase.

At minimum, require:
- a substantial set of independent resolved 5-minute markets (the project target is 500+ forward-paper signals before live review);
- positive out-of-sample net P&L after fees, latency, and slippage;
- probability calibration that is competitive with or better than the market;
- stable results across time and volatility regimes;
- acceptable drawdown and no single-period concentration of profits.

If those conditions are not met, the correct V2 decision is **NO TRADE**.
