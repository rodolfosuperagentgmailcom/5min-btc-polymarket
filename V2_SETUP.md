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

Expected result: all tests pass.

## 4. Start V2 paper observation

```bash
chmod +x scripts/btc5m_v2_ctl.sh
scripts/btc5m_v2_ctl.sh start --entry-timeout-min 35 --poll-sec 2
```

The V2 controller:
- never loads `.env` wallet credentials;
- does not submit orders;
- rejects `--execute`;
- reads `config/btc_5m_v2.yaml`;
- enforces the configured 90–150 second entry window;
- checks the selected outcome's CLOB spread and top-three ask depth;
- fails closed on stale/missing quote timestamps or repeated data errors.

Watch/stop it with:

```bash
scripts/btc5m_v2_ctl.sh logs
scripts/btc5m_v2_ctl.sh stop
```

## 5. Record public Polymarket market data

For a short test session:

```bash
PYTHONPATH=. python scripts/btc5m_v2_record.py --duration-sec 300 --output-root runtime/data
```

This records public market data only. It writes append-only raw JSONL, normalized Parquet CLOB snapshots, market metadata, and resolution information when available. No wallet/API trading credentials are loaded and no order path is used.

Recorder layout:

```text
runtime/data/
  YYYY-MM-DD/
    btc-updown-5m-.../
      clob_raw.jsonl
      clob_snapshots_part-000000.parquet
      metadata.json
      resolution.json   # when resolved/available
```

## 6. Enrich completed markets with final results

If a recording ended before the WebSocket delivered `market_resolved`, run:

```bash
PYTHONPATH=. python scripts/btc5m_v2_resolve.py --output-root runtime/data
```

## 7. Build the unified causal research dataset

```bash
PYTHONPATH=. python scripts/btc5m_v2_build_dataset.py \
  --input-root runtime/data \
  --output runtime/research/btc5m_features.parquet \
  --sample-interval-ms 1000
```

The dataset builder:
- samples each market causally at a minimum 1-second interval by default;
- computes UP/DOWN midpoints, normalized market probability, spread/depth and order-book imbalance features;
- computes causal 5s/15s/30s/60s midpoint/probability changes using only snapshots already known at that timestamp;
- attaches final UP/DOWN labels only after feature construction;
- reserves BTC reference/current/momentum/volatility/impulse columns as null until the verified settlement-aligned Chainlink BTC/USD 60-second TWAP report stream is available;
- never substitutes an unrelated spot/CEX feed for the settlement feed.

Output:

```text
runtime/research/btc5m_features.parquet
runtime/research/btc5m_features.parquet.json
```

The JSON sidecar reports row count, market count, resolved-row count, sampling interval, and whether settlement-aligned BTC features are populated.

## 8. Benchmark the actual legacy `.70` runner

Once you have resolved markets in the unified dataset:

```bash
PYTHONPATH=. python scripts/btc5m_v2_benchmark.py \
  --dataset runtime/research/btc5m_features.parquet \
  --threshold 0.70 \
  --min-seconds-left 60 \
  --output runtime/research/legacy_threshold_benchmark.json
```

This control benchmark intentionally mirrors the current public runner's behavior:
- first qualifying threshold signal per market;
- UP or DOWN qualifies when its ask is at least `.70`;
- if both qualify, it chooses the higher ask;
- it enforces only the legacy minimum 60 seconds remaining, with no upper entry-window bound;
- it holds the selected outcome to the final UP/DOWN label for the benchmark calculation.

The benchmark is explicitly **gross-only**. Fees, slippage, book participation, latency and fill probability are not yet included, so positive gross P&L must not be interpreted as a tradable edge.

## Safety rule

Do not add wallet keys or Polymarket trading API secrets for this phase. V2 remains paper/data-only until replay/backtesting, fee/slippage modeling, and forward-paper validation demonstrate a robust out-of-sample edge.
