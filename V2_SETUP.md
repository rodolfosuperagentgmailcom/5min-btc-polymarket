# BTC5M V2 Local Setup (Paper Only)

This setup is intentionally isolated from wallet credentials and live execution.

## 1. Clone your fork

```bash
git clone https://github.com/rodolfosuperagentgmailcom/5min-btc-polymarket.git
cd 5min-btc-polymarket
git checkout btc5m-v2-safety
```

## 2. Create a local virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-v2.txt
```

## 3. Run the safety tests

```bash
PYTHONPATH=. pytest -q tests/test_v2_safety.py
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

## 5. Watch output

```bash
scripts/btc5m_v2_ctl.sh logs
```

Common statuses:
- `observe_no_threshold_candidate` — no side has reached the benchmark threshold.
- `paper_candidate_blocked` — a candidate exists but fails a V2 safety gate.
- `paper_signal` — benchmark candidate passed market safety gates; still no live order is placed.
- `circuit_breaker` — repeated data errors stopped the run.

## 6. Stop the observer

```bash
scripts/btc5m_v2_ctl.sh stop
```

## Safety rule

Do not add wallet keys or Polymarket API secrets for this phase. V2 remains paper/data-only until the market recorder, replay backtester, fee/slippage model, and forward-paper validation are complete.
