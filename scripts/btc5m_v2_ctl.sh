#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$ROOT/runtime/v2"
PAPER_RUNNER="$ROOT/scripts/btc5m_v2_paper_live.py"
BENCHMARK_RUNNER="$ROOT/scripts/btc5m_v2_runner.py"
PY="${BTC5M_V2_PYTHON:-$ROOT/.venv/bin/python}"

PIDFILE="$RUNTIME_DIR/btc5m_v2.pid"
LATEST_LINK="$RUNTIME_DIR/latest.log"

mkdir -p "$RUNTIME_DIR"

if [[ ! -x "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "Python not found. Create .venv and install requirements-v2.txt" >&2
    exit 2
  fi
fi

usage() {
  cat <<'EOF'
Usage:
  btc5m_v2_ctl.sh start [--model PATH] [--stake-usd N] [--paper-equity-usd N] [--session-minutes N]
  btc5m_v2_ctl.sh benchmark [--threshold N] [--stake-usd N] [--entry-timeout-min N] [--poll-sec N]
  btc5m_v2_ctl.sh status
  btc5m_v2_ctl.sh stop
  btc5m_v2_ctl.sh logs

V2 SAFETY POLICY:
- `start` runs the trained-model ONLINE PAPER bot only.
- `benchmark` runs the legacy .70-threshold paper observer for comparison only.
- This controller never loads a wallet .env file.
- `--execute` is rejected. There is no live order path in this controller.
EOF
}

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1
}

_start_process() {
  local mode="$1"
  local log="$2"
  shift 2

  if is_running; then
    echo "already_running pid=$(cat "$PIDFILE")"
    return 0
  fi

  (
    cd "$ROOT"
    nohup "$@" >"$log" 2>&1 &
    echo $! >"$PIDFILE"
  )
  ln -sfn "$log" "$LATEST_LINK"
  sleep 1
  if ! is_running; then
    echo "failed_to_start mode=$mode log=$log" >&2
    tail -n 80 "$log" >&2 || true
    rm -f "$PIDFILE"
    return 2
  fi
  echo "started pid=$(cat "$PIDFILE") mode=$mode log=$log"
}

start() {
  local model="$ROOT/runtime/models/btc5m_logit_v1.json"
  local stake="5"
  local equity="1000"
  local session_minutes="0"
  local target_seconds_left="120"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) model="$2"; shift 2;;
      --stake-usd) stake="$2"; shift 2;;
      --paper-equity-usd) equity="$2"; shift 2;;
      --session-minutes) session_minutes="$2"; shift 2;;
      --target-seconds-left) target_seconds_left="$2"; shift 2;;
      --execute) echo "LIVE EXECUTION BLOCKED: V2 controller is paper-only" >&2; exit 2;;
      *) echo "Unknown arg: $1" >&2; usage; exit 2;;
    esac
  done

  if [[ ! -f "$model" ]]; then
    echo "trained_model_missing: $model" >&2
    echo "Build the ready dataset, then run scripts/btc5m_v2_train_model.py." >&2
    exit 2
  fi

  local ts log
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$RUNTIME_DIR/v2_model_paper_${ts}.log"
  echo "Starting BTC5M V2 MODEL PAPER bot. No wallet credentials are loaded."
  _start_process "model-paper" "$log" \
    "$PY" "$PAPER_RUNNER" \
      --model "$model" \
      --stake-usd "$stake" \
      --paper-equity-usd "$equity" \
      --session-minutes "$session_minutes" \
      --target-seconds-left "$target_seconds_left"
}

benchmark() {
  local threshold="0.70"
  local stake="5"
  local timeout="35"
  local poll="2"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --threshold) threshold="$2"; shift 2;;
      --stake-usd) stake="$2"; shift 2;;
      --entry-timeout-min) timeout="$2"; shift 2;;
      --poll-sec) poll="$2"; shift 2;;
      --execute) echo "LIVE EXECUTION BLOCKED: V2 controller is paper-only" >&2; exit 2;;
      *) echo "Unknown arg: $1" >&2; usage; exit 2;;
    esac
  done

  local ts log
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$RUNTIME_DIR/v2_benchmark_${ts}.log"
  echo "Starting legacy threshold BENCHMARK in PAPER mode."
  _start_process "benchmark-paper" "$log" \
    "$PY" "$BENCHMARK_RUNNER" \
      --threshold "$threshold" \
      --stake-usd "$stake" \
      --entry-timeout-min "$timeout" \
      --poll-sec "$poll"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "$PIDFILE")"
    echo "running pid=$pid mode=paper"
    ps -p "$pid" -o pid=,etime=,command=
  else
    echo "stopped"
  fi
  [[ -L "$LATEST_LINK" ]] && echo "latest_log=$(readlink "$LATEST_LINK")"
}

stop() {
  if ! is_running; then
    echo "already_stopped"
    rm -f "$PIDFILE"
    return 0
  fi
  local pid
  pid="$(cat "$PIDFILE")"
  kill "$pid" || true
  sleep 1
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -9 "$pid" || true
  fi
  rm -f "$PIDFILE"
  echo "stopped pid=$pid"
}

logs() {
  if [[ -L "$LATEST_LINK" ]]; then
    tail -n 160 "$(readlink "$LATEST_LINK")"
  else
    echo "no_logs"
  fi
}

cmd="${1:-}"
[[ -z "$cmd" ]] && { usage; exit 2; }
shift || true
case "$cmd" in
  start) start "$@" ;;
  benchmark) benchmark "$@" ;;
  status) status ;;
  stop) stop ;;
  logs) logs ;;
  *) usage; exit 2;;
esac
