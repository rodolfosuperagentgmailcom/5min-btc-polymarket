#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$ROOT/runtime/v2"
RUNNER="$ROOT/scripts/btc5m_v2_runner.py"
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
  btc5m_v2_ctl.sh start [--threshold N] [--stake-usd N] [--entry-timeout-min N] [--poll-sec N]
  btc5m_v2_ctl.sh status
  btc5m_v2_ctl.sh stop
  btc5m_v2_ctl.sh logs

V2 SAFETY POLICY:
- Paper/data-observation mode only.
- This controller never loads a wallet .env file.
- Live execution is intentionally unavailable until the safety/research gates are complete.
EOF
}

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1
}

start() {
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
      --execute) echo "LIVE EXECUTION BLOCKED in BTC5M V2 safety phase" >&2; exit 2;;
      *) echo "Unknown arg: $1" >&2; usage; exit 2;;
    esac
  done

  if is_running; then
    echo "already_running pid=$(cat "$PIDFILE")"
    return 0
  fi

  local ts log
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$RUNTIME_DIR/v2_paper_${ts}.log"

  echo "Starting BTC5M V2 in PAPER mode. No wallet credentials are loaded."
  (
    cd "$ROOT"
    nohup "$PY" "$RUNNER" \
      --threshold "$threshold" \
      --stake-usd "$stake" \
      --entry-timeout-min "$timeout" \
      --poll-sec "$poll" \
      >"$log" 2>&1 &
    echo $! >"$PIDFILE"
  )
  ln -sfn "$log" "$LATEST_LINK"
  sleep 1
  echo "started pid=$(cat "$PIDFILE") mode=paper log=$log"
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
    tail -n 120 "$(readlink "$LATEST_LINK")"
  else
    echo "no_logs"
  fi
}

cmd="${1:-}"
[[ -z "$cmd" ]] && { usage; exit 2; }
shift || true
case "$cmd" in
  start) start "$@" ;;
  status) status ;;
  stop) stop ;;
  logs) logs ;;
  *) usage; exit 2;;
esac
