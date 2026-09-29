#!/usr/bin/env bash
# Start the Forward & Backward pass demo, served through a Cloudflare quick tunnel.
# Meant for a machine you reach over SSH: the app is used from the public URL.
#
#   ./start.sh            serve + tunnel in the foreground (Ctrl+C stops both)
#   ./start.sh --detach   same, in the background; survives closing the SSH session
#   ./start.sh --status   show whether the detached app is running, and its URL
#   ./start.sh --stop     stop the detached app
#   ./start.sh --local    Flask dev server only, no tunnel (PORT, default 8080)
set -euo pipefail
cd "$(dirname "$0")"

RUN_DIR=".run"
PID_FILE="$RUN_DIR/tunnel.pid"
URL_FILE="$RUN_DIR/tunnel-url"
LOG_FILE="$RUN_DIR/tunnel.log"

running_pid() {
  local pid
  [[ -f "$PID_FILE" ]] || return 1
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then echo "$pid"; else return 1; fi
}

case "${1:-}" in
  ""|--tunnel)
    exec ./tunnel.sh
    ;;

  --detach)
    if pid=$(running_pid); then
      echo "Already running (pid $pid): $(cat "$URL_FILE" 2>/dev/null || echo 'URL not ready yet')"
      exit 0
    fi
    mkdir -p "$RUN_DIR"
    rm -f "$URL_FILE"
    # setsid puts the app in its own session so it outlives the SSH login;
    # nohup is the fallback where setsid is missing (macOS).
    if command -v setsid >/dev/null 2>&1; then
      setsid nohup ./tunnel.sh >"$LOG_FILE" 2>&1 < /dev/null &
    else
      nohup ./tunnel.sh >"$LOG_FILE" 2>&1 < /dev/null &
    fi
    echo $! > "$PID_FILE"
    echo "Starting in the background (log: $LOG_FILE)..."
    for _ in $(seq 1 120); do
      if [[ -s "$URL_FILE" ]]; then
        echo "MLP Training is live at: $(cat "$URL_FILE")"
        echo "A new URL can take up to a minute to start resolving."
        echo "Check with ./start.sh --status, stop with ./start.sh --stop"
        exit 0
      fi
      if ! running_pid >/dev/null; then
        echo "Error: failed to start. Log:" >&2
        cat "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
      fi
      sleep 0.5
    done
    echo "Still starting after 60s; check $LOG_FILE" >&2
    exit 1
    ;;

  --status)
    if pid=$(running_pid); then
      echo "Running (pid $pid): $(cat "$URL_FILE" 2>/dev/null || echo 'URL not ready yet')"
    else
      echo "Not running."
      exit 1
    fi
    ;;

  --stop)
    if pid=$(running_pid); then
      # TERM lets tunnel.sh's trap stop gunicorn and cloudflared cleanly.
      kill -TERM "$pid"
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      rm -f "$PID_FILE" "$URL_FILE"
      echo "Stopped."
    else
      rm -f "$PID_FILE"
      echo "Not running."
    fi
    ;;

  --local)
    if ! command -v uv >/dev/null 2>&1; then
      echo "Error: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
      exit 1
    fi
    export PORT="${PORT:-8080}"
    echo "Local only (no tunnel): http://127.0.0.1:${PORT}  —  Ctrl+C to stop"
    exec uv run python app.py
    ;;

  *)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
