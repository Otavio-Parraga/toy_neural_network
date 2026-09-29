#!/usr/bin/env bash
# Run the app with gunicorn and expose it through a Cloudflare quick tunnel.
# cloudflared prints a public https://<random>.trycloudflare.com URL.
set -euo pipefail
cd "$(dirname "$0")"

RUN_DIR=".run"            # URL file and logs (gitignored)
URL_FILE="$RUN_DIR/tunnel-url"
mkdir -p "$RUN_DIR"
rm -f "$URL_FILE"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Error: cloudflared is not installed.

Install it with one of:
  Debian/Ubuntu (.deb):
    curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared.deb
  macOS (Homebrew):
    brew install cloudflared
  Standalone binary (Linux amd64, no root needed):
    curl -L -o ~/.local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x ~/.local/bin/cloudflared

All releases: https://github.com/cloudflare/cloudflared/releases
MSG
  exit 1
fi

SERVER_PID=""
TUNNEL_PID=""
cleanup() {
  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap 'cleanup; rm -f "$URL_FILE"' EXIT
trap 'exit 130' INT TERM

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required (used to check the port and wait for the server)." >&2
  exit 1
fi

HEALTH_MARKER='"app":"toy-neural-network"'

# A port is free when nothing answers on it (curl exit code 7 = could not
# connect). Never reuse a busy port: the readiness check below would succeed
# against that service and cloudflared would publish it to the internet.
port_free() {
  local rc=0
  curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/" 2>/dev/null || rc=$?
  [[ "$rc" -eq 7 ]]
}

if [[ -n "${PORT:-}" ]]; then
  if ! port_free "$PORT"; then
    echo "Error: port ${PORT} on 127.0.0.1 is already in use by another process." >&2
    echo "Stop that process, choose another port (PORT=8090 $0), or unset PORT to pick one automatically." >&2
    exit 1
  fi
else
  # No port requested: take the first free one from 8090 upwards.
  PORT=""
  for p in $(seq 8090 8190); do
    if port_free "$p"; then PORT="$p"; break; fi
  done
  if [[ -z "$PORT" ]]; then
    echo "Error: no free port found in 8090-8190; set PORT explicitly." >&2
    exit 1
  fi
fi
HEALTH_URL="http://127.0.0.1:${PORT}/healthz"

# Exactly one worker: each visitor's network lives in an in-memory store
# inside the server process, so a second worker would not see it.
echo "Starting server on http://127.0.0.1:${PORT} ..."
uv run gunicorn app:app --bind "127.0.0.1:${PORT}" --workers 1 &
SERVER_PID=$!

# Wait (up to ~30s) until THIS app answers its health check.
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Error: server exited during startup." >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "$HEALTH_URL" 2>/dev/null | tr -d ' \n' | grep -qF "$HEALTH_MARKER"; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "$ready" -ne 1 ]]; then
  echo "Error: server did not become ready on port ${PORT}; not starting the tunnel." >&2
  exit 1
fi
# The health check answered, but make sure it was our server that did.
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Error: server exited during startup (is port ${PORT} in use?)." >&2
  exit 1
fi

echo "Starting Cloudflare quick tunnel..."
# 127.0.0.1 rather than localhost: localhost may resolve to ::1, where a
# different service could be listening. cloudflared logs to a file so the
# public URL can be picked out and shown instead of buried in its output.
TUNNEL_LOG="$RUN_DIR/cloudflared.log"
: > "$TUNNEL_LOG"
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

url=""
for _ in $(seq 1 60); do
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "Error: cloudflared exited. Its log:" >&2
    cat "$TUNNEL_LOG" >&2
    exit 1
  fi
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -n1 || true)
  [[ -n "$url" ]] && break
  sleep 0.5
done
if [[ -z "$url" ]]; then
  echo "Error: no trycloudflare.com URL after 30s. cloudflared log:" >&2
  cat "$TUNNEL_LOG" >&2
  exit 1
fi
echo "$url" > "$URL_FILE"
cat <<BANNER

============================================================
  MLP Training is live at:

    $url

  (URL saved in $URL_FILE; cloudflared log: $TUNNEL_LOG)
  A new URL can take up to a minute to start resolving.
  Stop: Ctrl+C
============================================================

BANNER

# Stay up until either process exits, then tear both down (via the EXIT trap).
# Background + wait keeps the INT/TERM traps responsive.
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
  wait -n || true
else
  # bash < 4.3 (e.g. macOS's /bin/bash 3.2) has no `wait -n`: poll instead.
  while kill -0 "$SERVER_PID" 2>/dev/null && kill -0 "$TUNNEL_PID" 2>/dev/null; do
    sleep 1
  done
fi

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Server exited; stopping the tunnel." >&2
else
  echo "Tunnel exited; stopping the server." >&2
fi
exit 1
