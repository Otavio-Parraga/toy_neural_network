#!/usr/bin/env bash
# Run the app with gunicorn and expose it through a Cloudflare quick tunnel.
# cloudflared prints a public https://<random>.trycloudflare.com URL.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"

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
trap cleanup EXIT
trap 'exit 130' INT TERM

# Exactly one worker: the network state lives in a global dict in process memory.
echo "Starting server on http://127.0.0.1:${PORT} ..."
uv run gunicorn app:app --bind "127.0.0.1:${PORT}" --workers 1 &
SERVER_PID=$!

# Wait for the server to accept connections (up to ~15s).
ready=0
for _ in $(seq 1 30); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Error: server exited during startup." >&2
    exit 1
  fi
  if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "$ready" -ne 1 ]]; then
  echo "Warning: server did not respond on port ${PORT} yet; starting tunnel anyway." >&2
fi

echo "Starting Cloudflare quick tunnel (look for the trycloudflare.com URL below)..."
# Run in the background and wait, so INT/TERM traps fire immediately.
cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" &
TUNNEL_PID=$!
wait "$TUNNEL_PID"
