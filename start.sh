#!/usr/bin/env bash
# Start the Forward & Backward pass demo locally.
#   ./start.sh            run the Flask dev server on $PORT (default 8080)
#   ./start.sh --tunnel   serve with gunicorn and expose it through a Cloudflare quick tunnel
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--tunnel" ]]; then
  exec ./tunnel.sh
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export PORT="${PORT:-8080}"
echo "============================================"
echo "  MLP Training — Forward & Backward Pass"
echo "  Open: http://127.0.0.1:${PORT}"
echo "  Stop: Ctrl+C"
echo "============================================"
exec uv run python app.py
