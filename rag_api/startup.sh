#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve script directory so it works from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
VENV_PATH="$SCRIPT_DIR/../mcp_env/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "❌ Virtualenv not found at $VENV_PATH"
  exit 1
fi
source "$VENV_PATH"

# Load runtime settings from config.json if present
CONFIG_FILE="$SCRIPT_DIR/config.json"
HOST="0.0.0.0"
PORT=8000
LOG_LEVEL="info"
RELOAD="false"
if command -v jq >/dev/null && [[ -f "$CONFIG_FILE" ]]; then
  HOST="$(jq -r '.host // "0.0.0.0"' "$CONFIG_FILE")"
  PORT="$(jq -r '.port // 8000' "$CONFIG_FILE")"
  LOG_LEVEL="$(jq -r '.log_level // "info"' "$CONFIG_FILE")"
  RELOAD="$(jq -r '.reload // false' "$CONFIG_FILE")"
fi

echo "🚀 Starting server on $HOST:$PORT (log_level=$LOG_LEVEL, reload=$RELOAD)"

# Ensure imports resolve relative to rag_api project
cd "$SCRIPT_DIR"

# Replace shell with uvicorn (better signal handling)
UVICORN_ARGS=(
  app.main:app
  --host "$HOST"
  --port "$PORT"
  --log-level "$LOG_LEVEL"
)

if [[ "$RELOAD" == "true" ]]; then
  UVICORN_ARGS+=(--reload)
fi

exec uvicorn "${UVICORN_ARGS[@]}"

