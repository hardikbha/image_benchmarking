#!/usr/bin/env bash
# =============================================================================
# start_bridge.sh
#
# Launches the DFToolBench-I bridge server.
# The bridge translates OpenCompass agent tool-call requests into HTTP calls
# to the tool server and returns results in the format OpenCompass expects.
#
# Architecture:
#   OpenCompass Agent  --[JSON-RPC/HTTP]--> Bridge Server  --[REST]--> Tool Server
#
# Usage:
#   bash scripts/start_bridge.sh [OPTIONS]
#
# Options:
#   --host HOST              Bridge bind address          (default: 127.0.0.1)
#   --port PORT              Bridge listening port        (default: 5001)
#   --tool-server-url URL    URL of the tool server       (default: http://127.0.0.1:5000)
#   --timeout SECONDS        Per-tool call timeout        (default: 120)
#   --log-level LEVEL        Logging level                (default: info)
#
# Environment variables:
#   BRIDGE_PORT              Override default bridge port
#   TOOL_SERVER_URL          Override default tool server URL
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
HOST="${BRIDGE_HOST:-127.0.0.1}"
PORT="${BRIDGE_PORT:-5001}"
TOOL_SERVER_URL="${TOOL_SERVER_URL:-http://127.0.0.1:5000}"
TIMEOUT="${BRIDGE_TIMEOUT:-120}"
LOG_LEVEL="${LOG_LEVEL:-info}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --------------------------------------------------------------------------- #
# Parse CLI arguments
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)              HOST="$2";             shift 2 ;;
        --port)              PORT="$2";             shift 2 ;;
        --tool-server-url)   TOOL_SERVER_URL="$2";  shift 2 ;;
        --timeout)           TIMEOUT="$2";          shift 2 ;;
        --log-level)         LOG_LEVEL="$2";        shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------- #
# Activate conda environment
# --------------------------------------------------------------------------- #
if [[ "${CONDA_DEFAULT_ENV:-}" != "dftoolbench" ]]; then
    echo "[INFO] Activating conda env: dftoolbench"
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate dftoolbench 2>/dev/null || {
        echo "[WARN] Could not activate 'dftoolbench' conda env. Proceeding with current Python."
    }
fi

# --------------------------------------------------------------------------- #
# Wait for the tool server to be ready before starting the bridge
# --------------------------------------------------------------------------- #
echo "[INFO] Waiting for tool server at ${TOOL_SERVER_URL}/health ..."
MAX_WAIT=60
ELAPSED=0
until curl -sf "${TOOL_SERVER_URL}/health" > /dev/null 2>&1; do
    if [[ ${ELAPSED} -ge ${MAX_WAIT} ]]; then
        echo "[ERROR] Tool server did not become ready within ${MAX_WAIT}s." >&2
        echo "        Start it first with: bash scripts/start_tool_server.sh" >&2
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "[INFO] Tool server is ready."

# --------------------------------------------------------------------------- #
# Create log directory
# --------------------------------------------------------------------------- #
mkdir -p "${REPO_ROOT}/logs"

# --------------------------------------------------------------------------- #
# Start bridge server
# --------------------------------------------------------------------------- #
PID_FILE="${REPO_ROOT}/.bridge_server.pid"

echo "============================================================"
echo " DFToolBench-I  —  Bridge Server"
echo "------------------------------------------------------------"
echo "  Bridge     : ${HOST}:${PORT}"
echo "  Tool server: ${TOOL_SERVER_URL}"
echo "  Timeout    : ${TIMEOUT}s"
echo "  Log level  : ${LOG_LEVEL}"
echo "============================================================"

exec python -m dftoolbench.bridge.server \
    --host "${HOST}" \
    --port "${PORT}" \
    --tool-server-url "${TOOL_SERVER_URL}" \
    --timeout "${TIMEOUT}" \
    --log-level "${LOG_LEVEL}" \
    --pid-file "${PID_FILE}" \
    2>&1 | tee "${REPO_ROOT}/logs/bridge_server.log"
