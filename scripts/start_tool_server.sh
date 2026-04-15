#!/usr/bin/env bash
# =============================================================================
# start_tool_server.sh
#
# Launches the DFToolBench-I HTTP tool server.
# The server exposes all 12 forensic tools as individual REST endpoints,
# each running in its own conda environment via the worker-dispatch mechanism.
#
# Usage:
#   bash scripts/start_tool_server.sh [OPTIONS]
#
# Options:
#   --host HOST        Bind address            (default: 0.0.0.0)
#   --port PORT        Listening port          (default: 5000)
#   --device DEVICE    Torch device string     (default: cuda:0)
#   --workers N        Gunicorn worker count   (default: 4)
#   --log-level LEVEL  Logging level           (default: info)
#   --tool-root DIR    DFTOOLBENCH_TOOL_ROOT   (default: ./tool_root)
#
# Environment variables (may be set externally to override defaults):
#   DFTOOLBENCH_TOOL_ROOT  — root directory containing CLI scripts & checkpoints
#   DFTOOLBENCH_DEVICE     — compute device (e.g. cuda:0, cpu)
#   TOOL_SERVER_PORT       — override default port
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
HOST="${HOST:-0.0.0.0}"
PORT="${TOOL_SERVER_PORT:-5000}"
DEVICE="${DFTOOLBENCH_DEVICE:-cuda:0}"
WORKERS="${TOOL_SERVER_WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_ROOT="${DFTOOLBENCH_TOOL_ROOT:-${REPO_ROOT}/tool_root}"

# --------------------------------------------------------------------------- #
# Parse CLI arguments
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2";       shift 2 ;;
        --port)      PORT="$2";       shift 2 ;;
        --device)    DEVICE="$2";     shift 2 ;;
        --workers)   WORKERS="$2";    shift 2 ;;
        --log-level) LOG_LEVEL="$2";  shift 2 ;;
        --tool-root) TOOL_ROOT="$2";  shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------- #
# Export environment variables consumed by the tool implementations
# --------------------------------------------------------------------------- #
export DFTOOLBENCH_TOOL_ROOT="${TOOL_ROOT}"
export DFTOOLBENCH_DEVICE="${DEVICE}"

# --------------------------------------------------------------------------- #
# Validate environment
# --------------------------------------------------------------------------- #
if [[ ! -d "${TOOL_ROOT}" ]]; then
    echo "[WARN] DFTOOLBENCH_TOOL_ROOT does not exist: ${TOOL_ROOT}"
    echo "       Run: python scripts/download_checkpoints.py --tools all"
fi

# Activate base conda environment (dftoolbench) if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "dftoolbench" ]]; then
    echo "[INFO] Activating conda env: dftoolbench"
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate dftoolbench 2>/dev/null || {
        echo "[WARN] Could not activate 'dftoolbench' conda env. Proceeding with current Python."
    }
fi

# --------------------------------------------------------------------------- #
# Start server
# --------------------------------------------------------------------------- #
echo "============================================================"
echo " DFToolBench-I  —  Tool Server"
echo "------------------------------------------------------------"
echo "  Host       : ${HOST}:${PORT}"
echo "  Device     : ${DEVICE}"
echo "  Workers    : ${WORKERS}"
echo "  Tool root  : ${TOOL_ROOT}"
echo "  Log level  : ${LOG_LEVEL}"
echo "============================================================"

# Write PID file so the stop/cleanup script can find the process
PID_FILE="${REPO_ROOT}/.tool_server.pid"

exec gunicorn \
    --bind "${HOST}:${PORT}" \
    --workers "${WORKERS}" \
    --worker-class "sync" \
    --timeout 600 \
    --log-level "${LOG_LEVEL}" \
    --access-logfile "${REPO_ROOT}/logs/tool_server_access.log" \
    --error-logfile  "${REPO_ROOT}/logs/tool_server_error.log"  \
    --pid "${PID_FILE}" \
    "dftoolbench.utils.tool_server:create_app()"
