#!/usr/bin/env bash
# =============================================================================
# run_evaluation.sh
#
# Main evaluation orchestration script for DFToolBench-I.
# Starts the tool server and bridge server, runs OpenCompass evaluation,
# then gracefully shuts down all background services.
#
# Usage:
#   bash scripts/run_evaluation.sh [OPTIONS]
#
# Options:
#   --model MODEL        Model identifier string (required)
#                        e.g. gpt-4o, claude-4-0-sonnet, llama-3-70b
#   --mode  MODE         Evaluation mode: react | e2e  (default: react)
#   --config CONFIG      Path to OpenCompass config    (default: auto-selected)
#   --output-dir DIR     Directory for results         (default: ./outputs/<timestamp>)
#   --device DEVICE      Torch device for tools        (default: cuda:0)
#   --tool-port PORT     Tool server port              (default: 5000)
#   --bridge-port PORT   Bridge server port            (default: 5001)
#   --num-workers N      Tool server gunicorn workers  (default: 4)
#   --skip-server        Skip starting servers (use already-running ones)
#   --dry-run            Print config and exit without running evaluation
#
# Environment variables:
#   OPENAI_API_KEY          Required for OpenAI models
#   ANTHROPIC_API_KEY       Required for Anthropic models
#   GOOGLE_API_KEY          Required for Google models
#   DFTOOLBENCH_TOOL_ROOT   Path to tool checkpoints & CLI scripts
#
# Exit codes:
#   0 — evaluation completed successfully
#   1 — configuration/argument error
#   2 — server startup failure
#   3 — evaluation failure
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
MODEL=""
MODE="react"
CONFIG=""
OUTPUT_DIR=""
DEVICE="${DFTOOLBENCH_DEVICE:-cuda:0}"
TOOL_PORT="${TOOL_SERVER_PORT:-5000}"
BRIDGE_PORT="${BRIDGE_PORT:-5001}"
NUM_WORKERS=4
SKIP_SERVER=false
DRY_RUN=false

TOOL_SERVER_PID=""
BRIDGE_SERVER_PID=""

# --------------------------------------------------------------------------- #
# Parse CLI arguments
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --mode)         MODE="$2";         shift 2 ;;
        --config)       CONFIG="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --tool-port)    TOOL_PORT="$2";    shift 2 ;;
        --bridge-port)  BRIDGE_PORT="$2";  shift 2 ;;
        --num-workers)  NUM_WORKERS="$2";  shift 2 ;;
        --skip-server)  SKIP_SERVER=true;  shift   ;;
        --dry-run)      DRY_RUN=true;      shift   ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "        Run: bash scripts/run_evaluation.sh --help" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------- #
# Validate required arguments
# --------------------------------------------------------------------------- #
if [[ -z "${MODEL}" ]]; then
    echo "[ERROR] --model is required." >&2
    echo "  Example: bash scripts/run_evaluation.sh --model gpt-4o --mode react" >&2
    exit 1
fi

if [[ "${MODE}" != "react" && "${MODE}" != "e2e" ]]; then
    echo "[ERROR] --mode must be 'react' or 'e2e', got: ${MODE}" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Auto-select OpenCompass config if not provided
# --------------------------------------------------------------------------- #
if [[ -z "${CONFIG}" ]]; then
    case "${MODE}" in
        react) CONFIG="${REPO_ROOT}/configs/eval_configs/react_agent.py" ;;
        e2e)   CONFIG="${REPO_ROOT}/configs/eval_configs/e2e_direct.py"  ;;
    esac
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG}" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Set output directory
# --------------------------------------------------------------------------- #
if [[ -z "${OUTPUT_DIR}" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${REPO_ROOT}/outputs/${MODEL}_${MODE}_${TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"

# --------------------------------------------------------------------------- #
# Print configuration summary
# --------------------------------------------------------------------------- #
echo "============================================================"
echo " DFToolBench-I  —  Evaluation"
echo "------------------------------------------------------------"
echo "  Model      : ${MODEL}"
echo "  Mode       : ${MODE}"
echo "  Config     : ${CONFIG}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Device     : ${DEVICE}"
echo "  Tool port  : ${TOOL_PORT}"
echo "  Bridge port: ${BRIDGE_PORT}"
echo "  Workers    : ${NUM_WORKERS}"
echo "============================================================"

if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY RUN] Exiting without running evaluation."
    exit 0
fi

# --------------------------------------------------------------------------- #
# Cleanup function — called on EXIT to stop background servers
# --------------------------------------------------------------------------- #
cleanup() {
    local exit_code=$?
    echo ""
    echo "[INFO] Cleaning up background services..."

    if [[ -n "${BRIDGE_SERVER_PID}" ]] && kill -0 "${BRIDGE_SERVER_PID}" 2>/dev/null; then
        echo "[INFO] Stopping bridge server (PID ${BRIDGE_SERVER_PID})"
        kill "${BRIDGE_SERVER_PID}" 2>/dev/null || true
        wait "${BRIDGE_SERVER_PID}" 2>/dev/null || true
    fi

    if [[ -n "${TOOL_SERVER_PID}" ]] && kill -0 "${TOOL_SERVER_PID}" 2>/dev/null; then
        echo "[INFO] Stopping tool server (PID ${TOOL_SERVER_PID})"
        kill "${TOOL_SERVER_PID}" 2>/dev/null || true
        wait "${TOOL_SERVER_PID}" 2>/dev/null || true
    fi

    # Clean up PID files
    rm -f "${REPO_ROOT}/.tool_server.pid" "${REPO_ROOT}/.bridge_server.pid"

    echo "[INFO] Cleanup complete. Exit code: ${exit_code}"
    exit ${exit_code}
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# Activate conda environment
# --------------------------------------------------------------------------- #
if [[ "${CONDA_DEFAULT_ENV:-}" != "dftoolbench" ]]; then
    echo "[INFO] Activating conda env: dftoolbench"
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate dftoolbench 2>/dev/null || \
        echo "[WARN] Could not activate 'dftoolbench' conda env."
fi

# --------------------------------------------------------------------------- #
# Step 1: Start tool server
# --------------------------------------------------------------------------- #
if [[ "${SKIP_SERVER}" == false ]]; then
    echo ""
    echo "[STEP 1/4] Starting tool server on port ${TOOL_PORT}..."
    TOOL_SERVER_PORT="${TOOL_PORT}" \
    DFTOOLBENCH_DEVICE="${DEVICE}" \
    TOOL_SERVER_WORKERS="${NUM_WORKERS}" \
    bash "${SCRIPTS_DIR}/start_tool_server.sh" \
        > "${LOG_DIR}/tool_server.log" 2>&1 &
    TOOL_SERVER_PID=$!
    echo "[INFO] Tool server PID: ${TOOL_SERVER_PID}"

    # Wait for tool server to be ready
    echo "[INFO] Waiting for tool server to become ready..."
    MAX_WAIT=90
    ELAPSED=0
    until curl -sf "http://127.0.0.1:${TOOL_PORT}/health" > /dev/null 2>&1; do
        if [[ ${ELAPSED} -ge ${MAX_WAIT} ]]; then
            echo "[ERROR] Tool server failed to start within ${MAX_WAIT}s." >&2
            echo "        Check logs: ${LOG_DIR}/tool_server.log" >&2
            exit 2
        fi
        sleep 3
        ELAPSED=$((ELAPSED + 3))
    done
    echo "[INFO] Tool server is ready."

    # --------------------------------------------------------------------------- #
    # Step 2: Start bridge server
    # --------------------------------------------------------------------------- #
    echo ""
    echo "[STEP 2/4] Starting bridge server on port ${BRIDGE_PORT}..."
    BRIDGE_PORT="${BRIDGE_PORT}" \
    TOOL_SERVER_URL="http://127.0.0.1:${TOOL_PORT}" \
    bash "${SCRIPTS_DIR}/start_bridge.sh" \
        > "${LOG_DIR}/bridge_server.log" 2>&1 &
    BRIDGE_SERVER_PID=$!
    echo "[INFO] Bridge server PID: ${BRIDGE_SERVER_PID}"

    # Wait for bridge to be ready
    echo "[INFO] Waiting for bridge server to become ready..."
    ELAPSED=0
    until curl -sf "http://127.0.0.1:${BRIDGE_PORT}/health" > /dev/null 2>&1; do
        if [[ ${ELAPSED} -ge 60 ]]; then
            echo "[ERROR] Bridge server failed to start within 60s." >&2
            echo "        Check logs: ${LOG_DIR}/bridge_server.log" >&2
            exit 2
        fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done
    echo "[INFO] Bridge server is ready."
else
    echo "[INFO] --skip-server: assuming tool server and bridge are already running."
fi

# --------------------------------------------------------------------------- #
# Step 3: Run OpenCompass evaluation
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 3/4] Running OpenCompass evaluation..."
echo "           Model  : ${MODEL}"
echo "           Config : ${CONFIG}"
echo "           Output : ${OUTPUT_DIR}"

python -m opencompass \
    "${CONFIG}" \
    --model "${MODEL}" \
    --work-dir "${OUTPUT_DIR}" \
    --bridge-url "http://127.0.0.1:${BRIDGE_PORT}" \
    --mode "${MODE}" \
    2>&1 | tee "${OUTPUT_DIR}/opencompass.log"

EVAL_EXIT=$?
if [[ ${EVAL_EXIT} -ne 0 ]]; then
    echo "[ERROR] OpenCompass evaluation exited with code ${EVAL_EXIT}." >&2
    exit 3
fi

# --------------------------------------------------------------------------- #
# Step 4: Compute and display summary metrics
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 4/4] Computing summary metrics..."
python -m dftoolbench.evaluation.summarise \
    --run-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/scores_summary.csv" \
    2>&1 | tee "${OUTPUT_DIR}/scoring.log"

echo ""
echo "============================================================"
echo " Evaluation complete."
echo " Results : ${OUTPUT_DIR}/scores_summary.csv"
echo " Full log: ${OUTPUT_DIR}/opencompass.log"
echo "============================================================"
