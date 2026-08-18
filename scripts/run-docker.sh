#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the StockTradingBot Docker container.
#
# Modes (default: dashboard):
#     ./scripts/run-docker.sh                     # start dashboard on :8501
#     ./scripts/run-docker.sh dry-run            # one-shot dry-run engine
#     ./scripts/run-docker.sh live               # live engine (interactive)
#     ./scripts/run-docker.sh backtest           # run a backtest
#     ./scripts/run-docker.sh bash               # drop into a shell
#
# All modes mount ./logs and ./data into the container so runtime state is
# preserved on the host.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-stocktradingbot}"
TAG="${TAG:-latest}"
MODE="${1:-ui}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Ensure host-side log/data dirs exist (mount target must exist).
mkdir -p logs data reports

# Common args: remove container on exit (-rm), mount host dirs, expose port.
COMMON_ARGS=(
    --rm
    -p 8501:8501
    -v "${REPO_ROOT}/logs:/app/logs"
    -v "${REPO_ROOT}/data:/app/data"
    -v "${REPO_ROOT}/reports:/app/reports"
)

# Load env file if present (best-effort; docker-compose is the richer path).
if [[ -f .env ]]; then
    COMMON_ARGS+=(--env-file .env)
fi

case "${MODE}" in
    ui|dashboard)
        echo "Starting Streamlit dashboard on http://localhost:8501 ..."
        docker run "${COMMON_ARGS[@]}" "${IMAGE_NAME}:${TAG}" \
            python main.py ui --port 8501
        ;;
    dry-run)
        echo "Starting dry-run engine (Ctrl-C to stop) ..."
        docker run "${COMMON_ARGS[@]}" -it "${IMAGE_NAME}:${TAG}" \
            python main.py dry-run
        ;;
    live)
        echo "Starting LIVE engine (interactive — confirm strategy at startup) ..."
        docker run "${COMMON_ARGS[@]}" -it "${IMAGE_NAME}:${TAG}" \
            python main.py live
        ;;
    backtest)
        echo "Running backtest ..."
        docker run "${COMMON_ARGS[@]}" "${IMAGE_NAME}:${TAG}" \
            python main.py backtest
        ;;
    bash|shell)
        echo "Opening shell in ${IMAGE_NAME}:${TAG} ..."
        docker run "${COMMON_ARGS[@]}" -it --entrypoint bash "${IMAGE_NAME}:${TAG}"
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Modes: ui | dry-run | live | backtest | bash" >&2
        exit 2
        ;;
esac