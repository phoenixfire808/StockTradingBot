#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the StockTradingBot Docker image.
#
# Usage:
#     ./scripts/build-docker.sh                    # build stocktradingbot:latest
#     ./scripts/build-docker.sh v1.2.3             # build stocktradingbot:v1.2.3
#     ./scripts/build-docker.sh latest --no-cache  # force rebuild
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-stocktradingbot}"
TAG="${1:-latest}"
NO_CACHE=""
if [[ "${2:-}" == "--no-cache" || "${1:-}" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Building ${IMAGE_NAME}:${TAG} from ${REPO_ROOT} ..."
docker build ${NO_CACHE} \
    -t "${IMAGE_NAME}:${TAG}" \
    -f Dockerfile \
    .

echo ""
echo "[OK] Image built: ${IMAGE_NAME}:${TAG}"
docker images "${IMAGE_NAME}:${TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"