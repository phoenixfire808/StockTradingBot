# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# StockTradingBot — multi-stage Docker image.
#
# Stage 1 (builder):  install build deps + Python packages into a virtualenv.
# Stage 2 (runtime):  slim image with the venv + source copied in. Exposes
#                     Streamlit on 8501 (the dashboard default).
#
# Build:
#     docker build -t stocktradingbot:latest .
#
# Run (one-shot dry-run):
#     docker run --rm -p 8501:8501 \
#         -v $(pwd)/logs:/app/logs \
#         -v $(pwd)/data:/app/data \
#         --env-file .env \
#         stocktradingbot:latest python main.py dry-run
#
# Run (dashboard):
#     docker run --rm -p 8501:8501 \
#         -v $(pwd)/logs:/app/logs \
#         -v $(pwd)/data:/app/data \
#         --env-file .env \
#         stocktradingbot:latest python main.py ui
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.13

# ── Stage 1: builder ────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

# System build deps for pandas/numpy wheels. Cleaned up before the runtime
# stage copies the venv out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps into a venv so we can copy it cleanly into the runtime stage.
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip wheel \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL maintainer="StockTradingBot" \
      org.opencontainers.image.source="https://github.com/your-org/stocktradingbot" \
      org.opencontainers.image.title="stocktradingbot" \
      org.opencontainers.image.description="Modular, agentic equity trading bot"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    STOCKTRADINGBOT_HOME=/app \
    LOG_LEVEL=INFO

# Runtime system deps only: tzdata for stable UTC handling + curl for healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tzdata \
        ca-certificates \
        curl \
 && ln -sf /usr/share/zoneinfo/UTC /etc/localtime \
 && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user for runtime safety.
RUN groupadd --system app \
 && useradd  --system --gid app --create-home --shell /bin/bash app \
 && mkdir -p /app/logs /app/data \
 && chown -R app:app /app

WORKDIR /app

# Copy the project source last to maximize Docker layer-cache hits.
COPY --chown=app:app . /app

USER app

# Streamlit default port.
EXPOSE 8501

# Lightweight healthcheck: a HEAD request against the Streamlit health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8501/_stcore/health || exit 1

# Default command launches the Streamlit dashboard. Override at `docker run`
# time with `python main.py live`, `python main.py dry-run`, etc.
CMD ["python", "main.py", "ui", "--port", "8501"]