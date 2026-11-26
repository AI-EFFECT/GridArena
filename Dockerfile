# syntax=docker/dockerfile:1

# Containerizes the GridArena FastAPI app. This does not replace the local
# `uvicorn gridarena.app:app --reload` workflow described in the README — it
# packages the same app so it can also be run as a container.
#
# Build:  docker build -t gridarena .
#         (GPU build: docker build --build-arg TORCH_INDEX_URL= -t gridarena .)
# Run:    docker run -p 8000:8000 -v gridarena-config:/app/config gridarena
#         then open http://localhost:8000 — without a database configured the
#         app opens the Setup page to choose one (saved in /app/config/.env).
#         A database can also be passed directly:
#         -e GRID_DB_DSN=postgresql://user:pass@host.docker.internal:5432/gridarena
FROM python:3.11-slim

# CPU-only PyTorch by default: the CUDA wheels add several GB and need a GPU
# runtime. Pass an empty value to use the default (CUDA) wheels from PyPI.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# build-essential: compiles any dependency without a prebuilt wheel for this
# platform. libgomp1: OpenMP runtime required at import time by torch/numpy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, without the project itself (the `-e .` line), so this
# heavy layer stays cached when only the source code changes.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$TORCH_INDEX_URL" ]; then pip install --index-url "$TORCH_INDEX_URL" torch; fi \
    && grep -v '^-e ' requirements.txt > /tmp/requirements-deps.txt \
    && pip install -r /tmp/requirements-deps.txt

# Non-root user that owns the app directory (the app writes run artifacts,
# the RAG index and its settings file under /app).
RUN useradd --create-home --uid 1000 appuser

# Copy the repo (secrets, data sets, venvs and caches are excluded via
# .dockerignore) and install the project itself in editable mode, matching a
# normal local checkout.
COPY --chown=appuser:appuser . .
RUN pip install --no-deps -e . \
    && mkdir -p /app/config /app/data/llm \
       /app/gridarena/rl/runs /app/gridarena/diffusion/runs /app/gridarena/pf_datagen/runs \
       /app/gridarena/benchmarks/submissions/runs /app/gridarena/llm/rag \
    && chown -R appuser:appuser /app/config /app/data \
       /app/gridarena/rl/runs /app/gridarena/diffusion/runs /app/gridarena/pf_datagen/runs \
       /app/gridarena/benchmarks/submissions/runs /app/gridarena/llm/rag

# Settings saved from the Setup / Settings pages live on a volume so they
# survive container re-creation. Trained models and assistant data too.
ENV GRIDARENA_ENV_FILE=/app/config/.env \
    LVGPLAY_DATA_DIR=/app/data/llm
VOLUME ["/app/config", "/app/data", \
        "/app/gridarena/rl/runs", "/app/gridarena/diffusion/runs", "/app/gridarena/pf_datagen/runs", \
        "/app/gridarena/benchmarks/submissions/runs"]

USER appuser

# The API listens on this port inside the container; publish it to reach the
# app from the host, e.g.: docker run -p 8000:8000 ...
EXPOSE 8000
ENV PORT=8000

# Runtime configuration (all optional at start — anything missing can be set
# from the Setup / Settings pages, which write /app/config/.env):
#   GRID_DB_DSN or DATABASE_URL — PostgreSQL connection. Use host.docker.internal
#                   (add --add-host=host.docker.internal:host-gateway on Linux)
#                   to reach a database running on the host.
#   REDIS_URL      (digital twin message broker; defaults to redis://localhost:6379,
#                   which will NOT resolve inside the container — override it)
#   LLM_API_KEY, LLM_API_URL, LLM_MODEL, LLM_VERIFY_SSL  (Chat Assistant)
#   LVGPLAY_API_KEY (required to save settings from the UI, when set)
#   GRIDARENA_SUBMISSION_SANDBOX (subprocess | docker) and related settings for
#                   running benchmark code submissions -- see
#                   gridarena/benchmarks/submissions/sandbox.py
# Values saved from the UI take precedence over -e values on the next start.

# Probes the home page, not /docs, so a broken UI or an unreachable database
# (the page then waits on the connection pool) marks the container unhealthy.
# Without a database configured, / redirects to /setup, which still answers.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/', timeout=2)" || exit 1

CMD ["sh", "-c", "exec uvicorn gridarena.app:app --host 0.0.0.0 --port ${PORT}"]
