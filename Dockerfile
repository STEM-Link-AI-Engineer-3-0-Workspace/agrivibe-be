# syntax=docker/dockerfile:1
# AgriVibe backend. Build context: this repository. Built by .github/workflows/deploy.yml (and copied from the
# Session 2 demo pack, docker/backend.Dockerfile, where it is built with an extra "pack" context).
# The same image runs on ECS Fargate and, through Lambda Web Adapter, on Lambda.

FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

# ---- build stage: resolve and install dependencies, then the project ----
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# 1. Lock file first: this layer is cached until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
# 2. Source second: a code change only rebuilds from here.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable
# Lambda only: boto3 lets the entrypoint read SSM parameters (ECS injects them itself).
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /app/.venv/bin/python boto3==1.40.0

# ---- runtime stage: no uv, no build cache, no source tree ----
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
# Lambda Web Adapter: an extension that turns Lambda events into HTTP requests. Ignored outside Lambda.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.0@sha256:b4da35991627bdac98a81c377d0cc28e6989687359576dfda9f0b64be835d648 /lambda-adapter /opt/extensions/lambda-adapter
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /app/.venv /app/.venv
COPY deploy/ssm_env.py /app/ssm_env.py
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HOME=/tmp \
    AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/health
USER app
EXPOSE 8000
# Used by Docker and Compose only. ECS and the ALB have their own health checks.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
# One worker: conversation state (InMemorySaver) lives in this process only.
CMD ["python", "/app/ssm_env.py", "uvicorn", "agrivibe.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
