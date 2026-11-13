# syntax=docker/dockerfile:1.7
# Multi-stage uv-based container image for Decision Forge.
# Build examples:
#   docker build -t decision-forge .
#   docker build --build-arg TORCH_BACKEND=cu126 -t decision-forge:cuda .
#   docker build --build-arg TORCH_BACKEND=xpu -t decision-forge:xpu .

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG TORCH_BACKEND=cpu
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=${TORCH_BACKEND}

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv venv /app/.venv && \
    if [ "${TORCH_BACKEND}" != "cpu" ] && [ "${TORCH_BACKEND}" != "xpu" ] && [ "${TORCH_BACKEND}" != "auto" ] && echo "${TORCH_BACKEND}" | grep -q "^cu"; then \
        uv pip install --python /app/.venv/bin/python --torch-backend "${TORCH_BACKEND}" --no-cache ".[cuda]"; \
    else \
        uv pip install --python /app/.venv/bin/python --torch-backend "${TORCH_BACKEND}" --no-cache .; \
    fi && \
    if [ "${TORCH_BACKEND}" = "xpu" ]; then \
        uv pip install --python /app/.venv/bin/python --torch-backend xpu --no-cache pytorch-triton-xpu; \
    fi

FROM python:3.12-slim-bookworm AS runtime

ARG TORCH_BACKEND=cpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libgomp1 && \
    if [ "${TORCH_BACKEND}" = "xpu" ]; then \
        apt-get install -y --no-install-recommends intel-opencl-icd libze1; \
    fi && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/sh decide

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

RUN mkdir -p /app/runs && chown -R decide:decide /app

USER decide

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8765

# Run `decide serve` by default; override CMD to run `decide` subcommands or `decide-setup --check`.
CMD ["decide", "serve", "--host", "0.0.0.0", "--port", "8765"]
