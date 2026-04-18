# Homie AI — Multi-stage Docker build
# Supports: headless server, spoke-only, full (with GPU)

# --- Stage 1: Base ---
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/

# --- Stage 2: Spoke (minimal — no model, no voice) ---
FROM base AS spoke

RUN pip install --no-cache-dir -e ".[mesh,app]"
ENV HOMIE_STORAGE_PATH=/data
ENV HOMIE_LLM_BACKEND=cloud
VOLUME /data
EXPOSE 8721
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from homie_core.mesh.health import MeshHealthChecker; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "homie_app.cli"]
CMD ["daemon", "--headless"]

# --- Stage 3: Full (with model support) ---
FROM base AS full

RUN pip install --no-cache-dir -e ".[mesh,app,model,storage,neural,network,email,docs]"
ENV HOMIE_STORAGE_PATH=/data
VOLUME /data
EXPOSE 8721 8765
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from homie_core.mesh.health import MeshHealthChecker; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "homie_app.cli"]
CMD ["daemon", "--headless"]
