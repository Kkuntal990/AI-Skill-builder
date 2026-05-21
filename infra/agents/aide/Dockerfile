# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# AIDE agent runtime image.
#
# Used by:
#   - deploy/k8s/helper-jupyter-1gpu.yaml   (interactive dev pod)
#   - infra/agents/aide/job-*.yaml          (Stage 2 trajectory Jobs — TBD)
#
# Built on amusing.ucsd.edu (native amd64 buildx), pushed to ghcr.io,
# pulled by Nautilus pods. See infra/agents/aide/README.md.
#
# Build:  make docker-agent
# Push:   make docker-push
# -----------------------------------------------------------------------------

ARG PYTORCH_TAG=2.7.1-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# git: clone AIDE. tini: signal handling on pod preemption. curl: probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tini ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# AIDE install. Pin AIDE_REF to a real SHA before any A/B run; `main` is OK
# for scaffold / smoke tests only.
#
# AIDE's requirements.txt is split with a comment divider:
#   - above '# agent requirements' = harness deps the AIDE code itself imports
#   - below = a kitchen-sink of "packages the agent-generated code might want
#             to import" — most have broken wheels (tensorpack, bayespy) or
#             are UI-only (streamlit). We install the harness block strictly
#             then curate the agent-side ML stack explicitly below.
# ---------------------------------------------------------------------------
ARG AIDE_REPO=https://github.com/WecoAI/aideml.git
ARG AIDE_REF=main

RUN git clone --depth 1 --branch "${AIDE_REF}" "${AIDE_REPO}" /opt/aide \
 && cd /opt/aide \
 # Capture exact SHA for trajectory provenance (manifest.json reads /opt/aide/.aide_sha).
 && git rev-parse HEAD > /opt/aide/.aide_sha \
 && awk '/^# agent requirements/{exit} {print}' requirements.txt \
        > /tmp/req_harness.txt \
 && pip install -r /tmp/req_harness.txt \
 && pip install --no-deps . \
 # Harness deps that AIDE imports but ships in the kitchen-sink section of
 # requirements.txt (caught by audit: aide imports `backoff`, `plotly`,
 # `dotenv` but the awk-strip put them below the divider).
 && pip install \
        backoff \
        plotly \
        python-dotenv \
 # Curated agent-side stack for PEFT trajectories. Versions left unpinned so
 # pip resolves against torch 2.7.1 from the base image.
 && pip install \
        "transformers>=4.45" \
        "peft>=0.13" \
        "trl>=0.11" \
        "datasets>=3" \
        "accelerate>=1.0"

# Harness-side Python package (editable for fast iteration of CLI/utils).
COPY pyproject.toml /workspace/pyproject.toml
COPY src            /workspace/src
RUN pip install -e /workspace

# AIDE-specific patches & shim. Sidecar is importable as `aide_sidecar`.
COPY infra/agents/aide/aide_sidecar /workspace/aide_sidecar
COPY infra/agents/aide/run_aide.py  /workspace/run_aide.py
ENV PYTHONPATH=/workspace

# Helper-pod extras (interactive Jupyter dev).
RUN pip install jupyterlab ipywidgets

# Entrypoint. The helper pod overrides `command:` to `jupyter lab`; this
# entrypoint is used by Job pods that run a full trajectory.
COPY infra/agents/aide/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
CMD []
