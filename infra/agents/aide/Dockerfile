# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# AIDE agent runtime image — single image, MLE-Bench-style frozen-lockfile.
#
# Used by:
#   - deploy/k8s/helper-jupyter-1gpu.yaml   (interactive dev pod, command=jupyter lab)
#   - infra/agents/aide/job.yaml.tmpl       (Stage 2 trajectory Jobs, command=entrypoint.sh)
#
# Architecture rationale (this file replaces a 3-layer cascade that broke):
#
#   Prior: pytorch/pytorch:2.7.1-cu126
#          + aideml git pin (with old datasets==2.1.0 / rich==13.7.0 pins)
#          + per-task `pip install -r requirements.txt` at trajectory startup
#       → vllm 0.6.6 in per-task reqs forced torch 2.7.1 → 2.5.1 downgrade,
#         broke torchaudio, broke transformers ABI, AIDE judge crashed.
#       → Three independent dep universes resolved at three different times;
#         pip never saw the full graph.
#
#   Now:   vllm/vllm-openai:v0.9.2 (vllm owns torch ABI; transformers 4.53.1
#                                   pinned to avoid the `aimv2` config clash
#                                   that breaks vllm import on transformers 5.x)
#          + ONE atomic pip install -r requirements.lock (280 packages,
#                                   resolved once in this same base image
#                                   via `requirements.in` → `pip freeze`).
#          + AIDE installed with --no-deps so its outdated pins do not
#            re-pin packages the lockfile already owns.
#          + Per-task requirements.txt deprecated (kept empty for backward
#            compat with entrypoint.sh's optional pip step).
#
# Build:  make docker-agent      (~15 min cold, ~3 min cache-hot on amusing)
# Push:   make docker-push       (multi-GB; needs ghcr write:packages PAT)
# -----------------------------------------------------------------------------

ARG VLLM_TAG=v0.9.2
FROM vllm/vllm-openai:${VLLM_TAG}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# vllm-openai ships only `python3`; symlink `python` for backward compat
# with our scripts (e.g. entrypoint.sh, run_aide.py shebang).
RUN if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3 /usr/bin/python; fi

# tini for proper signal forwarding from k8s SIGTERM into the entrypoint.
# (vllm-openai's base has bash + git + curl already.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# AIDE checkout. SHA-pinned for trajectory provenance.
# `git clone --branch` only accepts branch/tag names, not SHAs — pinning
# to a SHA needs init+fetch. We capture the exact SHA at /opt/aide/.aide_sha
# which manifest.json reads at trajectory startup.
# ---------------------------------------------------------------------------
ARG AIDE_REPO=https://github.com/WecoAI/aideml.git
ARG AIDE_REF=40dcf28fc3a39e93c7192acec0c9e2e9bffa973d

RUN mkdir -p /opt/aide && cd /opt/aide \
 && git init -q \
 && git remote add origin "${AIDE_REPO}" \
 && git fetch --depth 1 origin "${AIDE_REF}" \
 && git checkout -q FETCH_HEAD \
 && git rev-parse HEAD > /opt/aide/.aide_sha

# ---------------------------------------------------------------------------
# Frozen-lockfile install. Single atomic pip install of all 280 resolved
# packages — pip sees the full dep graph at once. Lockfile is generated
# from requirements.in by resolving in THIS SAME base image; see the
# top-of-file comment in requirements.in for the regen recipe.
#
# --no-deps so pip uses exactly the lockfile's pinned versions; transitive
# deps are already in the lockfile (we resolved with deps then froze).
# ---------------------------------------------------------------------------
COPY infra/agents/aide/requirements.lock /opt/requirements.lock
RUN pip install --no-deps -r /opt/requirements.lock

# AIDE itself: install --no-deps so its outdated pins (datasets==2.1.0,
# rich==13.7.0, jsonschema==4.19.2) do not override the lockfile's modern
# versions. AIDE's runtime imports work against datasets 4.x / rich 14.x
# (validated in helper pod under mvp-003).
RUN pip install --no-deps /opt/aide

# Harness Python package (mleval CLI + analyzer modules). Editable so
# bug fixes don't always require a rebuild during dev (the trajectory
# entrypoint re-imports them on every pod start).
COPY pyproject.toml /workspace/pyproject.toml
COPY src            /workspace/src
RUN pip install --no-deps -e /workspace

# AIDE-specific sidecar patches + shim. PYTHONPATH=/workspace makes
# `aide_sidecar` importable as a top-level package; run_aide.py loads
# the sidecar before AIDE imports.
COPY infra/agents/aide/aide_sidecar /workspace/aide_sidecar
COPY infra/agents/aide/run_aide.py  /workspace/run_aide.py
ENV PYTHONPATH=/workspace

# Snapshot the final installed package set into the image for trajectory
# provenance audit (manifest doesn't include this; reads on demand).
RUN pip freeze > /opt/requirements.freeze

# Entrypoint. Overrides vllm-openai's default `vllm serve` ENTRYPOINT.
# Helper pod overrides this further via `command: ["jupyter", "lab", ...]`.
COPY infra/agents/aide/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
CMD []
