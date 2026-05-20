# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Agent runtime image. Used by:
#   - deploy/k8s/job-*.yaml             (Stage 2 trajectory runs)
#   - deploy/k8s/helper-jupyter-*.yaml  (interactive debugging on Nautilus)
#
# Base image is version-matched to MLEvolve's requirements_ml.txt pins:
#   torch == 2.7.1
#   nvidia-cuda-runtime-cu12 == 12.6.77
# The `runtime` variant (vs `devel`) drops nvcc + headers — fine because every
# pip dep MLEvolve declares is available as a pre-built wheel. Switch to
# `-devel` if any pip install in the build fails with a "compiler not found"
# style error.
# -----------------------------------------------------------------------------

ARG PYTORCH_TAG=2.7.1-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# git: clone MLEvolve. jq: optional, useful in entrypoint. tini: handles SIGTERM
# cleanly when the pod is preempted. curl: grading-server health probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        jq \
        tini \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# MLEvolve install. Pin MLEVOLVE_REF to a real SHA in .env before any real
# A/B run; the default `main` is fine for Phase 0 / image smoke tests only.
# ---------------------------------------------------------------------------
ARG MLEVOLVE_REPO=https://github.com/InternScience/MLEvolve.git
ARG MLEVOLVE_REF=main

RUN git clone --depth 1 --branch "${MLEVOLVE_REF}" "${MLEVOLVE_REPO}" /opt/mlevolve \
 && cd /opt/mlevolve \
 # Upstream MLEvolve's requirements files contain declared-but-unused deps
 # with unresolvable pip resolution failures. Verified via GitHub code search
 # 2026-05-20: none of these are imported anywhere in MLEvolve's source
 # tree, so dropping is safe.
 #
 #   streamlit==1.40.2  → requires pillow<12, conflicts with pillow==12.0.0
 #   autogluon family   → transitively requires pyarrow<21, conflicts with
 #                         pyarrow==21.0.0 (required by datasets==4.1.1)
 #   umap==0.1.1        → no PyPI distribution available for Python 3.11
 #                         (the real UMAP is `umap-learn==0.5.9.post2`,
 #                         already declared on the next line)
 && sed -i '/^streamlit==/d' requirements_base.txt \
 && sed -i '/^autogluon/d'   requirements_ml.txt \
 && sed -i '/^umap==/d'      requirements_domain.txt \
 && pip install -r requirements_base.txt \
 && pip install -r requirements_ml.txt \
 # requirements_domain.txt is a kitchen-sink of vision / audio / time-series
 # / NLP packages, many of which are declared but never imported by the
 # MLEvolve harness itself (they're available for agent-generated code).
 # We install best-effort with `|| true` so a single missing distribution
 # doesn't abort the build. Any individual install failures are surfaced
 # in the build log; revisit if a downstream task actually needs one.
 && (pip install -r requirements_domain.txt || \
     pip install --no-deps -r requirements_domain.txt || true)

# Explicitly install the packages MLEvolve actually imports at runtime that
# either (a) aren't declared in any requirements_*.txt at all (`trl`,
# `mle-bench`), or (b) are declared in requirements_domain.txt but get
# skipped when the strict-then-fallback install above aborts mid-stream
# because of unrelated wheel-build failures (`faiss-cpu`, `rank_bm25`).
# Verified via runtime import test 2026-05-20 — all four import cleanly.
# Pinning matches what MLEvolve's own requirements would have asked for.
RUN pip install \
        trl \
        faiss-cpu==1.11.0 \
        rank_bm25==0.2.2 \
 && pip install git+https://github.com/openai/mle-bench.git

# MLEvolve is not packaged (no setup.py / pyproject.toml at the repo root), so
# we make it importable by adding /opt/mlevolve to PYTHONPATH and running
# scripts from that working directory at entry time.
ENV PYTHONPATH=/opt/mlevolve:${PYTHONPATH}

# Harness-side Python package. Editable install so iterating on src/ requires
# only an image rebuild of the COPY layers (cached pip layer above is reused).
COPY pyproject.toml /workspace/pyproject.toml
COPY src            /workspace/src
RUN pip install -e /workspace

# Helper for the interactive Jupyter pod (~50MB).
RUN pip install jupyterlab ipywidgets

# Entrypoint.
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
CMD []
