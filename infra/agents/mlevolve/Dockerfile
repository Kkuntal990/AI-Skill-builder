# MLEvolve runtime image — curated minimal install (Option C).
#
# Build philosophy: copied from MLE-Bench's environment/Dockerfile
# (https://github.com/openai/mle-bench/blob/main/environment/Dockerfile) +
# our prior AIDE Dockerfile. Both converged on the same answer:
#
#   1. One base image with prebuilt wheels (CUDA runtime inside torch's
#      own wheel — no nvcc, no cuda-toolkit apt package).
#   2. ONE atomic pip install of a hand-curated requirements.txt — no
#      kitchen-sink upstream lockfile (those fail 5+ ways: pillow vs
#      streamlit, pyarrow vs autogluon, distutils blinker, CUDA-version
#      compile errors on torchsort, etc.).
#   3. --no-deps for the agent source itself (or pip from a clean vendor
#      copy) so its declared deps don't re-pin packages we already own.
#
# Justification + alternatives considered → docs/env-build-details.md.

ARG VLLM_TAG=v0.9.2
FROM vllm/vllm-openai:${VLLM_TAG}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# vllm-openai ships only `python3`; symlink `python` for backward compat
# with our scripts.
RUN if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3 /usr/bin/python; fi

# System tools: tini for signal forwarding from k8s SIGTERM into the
# entrypoint, gettext-base for envsubst (config rendering),
# python3-blinker removed because the base image installs it as a
# distutils-system package that pip can't safely uninstall — and one
# of our pip deps WILL try to upgrade blinker as a transitive.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini ca-certificates curl jq gettext-base procps \
    && apt-get remove -y python3-blinker || true \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ----- Curated package install --------------------------------------------
# requirements.txt is hand-curated from MLEvolve's import graph (12 new
# packages on top of what vllm/vllm-openai already provides). pip resolves
# transitives ONCE here; the resolution surface is small enough that
# conflicts are unlikely.
#
# We DO NOT use upstream's requirements_{base,ml,domain}.txt because their
# 665-line lockfile is internally inconsistent (verified failures: pillow
# vs streamlit, pyarrow vs autogluon, umap typo) and assumes their CI
# env's package versions, not vllm-openai's.
COPY infra/agents/mlevolve/requirements.txt /opt/mlevolve-requirements.txt
RUN pip install --no-cache-dir -r /opt/mlevolve-requirements.txt

# ----- Vendored MLEvolve source ------------------------------------------
# Submodule contents COPYed directly (not pip-installed) so the upstream's
# relative imports (`from engine.executor import Interpreter`) resolve.
# The launcher cds into /workspace/mlevolve before invoking run.py.
COPY infra/agents/mlevolve/upstream /workspace/mlevolve

# ----- Our sidecar + entrypoint ------------------------------------------
COPY infra/agents/mlevolve/mlevolve_sidecar /workspace/mlevolve_sidecar
COPY infra/agents/mlevolve/run_mlevolve.py /workspace/run_mlevolve.py
COPY infra/agents/mlevolve/entrypoint.sh /workspace/entrypoint.sh
COPY infra/agents/mlevolve/config.yaml /workspace/mlevolve_config.yaml

# ----- Build-time regression guard: mlebench MUST NOT be importable ------
# C3 spike claim is that the mle-bench coupling is fully bypassable. If a
# transitive dep ever drags mlebench in, this fails the build instead of
# silently importing format_server at runtime.
RUN python -c "import importlib.util; \
    assert importlib.util.find_spec('mlebench') is None, \
        'mlebench was pulled in by a transitive dep — investigate before shipping'; \
    print('OK: mlebench not present')"

# ----- Build-time smoke: MLEvolve imports cleanly with our sidecar -------
# Catches "we forgot to install package X that MLEvolve imports at module
# load" before we burn a trajectory's image-pull cycle on the cluster.
# We import the same modules run.py does at startup, but in a directory
# where MLEvolve's `from config import load_cfg` (relative-style with
# config on path) resolves.
RUN cd /workspace/mlevolve && python -c "\
import sys; sys.path.insert(0, '/workspace'); \
import mlevolve_sidecar; \
from engine.executor import Interpreter; \
from engine.search_node import Journal; \
from engine.agent_search import AgentSearch; \
print('OK: MLEvolve + sidecar import cleanly')"

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
