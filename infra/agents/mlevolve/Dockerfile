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

# ----- De-Kaggle the vendored prompts ------------------------------------
# MLEvolve hardcodes a "Kaggle Grandmaster competing on a leaderboard"
# persona + a "data is in ./input / use sample_submission.csv" framing that
# derails the agent on our contract-only (HF-loaded, no-submission) tasks.
# no_submission_mode only gates output validation, not these prompts. This
# patch neutralizes the competition framing in the image copy (the submodule
# tree is reset by `git submodule update`, so we patch post-COPY). It asserts
# every required nudge was found+replaced, so upstream drift fails the build.
COPY infra/agents/mlevolve/patches/de_kaggle.py /workspace/de_kaggle.py
RUN python /workspace/de_kaggle.py

# ----- Our sidecar + entrypoint + analyzer -------------------------------
COPY infra/agents/mlevolve/mlevolve_sidecar /workspace/mlevolve_sidecar
COPY infra/agents/mlevolve/run_mlevolve.py /workspace/run_mlevolve.py
COPY infra/agents/mlevolve/entrypoint.sh /workspace/entrypoint.sh
COPY infra/agents/mlevolve/config.yaml /workspace/mlevolve_config.yaml

# Universal analyzer (adapter_mlevolve, stage_classifier, state_predicates).
# Entrypoint invokes via `python -m mleval.analyzer.X` after MLEvolve exits,
# so the mleval package must be on PYTHONPATH. We copy the source as-is
# (no pip install -e because we don't ship a pyproject.toml in this image
# layer; analyzer is pure-Python, no compiled extensions, so copying is
# enough).
COPY src/mleval /workspace/mleval
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}

# ----- Build-time regression guard: mlebench MUST NOT be importable ------
# C3 spike claim is that the mle-bench coupling is fully bypassable. If a
# transitive dep ever drags mlebench in, this fails the build instead of
# silently importing format_server at runtime.
RUN python -c "import importlib.util; \
    assert importlib.util.find_spec('mlebench') is None, \
        'mlebench was pulled in by a transitive dep — investigate before shipping'; \
    print('OK: mlebench not present')"

# ----- Build-time smoke: imports must work in the SAME environment as the
# entrypoint, not in an ad-hoc `python -c` shell.
#
# Earlier version of this smoke ran `cd /workspace/mlevolve && python -c
# "..."` which made MLEvolve's `llm/` package cwd-resolvable AND a false
# negative — it passed even though the actual entrypoint's
# `python /workspace/run_mlevolve.py` failed at module import time
# because sys.path[0] is the script's dir (`/workspace`), not cwd.
#
# Now: invoke the real entrypoint Python file with a dummy `main`
# replacement (so we don't actually launch MLEvolve, just verify imports).
# This catches sidecar-import-order bugs at build time.
COPY infra/agents/mlevolve/_smoke_imports.py /workspace/_smoke_imports.py
RUN python /workspace/_smoke_imports.py

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
