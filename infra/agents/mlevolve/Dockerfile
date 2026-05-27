# MLEvolve runtime image.
#
# Spike-stage build: smallest possible image that can run one MLEvolve
# trajectory on llama-inference without invoking the mle-bench grading server.
# Based on vllm/vllm-openai:v0.9.2 (CUDA 12.8, torch 2.7, transformers 4.53)
# to match the GPU environment our pilot tasks expect.
#
# Notable choices:
#   - MLEvolve upstream is vendored as a git submodule under ./upstream/ and
#     COPYed in directly (not pip-installed) so the import surface is the
#     pinned SHA we know, not whatever pip resolves.
#   - mlebench is NOT installed. The single import in MLEvolve's
#     engine/validation/format_server.py is bypassed via use_grading_server:
#     false in our config — that whole module never loads.
#   - FAISS + bge-base-en embedding model are NOT installed. We disable
#     use_global_memory in our config so memory_layer never imports them.
#   - coldstart JSONs ARE copied (~few hundred KiB) because run.py
#     unconditionally calls build_guidance_description when use_coldstart is
#     on; we toggle that off in config but ship the files anyway so the
#     codebase imports cleanly.
#
# This image is intentionally separate from the prior AIDE image
# (ghcr.io/kkuntal990/mleval-agent:dev). It will be pushed as
# ghcr.io/kkuntal990/mleval-agent-mlevolve:dev.

FROM vllm/vllm-openai:v0.9.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System tools we use in the entrypoint + diagnostics.
# Also: remove python3-blinker — the vllm/vllm-openai base ships it as a
# distutils-installed system package, and pip refuses to uninstall those
# ("Cannot uninstall blinker 1.4 — distutils installed project"). MLEvolve's
# requirements_base.txt pins blinker==1.9.0 which would trip that check.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash curl jq git gettext-base procps tini && \
    apt-get remove -y python3-blinker || true && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- MLEvolve upstream (vendored at infra/agents/mlevolve/upstream) ---------
# COPY the submodule contents straight into /workspace/mlevolve so the
# upstream's relative imports (`from engine.executor import Interpreter`)
# resolve. The launcher cds into /workspace/mlevolve before invoking run.py.
COPY infra/agents/mlevolve/upstream /workspace/mlevolve

# Install MLEvolve's pinned deps. base + ml + domain are 3 separate lockfiles
# upstream; install all three (no --no-deps; let pip resolve transitives so
# we don't get stuck on the same lockfile-vs-runtime drift that bit us with
# AIDE).
#
# Strip these lines from upstream requirements before install:
#   - mlebench       — defensive; not in their files but pinned out in case
#                      a future version adds it (our C3 spike asserts absence)
#   - streamlit      — leftover in req_base.txt:154 (no import streamlit in
#                      MLEvolve source); pinned at 1.40.2 caps pillow<12,
#                      conflicting with the same file's pillow==12.0.0.
#   - autogluon(.*)  — 7 lines in req_ml.txt (autogluon, autogluon.common,
#                      .core, .features, .multimodal, .tabular, .timeseries).
#                      autogluon-common 1.4.0 pins pyarrow<21, datasets 4.1.1
#                      pins pyarrow>=21 — upstream lockfile is internally
#                      inconsistent. autogluon is never imported in MLEvolve
#                      source (it's a library option for LLM-generated code
#                      to use, on tabular tasks). For the llama-inference
#                      spike: irrelevant. If/when we port to tabular tasks
#                      post-spike, fix properly (downgrade datasets or
#                      ditch autogluon).
RUN pip install --no-cache-dir --upgrade pip && \
    grep -vE "^(mlebench|streamlit|autogluon([._-].*)?)([=<>!~]|$)" /workspace/mlevolve/requirements_base.txt > /tmp/req_base.txt && \
    grep -vE "^(mlebench|streamlit|autogluon([._-].*)?)([=<>!~]|$)" /workspace/mlevolve/requirements_ml.txt > /tmp/req_ml.txt && \
    grep -vE "^(mlebench|streamlit|autogluon([._-].*)?)([=<>!~]|$)" /workspace/mlevolve/requirements_domain.txt > /tmp/req_domain.txt && \
    pip install --no-cache-dir -r /tmp/req_base.txt && \
    pip install --no-cache-dir -r /tmp/req_ml.txt && \
    pip install --no-cache-dir -r /tmp/req_domain.txt && \
    rm -f /tmp/req_base.txt /tmp/req_ml.txt /tmp/req_domain.txt

# --- Our sidecar (prompt logger + seed pin + openai-timeout) ---------------
COPY infra/agents/mlevolve/mlevolve_sidecar /workspace/mlevolve_sidecar
COPY infra/agents/mlevolve/run_mlevolve.py /workspace/run_mlevolve.py
COPY infra/agents/mlevolve/entrypoint.sh /workspace/entrypoint.sh
COPY infra/agents/mlevolve/config.yaml /workspace/mlevolve_config.yaml

# Tasks + skills + tools are mounted via PVC at runtime; nothing to bake in.

# Sanity: verify mlebench is NOT importable in this image (proves C3 ahead of
# runtime — fails the build if a transitive dep dragged it in).
RUN python -c "import importlib, sys; \
    assert importlib.util.find_spec('mlebench') is None, 'mlebench was installed by a transitive dep — fix the requirements grep'; \
    print('OK: mlebench not present')"

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/entrypoint.sh"]
