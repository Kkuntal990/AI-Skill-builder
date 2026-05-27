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

# System tools we use in the entrypoint + diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash curl jq git gettext-base procps tini && \
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
# AIDE). Strip mlebench out of any requirements line (it's not in upstream's
# files anyway, but defensive).
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v -i "mlebench" /workspace/mlevolve/requirements_base.txt > /tmp/req_base.txt && \
    grep -v -i "mlebench" /workspace/mlevolve/requirements_ml.txt > /tmp/req_ml.txt && \
    grep -v -i "mlebench" /workspace/mlevolve/requirements_domain.txt > /tmp/req_domain.txt && \
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
