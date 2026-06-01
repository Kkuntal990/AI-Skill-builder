#!/usr/bin/env bash
# Pre-warm a HuggingFace model into the shared PVC cache so trajectory pods
# don't pay download cost on first use.
#
# Usage:
#   ./warm_hf_model.sh <model_slug>
#
# Example:
#   ./warm_hf_model.sh BAAI/bge-base-en-v1.5
#   ./warm_hf_model.sh Qwen/Qwen2.5-3B-Instruct
#
# Idempotent — snapshot_download skips files already present.

set -euo pipefail

MODEL="${1:?usage: $0 <model_slug>}"
NAMESPACE="${KUBECTL_NS:-ecepxie}"
HELPER_POD="${HELPER_POD:-mleval-jupyter-1gpu}"

if ! kubectl -n "$NAMESPACE" get pod "$HELPER_POD" >/dev/null 2>&1; then
    echo "[warm_hf] ERROR: helper pod $HELPER_POD not found in $NAMESPACE" >&2
    exit 1
fi

echo "[warm_hf] downloading $MODEL into /results/.hf-cache/hf/hub/ via $HELPER_POD"

kubectl -n "$NAMESPACE" exec "$HELPER_POD" -- bash -c "
HF_HOME=/results/.hf-cache/hf python -c \"
from huggingface_hub import snapshot_download
import os
p = snapshot_download('$MODEL', cache_dir='/results/.hf-cache/hf/hub')
total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
print(f'warmed: $MODEL -> {p}')
print(f'size: {total/1e6:.1f} MB')
\"
"

echo "[warm_hf] done"
