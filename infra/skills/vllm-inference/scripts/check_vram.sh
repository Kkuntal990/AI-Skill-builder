#!/usr/bin/env bash
# Purpose: probe nvidia-smi for free VRAM per GPU; warn if any GPU is below THRESHOLD_GB.
# Usage: ./check_vram.sh [threshold_gb=16]
# Run before vLLM model load to confirm room for weights + KV cache.
set -euo pipefail

THRESHOLD_GB=${1:-16}  # Default 16 GB — a 7B model (~14 GB fp16) plus a small KV cache.
MIB_PER_GB=1024        # nvidia-smi reports MiB; 1 GiB = 1024 MiB.

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf "error: nvidia-smi not found (no NVIDIA driver?). vLLM requires a CUDA GPU.\n" >&2
  exit 2
fi

# --query-gpu is stable across driver versions; avoids parsing free-form nvidia-smi text.
if ! mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); then
  printf "error: nvidia-smi query failed.\n" >&2
  exit 2
fi

if (( ${#free_mib[@]} == 0 )); then
  printf "error: no GPUs detected.\n" >&2
  exit 2
fi

warned=0
for i in "${!free_mib[@]}"; do
  free_gb=$(( free_mib[i] / MIB_PER_GB ))
  if (( free_gb < THRESHOLD_GB )); then
    printf "GPU %d: %d GB free (below %d GB threshold) — model load may OOM\n" \
      "$i" "$free_gb" "$THRESHOLD_GB" >&2
    warned=1
  else
    printf "GPU %d: %d GB free — ok\n" "$i" "$free_gb"
  fi
done

exit "$warned"
