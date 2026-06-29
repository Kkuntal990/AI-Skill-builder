#!/usr/bin/env bash
# Purpose: probe nvidia-smi for free VRAM per GPU; fail if any GPU < THRESHOLD_GB.
# Usage: ./check_vram.sh [threshold_gb=16]   (run before `vllm serve` to catch OOM early)
set -euo pipefail

THRESHOLD_GB=${1:-16}  # Default 16 GB — floor for a 7B model in fp16 (~14 GB weights + KV cache).
MIB_PER_GB=1024        # nvidia-smi reports MiB; vLLM/HBM sizes are quoted in GiB.

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf "error: nvidia-smi not found — no NVIDIA driver visible in this container/host\n" >&2
  exit 2
fi

# --query-gpu is stable across driver versions; parsing the human table is not.
if ! mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); then
  printf "error: nvidia-smi query failed — GPU may be unreachable\n" >&2
  exit 2
fi

if (( ${#free_mib[@]} == 0 )); then
  printf "error: nvidia-smi reported no GPUs\n" >&2
  exit 2
fi

below_threshold=0
for i in "${!free_mib[@]}"; do
  free_gb=$(( free_mib[i] / MIB_PER_GB ))
  if (( free_gb < THRESHOLD_GB )); then
    printf "GPU %d: %d GB free (below %d GB threshold — OOM risk)\n" "$i" "$free_gb" "$THRESHOLD_GB" >&2
    below_threshold=1
  else
    printf "GPU %d: %d GB free — ok\n" "$i" "$free_gb"
  fi
done

exit "$below_threshold"
