#!/usr/bin/env bash
# Purpose: Query nvidia-smi for free VRAM on all GPUs; warn if any device is below threshold.
# Usage: ./check_vram.sh [threshold_gb=16]
# Runs before: vllm serve or LLM() to confirm sufficient memory headroom.
set -euo pipefail

THRESHOLD_GB=${1:-16}  # 16 GB — minimum headroom for typical vLLM model serving.

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf "error: nvidia-smi not found — no NVIDIA driver or not in PATH.\n" >&2
  exit 2
fi

mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)

if [[ ${#free_mib[@]} -eq 0 ]]; then
  printf "error: nvidia-smi returned no GPU data. Is a GPU present and driver loaded?\n" >&2
  exit 2
fi

warned=0
for i in "${!free_mib[@]}"; do
  free_gb=$(( free_mib[i] / 1024 ))
  if (( free_gb < THRESHOLD_GB )); then
    printf "error: GPU %d: %d GB free — below %d GB threshold. Free VRAM before launching vLLM.\n" \
      "$i" "$free_gb" "$THRESHOLD_GB" >&2
    warned=1
  else
    printf "GPU %d: %d GB free — ok (threshold: %d GB)\n" "$i" "$free_gb" "$THRESHOLD_GB"
  fi
done

exit "$warned"
