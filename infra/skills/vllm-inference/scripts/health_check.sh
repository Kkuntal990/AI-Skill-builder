#!/usr/bin/env bash
# Purpose: Poll vLLM /health endpoint until server is ready or timeout expires.
#          Prints round-trip latency on success.
# Usage:   ./health_check.sh [host=http://localhost:8000] [timeout_sec=120]

set -euo pipefail

HOST="${1:-http://localhost:8000}"
TIMEOUT_SEC="${2:-120}"  # 120 s — generous for model weight loading on first start.
POLL_INTERVAL=2          # Seconds between retries — avoids hammering the server.

HEALTH_URL="${HOST}/health"
elapsed=0

if ! command -v curl >/dev/null 2>&1; then
  printf "error: curl not found — install curl and retry\n" >&2
  exit 2
fi

printf "Polling %s (timeout %ds) ...\n" "$HEALTH_URL" "$TIMEOUT_SEC"

while true; do
  start_ns=$(date +%s%N)

  http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH_URL" 2>/dev/null || true)

  end_ns=$(date +%s%N)
  latency_ms=$(( (end_ns - start_ns) / 1000000 ))

  if [[ "$http_code" == "200" ]]; then
    printf "vLLM server is ready — /health returned 200 in %d ms\n" "$latency_ms"
    exit 0
  fi

  if (( elapsed >= TIMEOUT_SEC )); then
    printf "error: server not ready after %ds (last HTTP status: %s)\n" \
      "$TIMEOUT_SEC" "${http_code:-no response}" >&2
    printf "hint: check 'vllm serve' logs for startup errors\n" >&2
    exit 1
  fi

  printf "  [%3ds] status=%s — retrying in %ds...\n" \
    "$elapsed" "${http_code:-no response}" "$POLL_INTERVAL"

  sleep "$POLL_INTERVAL"
  elapsed=$(( elapsed + POLL_INTERVAL ))
done
