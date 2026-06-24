#!/usr/bin/env bash
# Purpose: poll vLLM's /health until HTTP 200, then list /v1/models.
# Usage: ./wait_for_server.sh [base_url=http://localhost:8000] [timeout_sec=300]
set -euo pipefail

BASE_URL=${1:-http://localhost:8000}  # vLLM OpenAI server defaults to port 8000.
TIMEOUT_SEC=${2:-300}                 # 300s covers cold model-weight load for 7B-class checkpoints.
POLL_INTERVAL_SEC=2                   # Tight enough to catch readiness, loose enough to avoid hammering.

if ! command -v curl >/dev/null 2>&1; then
  printf "error: curl not found (cannot probe %s)\n" "$BASE_URL" >&2
  exit 2
fi

printf "Waiting for vLLM at %s (timeout %ds)...\n" "$BASE_URL" "$TIMEOUT_SEC"

elapsed=0
while (( elapsed < TIMEOUT_SEC )); do
  # --fail makes curl exit non-zero on non-2xx; -s/-o discard the (empty) body.
  if curl -fsS -o /dev/null "${BASE_URL}/health" 2>/dev/null; then
    printf "Server healthy after %ds.\n" "$elapsed"
    printf "Available models:\n"
    if command -v jq >/dev/null 2>&1; then
      curl -fsS "${BASE_URL}/v1/models" | jq -r '.data[].id'
    else
      # jq absent — emit raw JSON so the caller still sees the model list.
      curl -fsS "${BASE_URL}/v1/models"
      printf "\n"
    fi
    exit 0
  fi
  sleep "$POLL_INTERVAL_SEC"
  elapsed=$(( elapsed + POLL_INTERVAL_SEC ))
done

printf "error: server at %s not healthy within %ds\n" "$BASE_URL" "$TIMEOUT_SEC" >&2
exit 1
