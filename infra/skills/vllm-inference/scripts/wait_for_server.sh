#!/usr/bin/env bash
# Purpose: poll vLLM's /health until 200, then GET /v1/models to confirm a model loaded.
# Usage: ./wait_for_server.sh [base_url=http://localhost:8000] [timeout_sec=300]
set -euo pipefail

BASE_URL=${1:-http://localhost:8000}
TIMEOUT_SEC=${2:-300}      # 5 min — covers cold weight load for a 7B model on one GPU.
POLL_INTERVAL_SEC=2        # vLLM startup is slow; 2s avoids hammering while staying responsive.

if ! command -v curl >/dev/null 2>&1; then
  printf "error: curl not found\n" >&2
  exit 2
fi

# Phase 1: wait for /health to return HTTP 200 (vLLM reports ready here).
deadline=$(( SECONDS + TIMEOUT_SEC ))
until code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/health" 2>/dev/null) && [ "$code" = "200" ]; do
  if (( SECONDS >= deadline )); then
    printf "error: %s/health not ready after %ds (last code: %s)\n" "$BASE_URL" "$TIMEOUT_SEC" "${code:-none}" >&2
    exit 1
  fi
  sleep "$POLL_INTERVAL_SEC"
done
printf "%s/health is ready\n" "$BASE_URL"

# Phase 2: confirm at least one model is served via /v1/models.
body=$(curl -s "${BASE_URL}/v1/models")
if command -v jq >/dev/null 2>&1; then
  model_id=$(printf '%s' "$body" | jq -r '.data[0].id // empty')
  if [ -z "$model_id" ]; then
    printf "error: /v1/models returned no model\n%s\n" "$body" >&2
    exit 1
  fi
  printf "model loaded: %s\n" "$model_id"
else
  # No jq — fall back to a substring check so the script still works unattended.
  if ! printf '%s' "$body" | grep -q '"object"[[:space:]]*:[[:space:]]*"list"'; then
    printf "error: /v1/models did not return a model list\n%s\n" "$body" >&2
    exit 1
  fi
  printf "model list returned (install jq for the model id)\n"
fi

exit 0
