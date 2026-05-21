#!/usr/bin/env bash
# AIDE pod entrypoint. Used by Job pods (helper pod overrides `command:` to
# run `jupyter lab` directly). Sets env, runs the shim, finalizes outputs.
#
# Exit codes:
#   0   completed
#   1   missing required env / image bug
#   124 wall-clock cap (from `timeout`)
#   130 SIGINT/SIGTERM (forwarded by tini)
#   *   any other failure from AIDE

set -euo pipefail

require() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[entrypoint] FATAL: required env var '$name' is unset" >&2
        exit 1
    fi
}

require MLEVAL_RUN_ID
require MLEVAL_TRAJECTORY_ID
require TASK
require SEED
require TIME_LIMIT_SECONDS
require MLEVAL_OUTPUT_DIR
require MLEVAL_TASK_DATA_DIR
require MLEVAL_TASK_INSTRUCTION_PATH
require MLEVAL_LLM_MODEL
require OPENROUTER_API_KEY

: "${CELL:=without_skill}"
: "${STEP_LIMIT:=20}"
: "${MEMORY_INDEX:=0}"

mkdir -p "$MLEVAL_OUTPUT_DIR" "$MLEVAL_OUTPUT_DIR/agent_logs"
export MLEVAL_PROMPTS_LOG="$MLEVAL_OUTPUT_DIR/prompts.jsonl"

START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date -u +%s)

echo "[entrypoint] run_id=$MLEVAL_RUN_ID trajectory_id=$MLEVAL_TRAJECTORY_ID"
echo "[entrypoint] task=$TASK cell=$CELL seed=$SEED time_limit=${TIME_LIMIT_SECONDS}s steps=$STEP_LIMIT"

set +e
CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" \
timeout --foreground --signal=TERM --kill-after=10s "${TIME_LIMIT_SECONDS}s" \
    python /workspace/run_aide.py \
        data_dir="$MLEVAL_TASK_DATA_DIR" \
        desc_file="$MLEVAL_TASK_INSTRUCTION_PATH" \
        log_dir="$MLEVAL_OUTPUT_DIR/aide_logs" \
        workspace_dir="$MLEVAL_OUTPUT_DIR/aide_workspace" \
        exp_name="$MLEVAL_TRAJECTORY_ID" \
        agent.code.model="$MLEVAL_LLM_MODEL" \
        agent.code.base_url="https://openrouter.ai/api/v1" \
        agent.code.api_key="$OPENROUTER_API_KEY" \
        agent.code.temp=0 \
        agent.feedback.model="$MLEVAL_LLM_MODEL" \
        agent.feedback.base_url="https://openrouter.ai/api/v1" \
        agent.feedback.api_key="$OPENROUTER_API_KEY" \
        agent.feedback.temp=0 \
        agent.steps="$STEP_LIMIT" \
        2>&1 | tee "$MLEVAL_OUTPUT_DIR/agent_logs/run.log"
AGENT_EXIT=${PIPESTATUS[0]}
set -e

END_EPOCH=$(date -u +%s)
END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
WALL=$((END_EPOCH - START_EPOCH))

case "$AGENT_EXIT" in
    0)   STATUS=completed   ;;
    124) STATUS=time_capped ;;
    130) STATUS=interrupted ;;
    *)   STATUS=crashed     ;;
esac

if [[ "$CELL" == "with_skill" ]]; then
    WITH_SKILL_JSON=true
else
    WITH_SKILL_JSON=false
fi

cat > "$MLEVAL_OUTPUT_DIR/manifest.json" <<JSON
{
  "schema_version": "1.0",
  "run_id": "${MLEVAL_RUN_ID}",
  "trajectory_id": "${MLEVAL_TRAJECTORY_ID}",
  "task": { "name": "${TASK}" },
  "cell": { "with_skill": ${WITH_SKILL_JSON} },
  "seed": ${SEED},
  "agent": { "name": "aide", "llm_model": "${MLEVAL_LLM_MODEL}" },
  "timestamps": {
    "started_at": "${START_ISO}",
    "ended_at": "${END_ISO}",
    "wall_clock_sec": ${WALL}
  },
  "result": {
    "status": "${STATUS}",
    "exit_code": ${AGENT_EXIT}
  }
}
JSON

echo "[entrypoint] finished in ${WALL}s (exit ${AGENT_EXIT}, status=${STATUS})"
exit "$AGENT_EXIT"
