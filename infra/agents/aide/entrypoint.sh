#!/usr/bin/env bash
# AIDE pod entrypoint. Used by Job pods (helper pod overrides `command:` to
# run `jupyter lab` directly). Sets env, runs the shim, writes manifest, then
# invokes our post-trajectory analyzer chain (adapter -> classifier -> preds).
#
# Exit codes:
#   0   completed
#   1   missing required env / image bug
#   124 wall-clock cap (from `timeout`)
#   130 SIGINT/SIGTERM (forwarded by tini; also "preempted" path)
#   *   any other failure from AIDE

set -eo pipefail

require() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[entrypoint] FATAL: required env var '$name' is unset" >&2
        exit 1
    fi
}

# Canonical key name from the plugin contract is MLEVAL_LLM_API_KEY; for
# backward-compat with our smoke tests we also accept OPENROUTER_API_KEY.
: "${MLEVAL_LLM_API_KEY:=${OPENROUTER_API_KEY:-}}"
: "${OPENROUTER_API_KEY:=$MLEVAL_LLM_API_KEY}"
export MLEVAL_LLM_API_KEY OPENROUTER_API_KEY

require MLEVAL_RUN_ID
require MLEVAL_TRAJECTORY_ID
require TASK
require SEED
require TIME_LIMIT_SECONDS
require MLEVAL_OUTPUT_DIR
require MLEVAL_TASK_DATA_DIR
require MLEVAL_TASK_INSTRUCTION_PATH
require MLEVAL_LLM_MODEL
require MLEVAL_LLM_API_KEY

: "${CELL:=without_skill}"
: "${STEP_LIMIT:=20}"
: "${MEMORY_INDEX:=0}"
: "${MLEVAL_SKILL_PATH:=}"
: "${GENERATE_REPORT:=false}"

mkdir -p "$MLEVAL_OUTPUT_DIR" "$MLEVAL_OUTPUT_DIR/agent_logs"
export MLEVAL_PROMPTS_LOG="$MLEVAL_OUTPUT_DIR/prompts.jsonl"

# Route AIDE's openai backend at OpenRouter. With OPENAI_BASE_URL set AND the
# model not matching gpt-*/o*/codex, AIDE picks `use_chat_api=true` which uses
# chat.completions.create (OpenRouter-compatible). Without this, AIDE falls
# back to `responses.create` which is OpenAI-only and 401s.
# See aide/backend/backend_openai.py:74.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$MLEVAL_LLM_API_KEY}"

# ---- early metadata + preempt-safe partial manifest --------------------

AIDE_SHA="$(cat /opt/aide/.aide_sha 2>/dev/null || echo unknown)"
HOSTNAME_K8S="${HOSTNAME:-unknown}"
NODE_NAME="${KUBE_NODE_NAME:-unknown}"
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date -u +%s)

INSTRUCTION_SHA="$(sha256sum "$MLEVAL_TASK_INSTRUCTION_PATH" 2>/dev/null | awk '{print $1}')"
SKILL_SHA=""
if [[ -n "$MLEVAL_SKILL_PATH" && -f "$MLEVAL_SKILL_PATH" ]]; then
    SKILL_SHA="$(sha256sum "$MLEVAL_SKILL_PATH" | awk '{print $1}')"
fi
if [[ "$CELL" == "with_skill" ]]; then
    WITH_SKILL_JSON=true
else
    WITH_SKILL_JSON=false
fi

write_manifest() {
    local status="$1"
    local exit_code="$2"
    local end_iso end_epoch wall
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    end_epoch=$(date -u +%s)
    wall=$((end_epoch - START_EPOCH))
    cat > "$MLEVAL_OUTPUT_DIR/manifest.json" <<JSON
{
  "schema_version": "1.0",
  "run_id": "${MLEVAL_RUN_ID}",
  "trajectory_id": "${MLEVAL_TRAJECTORY_ID}",
  "task": {
    "name": "${TASK}",
    "instruction_path": "${MLEVAL_TASK_INSTRUCTION_PATH}",
    "instruction_sha256": "${INSTRUCTION_SHA}",
    "data_dir": "${MLEVAL_TASK_DATA_DIR}"
  },
  "cell": {
    "with_skill": ${WITH_SKILL_JSON},
    "skill_path": "${MLEVAL_SKILL_PATH}",
    "skill_sha256": "${SKILL_SHA}"
  },
  "seed": ${SEED},
  "agent": {
    "name": "aide",
    "version": "${AIDE_SHA}",
    "container_image": "${HOSTNAME_K8S}",
    "llm_model": "${MLEVAL_LLM_MODEL}"
  },
  "pod": {
    "node": "${NODE_NAME}",
    "hostname": "${HOSTNAME_K8S}"
  },
  "timestamps": {
    "started_at": "${START_ISO}",
    "ended_at": "${end_iso}",
    "wall_clock_sec": ${wall}
  },
  "result": {
    "status": "${status}",
    "exit_code": ${exit_code}
  }
}
JSON
}

# Preempt-safe trap: if the pod is SIGTERMed (kubelet preempt, node drain,
# OOM-killer cascade), write a manifest BEFORE we get SIGKILLed.
on_signal() {
    write_manifest "interrupted" 130
    echo "[entrypoint] received signal — partial manifest written" >&2
    exit 130
}
trap on_signal SIGTERM SIGINT

echo "[entrypoint] run_id=$MLEVAL_RUN_ID trajectory_id=$MLEVAL_TRAJECTORY_ID"
echo "[entrypoint] task=$TASK cell=$CELL seed=$SEED time=${TIME_LIMIT_SECONDS}s steps=$STEP_LIMIT"
echo "[entrypoint] aide_sha=$AIDE_SHA node=$NODE_NAME"

# ---- run AIDE ----------------------------------------------------------

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
        agent.code.temp=0 \
        agent.feedback.model="$MLEVAL_LLM_MODEL" \
        agent.feedback.temp=0 \
        agent.steps="$STEP_LIMIT" \
        generate_report="$GENERATE_REPORT" \
        2>&1 | tee "$MLEVAL_OUTPUT_DIR/agent_logs/run.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
AGENT_EXIT=${PIPE_STATUS[0]}
TEE_EXIT=${PIPE_STATUS[1]}
set -e

case "$AGENT_EXIT" in
    0)   STATUS=completed   ;;
    124) STATUS=time_capped ;;
    130) STATUS=interrupted ;;
    *)   STATUS=crashed     ;;
esac

# tee failing means the PVC is full / mount is stale — flag it.
if [[ "$TEE_EXIT" -ne 0 ]]; then
    echo "[entrypoint] WARNING: tee returned $TEE_EXIT (PVC full or mount stale)" >&2
    STATUS="${STATUS}_log_truncated"
fi

write_manifest "$STATUS" "$AGENT_EXIT"

# ---- post-trajectory analyzer chain ------------------------------------
# Best-effort: a failed analyzer should not change the trajectory's exit
# status. Each stage writes its own output that the next stage reads.

run_analyzer() {
    local name="$1"; shift
    if "$@"; then
        echo "[entrypoint] $name OK"
    else
        echo "[entrypoint] $name FAILED (continuing)" >&2
    fi
}

run_analyzer adapter_aide      python -m mleval.analyzer.adapter_aide      "$MLEVAL_OUTPUT_DIR"
run_analyzer stage_classifier  python -m mleval.analyzer.stage_classifier  "$MLEVAL_OUTPUT_DIR"
run_analyzer state_predicates  python -m mleval.analyzer.state_predicates  "$MLEVAL_OUTPUT_DIR"

echo "[entrypoint] finished status=${STATUS} exit=${AGENT_EXIT}"
exit "$AGENT_EXIT"
