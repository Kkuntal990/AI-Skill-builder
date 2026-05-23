#!/usr/bin/env bash
# AIDE pod entrypoint. Used by Job pods (helper pod overrides `command:` to
# run `jupyter lab` directly).
#
# Flow:
#   1. Validate env
#   2. pip install per-task + per-skill requirements into PVC-cached pip dir
#   3. Run AIDE (timeout-wrapped)
#   4. Always: write_manifest + run analyzer chain — even on SIGTERM/SIGKILL-pending
#
# Exit codes:
#   0   completed
#   1   missing required env / image bug / pip install failed
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
: "${MLEVAL_TASK_REQS_PATH:=}"
: "${MLEVAL_SKILL_REQS_PATH:=}"
: "${GENERATE_REPORT:=false}"
: "${MLEVAL_LLM_TIMEOUT_SEC:=120}"
: "${MLEVAL_PIP_CACHE_DIR:=/results/.pip-cache}"
export MLEVAL_LLM_TIMEOUT_SEC

mkdir -p "$MLEVAL_OUTPUT_DIR" "$MLEVAL_OUTPUT_DIR/agent_logs" "$MLEVAL_PIP_CACHE_DIR"
export MLEVAL_PROMPTS_LOG="$MLEVAL_OUTPUT_DIR/prompts.jsonl"

# Route AIDE's openai backend at OpenRouter. With OPENAI_BASE_URL set AND the
# model not matching gpt-*/o*/codex, AIDE picks `use_chat_api=true` which uses
# chat.completions.create (OpenRouter-compatible).
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

# Defaults that get overwritten when AIDE finishes (or signal is received).
STATUS="crashed"
AGENT_EXIT=1
SIGNAL_RECEIVED=false
ANALYZER_DONE=false

write_manifest() {
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
    "data_dir": "${MLEVAL_TASK_DATA_DIR}",
    "requirements_path": "${MLEVAL_TASK_REQS_PATH}"
  },
  "cell": {
    "name": "${CELL}",
    "with_skill": ${WITH_SKILL_JSON},
    "skill_path": "${MLEVAL_SKILL_PATH}",
    "skill_sha256": "${SKILL_SHA}",
    "skill_requirements_path": "${MLEVAL_SKILL_REQS_PATH}"
  },
  "seed": ${SEED},
  "agent": {
    "name": "aide",
    "version": "${AIDE_SHA}",
    "container_image": "${HOSTNAME_K8S}",
    "llm_model": "${MLEVAL_LLM_MODEL}",
    "llm_timeout_sec": ${MLEVAL_LLM_TIMEOUT_SEC}
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
    "status": "${STATUS}",
    "exit_code": ${AGENT_EXIT},
    "signal_received": ${SIGNAL_RECEIVED}
  }
}
JSON
}

run_analyzer() {
    local name="$1"; shift
    if timeout 30s "$@"; then
        echo "[entrypoint] $name OK"
    else
        echo "[entrypoint] $name FAILED rc=$? (continuing)" >&2
    fi
}

finalize() {
    # Idempotent: safe to call from both the trap and the normal exit path.
    if [[ "$ANALYZER_DONE" == "true" ]]; then
        return 0
    fi
    write_manifest
    run_analyzer adapter_aide      python -m mleval.analyzer.adapter_aide      "$MLEVAL_OUTPUT_DIR"
    run_analyzer stage_classifier  python -m mleval.analyzer.stage_classifier  "$MLEVAL_OUTPUT_DIR"
    run_analyzer state_predicates  python -m mleval.analyzer.state_predicates  "$MLEVAL_OUTPUT_DIR"
    ANALYZER_DONE=true
}

# Preempt-safe trap: SIGTERM (kubelet preempt, node drain, kubectl delete)
# sets a flag and lets the AIDE loop unwind. The post-AIDE block then writes
# the manifest with status=interrupted and runs analyzers within the
# remaining terminationGracePeriodSeconds window (default 30s; we set 90s
# in the Job templates to give analyzers room).
on_signal() {
    SIGNAL_RECEIVED=true
    STATUS="interrupted"
    AGENT_EXIT=130
    trap '' SIGTERM SIGINT  # don't recurse
    echo "[entrypoint] signal received — will finalize with status=interrupted" >&2
}
trap on_signal SIGTERM SIGINT

echo "[entrypoint] run_id=$MLEVAL_RUN_ID trajectory_id=$MLEVAL_TRAJECTORY_ID"
echo "[entrypoint] task=$TASK cell=$CELL seed=$SEED time=${TIME_LIMIT_SECONDS}s steps=$STEP_LIMIT"
echo "[entrypoint] aide_sha=$AIDE_SHA node=$NODE_NAME"
echo "[entrypoint] llm=$MLEVAL_LLM_MODEL timeout=${MLEVAL_LLM_TIMEOUT_SEC}s"

# ---- per-task / per-skill pip install ----------------------------------
# Both cells get the same library universe so the comparison isolates the
# effect of the skill's prompt text (not its library availability). Wheels
# are cached on the PVC so subsequent trajectories reuse the downloads.

install_reqs() {
    local label="$1" reqs="$2"
    if [[ -z "$reqs" ]]; then
        echo "[entrypoint] no $label requirements declared (skip)"
        return 0
    fi
    if [[ ! -f "$reqs" ]]; then
        echo "[entrypoint] WARNING: $label requirements file $reqs missing (skip)" >&2
        return 0
    fi
    if [[ ! -s "$reqs" ]] || ! grep -qE '^[^#[:space:]]' "$reqs"; then
        echo "[entrypoint] $label requirements file $reqs is empty (skip)"
        return 0
    fi
    echo "[entrypoint] installing $label requirements from $reqs"
    pip install --no-warn-script-location --cache-dir "$MLEVAL_PIP_CACHE_DIR" --quiet -r "$reqs"
}

install_reqs task  "$MLEVAL_TASK_REQS_PATH"
install_reqs skill "$MLEVAL_SKILL_REQS_PATH"

# Record exact installed-package state for reproducibility.
pip freeze > "$MLEVAL_OUTPUT_DIR/pip_freeze.txt" 2>/dev/null || true

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
RAW_AGENT_EXIT=${PIPE_STATUS[0]}
TEE_EXIT=${PIPE_STATUS[1]}
set -e

# Trap may have already set STATUS/AGENT_EXIT to interrupted/130. Only
# overwrite from AIDE's exit code if we didn't receive a signal.
if [[ "$SIGNAL_RECEIVED" != "true" ]]; then
    AGENT_EXIT="$RAW_AGENT_EXIT"
    case "$AGENT_EXIT" in
        0)   STATUS=completed   ;;
        124) STATUS=time_capped ;;
        130) STATUS=interrupted ;;
        *)   STATUS=crashed     ;;
    esac
fi

if [[ "$TEE_EXIT" -ne 0 ]]; then
    echo "[entrypoint] WARNING: tee returned $TEE_EXIT (PVC full or mount stale)" >&2
    STATUS="${STATUS}_log_truncated"
fi

# ---- always write manifest + run analyzer chain ------------------------
finalize

echo "[entrypoint] finished status=${STATUS} exit=${AGENT_EXIT}"
exit "$AGENT_EXIT"
