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
# kills the AIDE process group, then lets the script fall through to
# finalize() (which writes manifest + runs analyzers).
#
# Critical: bash `wait` is signal-interruptible, BUT a foreground command in
# a pipeline (e.g. `timeout ... | tee ...`) blocks the trap from running
# until the pipeline finishes. We work around this by running the AIDE
# pipeline backgrounded in its OWN process group (via setsid) and waiting
# with the bash `wait` builtin. Then this trap actively SIGTERMs the
# process group when k8s asks us to shut down.
on_signal() {
    SIGNAL_RECEIVED=true
    STATUS="interrupted"
    AGENT_EXIT=130
    trap '' SIGTERM SIGINT  # don't recurse
    echo "[entrypoint] signal received — killing AIDE process group ${AGENT_PGID:-?}" >&2
    if [[ -n "$AGENT_PGID" ]]; then
        kill -TERM -- "-${AGENT_PGID}" 2>/dev/null || true
        # Brief grace for AIDE to finish the in-flight LLM call + tee flush.
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            sleep 1
            kill -0 -- "-${AGENT_PGID}" 2>/dev/null || break
        done
        kill -KILL -- "-${AGENT_PGID}" 2>/dev/null || true
    fi
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

LOG_FILE="$MLEVAL_OUTPUT_DIR/agent_logs/run.log"
EXIT_FILE="$MLEVAL_OUTPUT_DIR/.agent_exit"
rm -f "$EXIT_FILE"

set +e
# setsid puts the entire AIDE pipeline (timeout → python → tee) into a NEW
# session/process group. We background it so the parent shell can `wait` on
# it; bash `wait` IS signal-interruptible. When the trap fires, it sends
# SIGTERM to the whole group via `kill -- -$AGENT_PGID`, killing timeout +
# python + tee cleanly, then `wait` returns and the script continues to
# finalize(). PIPESTATUS only valid in the inner shell, so the inner shell
# writes the exit codes to $EXIT_FILE which we source after wait returns.
CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" setsid bash -c '
    timeout --foreground --signal=TERM --kill-after=10s "${1}s" \
        python /workspace/run_aide.py \
            data_dir="$2" \
            desc_file="$3" \
            log_dir="$4" \
            workspace_dir="$5" \
            exp_name="$6" \
            agent.code.model="$7" \
            agent.code.temp=0 \
            agent.feedback.model="$7" \
            agent.feedback.temp=0 \
            agent.steps="$8" \
            generate_report="$9" \
            2>&1 | tee "${10}"
    printf "INNER_AGENT_EXIT=%s\nINNER_TEE_EXIT=%s\n" \
        "${PIPESTATUS[0]}" "${PIPESTATUS[1]}" > "${11}"
' bash \
    "$TIME_LIMIT_SECONDS" \
    "$MLEVAL_TASK_DATA_DIR" \
    "$MLEVAL_TASK_INSTRUCTION_PATH" \
    "$MLEVAL_OUTPUT_DIR/aide_logs" \
    "$MLEVAL_OUTPUT_DIR/aide_workspace" \
    "$MLEVAL_TRAJECTORY_ID" \
    "$MLEVAL_LLM_MODEL" \
    "$STEP_LIMIT" \
    "$GENERATE_REPORT" \
    "$LOG_FILE" \
    "$EXIT_FILE" &
AGENT_PID=$!
# PGID == PID for a session leader created by setsid. ps confirms.
AGENT_PGID=$(ps -o pgid= -p "$AGENT_PID" 2>/dev/null | tr -d ' ')
[[ -z "$AGENT_PGID" ]] && AGENT_PGID="$AGENT_PID"
echo "[entrypoint] AIDE PID=$AGENT_PID PGID=$AGENT_PGID"

# Memory sampler — code review M3 / answers "why did we OOM at N GiB" by
# logging actual peak RSS + pod-level memory + GPU util every 5s while AIDE
# is alive. Self-terminates when AGENT_PID exits; no separate kill needed.
# Background subshell, set -e doesn't propagate.
SAMPLER_LOG="$MLEVAL_OUTPUT_DIR/mem_sample.csv"
echo "ts_unix,pgrp_rss_kb,pod_mem_used_kb,gpu_util_pct,gpu_mem_used_mib" > "$SAMPLER_LOG"
( while kill -0 "$AGENT_PID" 2>/dev/null; do
    ts=$(date +%s)
    rss=$(ps -eo pid,pgid,rss --no-headers 2>/dev/null \
          | awk -v pgid="$AGENT_PGID" '$2==pgid {s+=$3} END{print s+0}')
    pod=$(awk '/^MemTotal:/{mt=$2} /^MemAvailable:/{ma=$2} END{print mt-ma}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
          | head -1 | tr -d ' ')
    [[ -z "$gpu" ]] && gpu="0,0"
    echo "${ts},${rss},${pod},${gpu}" >> "$SAMPLER_LOG"
    sleep 5
done ) &
SAMPLER_PID=$!

wait "$AGENT_PID"
WAIT_EXIT=$?
set -e

# Recover the real pipeline exit codes from the inner shell's $EXIT_FILE.
INNER_AGENT_EXIT=
INNER_TEE_EXIT=
if [[ -f "$EXIT_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$EXIT_FILE"
fi
RAW_AGENT_EXIT="${INNER_AGENT_EXIT:-$WAIT_EXIT}"
TEE_EXIT="${INNER_TEE_EXIT:-0}"

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
