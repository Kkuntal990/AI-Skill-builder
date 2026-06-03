#!/usr/bin/env bash
# MLEvolve trajectory entrypoint.
#
# Contract (mirrors AIDE's so the orchestrator stays agent-agnostic):
#
#   In env:
#     MLEVAL_RUN_ID, MLEVAL_TRAJECTORY_ID, TASK, CELL, SEED
#     TIME_LIMIT_SECONDS, STEP_LIMIT
#     MLEVAL_LLM_MODEL, MLEVAL_LLM_API_KEY (or OPENAI_API_KEY)
#     OPENAI_BASE_URL (OpenAI-compatible endpoint, e.g. OpenRouter)
#     MLEVAL_OUTPUT_DIR, MLEVAL_TASK_INSTRUCTION_PATH, MLEVAL_TASK_DATA_DIR
#     MLEVAL_SKILL_PATH (optional)
#
#   Out (in MLEVAL_OUTPUT_DIR):
#     manifest.json, prompts.jsonl, mem_sample.csv, pip_freeze.txt
#     mlevolve_runs/<ts>_<exp>/{journal.json,metric.txt,workspaces/}
#     trajectory.jsonl  (post-run, adapter_mlevolve)
#
# Spike-mode choices:
#   - use_grading_server: false in config → mlebench is never imported
#   - use_global_memory: false → no FAISS / bge embedding
#   - use_coldstart: false → coldstart classifier never runs
#   - LLM API key flows via OPENAI_API_KEY env (the OpenAI client reads it
#     when api_key="" in config — see mlevolve_sidecar/openai_apikey_env.py).

set -uo pipefail

OUT_DIR="${MLEVAL_OUTPUT_DIR:?MLEVAL_OUTPUT_DIR required}"
TASK="${TASK:?TASK required}"
CELL="${CELL:?CELL required}"
SEED="${SEED:?SEED required}"
TRAJECTORY_ID="${MLEVAL_TRAJECTORY_ID:?MLEVAL_TRAJECTORY_ID required}"
RUN_ID="${MLEVAL_RUN_ID:?MLEVAL_RUN_ID required}"
INSTRUCTION_PATH="${MLEVAL_TASK_INSTRUCTION_PATH:?MLEVAL_TASK_INSTRUCTION_PATH required}"
DATA_DIR="${MLEVAL_TASK_DATA_DIR:?MLEVAL_TASK_DATA_DIR required}"
# Required even though config.yaml has a default — empty model name silently
# breaks MLEvolve's per-stage dispatch. Caught here, not after a 5-min image
# pull and a wasted run.
MLEVAL_LLM_MODEL="${MLEVAL_LLM_MODEL:?MLEVAL_LLM_MODEL required}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-1800}"
STEP_LIMIT="${STEP_LIMIT:-5}"
LLM_TIMEOUT_SEC="${MLEVAL_LLM_TIMEOUT_SEC:-120}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
# Per-exec subprocess timeout (MLEvolve's `exec.timeout`). DECOUPLED from
# TIME_LIMIT_SECONDS so a single training exec can't consume the whole
# trajectory budget — defaults to TIME_LIMIT_SECONDS / 2 (one full
# attempt + one debug retry within the wall budget). For tasks that need
# a full training+eval pass in a single exec (e.g. SAMSum at 40 min),
# pass MLEVAL_EXEC_TIMEOUT_SEC explicitly with TIME_LIMIT_SECONDS at
# least 2x larger.
MLEVAL_EXEC_TIMEOUT_SEC="${MLEVAL_EXEC_TIMEOUT_SEC:-$((TIME_LIMIT_SECONDS / 2))}"

# MLEvolve reads OPENAI_API_KEY; map MLEVAL_LLM_API_KEY into it if set.
export OPENAI_API_KEY="${MLEVAL_LLM_API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_BASE_URL MLEVAL_LLM_MODEL

# prompt_logger reads MLEVAL_PROMPTS_LOG at import time; without this it
# resolves to ``./prompts.jsonl`` against cwd (/workspace/mlevolve), which
# would silently lose every prompt. Job manifest sets this too, but the
# helper-pod smoke path goes through the entrypoint, so set it here.
export MLEVAL_PROMPTS_LOG="${MLEVAL_PROMPTS_LOG:-$OUT_DIR/prompts.jsonl}"

mkdir -p "$OUT_DIR" "$OUT_DIR/agent_logs"

echo "[entrypoint] run_id=$RUN_ID trajectory_id=$TRAJECTORY_ID"
echo "[entrypoint] task=$TASK cell=$CELL seed=$SEED steps=${STEP_LIMIT}"
echo "[entrypoint] wall_cap=${TIME_LIMIT_SECONDS}s (graceful) | per-exec=${MLEVAL_EXEC_TIMEOUT_SEC}s | llm=${MLEVAL_LLM_MODEL} llm_timeout=${LLM_TIMEOUT_SEC}s base=${OPENAI_BASE_URL}"

# Full pip freeze saved to OUT_DIR for post-mortem analysis (which versions
# the trajectory actually ran against). Not surfaced to the agent — we
# rely on upstream MLEvolve's stock 15-package env hint
# (agents/prompts/environment.py) for prompt-level package guidance.
# Spike-009 ImportError protection comes from requirements.txt pinning to
# mid-2024 versions that match the LLM's training-data prior.
pip freeze > "$OUT_DIR/pip_freeze.txt" 2>/dev/null || true

# -------- Build MLEvolve's expected dataset layout ----------------------
# MLEvolve assumes <dataset_dir>/<exp_id>/prepared/public/{input,description.md}.
# We don't have an mle-bench tree, so synthesise the minimum shape here.
# Symlink (don't copy) the task data, copy description so any LLM-generated
# code that mutates it doesn't corrupt the PVC source.
DATASET_DIR="$OUT_DIR/_dataset_shim"
EXP_ID="${TASK//[^a-zA-Z0-9_-]/_}-${TRAJECTORY_ID//[^a-zA-Z0-9_-]/_}"
PUBLIC_DIR="$DATASET_DIR/$EXP_ID/prepared/public"
mkdir -p "$PUBLIC_DIR/input"

for entry in "$DATA_DIR"/*; do
    [ -e "$entry" ] || continue
    ln -sf "$entry" "$PUBLIC_DIR/input/$(basename "$entry")"
done
cp -f "$INSTRUCTION_PATH" "$PUBLIC_DIR/description.md"

# Optional skill(s): forward MLEVAL_SKILL_PATHS (preferred, colon-separated)
# or MLEVAL_SKILL_PATH (singular, back-compat) to the sidecar's
# skill_retriever. The retriever loads SKILL.md + references/*.md for each
# resolved skill at import time and patches get_prompt_environment to splice
# an L1 catalog + L2 full body into prompt["Instructions"]. This injection
# slot is reached by both the regular draft path and the stepwise StepAgent
# / MetaAgent (which copy prompt_base["Instructions"]).
if [ -n "${MLEVAL_SKILL_PATHS:-}" ]; then
    export MLEVAL_SKILL_PATHS
    echo "[entrypoint] skill paths: $MLEVAL_SKILL_PATHS"
elif [ -n "${MLEVAL_SKILL_PATH:-}" ] && [ -e "$MLEVAL_SKILL_PATH" ]; then
    export MLEVAL_SKILL_PATH
    echo "[entrypoint] skill: $MLEVAL_SKILL_PATH (singular, back-compat)"
fi

# -------- Render config + overwrite MLEvolve's default ------------------
# load_cfg() defaults to config/config.yaml in MLEvolve's source tree.
# We overwrite that file with our runtime-rendered version, then run.
RUN_CFG="$OUT_DIR/_runtime_config.yaml"
RUNS_DIR="$OUT_DIR/mlevolve_runs"
mkdir -p "$RUNS_DIR"

export EXP_ID DATASET_DIR PUBLIC_DIR RUNS_DIR \
       STEP_LIMIT TIME_LIMIT_SECONDS MLEVAL_EXEC_TIMEOUT_SEC LLM_TIMEOUT_SEC \
       MLEVAL_LLM_MODEL OPENAI_BASE_URL TRAJECTORY_ID
envsubst < /workspace/mlevolve_config.yaml > "$RUN_CFG"
cp -f "$RUN_CFG" /workspace/mlevolve/config/config.yaml

# -------- Memory sampler -----------------------------------------------
SAMPLER_LOG="$OUT_DIR/mem_sample.csv"
echo "ts_unix,pgrp_rss_kb,pod_mem_used_kb,gpu_util_pct,gpu_mem_used_mib" > "$SAMPLER_LOG"

# -------- Launch MLEvolve ----------------------------------------------
INNER_LOG="$OUT_DIR/agent_logs/mlevolve_stdout.log"
cd /workspace/mlevolve

# setsid gives the agent tree its own session for clean killpg cleanup
# (MLEvolve uses subprocess.Popen per step, but agent itself still needs
# its own session so the entrypoint can signal-group it on SIGTERM).
setsid python3 /workspace/run_mlevolve.py \
    > "$INNER_LOG" 2>&1 &
AGENT_PID=$!
AGENT_PGID=$(ps -o pgid= -p "$AGENT_PID" | tr -d ' ' || echo "$AGENT_PID")
echo "[entrypoint] mlevolve PID=$AGENT_PID PGID=$AGENT_PGID"

(
    while kill -0 "$AGENT_PID" 2>/dev/null; do
        ts=$(date +%s)
        rss=$(ps -eo pid,pgid,rss --no-headers 2>/dev/null \
              | awk -v pgid="$AGENT_PGID" '$2==pgid {s+=$3} END{print s+0}')
        pod=$(awk '/^MemTotal:/{mt=$2} /^MemAvailable:/{ma=$2} END{print mt-ma}' /proc/meminfo)
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
              | head -1 | tr -d ' ')
        [ -z "$gpu" ] && gpu="0,0"
        echo "${ts},${rss},${pod},${gpu}" >> "$SAMPLER_LOG"
        sleep 5
    done
) &
SAMPLER_PID=$!

# Wall-clock watchdog — graceful soft kill of the agent PGID after
# TIME_LIMIT_SECONDS. MLEvolve's `agent.time_limit` is consulted by the
# search loop for branching decisions but does NOT gate the main loop
# (verified at upstream run.py:137 — the `while completed < total_steps`
# check has no time clause). Without this watchdog, the only wall-clock
# cap is the K8s activeDeadlineSeconds, which SIGKILLs the pod and skips
# post-run analyzer + manifest write.
#
# The watchdog sends SIGTERM to the agent's process group, then SIGKILL
# after a 60s grace. The entrypoint's existing SIGTERM trap forwards to
# PGID, but the watchdog runs in parallel so it works whether the trap
# fires or not. Analyzer + manifest then run as normal because the wait
# below returns and we continue to the post-run section.
(
    sleep "$TIME_LIMIT_SECONDS"
    if kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "[entrypoint] WALL CAP ($TIME_LIMIT_SECONDS sec) reached — SIGTERM to PGID=$AGENT_PGID"
        kill -TERM -"$AGENT_PGID" 2>/dev/null || true
        sleep 60
        if kill -0 "$AGENT_PID" 2>/dev/null; then
            echo "[entrypoint] WALL CAP — SIGKILL after 60s grace"
            kill -KILL -"$AGENT_PGID" 2>/dev/null || true
        fi
    fi
) &
WATCHDOG_PID=$!

_term() {
    echo "[entrypoint] received signal, forwarding to mlevolve PGID=$AGENT_PGID"
    kill -TERM -"$AGENT_PGID" 2>/dev/null || kill -TERM "$AGENT_PID" 2>/dev/null || true
}
trap _term TERM INT

# wait returns on signal interrupt (bash sets exit > 128 + signum) — restart
# until the agent actually dies (or watchdog SIGKILLs it). Spike-012 lesson:
# without this loop, the FIRST SIGTERM caused wait to return with $? = 143
# while the agent was still mid-PyTorch-op; the analyzer chain then ran
# while the agent was alive, race-conditioned, and never produced
# trajectory.jsonl / manifest.json.
while true; do
    wait "$AGENT_PID" 2>/dev/null
    INNER_EXIT=$?
    # bash wait returns >128 on signal interrupt; if the agent is still alive
    # (kill -0 succeeds), retry wait. If it's dead, fall through.
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        break
    fi
    echo "[entrypoint] wait interrupted by signal (exit=$INNER_EXIT) but agent PID=$AGENT_PID still alive — retrying"
done

# Disable signal forwarding during cleanup so additional SIGTERMs from K8s
# (or stray watchdog signals) cannot interrupt the analyzer chain. We have
# K8s terminationGracePeriodSeconds=90 to finish manifest + trajectory.jsonl.
trap '' TERM INT

kill "$SAMPLER_PID" 2>/dev/null || true
kill "$WATCHDOG_PID" 2>/dev/null || true

echo "[entrypoint] mlevolve exited with $INNER_EXIT"

# -------- Post-run analyzer chain --------------------------------------
python3 -m mleval.analyzer.adapter_mlevolve "$OUT_DIR" 2>&1 | tail -10 || \
    echo "[entrypoint] adapter_mlevolve failed; trajectory.jsonl absent"
python3 -m mleval.analyzer.stage_classifier "$OUT_DIR" 2>&1 | tail -5 || \
    echo "[entrypoint] stage_classifier failed (non-fatal)"

# -------- Manifest -----------------------------------------------------
python3 <<PYEOF || echo "[entrypoint] manifest write failed (non-fatal)"
import json, os, socket, time
out = os.environ['MLEVAL_OUTPUT_DIR']
m = {
    'schema_version': '1.0',
    'run_id': os.environ['MLEVAL_RUN_ID'],
    'trajectory_id': os.environ['MLEVAL_TRAJECTORY_ID'],
    'task': {'name': os.environ['TASK']},
    'cell': {'name': os.environ['CELL']},
    'seed': int(os.environ['SEED']),
    'agent': {
        'name': 'mlevolve',
        'version': 'vendored-26bde89',
        'llm_model': os.environ['MLEVAL_LLM_MODEL'],
    },
    'pod': {'hostname': socket.gethostname(), 'node': os.environ.get('KUBE_NODE_NAME', 'unknown')},
    'timestamps': {'ended_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
    'result': {'exit_code': $INNER_EXIT, 'status': 'completed' if $INNER_EXIT == 0 else 'crashed'},
}
with open(os.path.join(out, 'manifest.json'), 'w') as f:
    json.dump(m, f, indent=2)
print('[entrypoint] manifest written')
PYEOF

echo "[entrypoint] finished exit=$INNER_EXIT"
exit "$INNER_EXIT"
