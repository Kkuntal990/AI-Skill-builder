#!/usr/bin/env bash
# Pod entrypoint for an mleval trajectory run.
#
# Translates our pod env vars (MLEVAL_*, TASK, CELL, SEED, ...) into the
# config MLEvolve expects, starts the grading-server sidecar, runs the agent,
# runs MLEvolve's submission-fusion post-step, and finalizes outputs in
# $MLEVAL_OUTPUT_DIR.
#
# MLEvolve's documented entry point is `run_single_task.sh` (a shell wrapper)
# but it hardcodes TIME_LIMIT_SECS and assumes the MLE-Bench data layout. We
# replicate its behavior inline so we can:
#   * honor $TIME_LIMIT_SECONDS from our Job spec
#   * point at task data prepared outside MLE-Bench
#   * inject our skill cell wiring later (post-MVP)
#
# Exit codes (mostly forwarded from MLEvolve):
#   0   completed successfully
#   1   missing required env var / image bug
#   124 wall-clock cap exceeded
#   130 interrupted (SIGINT/SIGTERM forwarded by tini)
#   *   any other error from run.py

set -euo pipefail

# ---- required env vars ----------------------------------------------------

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
require MLEVAL_PLAYGROUND_DIR

: "${CELL:=without_skill}"
: "${MLEVAL_LLM_MODEL:=}"
: "${STEP_LIMIT:=500}"
: "${MEMORY_INDEX:=0}"
: "${GRADING_SERVER_ID:=111}"
: "${CPUS_PER_TASK:=8}"           # bumped from MLEvolve's default 21 to fit
                                  # debug-trl-grpo's CPU budget. Override
                                  # in the Job spec for larger tasks.

GRADING_SERVER_PORT=$((5005 + GRADING_SERVER_ID))
export MEMORY_INDEX GRADING_SERVER_PORT

echo "[entrypoint] run_id=$MLEVAL_RUN_ID trajectory_id=$MLEVAL_TRAJECTORY_ID"
echo "[entrypoint] task=$TASK cell=$CELL seed=$SEED time_limit=${TIME_LIMIT_SECONDS}s"
echo "[entrypoint] cpus=$CPUS_PER_TASK steps=$STEP_LIMIT grading_port=$GRADING_SERVER_PORT"

# ---- seed pinning ---------------------------------------------------------

python3 - <<'PYEOF'
import os
import random

import numpy as np
import torch

seed = int(os.environ["SEED"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"[entrypoint] seeds pinned to {seed}", flush=True)
PYEOF

# ---- output dir layout (matches infra/agents/_interface.md §3) -----------

mkdir -p \
    "$MLEVAL_OUTPUT_DIR" \
    "$MLEVAL_OUTPUT_DIR/code" \
    "$MLEVAL_OUTPUT_DIR/agent_native_logs" \
    "$MLEVAL_PLAYGROUND_DIR"

START_EPOCH=$(date -u +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---- working dir & MLEvolve env ------------------------------------------

cd /opt/mlevolve

# DATASET_DIR is consumed by both launch_server.sh and run.py. MLEvolve
# expects the MLE-Bench layout: ${DATASET_DIR}/${EXP_ID}/prepared/public/
export DATASET_DIR="$MLEVAL_TASK_DATA_DIR"

# Format TIME_LIMIT like "Hhrs Mmins Ssecs" — run.py logs it for humans.
format_time() {
    local t=$1
    echo "$((t / 3600))hrs $(((t % 3600) / 60))mins $((t % 60))secs"
}
export TIME_LIMIT
TIME_LIMIT="$(format_time "$TIME_LIMIT_SECONDS")"
export STEP_LIMIT

# ---- grading server sidecar (background) ---------------------------------

echo "[entrypoint] starting grading server on port $GRADING_SERVER_PORT"
bash /opt/mlevolve/launch_server.sh "$GRADING_SERVER_ID" \
    > "$MLEVAL_OUTPUT_DIR/agent_native_logs/grading_server.out" 2>&1 || true

# Health-poll the server (MLEvolve waits ~30s; we match).
WAITED=0
while (( WAITED < 30 )); do
    if curl -sf "http://127.0.0.1:${GRADING_SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "[entrypoint] grading server ready"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
if (( WAITED >= 30 )); then
    echo "[entrypoint] WARNING: grading server health probe timed out; continuing anyway" >&2
fi

# ---- MLEvolve agent loop --------------------------------------------------
#
# OmegaConf-style positional `key=value` overrides, mirroring upstream's
# run_single_task.sh. We add `agent.seed=$SEED` so the seed pinned above
# also flows into MLEvolve's own RNGs (set_global_seed is called from
# run.py and reads cfg.agent.seed).

EXP_ID="$TASK"
DATA_DIR="${DATASET_DIR}/${EXP_ID}/prepared/public"
DESC_FILE="${DATA_DIR}/description.md"

if [[ ! -d "$DATA_DIR" || ! -f "$DESC_FILE" ]]; then
    echo "[entrypoint] WARNING: expected MLE-Bench layout not found:" >&2
    echo "             $DATA_DIR / $DESC_FILE" >&2
    echo "             Continuing — run.py may fail or you may have prepared" >&2
    echo "             a flat layout for this task." >&2
fi

set +e
CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" \
timeout --foreground --signal=TERM --kill-after=10s "${TIME_LIMIT_SECONDS}s" \
    python run.py \
        exp_id="$EXP_ID" \
        dataset_dir="$DATASET_DIR" \
        data_dir="$DATA_DIR" \
        desc_file="$DESC_FILE" \
        exp_name="$MLEVAL_TRAJECTORY_ID" \
        agent.seed="$SEED" \
        start_cpu_id=0 \
        cpu_number="$CPUS_PER_TASK" \
        2>&1 | tee "$MLEVAL_OUTPUT_DIR/agent_native_logs/run.log"
AGENT_EXIT=${PIPESTATUS[0]}
set -e

# ---- submission fusion (MLEvolve's post-step) ----------------------------

if [[ "$AGENT_EXIT" -eq 0 || "$AGENT_EXIT" -eq 124 ]]; then
    echo "[entrypoint] running submission fusion"
    python utils/submission_fusion_utils.py \
        --task_id "$EXP_ID" \
        --exp_name "$MLEVAL_TRAJECTORY_ID" \
        2>&1 | tee "$MLEVAL_OUTPUT_DIR/agent_native_logs/submission_fusion.log" || true
fi

END_EPOCH=$(date -u +%s)
END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
WALL_CLOCK=$((END_EPOCH - START_EPOCH))

# ---- output finalization --------------------------------------------------

# Copy any submission MLEvolve produced into the pod's PVC-backed output dir.
shopt -s nullglob globstar
for s in "$MLEVAL_PLAYGROUND_DIR"/submission.* "$MLEVAL_PLAYGROUND_DIR"/**/submission.* /opt/mlevolve/workspaces/**/submission.*; do
    cp -v "$s" "$MLEVAL_OUTPUT_DIR/" 2>/dev/null || true
done
shopt -u nullglob globstar

# Derive JSON-safe fields.
if [[ "$CELL" == "with_skill" ]]; then
    WITH_SKILL_JSON=true
else
    WITH_SKILL_JSON=false
fi

case "$AGENT_EXIT" in
    0)   STATUS=completed   ;;
    124) STATUS=time_capped ;;
    130) STATUS=interrupted ;;
    *)   STATUS=crashed     ;;
esac

# Minimal manifest. Full schema (per infra/agents/_interface.md §5) lands
# once the in-pod adapter (#61) is wired up post-MVP.
cat > "$MLEVAL_OUTPUT_DIR/manifest.json" <<JSON
{
  "schema_version": "1.0",
  "run_id": "${MLEVAL_RUN_ID}",
  "trajectory_id": "${MLEVAL_TRAJECTORY_ID}",
  "task": { "name": "${TASK}" },
  "cell": { "with_skill": ${WITH_SKILL_JSON} },
  "seed": ${SEED},
  "timestamps": {
    "started_at": "${START_ISO}",
    "ended_at": "${END_ISO}",
    "wall_clock_sec": ${WALL_CLOCK}
  },
  "result": {
    "status": "${STATUS}",
    "exit_code": ${AGENT_EXIT}
  }
}
JSON

# ---- shutdown grading server ---------------------------------------------

PID_FILE="/opt/mlevolve/grading_servers/grading_server_${GRADING_SERVER_ID}.pid"
if [[ -f "$PID_FILE" ]]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
fi

echo "[entrypoint] finished in ${WALL_CLOCK}s (exit ${AGENT_EXIT}, status=${STATUS})"
exit "$AGENT_EXIT"
