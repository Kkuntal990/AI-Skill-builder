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

export TASK

# GOLD references (id + answer). Used ONLY by the post-run held-out grader,
# which runs in THIS entrypoint (parent process) — NEVER by the agent. Under the
# held-out design (docs/eval/task-authoring.md C3) we must NOT export this to the
# agent's node subprocesses, or self-validation would hand them the path to the
# answers. Kept as a plain (un-exported) shell var.
REFS_PATH="${MLEVAL_TASK_REFS_PATH:-$(dirname "$DATA_DIR")/refs/test_refs.csv}"

# Pin the metric optimization direction (mlevolve_sidecar/metric_direction.py).
# MLEvolve's LLM determine_metric_direction nondeterministically flips the
# maximize/minimize boolean (spike-026: inverted the without-skill search).
# Every harness task maximizes its metric (gsm8k exact-match, samsum ROUGE-L),
# so pin maximize by default; override per-task with MLEVAL_METRIC_MAXIMIZE=0
# for a future error/loss metric. Empty => fall back to MLEvolve's LLM guess.
export MLEVAL_METRIC_MAXIMIZE="${MLEVAL_METRIC_MAXIMIZE:-1}"
echo "[entrypoint] metric direction pinned: MLEVAL_METRIC_MAXIMIZE=$MLEVAL_METRIC_MAXIMIZE (1=maximize)"

# PUBLIC id-set for the in-run format validator (mleval.grader.validate — the
# MLE-Bench validate_submission affordance, format-only, no score). Prefer the
# agent-facing sample_submission.csv (id column, NO targets) so node subprocesses
# can self-check format without ever seeing the gold path. Back-compat: tasks
# staged before the split have no public sample_submission.csv — fall back to
# exporting the gold refs (the validator reads only its id column, so no leak via
# the validator itself, though the path is then visible — pre-C3 behaviour).
if [ -f "$DATA_DIR/sample_submission.csv" ]; then
    export MLEVAL_TASK_IDSET_PATH="$DATA_DIR/sample_submission.csv"
    echo "[entrypoint] validator id-set (public): $MLEVAL_TASK_IDSET_PATH; gold refs withheld from agent env"
else
    export MLEVAL_TASK_REFS_PATH="$REFS_PATH"
    echo "[entrypoint] no public id-set; exporting refs for validator (back-compat): $REFS_PATH"
fi
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

# description.md = the task instruction ONLY (clean, task-specific text).
#
# We intentionally do NOT prepend the shared benchmark/harness rules here.
# MLEvolve delivers harness info (resource budget, submission contract,
# anti-cheat) through its OWN per-node channel — agents/prompts/impl_guideline.py
# (get_impl_guideline), injected into every codegen prompt — NOT through the task
# description. This mirrors how MLEvolve+MLE-Bench actually runs (desc_file =
# description.md only; instructions.txt is never loaded). Our eval-specific rules
# (held-out test, validate tool, submission-is-the-score, no-train-on-test) are
# appended to that same impl_guideline via mlevolve_sidecar/eval_harness.py
# (injected by skill_injector's wrapper), so they reach BOTH cells identically
#
# Why this matters: MLEvolve USED to run an LLM "clean_task_desc" rewrite over
# description.md at init — nondeterministic and prone to gutting the task
# (spike-025: hallucinated "Unihandecode Ecosphere"; mvp-029 with-skill: cleaned
# to ~empty -> agent built a tabular regressor, all nodes invalid). That LLM
# rewrite is now BYPASSED at build time (patches/de_kaggle.py), so description.md
# reaches the agent VERBATIM and identically across cells/runs. We still keep
# description.md = clean task only (harness rules ride the impl_guideline).
cp -f "$INSTRUCTION_PATH" "$PUBLIC_DIR/description.md"
echo "[entrypoint] description.md = task instruction only (harness via impl_guideline)"

# Optional skill(s): the sidecar's skill_retriever loads a library at import
# time and skill_injector patches the 4 codegen agents (draft/improve/debug/
# evolution) — Tier-0 catalog into every node + a per-node model selector that
# loads the relevant skill(s)+references (Anthropic progressive disclosure).
# Preferred: MLEVAL_SKILL_LIBRARY (a directory of */SKILL.md, all available).
# Back-compat: MLEVAL_SKILL_PATHS (colon-separated) / MLEVAL_SKILL_PATH (one).
if [ -n "${MLEVAL_SKILL_LIBRARY:-}" ] && [ -d "$MLEVAL_SKILL_LIBRARY" ]; then
    export MLEVAL_SKILL_LIBRARY
    echo "[entrypoint] skill library: $MLEVAL_SKILL_LIBRARY"
elif [ -n "${MLEVAL_SKILL_PATHS:-}" ]; then
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

# Trajectory wall-clock start, stamped into the manifest below.
RUN_START_EPOCH=$(date +%s)

# setsid gives the agent tree its own session for clean killpg cleanup
# (MLEvolve uses subprocess.Popen per step, but agent itself still needs
# its own session so the entrypoint can signal-group it on SIGTERM).
setsid python3 /workspace/run_mlevolve.py \
    > "$INNER_LOG" 2>&1 &
AGENT_PID=$!
# setsid makes the agent a session+group leader, so its PGID == its PID.
# But `ps` can race the child's setsid() call and momentarily report the
# entrypoint's OWN pgid. If we then signalled that pgid we'd signal the
# entrypoint shell itself, re-entering the trap forever (the spike-027 2655x
# "received signal" storm). Guard: if the resolved pgid is empty or equals the
# entrypoint's own group, fall back to AGENT_PID (which setsid guarantees IS
# the agent's pgid).
SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ' 2>/dev/null)
AGENT_PGID=$(ps -o pgid= -p "$AGENT_PID" | tr -d ' ' 2>/dev/null)
if [ -z "$AGENT_PGID" ] || [ "$AGENT_PGID" = "$SELF_PGID" ]; then
    AGENT_PGID="$AGENT_PID"
fi
echo "[entrypoint] mlevolve PID=$AGENT_PID PGID=$AGENT_PGID (self_pgid=$SELF_PGID)"

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
WATCHDOG_SENTINEL="$OUT_DIR/.watchdog_fired"
rm -f "$WATCHDOG_SENTINEL"
(
    sleep "$TIME_LIMIT_SECONDS"
    if kill -0 "$AGENT_PID" 2>/dev/null; then
        # Mark this as a graceful time-budget stop (an expected, successful
        # harvest — NOT a crash). The main shell reads this to exit 0 so the
        # Job Completes instead of retrying from scratch.
        : > "$WATCHDOG_SENTINEL"
        echo "[entrypoint] WALL CAP ($TIME_LIMIT_SECONDS sec) reached — SIGTERM to PGID=$AGENT_PGID"
        kill -TERM -"$AGENT_PGID" 2>/dev/null || true
        sleep 30
        if kill -0 "$AGENT_PID" 2>/dev/null; then
            echo "[entrypoint] WALL CAP — SIGKILL after 30s grace"
            kill -KILL -"$AGENT_PGID" 2>/dev/null || true
        fi
    fi
) &
WATCHDOG_PID=$!

# Re-entrancy guard: forward exactly once. Even with a correct PGID, K8s + the
# watchdog can each deliver SIGTERM, and a trap that re-signals can re-enter
# itself. We only ever need to forward the kill ONCE; after that, ignore further
# TERM/INT so the wait loop + watchdog SIGKILL backstop finish the teardown
# without a signal storm.
_TERM_FIRED=0
_term() {
    [ "$_TERM_FIRED" = "1" ] && return
    _TERM_FIRED=1
    trap '' TERM INT
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
# State predicates → state.json (generic + per-task next to instruction.md).
# Without this the analyzer's predicate pass-rates are blank.
python3 -m mleval.analyzer.state_predicates "$OUT_DIR" 2>&1 | tail -3 || \
    echo "[entrypoint] state_predicates failed (non-fatal)"

# -------- Held-out grader (trustworthy A/B metric) ---------------------
# Scores the predictions MLEvolve preserved for the best node
# (mlevolve_runs/<ts>/workspace/best_submission/submission.csv, present because
# config has no_submission_mode:False) against held-out references — NOT the
# self-reported stdout number. Writes held_out_score.json. By contract the
# grader never raises and exits 0, so it cannot abort the manifest write below.
# Refs live at <data-root>/refs/test_refs.csv (sibling of the symlinked data
# dir); $REFS_PATH (gold, parent-only) was resolved near the top and is NOT in
# the agent's environment under C3. This grader runs in the entrypoint, so it
# can read the gold answers the agent never sees.
python3 -m mleval.grader "$OUT_DIR" --task "$TASK" --refs "$REFS_PATH" 2>&1 | tail -5 || \
    echo "[entrypoint] grader failed (non-fatal)"

# -------- Manifest -----------------------------------------------------
# A watchdog-triggered stop is an EXPECTED time-budget harvest, not a failure:
# treat it as a completed run (status=completed, exit 0) so the Job does not
# retry-from-scratch and discard the graded best node.
if [ -f "$WATCHDOG_SENTINEL" ]; then WATCHDOG_FIRED=1; else WATCHDOG_FIRED=0; fi
python3 <<PYEOF || echo "[entrypoint] manifest write failed (non-fatal)"
import json, os, socket, time
out = os.environ['MLEVAL_OUTPUT_DIR']
started = $RUN_START_EPOCH
now = time.time()
watchdog_fired = bool($WATCHDOG_FIRED)
inner_exit = $INNER_EXIT
m = {
    'schema_version': '1.0',
    'run_id': os.environ['MLEVAL_RUN_ID'],
    'trajectory_id': os.environ['MLEVAL_TRAJECTORY_ID'],
    'task': {'name': os.environ['TASK']},
    'cell': {
        'name': os.environ['CELL'],
        # skill_path lets the analyzer resolve which skill was injected (for
        # skill_api_adoption); empty for without_skill / library-routing mode.
        'skill_path': os.environ.get('MLEVAL_SKILL_PATH', ''),
        'skill_library': os.environ.get('MLEVAL_SKILL_LIBRARY', ''),
    },
    'seed': int(os.environ['SEED']),
    'agent': {
        'name': 'mlevolve',
        'version': 'vendored-26bde89',
        'llm_model': os.environ['MLEVAL_LLM_MODEL'],
    },
    'pod': {'hostname': socket.gethostname(), 'node': os.environ.get('KUBE_NODE_NAME', 'unknown')},
    'timestamps': {
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)),
        'ended_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)),
        'wall_clock_sec': int(now - started),
    },
    'result': {
        'exit_code': inner_exit,
        # time-budget stop OR clean finish → completed; anything else → crashed.
        'status': 'completed' if (inner_exit == 0 or watchdog_fired) else 'crashed',
        'stopped_by': 'watchdog_time_limit' if watchdog_fired else ('clean' if inner_exit == 0 else 'crash'),
    },
}
with open(os.path.join(out, 'manifest.json'), 'w') as f:
    json.dump(m, f, indent=2)
print('[entrypoint] manifest written')
PYEOF

# Exit 0 on a watchdog time-limit stop: the agent hit its wall budget and we
# successfully harvested + graded the best node. A non-zero exit here would
# make K8s mark the pod Failed and backoffLimit would retry FROM SCRATCH,
# discarding the graded result and burning another full budget (spike-018,
# 2026-06-09). A genuine crash (no sentinel) still propagates its exit code so
# backoffLimit can retry transient failures / evictions.
if [ -f "$WATCHDOG_SENTINEL" ]; then
    echo "[entrypoint] finished via watchdog time-limit (agent exit=$INNER_EXIT) — harvested + graded; exiting 0 (Job completes)"
    exit 0
fi
echo "[entrypoint] finished exit=$INNER_EXIT"
exit "$INNER_EXIT"
