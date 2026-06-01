---
name: refresh-mleval-pvc
description: Synchronize the mleval-results PVC with current local repo state — stage task instructions, skills, prompt overlays; remove obsolete files; pre-warm HuggingFace cache (models + datasets); verify the helper pod is healthy. Use whenever local edits to infra/tasks/, infra/skills/, infra/agents/<name>/requirements.txt, or a config change (e.g. new embedding model) would otherwise be invisible to a trajectory pod. Also use when adding a new model to keep on the PVC for future trajectories.
---

# Refresh mleval-results PVC

The mleval-results PVC is the single source of truth that **trajectory pods read** at runtime. Image builds do NOT push task data — runtime pods read from `/results/data/<task>/`. Without an explicit refresh, edits to `infra/tasks/*/instruction.md`, `infra/tasks/*/prompt_overlay.yaml`, or any other runtime-mounted file stay invisible to new trajectories.

This skill encodes the recurring workflow so you don't bleed compute on stale data again.

## When to invoke

Refresh the PVC after:

- Editing `infra/tasks/<task>/instruction.md` or other task data
- Editing `infra/tasks/<task>/prompt_overlay.yaml` (or removing it)
- Editing `infra/skills/<skill>/SKILL.md` or `references/`
- Adding a new HuggingFace model that trajectories will load
- Switching `memory_embedding_model_path` in `config.yaml`
- Cleaning up obsolete files after deleting a task or skill

Do NOT refresh PVC for changes to: `mlevolve_sidecar/` (those go in the image), `Dockerfile`, `requirements.txt` (those go in the image), `entrypoint.sh` (in the image). Those changes need image rebuild + push, not PVC sync.

## Prerequisites

- Local clone of the repo at `~/AI-Skill-builder` (or equivalent on amusing — use `ad-kkokate@amusing.ucsd.edu`)
- `kubectl` configured for the `ecepxie` namespace on Nautilus
- `.env` sourced (provides `GPU_TYPE`, `GPU_PRODUCT`, `IMAGE_REGISTRY`, etc. for helper-pod manifest substitution)

## Step 1 — Ensure helper pod is running

The helper pod (`mleval-jupyter-1gpu`) mounts the same PVC at `/results` as trajectory pods. It's the cleanest way to read/write PVC contents without spinning up a Job.

```bash
# Check status
kubectl -n ecepxie get pod mleval-jupyter-1gpu

# If Error / NotFound / >24h old → redeploy
cd ~/AI-Skill-builder && set -a && source .env && set +a
kubectl -n ecepxie delete pod mleval-jupyter-1gpu --ignore-not-found
envsubst < deploy/k8s/helper-jupyter-1gpu.yaml | kubectl -n ecepxie apply -f -

# Wait for ready
until [ "$(kubectl -n ecepxie get pod mleval-jupyter-1gpu -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)" = "true" ]; do sleep 10; done
```

The helper pod is **free to redeploy** — only Jobs (real trajectory runs) need careful approval.

## Step 2 — Stage local files to PVC

For each changed file, push via `kubectl cp`. Paths on the PVC mirror the local repo layout under `/results/data/<task>/` and `/results/skills/<skill>/` (or however the previous staging laid them out — verify with `ls` first).

```bash
# Single file push
kubectl -n ecepxie cp infra/tasks/samsum/instruction.md \
    mleval-jupyter-1gpu:/results/data/samsum/instruction.md

# Multiple files (whole task dir)
for f in infra/tasks/samsum/*.md infra/tasks/samsum/*.yaml; do
    [ -f "$f" ] || continue
    kubectl -n ecepxie cp "$f" "mleval-jupyter-1gpu:/results/data/samsum/$(basename $f)"
done

# Verify the push landed
kubectl -n ecepxie exec mleval-jupyter-1gpu -- stat /results/data/samsum/instruction.md
```

If a file was DELETED locally (e.g. you removed `prompt_overlay.yaml` because we adopted MLE-Bench parity), also remove it from PVC:

```bash
kubectl -n ecepxie exec mleval-jupyter-1gpu -- rm -fv /results/data/samsum/prompt_overlay.yaml
```

## Step 3 — Warm HuggingFace cache

Trajectory pods set `HF_HOME=/results/.hf-cache/hf` so any cached model/dataset persists across trajectories. First-trajectory download cost can be tens of minutes for a 3B+ model.

```bash
# Pre-download a model into the shared cache
kubectl -n ecepxie exec mleval-jupyter-1gpu -- bash -c "
HF_HOME=/results/.hf-cache/hf python -c \"
from huggingface_hub import snapshot_download
p = snapshot_download('<MODEL_SLUG>', cache_dir='/results/.hf-cache/hf/hub')
print('warmed at:', p)
import os
total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
print(f'size: {total/1e6:.1f} MB')
\"
"
```

Replace `<MODEL_SLUG>` with e.g. `BAAI/bge-base-en-v1.5`, `Qwen/Qwen2.5-3B-Instruct`, `sentence-transformers/all-MiniLM-L6-v2`.

Verify with:

```bash
kubectl -n ecepxie exec mleval-jupyter-1gpu -- bash -c "
for m in models--BAAI--bge-base-en-v1.5 models--Qwen--Qwen2.5-3B-Instruct; do
    if [ -d /results/.hf-cache/hf/hub/\$m ]; then
        du -sh /results/.hf-cache/hf/hub/\$m
    else
        echo \"NOT WARMED: \$m\"
    fi
done
"
```

## Step 4 — End-to-end smoke (when adding embedding models)

If you added a model the `sentence-transformers` library will load, verify it actually instantiates from cache:

```bash
kubectl -n ecepxie exec mleval-jupyter-1gpu -- bash -c "
HF_HOME=/results/.hf-cache/hf python -c \"
import time; t0 = time.time()
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('<MODEL_SLUG>')
print(f'loaded in {time.time()-t0:.1f}s; dim={m.get_sentence_embedding_dimension()}')
\"
"
```

`< 60s` load time confirms it's coming from cache (vs ~5-10 min for download).

## Step 5 — Cleanup of obsolete staging

If a refactor deleted local files, also delete them from PVC to prevent footguns:

```bash
# Find files in /results/data/<task>/ that no longer have a local counterpart
LOCAL_TASK="infra/tasks/samsum"
PVC_TASK="/results/data/samsum"

kubectl -n ecepxie exec mleval-jupyter-1gpu -- ls "$PVC_TASK" 2>/dev/null \
    | while read f; do
        if [ ! -e "$LOCAL_TASK/$f" ]; then
            echo "STALE on PVC (not in local): $f"
        fi
    done
```

Manually delete confirmed-stale files. Don't auto-delete — you might miss something the orchestrator needs.

## Common pitfalls

### Trajectory prompts.jsonl bleeds across pod restarts

If you `kubectl delete job <traj_id>` then relaunch the SAME `--cell --seed` for the same RUN_ID, both pods share the same trajectory_id and append to the same `prompts.jsonl`. The new pod's prompts are appended AFTER the old pod's. Solutions:

- Bump `MLEVAL_RUN_ID` in `.env` (cheapest)
- Or delete `/results/$MLEVAL_RUN_ID/$TRAJECTORY_ID/` before relaunch

### kubectl cp can silently fail on tarball-mode

If the destination directory doesn't exist, `kubectl cp` may write to a parent. Always verify with `stat` or `ls` after.

### HF cache has multiple cache dirs

`HF_HOME=/results/.hf-cache/hf` is the modern unified dir, but older code may use `TRANSFORMERS_CACHE=/results/.hf-cache/hf/transformers` or `HF_DATASETS_CACHE`. The `job.yaml.tmpl` sets all three; the helper pod manifest may set only HF_HOME. Override per command if needed.

### sentence-transformers ALSO caches separately

If you see double the disk usage on `du`, `sentence-transformers` writes additional copies under `~/.cache/torch/sentence_transformers/`. Setting `HF_HOME` correctly avoids this, but verify.

## One-shot variant for "just push task data"

The most common refresh case — task description edits, no model warming:

```bash
TASK=samsum   # change as needed
cd ~/AI-Skill-builder
for f in infra/tasks/$TASK/*.md infra/tasks/$TASK/*.yaml; do
    [ -f "$f" ] || continue
    kubectl -n ecepxie cp "$f" "mleval-jupyter-1gpu:/results/data/$TASK/$(basename $f)"
done
kubectl -n ecepxie exec mleval-jupyter-1gpu -- ls -la /results/data/$TASK/
```

## Verification checklist before launching a trajectory

- [ ] Helper pod is `1/1 Running`
- [ ] `instruction.md` on PVC matches local (`diff` via `kubectl exec cat`)
- [ ] Stale files deleted (no `prompt_overlay.yaml` if local was deleted)
- [ ] All HF models the trajectory needs are warm (Qwen, bge, etc.)
- [ ] `MLEVAL_RUN_ID` in `.env` is fresh (not a previous run's ID)
- [ ] If launching a paired A/B, `--cells with_skill without_skill` not just one

Once these pass, the trajectory pod sees current data and won't waste compute on a stale-state run.
