# `docs/eval/ops.md` — Stage 2 runtime playbook

How to operate the Stage 2 A/B framework on UCSD Nautilus NRP. Pair with [`stage2.md`](./stage2.md) for the methodology and [`infra/agents/_interface.md`](../../infra/agents/_interface.md) for the contract.

## NRP cluster policies (must read)

Source: <https://nrp.ai/documentation/userdocs/start/policies/>. Violating these can get the namespace banned. The Job templates already encode compliance; the table here exists so changes don't drift.

| Policy | How we comply |
|---|---|
| **CPU/Memory limits must stay within 20% of requests** (== request when running >100 pods) | Both templates use `request == limit`. |
| **No more than 4 pods may simultaneously have CPU usage <20% of requested *or* memory usage <20% of requested** | Pilot template uses the exempt tier (≤1 CPU + ≤2 GiB) → enforcement doesn't apply. Full PEFT template sized so training fills the resource ask. |
| **GPU utilization must exceed 40%, ideally near 100%** | CPU-only tasks (tabular pilots) use `--profile cpu` → no GPU requested. GPU is only allocated for tasks that actually use CUDA. |
| **CPU-only pods should add `priorityClassName: opportunistic` + node anti-affinity to GPU nodes** | `job_cpu.yaml.tmpl` has both. |
| **Interactive pods destroyed after 6h** | `mleval-jupyter-1gpu` is short-lived; redeploy via `make k8s-apply-helper` and tear down between sweeps. |
| **`sleep infinity` / non-terminating commands = ban** | Entrypoint runs MLEvolve then exits; no sleeps. |
| **A100 default quota = 0** | We target `rtxa6000` only (set via `GPU_TYPE` in `.env`). A100 needs a separate request. |

When in doubt: shrink CPU/RAM to the exempt tier (1 CPU / 2 GiB) and drop the GPU. That's always safe.

## Choosing a profile

| Profile | Use when | Resources |
|---|---|---|
| `cpu` (default for pilots) | Task does NOT use CUDA (tabular, small NLP, smoke runs) | 1 CPU / 2 GiB, no GPU, opportunistic priority, anti-affinity to GPU nodes |
| `gpu` (default for `make ab-*`) | Task fine-tunes / runs CUDA models | 1× rtxa6000 + 2 CPU / 8 GiB (right-sized from 16 GiB — observed CPU-RAM peak ~2.7 GB even with global memory on) |

Override per-invocation: `make ab-plan PROFILE=cpu TASK=house-prices ...`.

## One-time setup (per fresh `.env`)

```bash
# 1. Config — copy template, fill in secrets
cp .env.example .env
$EDITOR .env
make config              # mask-print: confirms OPENROUTER_API_KEY=<set>, K8S_NAMESPACE, etc.

# 2. Cluster pre-flight (read-only)
kubectl config current-context           # expect: nautilus
kubectl -n "$K8S_NAMESPACE" describe quota
kubectl get storageclass | grep cephfs

# 3. Provision PVC (one-time per namespace; 1Ti CephFS RWX)
make k8s-apply-pvc
kubectl -n "$K8S_NAMESPACE" get pvc mleval-results -o wide   # expect: Bound

# 4. Provision secrets
make k8s-secret              # OpenRouter + HF tokens from .env
make k8s-ghcr-pull-secret    # GitHub PAT (read:packages, classic) for private image pull

# 5. Build + push image (on amusing). Use the build-mleval-image skill, or:
ssh ad-kkokate@amusing.ucsd.edu
cd ~/AI-Skill-builder && git checkout mlevolve-smoke && git pull && \
  git submodule update --init --recursive infra/agents/mlevolve/upstream && \
  set -a && source .env && set +a && \
  make docker-mlevolve && make docker-mlevolve-push   # Dockerfile runs _smoke_imports.py
exit
```

**Reusable skills** (in `agents/ai-skill-builder/skills/`):
- **build-mleval-image** — rebuild+push on amusing after an in-image change (sidecar, Dockerfile, requirements, entrypoint, `src/mleval/analyzer/*`).
- **refresh-mleval-pvc** — sync task/skill data + warm HF cache onto the PVC after editing `infra/tasks/*` or `infra/skills/*` (runtime-mounted, NOT in the image).
- **monitor-mleval-job** — adaptive-cadence background watch of a live A/B sweep; quiet unless an anomaly (crash, off-task agent, stall).

## Per-sweep setup

```bash
# Bump MLEVAL_RUN_ID in .env so per-trajectory PVC paths stay disjoint
$EDITOR .env

# Stage task data + skill onto the PVC (one-time per task/skill)
# See infra/tasks/README.md "Staging task data onto the PVC" for the
# alpine-pod cp recipe.

# Warm the pip cache (saves ~30-60s on every trajectory's first launch).
# As of 2026-05-25 this is a near-no-op because per-task requirements.txt
# files are now empty (all ML deps live in the base image's curated
# infra/agents/mlevolve/requirements.txt). Still safe
# to run; only meaningful if you add a niche package to a task's reqs.
make pip-warm TASK=llama-inference SKILL=vllm-inference

# (GPU profile only) Warm the HF cache so model weights are downloaded
# once per sweep, not once per trajectory. NOT YET AUTOMATED — see the
# "Open decisions remaining" section in stage2.md. Workaround: launch
# one smoke trajectory before the parallel A/B and let it populate
# /results/.hf-cache/ on its own.
```

## Interactive dev pod (smoke tests, prototyping)

```bash
make k8s-apply-helper
kubectl -n "$K8S_NAMESPACE" port-forward pod/mleval-jupyter-1gpu 8888:8888
# open http://localhost:8888/?token=mleval-dev

# tear down (frees 1 GPU from the quota)
make k8s-delete-helper
```

## A/B sweep

### Preview (no cluster touch)

```bash
make ab-plan TASK=mytask SEEDS="0 1" \
    SKILL_PATH=/results/skills/peft-tuning/SKILL.md \
    TIME_LIMIT=3600 STEP_LIMIT=20
```

Prints the trajectories that would be applied. Inspect, confirm, then:

### Apply (live, deploys k8s Jobs)

```bash
make ab-apply TASK=mytask SEEDS="0 1" \
    SKILL_PATH=/results/skills/peft-tuning/SKILL.md \
    TIME_LIMIT=3600 STEP_LIMIT=20
```

Orchestrator:
- Renders `infra/agents/mlevolve/job.yaml.tmpl` per (task × cell × seed)
- Skips trajectories whose Job already exists (idempotent)
- Applies via `kubectl apply -f -`

### Watch progress

```bash
kubectl -n "$K8S_NAMESPACE" get jobs -l app=mleval,run_id=$MLEVAL_RUN_ID -w
kubectl -n "$K8S_NAMESPACE" logs -f -l app=mleval,run_id=$MLEVAL_RUN_ID --tail=20
```

Or block in the orchestrator:

```bash
make ab-wait TASK=mytask SEEDS="0 1" \
    SKILL_PATH=/results/skills/peft-tuning/SKILL.md
```

### Pull results from the PVC

```bash
mkdir -p ./pulled-results/$MLEVAL_RUN_ID

# Spin a one-shot mount pod
kubectl -n "$K8S_NAMESPACE" run pvc-shell --rm -it --restart=Never \
    --image=alpine \
    --overrides='{"spec":{"containers":[{"name":"pvc-shell","image":"alpine","stdin":true,"tty":true,"volumeMounts":[{"name":"r","mountPath":"/r"}]}],"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"mleval-results"}}]}}' \
    -- /bin/sh
# inside: tar czf - /r/$MLEVAL_RUN_ID  (then exit and reattach to copy)

# Or per-trajectory cp:
kubectl -n "$K8S_NAMESPACE" cp pvc-shell:/r/$MLEVAL_RUN_ID ./pulled-results/$MLEVAL_RUN_ID
```

### Aggregate

```bash
make aggregate-run RUN_DIR=./pulled-results/$MLEVAL_RUN_ID
# writes ./pulled-results/$MLEVAL_RUN_ID/report.{md,json}
```

## Iterating on the analyzer locally

If you have a pulled trajectory and want to re-run the analyzer chain (e.g., after updating `stage_classifier.py`):

```bash
make analyze-trajectory DIR=./pulled-results/$MLEVAL_RUN_ID/<trajectory_id>
```

Re-writes `trajectory.jsonl` and `state.json` in place. Safe to run repeatedly.

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| `ErrImagePull` immediately after apply | ghcr.io image is private + pull secret missing | `kubectl -n "$K8S_NAMESPACE" get secret ghcr-pull`; rerun `make k8s-ghcr-pull-secret` |
| Pod `Pending` >2 min | GPU quota exhausted or taint mismatch | `kubectl describe pod <name>` events; `kubectl describe quota` |
| Job runs forever, no LLM activity | Agent hung in `rich.Status(...)` when stdout isn't a TTY | Our entrypoint already passes `TERM=dumb`; if it recurs, check `kubectl logs` for the last printed line |
| `429` from OpenRouter | Free-tier model rate-limited | Switch `MLEVAL_LLM_MODEL` to a paid slug; smoke-test from helper pod first |
| `401` from OpenAI responses API | `report.model=gpt-4.1` default routes to OpenAI's `responses.create` (not OpenRouter) | We default `generate_report=false`; ensure entrypoint passes it |
| `trajectory.jsonl` missing after pod completes | Analyzer chain crashed silently | Inspect `agent_logs/run.log` for `[entrypoint] adapter_mlevolve FAILED` |
| PVC fills up | Working-dir snapshots include large model checkpoints | MLEvolve executes each node in its own subprocess; prune old `runfile_*` outputs on the PVC |

## Cleanup between sweeps

```bash
# delete completed Jobs (keep their results on the PVC)
kubectl -n "$K8S_NAMESPACE" delete jobs -l app=mleval,run_id=$MLEVAL_RUN_ID

# delete old trajectory outputs on PVC (use mount pod)
kubectl -n "$K8S_NAMESPACE" run pvc-shell --rm -it ... -- /bin/sh
rm -rf /r/<old-run-id>/
```

## What the user holds vs. what the orchestrator owns

| Owned by user | Owned by orchestrator |
|---|---|
| `.env` values (PAT, API key, run id, model) | k8s Job lifecycle (apply, wait, idempotency) |
| Task choice + data staging | Per-trajectory env rendering from `.env` + CLI flags |
| Skill choice + SKILL.md authoring | Analyzer chain invocation inside the pod |
| Final aggregate review | Cross-trajectory aggregation (`mleval.analyzer.aggregate`) |
| Decision to apply (`--apply` flag) | Nothing applies until explicit user invocation |
