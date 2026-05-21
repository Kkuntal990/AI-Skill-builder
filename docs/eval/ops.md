# `docs/eval/ops.md` — Stage 2 runtime playbook

How to operate the Stage 2 A/B framework on UCSD Nautilus NRP. Pair with [`stage2.md`](./stage2.md) for the methodology and [`infra/agents/_interface.md`](../../infra/agents/_interface.md) for the contract.

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

# 5. Build + push image (on amusing; see CLAUDE.md "Container image" section)
ssh ad-kkokate@amusing.ucsd.edu
cd ~/AI-Skill-builder && git pull
make docker-agent && make docker-push
exit
```

## Per-sweep setup

```bash
# Bump MLEVAL_RUN_ID in .env so per-trajectory PVC paths stay disjoint
$EDITOR .env

# Stage task data + skill onto the PVC (one-time per task/skill)
# See infra/tasks/README.md "Staging task data onto the PVC" for the
# alpine-pod cp recipe.
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
- Renders `infra/agents/aide/job.yaml.tmpl` per (task × cell × seed)
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
| Job runs forever, no LLM activity | AIDE hung in `rich.Status(...)` when stdout isn't a TTY | Our entrypoint already passes `TERM=dumb`; if it recurs, check `kubectl logs` for the last printed line |
| `429` from OpenRouter | Free-tier model rate-limited | Switch `MLEVAL_LLM_MODEL` to a paid slug; smoke-test from helper pod first |
| `401` from OpenAI responses API | `report.model=gpt-4.1` default routes to OpenAI's `responses.create` (not OpenRouter) | We default `generate_report=false`; ensure entrypoint passes it |
| `trajectory.jsonl` missing after pod completes | Analyzer chain crashed silently | Inspect `agent_logs/run.log` for `[entrypoint] adapter_aide FAILED` |
| PVC fills up | Working-dir snapshots include large model checkpoints | Tune `SKIP_GLOBS` in `aide_sidecar/interpreter_patch.py` |

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
