# `infra/tasks/` — Task scaffolds

Each task lives in `infra/tasks/<task-name>/` and provides:

| File | Required | Purpose |
|---|---|---|
| `instruction.md` | yes | Agent `desc_file` — natural-language goal + eval criterion + data description |
| `data/` | yes | Read-only mount inside the pod at `/results/data/<task>/data/` (staged separately) |
| `predicates.py` | optional | `PREDICATES: dict[str, Callable[[Path], bool]]` — task-specific assertions |
| `README.md` | optional | How to stage the data, where it came from, license notes |

## How a task gets exercised

1. Orchestrator (`infra/orchestrator/run_ab.py`) creates a k8s Job per `(task, cell, seed)`.
2. The Job pod mounts the PVC at `/results`. Task data must already be at
   `/results/data/<task>/{instruction.md, data/}` — staged via `kubectl cp`
   or an initContainer (see `docs/eval/ops.md`).
3. `entrypoint.sh` builds the agent's expected dataset layout from
   `/results/data/<task>/instruction.md` and `/results/data/<task>/data/`.
4. After the agent finishes, the analyzer chain runs
   (`adapter_<agent>` → `stage_classifier` → `state_predicates`).
5. `state_predicates` looks for `infra/tasks/<task>/predicates.py` (via
   `MLEVAL_TASK_INSTRUCTION_PATH`'s parent dir on the PVC).

## Staging task data onto the PVC

Run this once per task, from the Mac (no cluster compute):

```bash
# 1. Spin up an alpine pod that mounts only the PVC (no GPU, no agent image)
kubectl -n $K8S_NAMESPACE run pvc-shell --rm -it --restart=Never \
    --image=alpine \
    --overrides='{"spec":{"containers":[{"name":"pvc-shell","image":"alpine","stdin":true,"tty":true,"volumeMounts":[{"name":"r","mountPath":"/results"}]}],"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"mleval-results"}}]}}' -- /bin/sh

# 2. Inside the pod
mkdir -p /results/data/<task>/data

# 3. From a separate Mac terminal, cp data + instruction + predicates into the pod
kubectl -n $K8S_NAMESPACE cp infra/tasks/<task>/instruction.md pvc-shell:/results/data/<task>/instruction.md
kubectl -n $K8S_NAMESPACE cp infra/tasks/<task>/predicates.py  pvc-shell:/results/data/<task>/predicates.py
kubectl -n $K8S_NAMESPACE cp infra/tasks/<task>/data           pvc-shell:/results/data/<task>/data
```

After this, every (task × cell × seed) Job picks up the same staged data.

## Starting a new task

Copy `_template/`:

```bash
cp -r infra/tasks/_template infra/tasks/<your-task>
$EDITOR infra/tasks/<your-task>/instruction.md
$EDITOR infra/tasks/<your-task>/predicates.py
```

`instruction.md` follows the agent's free-form `desc_file` format. The minimal
template has `## Goal`, `## Evaluation`, `## Data description` sections.
