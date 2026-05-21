# `deploy/k8s/` — Kubernetes manifests (agent-agnostic)

All manifests target Nautilus NRP. **Nothing here is applied automatically.**
Every k8s operation is fronted by a `make` target that reads `.env` and
substitutes config at apply time.

Per-agent Job manifests live under `infra/agents/<name>/`, not here.

## Files

| File | Purpose | How to apply |
|---|---|---|
| `pvc.yaml` | Shared PVC for trajectory outputs (1Ti CephFS RWX). | `make k8s-apply-pvc` |
| `secret.template.yaml` | **Reference only.** Documents the Secret keys the pods expect. | `make k8s-secret` (provisions from `.env`) |
| `helper-jupyter-1gpu.yaml` | Single-RTX-A6000 Jupyter pod for interactive dev. | `make k8s-apply-helper` |

YAML files contain `${VAR}` placeholders that get filled in by `envsubst` at
apply time. Run `make config` to see exactly what values will be substituted.

## First-time setup

```bash
# 1. Config — copy the template and fill in your values
cp .env.example .env
$EDITOR .env
make config              # sanity-check what's set

# 2. Pre-flight on the cluster
kubectl get gpus -A
kubectl -n "$K8S_NAMESPACE" describe quota

# 3. Provision the PVC (one-time per namespace)
make k8s-apply-pvc

# 4. Provision the secrets (rerun whenever you rotate keys in .env)
make k8s-secret              # OpenRouter + HF tokens
make k8s-ghcr-pull-secret    # private ghcr.io image pull
```

## Interactive dev on a real GPU node

```bash
make k8s-apply-helper
kubectl -n "$K8S_NAMESPACE" port-forward pod/mleval-jupyter-1gpu 8888:8888
# open http://localhost:8888/?token=mleval-dev

# tear down when done:
make k8s-delete-helper
```

## Where each variable goes

| `.env` var | YAML field(s) | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `Secret/mleval-secrets` (via `make k8s-secret`) | Never enters a YAML file |
| `HF_TOKEN` | `Secret/mleval-secrets` | Optional |
| `GHCR_READ_TOKEN` | `Secret/ghcr-pull` (via `make k8s-ghcr-pull-secret`) | classic `read:packages` PAT |
| `K8S_NAMESPACE` | `kubectl -n` flag (not in YAML) | Used by every k8s target |
| `IMAGE_REGISTRY` / `IMAGE_NAME` / `IMAGE_TAG` | `containers[*].image` | Same image used by helper pod + Jobs |
| `GPU_TYPE` | `resources.{limits,requests}` | e.g. `nvidia.com/rtxa6000` |
| `MLEVAL_LLM_MODEL` | Job `env: MLEVAL_LLM_MODEL` | Passed into the agent container |
| `MLEVAL_RUN_ID` | Job `env: MLEVAL_RUN_ID`, `MLEVAL_OUTPUT_DIR` | Identifies the sweep / output subdir |
| `DEFAULT_SEED` | Job `env: SEED`, trajectory id | Override per-run with `make TARGET DEFAULT_SEED=1` |

## Conventions

- **Tolerations** match the reference jupyter-1gpu pod (`nautilus.io/reservation`,
  `nautilus.io/hardware=large-gpu`, `nvidia.com/gpu` PreferNoSchedule).
  Adjust if your namespace's reservation tolerations differ.
- **No namespace in YAML** — every command uses `kubectl -n $K8S_NAMESPACE`.
- **No image tag in YAML** — every image reference is `${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}`.
- **Image pull secret**: every pod that pulls our private image sets
  `imagePullSecrets: [{name: ghcr-pull}]`. Provisioned once per namespace
  via `make k8s-ghcr-pull-secret`.
