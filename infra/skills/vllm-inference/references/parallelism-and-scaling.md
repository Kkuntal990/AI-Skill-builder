# Parallelism and Scaling

Covers tensor, pipeline, expert, data, and context parallelism strategies in vLLM, plus multi-node serving with Ray and KubeRay.

## Contents

- Parallelism Strategy Overview
- Tensor Parallelism
- Pipeline Parallelism
- Expert Parallelism
- Data Parallelism
- Context Parallelism
- Combining Parallelism Strategies
- Multi-Node Serving with Ray
- KubeRay Cluster Setup
- Choosing the Right Strategy

---

## Parallelism Strategy Overview

vLLM supports five parallelism axes that can be combined to scale inference across GPUs and nodes:

| Strategy | Flag | Splits | Best for |
|---|---|---|---|
| Tensor (TP) | `--tensor-parallel-size` | Weight matrices across GPUs | Single-node, large models |
| Pipeline (PP) | `--pipeline-parallel-size` | Layers across GPUs/nodes | Multi-node, latency-tolerant |
| Expert (EP) | `--expert-parallel-size` | MoE expert routing across GPUs | MoE models (DeepSeek, Mixtral) |
| Data (DP) | `--data-parallel-size` | Request batches across replicas | High-throughput serving |
| Context (CP) | `--context-parallel-size` | Sequence length across GPUs | Very long context inference |

All flags are passed to `vllm serve` or `LLM(...)` constructor.

---

## Tensor Parallelism

Tensor parallelism (TP) shards individual weight matrices across GPUs within a node. Each GPU holds a slice of every layer and communicates via all-reduce after each operation.

**Basic usage — offline inference:**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
)
outputs = llm.generate(["Hello, my name is"])
```

**Online serving:**

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

- TP requires all GPUs to be reachable via fast interconnect (NVLink preferred).
- TP size must evenly divide the number of attention heads.
- Within a single node, prefer TP over PP to avoid pipeline bubbles.

---

## Pipeline Parallelism

Pipeline parallelism (PP) partitions the model's layers into stages, each stage assigned to one or more GPUs. Micro-batches flow through stages sequentially.

**Online serving across two nodes (4 GPUs each, PP=2, TP=4):**

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

- PP introduces pipeline bubbles; pair with TP to keep GPU utilization high.
- PP is required when the model does not fit on a single node's GPU memory even with TP.
- Each pipeline stage must hold at least one transformer layer.

---

## Expert Parallelism

Expert parallelism (EP) distributes Mixture-of-Experts (MoE) expert weights across GPUs. Each GPU holds a subset of experts; the router dispatches tokens to the correct GPU via all-to-all communication.

**DeepSeek-R1 with expert parallelism:**

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8
```

**Offline:**

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-R1",
    tensor_parallel_size=8,
    expert_parallel_size=8,
)
```

- EP is only meaningful for MoE models.
- EP size must evenly divide the number of experts.
- EP can be combined with TP; the total world size equals `TP × EP`.
- Elastic expert parallelism (EPLB) allows dynamic rebalancing of expert load at runtime — see `vllm.distributed.elastic_ep`.

---

## Data Parallelism

Data parallelism (DP) runs multiple independent engine replicas, each handling a separate slice of incoming requests. A supervisor process load-balances across replicas.

**Online serving with DP:**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --data-parallel-size 4
```

**Offline with DP:**

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data_parallel_size=4,
)
```

- Each DP replica is a full model copy; total GPU count = `DP × TP × PP`.
- DP maximizes throughput for request-heavy workloads where a single replica is the bottleneck.
- The `dp_supervisor` module (`vllm.entrypoints.openai.dp_supervisor`) manages replica health and routing.
- See `examples/features/data_parallel.py` for a runnable example.

---

## Context Parallelism

Context parallelism (CP) splits the sequence dimension across GPUs so that very long sequences (e.g., 1 M tokens) can be processed without exhausting KV-cache memory on a single device.

**Serving with context parallelism:**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --context-parallel-size 2 \
    --tensor-parallel-size 4
```

**Offline:**

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    context_parallel_size=2,
    tensor_parallel_size=4,
)
```

- CP requires ring-attention or equivalent; not all attention backends support it — check `Attention Backend Feature Support` in the developer docs.
- CP is orthogonal to TP; both can be active simultaneously.
- Typical use: models with context windows ≥ 128 K tokens where KV cache per GPU is the bottleneck.
- See `examples/features/context_extension.py` and the `Context Parallel Deployment` guide.

---

## Combining Parallelism Strategies

The total world size (number of GPU processes) is:

```
world_size = tensor_parallel_size × pipeline_parallel_size × data_parallel_size
```

Expert parallelism replaces or extends TP within MoE layers; context parallelism operates within the TP group.

**Example — 16-GPU cluster, large MoE model:**

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --expert-parallel-size 4 \
    --data-parallel-size 2
```

**Recommended starting points:**

| Scenario | Recommended combo |
|---|---|
| 70 B dense, 8 GPUs, single node | `--tensor-parallel-size 8` |
| 405 B dense, 2 × 8-GPU nodes | `--tensor-parallel-size 8 --pipeline-parallel-size 2` |
| MoE (e.g., DeepSeek), 8 GPUs | `--tensor-parallel-size 4 --expert-parallel-size 4` |
| High-QPS, small model | `--data-parallel-size N` |
| 1 M-token context | `--context-parallel-size 2 --tensor-parallel-size 4` |

---

## Multi-Node Serving with Ray

Ray is vLLM's primary distributed runtime for multi-node deployments. vLLM auto-detects Ray when it is installed and uses it to spawn workers across nodes.

### Starting a Ray cluster manually

**On the head node:**

```bash
ray start --head
```

**On each worker node:**

```bash
ray start --address=<HEAD_NODE_IP>:6379
```

Verify the cluster:

```bash
ray status
```

### Launching vLLM on the Ray cluster

Once the cluster is up, run `vllm serve` on the head node — Ray handles worker placement automatically:

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```

### Ray Serve deployment

For production, wrap vLLM in a Ray Serve deployment for autoscaling and health management. See `examples/ray_serving/multi_node_serving.py`:

```python
import ray
from ray import serve
from vllm import LLM, SamplingParams

@serve.deployment(
    ray_actor_options={"num_gpus": 8},
    num_replicas=1,
)
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=8,
        )

    async def __call__(self, request):
        prompts = await request.json()
        outputs = self.llm.generate(prompts["prompts"])
        return [o.outputs[0].text for o in outputs]

app = VLLMDeployment.bind()
```

Deploy:

```bash
serve run multi_node_serving:app
```

### Batch inference with Ray

`examples/ray_serving/batch_llm_inference.py` shows offline batch inference distributed across a Ray cluster:

```python
import ray
from vllm import LLM, SamplingParams

@ray.remote(num_gpus=4)
def run_inference(prompts):
    llm = LLM(model="meta-llama/Llama-3.1-8B", tensor_parallel_size=4)
    params = SamplingParams(temperature=0.8, max_tokens=256)
    return llm.generate(prompts, params)

ray.init()
result_ref = run_inference.remote(["Tell me about Ray."])
results = ray.get(result_ref)
```

### Running a Ray cluster via script

`examples/ray_serving/run_cluster.py` / `run_cluster.sh` automates head + worker startup for common cloud environments.

---

## KubeRay Cluster Setup

KubeRay is the Kubernetes operator for Ray clusters. vLLM's KubeRay integration is documented under `Integrations → KubeRay`.

### Minimal RayCluster manifest

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: vllm-cluster
spec:
  rayVersion: "2.40.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: vllm/vllm-openai:latest
            resources:
              limits:
                nvidia.com/gpu: "0"
  workerGroupSpecs:
    - replicas: 2
      groupName: gpu-workers
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-worker
              image: vllm/vllm-openai:latest
              resources:
                limits:
                  nvidia.com/gpu: "8"
```

Apply:

```bash
kubectl apply -f raycluster.yaml
kubectl get raycluster
```

### Submitting a vLLM job to KubeRay

```bash
ray job submit \
  --address http://<RAY_DASHBOARD_HOST>:8265 \
  -- vllm serve meta-llama/Llama-3.1-70B-Instruct \
       --tensor-parallel-size 8
```

### Elastic expert parallelism on KubeRay

The `examples/ray_serving/elastic_ep.py` example demonstrates dynamic expert rebalancing (EPLB) within a KubeRay-managed cluster. EPLB monitors per-expert load and migrates experts between workers without restarting the server.

---

## Choosing the Right Strategy

Use this decision tree as a starting point:

1. **Does the model fit on one GPU?**
   - Yes → no parallelism needed; optionally add `--data-parallel-size` for throughput.
   - No → continue.

2. **Does it fit on one node with TP?**
   - Yes → `--tensor-parallel-size <num_gpus>`.
   - No → add `--pipeline-parallel-size` to span nodes.

3. **Is it a MoE model?**
   - Yes → add `--expert-parallel-size`; tune so `TP × EP = available GPUs per node`.

4. **Is context length the bottleneck (KV cache OOM on long sequences)?**
   - Yes → add `--context-parallel-size 2` (or higher).

5. **Is request throughput the bottleneck after fitting the model?**
   - Yes → add `--data-parallel-size` to replicate across more GPUs/nodes.

**Key constraints to remember:**
- `TP` must divide the number of attention heads evenly.
- `EP` must divide the number of experts evenly.
- Total GPUs must equal `TP × PP × DP` (EP is within the TP group for MoE layers).
- Cross-node communication favors PP over TP due to lower bandwidth requirements between stages.
