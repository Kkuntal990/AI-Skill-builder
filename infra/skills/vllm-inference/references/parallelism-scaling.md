# Parallelism and Scaling

How to distribute a vLLM model across multiple GPUs and nodes (tensor, pipeline, data, and expert parallelism), serve MoE models, conserve GPU memory, and tune throughput vs. latency.

## Contents

- Choosing a parallelism strategy
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Data parallelism (DP)
- Expert parallelism (EP) for MoE
- Expert Parallel Load Balancer (EPLB)
- Multi-node serving
- Conserving GPU memory
- Optimization and tuning
- NUMA binding and CPU provisioning

## Choosing a parallelism strategy

vLLM's selection guide, from cheapest to most distributed:

- **Single GPU** — model fits on one GPU; no distributed inference.
- **Single-node multi-GPU** — model exceeds one GPU but fits one node → **tensor parallelism**.
- **Multi-node multi-GPU** — model needs multiple nodes → combine **tensor + pipeline parallelism**.

Rule of thumb for distributed deployment: set `tensor_parallel_size` to the number of GPUs **per node**, and `pipeline_parallel_size` to the number of **nodes**.

The four axes serve different goals: TP shards parameters *within* a layer to fit a model / free KV-cache room; PP splits *layers* across GPUs for very large models; DP *replicates* the model to raise throughput; EP places MoE experts on separate GPUs.

## Tensor parallelism (TP)

Shards model parameters across GPUs within each layer. Use when the model is too big for one GPU, or to reduce memory pressure and leave more room for KV cache.

Python API:

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

Server CLI:

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

To control which physical devices are used, set `CUDA_VISIBLE_DEVICES` — do **not** call `torch.accelerator.set_device_index()` before vLLM initializes.

## Pipeline parallelism (PP)

Distributes model *layers* across GPUs. Combine with TP for very large models; typically TP within a node, PP across nodes.

```bash
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

Python:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
)
```

## Data parallelism (DP)

Replicates the **entire model** across separate instances/GPU sets, each processing independent batches. Raises throughput; works for dense and MoE models. Set `data_parallel_size=N` (Python) or `--data-parallel-size` / `-dp` (CLI).

> `--max-num-seqs` applies **per DP rank**, so total throughput scales with deployment size.

**Internal load balancing** — one API endpoint, vLLM balances internally. Single node:

```
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

Multi-node (DP=4 split across two nodes, 2 ranks local to each):

```
# Node 0 (10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345

# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

With the Ray backend you launch once (no per-node commands, no `--data-parallel-address`); set `VLLM_RAY_DP_PACK_STRATEGY="span"` for multi-node:

```
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

At larger DP sizes the single API server becomes a bottleneck — scale it with `--api-server-count`, or switch to **hybrid load balancing** (`--data-parallel-hybrid-lb`, per-node API servers behind an external upstream LB).

**External load balancing** — each rank is an independent endpoint, routed by something upstream:

```
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

## Expert parallelism (EP) for MoE

Places MoE experts on separate GPUs for locality and throughput. EP works best combined with DP. The expert-parallel world size is `EP_SIZE = TP_SIZE × DP_SIZE`. Enable with `enable_expert_parallel=True` (Python) or `--enable-expert-parallel` (CLI) — this uses EP instead of TP for MoE layers.

Attention-layer behavior under EP: with `TP = 1`, attention weights replicate across DP ranks; with `TP > 1`, attention shards via TP within each DP group.

Single node (DeepSeek-V3, DP=8):

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

Pick the all-to-all communication backend with `--all2all-backend`: `allgather_reducescatter` (default), `deepep_high_throughput` (multi-node prefill), `deepep_low_latency` (multi-node decode), `flashinfer_nvlink_one_sided` / `flashinfer_nvlink_two_sided` (MNNVL systems). DeepEP/DeepGEMM must be installed first.

Performance levers: `--enable-dbo` overlaps all-to-all with compute; `--async-scheduling` overlaps scheduling with execution.

## Expert Parallel Load Balancer (EPLB)

Rebalances token distribution across experts. Enable with `--enable-eplb`; configure via `--eplb-config` (JSON) or dotted args. Key params: `window_size` (default 1000), `step_interval` (3000), `num_redundant_experts` (0), `log_balancedness` (false), `use_async` (true).

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

Equivalent dotted form: `--eplb-config.window_size 1000 --eplb-config.step_interval 3000 ...`.

## Multi-node serving

**Ray cluster** — start the cluster, then serve. Head and worker setup:

```bash
# Head node
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<HEAD_NODE_IP>

# Worker node
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

Then serve across the cluster (TP within node × PP across nodes):

```bash
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

**Multiprocessing backend** (no Ray) — launch one process per node:

```bash
# Head node
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_NODE_IP>

# Worker node
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_NODE_IP> --headless
```

For high-bandwidth GPUDirect RDMA, give the container shared memory and `IPC_LOCK`. Docker:

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

Kubernetes pod spec equivalent:

```yaml
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      resources:
        limits:
          nvidia.com/gpu: 8
        requests:
          nvidia.com/gpu: 8
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
```

## Conserving GPU memory

When a model won't fit or you hit OOM, apply these in roughly this order.

**Shard across GPUs** with TP (also frees room for KV cache):

```python
from vllm import LLM
llm = LLM(model="ibm-granite/granite-3.1-8b-instruct", tensor_parallel_size=2)
```

**Cap context and batch size** — the biggest single-knob KV-cache savings:

```python
from vllm import LLM
llm = LLM(model="adept/fuyu-8b", max_model_len=2048, max_num_seqs=2)
```

**Quantize** — load a pre-quantized model or set the `quantization` option for dynamic quantization.

**Trim CUDA graphs** — capture fewer batch sizes:

```python
from vllm import LLM
from vllm.config import CompilationConfig, CompilationMode

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
    ),
)
```

Or disable graph capture entirely (saves memory, costs throughput):

```python
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)
```

**Multi-modal models** — drop unused modalities or cap items per prompt with `limit_mm_per_prompt`:

```python
llm = LLM(
    model="google/gemma-3-27b-it",
    limit_mm_per_prompt={"image": 0},
)
```

Other cache knobs: `mm_processor_cache_gb` (default 4 GiB; set `0` to disable) and `VLLM_CPU_KVCACHE_SPACE` for the CPU backend.

## Optimization and tuning

**Optimization levels** trade startup time for performance: `-O0` (no opt, fastest start), `-O1` (simple compile + PIECEWISE cudagraphs), `-O2` (default; FULL_AND_PIECEWISE cudagraphs), `-O3` (aggressive; currently equals `-O2`).

**Preemption** — if you see `Sequence group N is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space`, the engine is starved for KV cache. Mitigate by: raising `gpu_memory_utilization`; lowering `max_num_seqs` or `max_num_batched_tokens`; raising `tensor_parallel_size`; or raising `pipeline_parallel_size`. (Default mode in V1 is `RECOMPUTE`, not `SWAP`.)

**Chunked prefill** (on by default in V1) batches prefill chunks with decode requests. Tune via `max_num_batched_tokens`:

- smaller (e.g. `2048`) → better inter-token latency;
- higher → better time-to-first-token; `>8192` recommended for throughput on smaller models.

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

> When chunked prefill is **disabled**, `max_num_batched_tokens` must be greater than `max_model_len`, or vLLM may crash at startup.

**Batch-level DP for multi-modal encoders** — TP gives little gain on small encoders but adds all-reduce overhead. Use data parallelism for the encoder instead:

```python
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    tensor_parallel_size=4,
    mm_encoder_tp_mode="data",
)
```

**API server scale-out** — relieve front-end bottlenecks (input processing across processes):

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4 -dp 2
```

## NUMA binding and CPU provisioning

On multi-socket nodes, bind workers to NUMA nodes to cut cross-socket traffic. CLI auto-detection:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind
```

Explicit node and CPU mapping (one entry per TP worker):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1 \
  --numa-bind-cpus 0-3 4-7 48-51 52-55
```

CLI usage enables the `spawn` multiprocessing method automatically; for the Python API set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

**Provision enough CPU cores.** vLLM V1 runs 1 API server + 1 engine core + N GPU worker processes, so **minimum physical cores = 2 + N** (N = number of GPUs). With data parallelism / multiple API servers: `A + DP + N + (1 if DP > 1 else 0)`. If hyperthreading is on, double the vCPU count — `2 × (2 + N)`. Under-provisioning CPU throttles input processing, scheduling, and output handling.
