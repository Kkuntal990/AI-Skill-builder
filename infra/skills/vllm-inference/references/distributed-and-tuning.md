# Distributed Inference & Performance Tuning

How to spread a model across multiple GPUs and nodes (tensor / pipeline / data / expert parallelism), how to fit a model that's too big for the VRAM you have, and how to trade latency against throughput.

## Contents

- Choosing a distributed strategy
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Data parallelism (DP)
- Expert parallelism (EP) for MoE
- Multi-node serving with Ray
- Conserving GPU memory
- Preemption and KV-cache pressure
- Chunked prefill: the throughput/latency dial
- Tuning checklist

## Choosing a distributed strategy

Pick the smallest topology that fits the model; every extra dimension adds communication cost.

- **Single GPU** — if the weights + KV cache fit in one GPU, do not use distributed inference at all.
- **Single node, multiple GPUs (tensor parallel)** — model too large for one GPU but fits in one node. Set `tensor_parallel_size` to the number of GPUs.
- **Multi-node (tensor + pipeline parallel)** — model too large for one node. Set `tensor_parallel_size` to the number of GPUs *per node* and `pipeline_parallel_size` to the number of nodes.

Rule of thumb: `tensor_parallel_size × pipeline_parallel_size` must equal the total number of GPUs. TP is communication-heavy (an all-reduce per layer) so keep it inside a node with fast interconnect (NVLink); reach for PP to cross node boundaries where bandwidth is lower.

## Tensor parallelism (TP)

Shards the weights of each layer across GPUs. This is the default way to serve a model larger than one GPU.

Offline:

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

Online:

```bash
vllm serve facebook/opt-13b --tensor-parallel-size 4
```

**Typical next step:** if you still OOM at load time, the weights don't fit even sharded — add pipeline parallelism or quantize. On a single node vLLM uses Python multiprocessing for the workers by default; force Ray with `distributed_executor_backend="ray"` (or `--distributed-executor-backend ray`).

## Pipeline parallelism (PP)

Splits the model's layers across GPUs (each GPU owns a contiguous block of layers). Communication is point-to-point between adjacent stages rather than an all-reduce, so it tolerates slower interconnects — that's why it's the cross-node dimension. Combine it with TP:

```bash
vllm serve gpt2 --tensor-parallel-size 4 --pipeline-parallel-size 2
```

This uses 8 GPUs: 4-way TP within each of 2 pipeline stages. PP alone improves capacity, not single-request latency — in-flight microbatching keeps the stages busy across concurrent requests.

## Data parallelism (DP)

Replicates the whole model across GPU groups and routes different requests to different replicas. Raises throughput, not capacity — each replica must already fit (possibly via its own TP). It is the standard pairing with expert parallelism for MoE models.

```bash
# 4 independent replicas on a single node
vllm serve $MODEL --data-parallel-size 4

# 2 replicas, each sharded 2-way with TP
vllm serve $MODEL --data-parallel-size 2 --tensor-parallel-size 2
```

## Expert parallelism (EP) for MoE

For Mixture-of-Experts models, EP distributes the experts across GPUs instead of replicating them. Enable it on top of a DP deployment so the attention layers run data-parallel while the experts run expert-parallel:

```bash
vllm serve $MODEL --enable-expert-parallel --data-parallel-size 8
```

EP introduces an all-to-all exchange to route tokens to their experts. The backend used for that exchange is selectable via the `VLLM_ALL2ALL_BACKEND` environment variable if you need to tune dispatch/combine behavior for your interconnect.

## Multi-node serving with Ray

To span nodes, start a Ray cluster first, then launch a single `vllm serve` against it. vLLM ships a `run_cluster.sh` helper that starts the containers and joins them.

On the head node:

```bash
bash run_cluster.sh \
    vllm/vllm-openai \
    ip_of_head_node \
    --head \
    /path/to/the/huggingface/home/in/this/node \
    -e VLLM_HOST_IP=ip_of_this_node
```

On every worker node (same `ip_of_head_node`, change `--head` to `--worker` and set the worker's own `VLLM_HOST_IP`):

```bash
bash run_cluster.sh \
    vllm/vllm-openai \
    ip_of_head_node \
    --worker \
    /path/to/the/huggingface/home/in/this/node \
    -e VLLM_HOST_IP=ip_of_this_node
```

Then, from any node, serve as if it were one big machine:

```bash
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```

**Gotchas:** every node must reach the same model path (mount the HF home consistently); `VLLM_HOST_IP` must be set per node and reachable; and the GPUs/interconnect should be homogeneous across nodes or the slowest stage gates throughput.

## Conserving GPU memory

When a model won't fit, apply these in roughly increasing order of performance cost.

**Shard across GPUs (tensor parallelism).** Splits weights *and* KV cache:

```python
from vllm import LLM
llm = LLM(model="ibm-granite/granite-3.1-8b-instruct", tensor_parallel_size=2)
```

**Use a quantized model.** FP8/INT8/INT4/GPTQ/AWQ/GGUF roughly halve or quarter the weight footprint (see the quantization reference).

**Cap context length and batch size.** The KV cache scales with both `max_model_len` and the number of concurrent sequences `max_num_seqs`:

```python
from vllm import LLM
llm = LLM(model="adept/fuyu-8b", max_model_len=2048, max_num_seqs=2)
```

**Disable CUDA graphs.** Captured graphs cost extra memory; turning them off frees it at the price of slower execution:

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)
```

For a middle ground, capture graphs only for a small set of batch sizes via `compilation_config` instead of disabling entirely.

**Adjust the KV-cache budget.** `gpu_memory_utilization` (default `0.9`) is the fraction of each GPU vLLM may claim for weights + cache; raise it to squeeze out more KV cache, lower it to leave room for other processes:

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.95)
```

**Limit or drop multimodal inputs.** Cap per-prompt media (or set a modality to `0` to disable it) to shrink the multimodal cache:

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct", limit_mm_per_prompt={"image": 1})
```

**CPU offloading.** Spill part of the weights to host RAM, treating GPU+CPU as one pool — last resort, since it's bottlenecked by the CPU-GPU link:

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", cpu_offload_gb=10)
```

## Preemption and KV-cache pressure

When concurrent requests need more KV-cache blocks than exist, vLLM **preempts** the lowest-priority requests, evicts their blocks, and recomputes them later. You'll see a warning like `WARNING ... Sequence group ... is preempted by ...`. Occasional preemption is fine; frequent preemption tanks throughput. Mitigations, in order:

- Increase `gpu_memory_utilization` (more blocks).
- Decrease `max_num_seqs` or `max_num_batched_tokens` (less concurrent demand).
- Increase `tensor_parallel_size` (more aggregate VRAM for cache).
- Increase `pipeline_parallel_size` (spreads weights, freeing cache room).

## Chunked prefill: the throughput/latency dial

Chunked prefill breaks a large prompt's prefill into chunks and batches them alongside ongoing decode steps (enabled by default in V1). The single most important latency/throughput knob is `max_num_batched_tokens` — the token budget per scheduler step:

- **Larger `max_num_batched_tokens`** → more prefill packed per step → better **TTFT** (time to first token) and higher **throughput**, at the cost of worse **ITL** (inter-token latency), since big prefills interrupt decodes.
- **Smaller `max_num_batched_tokens`** → fewer prefills interrupting decodes → better **ITL** for interactive serving, at the cost of TTFT.

For throughput-oriented batch workloads, set it well above the default (e.g. `> 8096`) on high-memory GPUs:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-num-batched-tokens 16384
```

Tune `max_num_batched_tokens` together with `max_num_seqs`: the first caps tokens per step, the second caps concurrent sequences. Both feed back into KV-cache pressure (see preemption above).

## Tuning checklist

1. **Fit first.** Get the model loading with TP/quantization/`gpu_memory_utilization` before chasing speed.
2. **Watch for preemption warnings** in the logs — they mean the cache is the bottleneck, not compute.
3. **Name your objective.** Latency-sensitive (low ITL/TTFT) → smaller `max_num_batched_tokens`, fewer concurrent seqs. Throughput-sensitive (offline batch) → large `max_num_batched_tokens`, high `max_num_seqs`, high `gpu_memory_utilization`.
4. **Keep TP inside a node, PP across nodes**; add DP to scale throughput once a single replica is tuned.
5. **For MoE**, combine DP (attention) + EP (experts) rather than scaling TP alone.
