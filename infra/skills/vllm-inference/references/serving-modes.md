# Serving Modes

Covers the four main ways to deploy vLLM: offline inference via the `LLM` class, the OpenAI-compatible HTTP server, disaggregated prefill, and data-parallel / context-parallel scaling.

## Contents

- Offline Inference with the LLM Class
- Online Serving with the OpenAI-Compatible Server
- Disaggregated Prefill
- Data-Parallel Deployment
- Context-Parallel Deployment
- Choosing Between Modes

---

## Offline Inference with the LLM Class

Use the `LLM` class when you want to run batch inference inside a Python process — no HTTP server, no network overhead.

### Basic usage

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`llm.generate()` returns a list of `RequestOutput` objects. Each has `.prompt` and `.outputs` (a list of `CompletionOutput`).

### Typical next steps after offline inference

- Inspect `output.outputs[0].token_ids` for token-level results.
- Pass `SamplingParams(n=4)` to get multiple completions per prompt.
- Use `llm.chat()` for chat-formatted inputs instead of raw strings.

### Async streaming variant

For long-running pipelines that need streaming without a server:

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs

engine_args = AsyncEngineArgs(model="facebook/opt-125m")
engine = AsyncLLMEngine.from_engine_args(engine_args)
```

`AsyncLLMEngine` exposes `engine.generate()` as an async generator, yielding partial `RequestOutput` objects as tokens are produced.

### LLM Engine example (lower-level)

```python
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.utils import FlexibleArgumentParser

def create_test_prompts():
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.1)),
    ]

def process_requests(engine, test_prompts):
    request_id = 0
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, params)
            request_id += 1
        request_outputs = engine.step()
        for output in request_outputs:
            if output.finished:
                print(output)

def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)
```

Use `LLMEngine` directly when you need fine-grained control over the request loop (e.g., dynamic prompt injection, custom scheduling logic).

---

## Online Serving with the OpenAI-Compatible Server

vLLM ships an HTTP server that implements the OpenAI Chat Completions, Completions, and Embeddings APIs.

### Starting the server

```bash
vllm serve facebook/opt-125m
```

Or with explicit options:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123
```

### Querying with the OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "Write a haiku about recursion in programming."}
    ]
)
print(completion.choices[0].message.content)
```

### Streaming responses

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Count to 10."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Batched chat completions (online)

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

async def main():
    tasks = [
        client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": f"Tell me a fact about {topic}"}],
        )
        for topic in ["the moon", "the ocean", "volcanoes"]
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r.choices[0].message.content)

asyncio.run(main())
```

### Token generation client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

# Raw completions endpoint (non-chat)
response = client.completions.create(
    model="facebook/opt-125m",
    prompt="San Francisco is a",
    max_tokens=64,
    temperature=0,
)
print(response.choices[0].text)
```

### Key server CLI flags

| Flag | Purpose |
|---|---|
| `--model` | HuggingFace model ID or local path |
| `--tensor-parallel-size` / `-tp` | Number of GPUs for tensor parallelism |
| `--data-parallel-size` / `-dp` | Number of data-parallel replicas |
| `--max-model-len` | Override context window length |
| `--api-key` | Require this key on all requests |
| `--ssl-certfile` / `--ssl-keyfile` | Enable HTTPS |

---

## Disaggregated Prefill

Disaggregated prefill separates the **prefill** (prompt processing) and **decode** (token generation) phases onto different vLLM instances, connected via a KV-transfer connector. This is experimental.

### Why disaggregate?

Prefill is compute-bound; decode is memory-bandwidth-bound. Running them on separate hardware lets each be tuned independently and avoids head-of-line blocking in mixed workloads.

### Basic two-process setup

**Prefill instance** (sends KV cache to decode instance):

```bash
VLLM_DISAGG_PREFILL_ROLE=prefill \
vllm serve facebook/opt-125m \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

**Decode instance** (receives KV cache):

```bash
VLLM_DISAGG_PREFILL_ROLE=decode \
vllm serve facebook/opt-125m \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

### Python script pattern (v1 API)

```python
# disaggregated_prefill.py
import subprocess, os

prefill_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
decode_env  = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

prefill_proc = subprocess.Popen(
    ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--model", "facebook/opt-125m",
     "--port", "8100",
     "--kv-transfer-config",
     '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer",'
     '"kv_rank":0,"kv_parallel_size":2}'],
    env=prefill_env,
)

decode_proc = subprocess.Popen(
    ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--model", "facebook/opt-125m",
     "--port", "8200",
     "--kv-transfer-config",
     '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer",'
     '"kv_rank":1,"kv_parallel_size":2}'],
    env=decode_env,
)
```

A proxy (or load balancer) routes prefill requests to port 8100 and decode requests to port 8200.

### Available KV connectors

| Connector | Transport | Notes |
|---|---|---|
| `PyNcclConnector` | NCCL (GPU-to-GPU) | Default for single-node |
| `MooncakeConnector` | RDMA / shared memory | Multi-node, low latency |
| `LMCacheConnector` | CPU offload + network | Longer KV retention |
| `NixlConnector` | NIXL protocol | See NixlConnector compatibility matrix |

### FlexKV connector (dynamic routing)

```python
# flexkv_connector.py — routes KV based on request metadata
kv_transfer_config = {
    "kv_connector": "FlexKVConnector",
    "kv_role": "kv_both",   # instance can act as producer or consumer
}
```

`FlexKVConnector` lets a single instance act as both producer and consumer, enabling more flexible topologies.

---

## Data-Parallel Deployment

Data parallelism (DP) runs multiple full model replicas and distributes requests across them. Use it to scale throughput when a single replica saturates.

### Single-node DP via the server

```bash
vllm serve facebook/opt-125m \
    --data-parallel-size 4 \
    --tensor-parallel-size 1
```

This starts 4 replicas on the same node. The built-in DP supervisor (`dp_supervisor.py`) load-balances requests.

### Multi-node DP with Ray

```python
# ray_serving/run_cluster.py pattern
import ray
from vllm import LLM

ray.init(address="auto")

@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(self, model):
        self.llm = LLM(model=model)

    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params)

workers = [VLLMWorker.remote("facebook/opt-125m") for _ in range(4)]
```

### Ray Serve integration

```python
# ray_serving/batch_llm_inference.py pattern
from ray import serve
from vllm import LLM, SamplingParams

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(model="facebook/opt-125m")

    async def __call__(self, request):
        data = await request.json()
        params = SamplingParams(**data.get("sampling_params", {}))
        outputs = self.llm.generate(data["prompts"], params)
        return [o.outputs[0].text for o in outputs]

app = VLLMDeployment.bind()
```

Deploy with:

```bash
serve run ray_serving/batch_llm_inference:app
```

### DP + TP combined

```bash
# 2 DP replicas, each using 4-way tensor parallelism = 8 GPUs total
vllm serve meta-llama/Llama-3-70b-instruct \
    --data-parallel-size 2 \
    --tensor-parallel-size 4
```

DP and TP compose: `total_gpus = dp_size × tp_size`.

---

## Context-Parallel Deployment

Context parallelism (CP) splits the **sequence dimension** of the KV cache across GPUs within a single request. This extends the effective context length beyond what fits on one GPU.

### Enabling context parallelism

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --context-parallel-size 2
```

With `--context-parallel-size 2`, each request's KV cache is split across 2 GPUs. Combined with `--tensor-parallel-size 4`, this uses 8 GPUs per replica.

### When to use context parallelism

- Requests regularly exceed 32K tokens.
- A single GPU cannot hold the full KV cache for the target sequence length.
- You want to keep batch size high while serving very long contexts.

### Context parallelism vs. tensor parallelism

| Dimension | Tensor Parallel | Context Parallel |
|---|---|---|
| Splits | Model weights | KV cache (sequence) |
| Communication | All-reduce on activations | Ring attention across sequence chunks |
| Best for | Large models that don't fit on one GPU | Very long sequences |

Both can be combined with data parallelism.

---

## Choosing Between Modes

| Scenario | Recommended mode |
|---|---|
| Batch processing, scripting, evaluation | Offline `LLM` class |
| Production API serving, multi-client | Online OpenAI-compatible server |
| Mixed prefill/decode workloads, SLA isolation | Disaggregated prefill |
| Throughput scaling across many GPUs/nodes | Data-parallel deployment |
| Very long context (>32K tokens) per request | Context-parallel deployment |

Modes compose: a disaggregated setup can itself use data parallelism on the decode side, and each replica can use tensor parallelism internally. Start with the simplest mode that meets your latency and throughput requirements, then layer in parallelism as needed.
