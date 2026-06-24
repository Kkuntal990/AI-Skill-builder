# Online Serving with vLLM

How to launch vLLM's OpenAI-compatible HTTP server with `vllm serve`, the endpoints it exposes, the server/engine flags that shape it, and how to run it under Docker and Kubernetes.

## Contents

- Starting the server
- OpenAI-compatible endpoints
- Querying the server
- Core server & engine arguments
- Distributed serving flags
- Serving LoRA adapters
- Structured outputs and tool calling
- Docker deployment
- Kubernetes deployment
- Health, metrics, and observability

## Starting the server

`vllm serve` boots an async engine behind a FastAPI app that mimics the OpenAI REST API. The only required argument is the model (a Hugging Face repo id or a local path):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

This downloads the model (if not cached), loads it onto the GPU, and listens on `http://0.0.0.0:8000`. Typical next step: hit `GET /v1/models` to confirm the served model name, then send a chat request.

Common launch shape with the flags you will almost always set:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name llama3-8b \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --api-key token-abc123
```

`--served-model-name` decouples the public name clients pass in `"model": ...` from the checkpoint path. `--api-key` makes the server require `Authorization: Bearer token-abc123`.

## OpenAI-compatible endpoints

Once running, the server exposes (subset, all under the configured host:port):

| Method & path | Purpose |
|---|---|
| `GET /v1/models` | List served model name(s) and any LoRA modules |
| `POST /v1/completions` | Legacy text completion |
| `POST /v1/chat/completions` | Chat completion (applies the model's chat template) |
| `POST /v1/embeddings` | Embeddings (embedding/pooling models only) |
| `POST /score` | Cross-encoder / reranker scoring |
| `POST /pooling` | Raw pooled hidden states |
| `POST /tokenize`, `POST /detokenize` | Tokenizer round-trips |
| `POST /v1/audio/transcriptions` | Speech-to-text (transcription models) |
| `GET /health` | Liveness/readiness probe |
| `GET /metrics` | Prometheus metrics |

The server also supports streaming for completions and chat completions via `"stream": true`, returning server-sent events.

## Querying the server

The server is drop-in compatible with the official `openai` Python client — point `base_url` at the vLLM host:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",   # use "EMPTY" if --api-key was not set
)

completion = client.chat.completions.create(
    model="llama3-8b",
    messages=[{"role": "user", "content": "Summarize PagedAttention in one line."}],
)
print(completion.choices[0].message.content)
```

Same call over `curl`, useful for smoke-testing a fresh pod:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer token-abc123" \
    -d '{
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 64
    }'
```

Typical next step: wrap this in a readiness loop that polls `GET /health` until 200 before sending real traffic.

## Core server & engine arguments

`vllm serve` accepts both server arguments (HTTP/serving layer) and engine arguments (the same knobs as the offline `LLM(...)` constructor). The ones that matter most for fitting a model on a GPU and shaping throughput:

| Flag | Effect |
|---|---|
| `--max-model-len <int>` | Cap context length; lower it to shrink KV-cache memory and avoid OOM at load |
| `--gpu-memory-utilization <0–1>` | Fraction of GPU memory vLLM may claim for weights + KV cache (default `0.9`) |
| `--dtype {auto,half,bfloat16,float16,float32}` | Weight/compute dtype; `auto` follows the checkpoint |
| `--quantization <method>` | e.g. `awq`, `gptq`, `fp8`, `bitsandbytes`, `compressed-tensors` |
| `--max-num-seqs <int>` | Max concurrent sequences in a batch (continuous batching width) |
| `--max-num-batched-tokens <int>` | Token budget per scheduler step; raise for throughput, lower for latency |
| `--enable-prefix-caching` | Reuse KV cache across requests sharing a prompt prefix |
| `--trust-remote-code` | Allow custom modeling code from the HF repo |
| `--download-dir <path>` | Where to cache weights (point at a shared volume to download once) |
| `--chat-template <file>` | Override the Jinja chat template (needed when a model ships none) |
| `--seed <int>` | Seed engine RNG for reproducibility |

Inspect the full, version-accurate list on a given build with:

```bash
vllm serve --help
```

Typical next step: when you hit a load-time OOM, lower `--max-model-len` first, then `--gpu-memory-utilization`, before reaching for quantization.

## Distributed serving flags

For models too large for one GPU, or to raise throughput across GPUs:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

- `--tensor-parallel-size <N>` shards each layer across N GPUs on one node (the primary scaling knob).
- `--pipeline-parallel-size <N>` splits layers into N stages, can span nodes.
- `--data-parallel-size <N>` replicates the model for more aggregate throughput.

The product of these must match the GPUs you intend to use. Multi-GPU on a single node generally needs `--ipc=host` (Docker) or sufficient shared memory (Kubernetes) so NCCL can communicate.

## Serving LoRA adapters

Serve a base model with one or more LoRA adapters selectable per request by name:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules sql-adapter=/adapters/sql alpaca=/adapters/alpaca
```

Each adapter then appears as its own entry in `GET /v1/models`; a request selects one by passing its name in the `"model"` field. Typical next step: confirm an adapter loaded by listing models before routing traffic to it.

## Structured outputs and tool calling

Constrain generation to a schema or grammar via the structured-outputs path, and enable function/tool calling with a model-specific parser:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

Structured outputs are requested per call (JSON schema / regex / grammar in the request body); `--enable-auto-tool-choice` plus the matching `--tool-call-parser` lets the server emit OpenAI-style `tool_calls`. Reasoning models use `--reasoning-parser` to split chain-of-thought from the final answer.

## Docker deployment

vLLM ships the `vllm/vllm-openai` image, whose entrypoint is `vllm serve` — arguments after the image name are passed straight through:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

Notes that bite people:

- `--ipc=host` (or a large `--shm-size`) is required; otherwise tensor-parallel NCCL hangs on too-small `/dev/shm`.
- Mounting `~/.cache/huggingface` persists downloaded weights across container restarts.
- Pin a release tag (e.g. `vllm/vllm-openai:v0.9.2`) rather than `latest` for reproducible deployments.

## Kubernetes deployment

The minimal pattern is a `Deployment` running the image plus a `Service` to expose it. Request GPUs via the `nvidia.com/gpu` resource and back the HF cache with a volume so weights download once per node:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels: { app: vllm-server }
  template:
    metadata:
      labels: { app: vllm-server }
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model", "mistralai/Mistral-7B-v0.1"]
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef: { name: hf-token, key: token }
          volumeMounts:
            - name: cache
              mountPath: /root/.cache/huggingface
            - name: shm
              mountPath: /dev/shm
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 60
            periodSeconds: 10
      volumes:
        - name: cache
          persistentVolumeClaim: { claimName: hf-cache }
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: "2Gi" }
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector: { app: vllm-server }
  ports:
    - port: 80
      targetPort: 8000
```

Key points: the `/dev/shm` `emptyDir` is the Kubernetes analogue of Docker's `--ipc=host` (needed for multi-GPU NCCL); the `readinessProbe` on `/health` keeps the Service from routing before the model finishes loading (cold loads of large checkpoints can take minutes). For production fleets, vLLM also documents Helm charts and integrations such as the vLLM production stack, KServe, and KubeRay.

## Health, metrics, and observability

- `GET /health` returns 200 once the engine is ready — wire it to liveness and readiness probes.
- `GET /metrics` exposes Prometheus metrics (request counts, latencies, KV-cache usage, running/waiting queue depth) for scraping by Prometheus + Grafana dashboards.

Typical next step: scrape `/metrics` to watch `num_requests_running` and KV-cache utilization under load, and tune `--max-num-seqs` / `--max-num-batched-tokens` from what you observe.
