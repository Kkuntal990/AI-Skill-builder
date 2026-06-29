# vLLM Online Serving

How to launch vLLM's OpenAI-compatible HTTP server with `vllm serve`, call its completions/chat/embeddings endpoints, configure engine and server arguments, and control request logging.

## Contents

- Starting the server with `vllm serve`
- Engine arguments vs. server arguments
- Common server flags
- Chat Completions endpoint
- Completions endpoint
- Embeddings endpoint
- Calling the server with the OpenAI Python client
- vLLM-specific sampling parameters
- Utility endpoints: models, health, metrics
- Request and stats logging
- API-key authentication

## Starting the server with `vllm serve`

The server is launched from the CLI by passing a Hugging Face model id (or local path):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

This loads the model, starts an OpenAI-compatible HTTP server on `http://0.0.0.0:8000`, and exposes `/v1/completions`, `/v1/chat/completions`, and (for pooling models) `/v1/embeddings`. The process stays in the foreground until killed.

Typical next step: in another shell, hit `/v1/models` to confirm the server is up and learn the model id clients should send (see below).

To shard a large model across GPUs, add tensor parallelism:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

`--tensor-parallel-size` splits the model over N GPUs; `--max-model-len` caps the context window (lower it if KV-cache allocation OOMs); `--gpu-memory-utilization` is the fraction of VRAM vLLM may claim for weights + KV cache.

## Engine arguments vs. server arguments

`vllm serve --help` lists two groups of flags:

- **Engine arguments** configure the inference engine itself — which model to load, parallelism, dtype, quantization, KV-cache and scheduling limits. Examples: `--tensor-parallel-size`, `--max-model-len`, `--dtype`, `--quantization`, `--max-num-seqs`, `--enforce-eager`, `--trust-remote-code`, `--seed`. These are the same arguments the offline `LLM(...)` constructor accepts.
- **Server (frontend) arguments** configure the HTTP layer — host/port, auth, chat template, tool-calling parsers, logging. Examples: `--host`, `--port`, `--api-key`, `--served-model-name`, `--chat-template`, `--response-role`, `--enable-auto-tool-choice`, `--disable-log-requests`.

Inspect the full set for your build with:

```bash
vllm serve --help
```

## Common server flags

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen-small \
    --max-num-seqs 256 \
    --dtype bfloat16
```

- `--host` / `--port` — bind address and port (default `0.0.0.0:8000`).
- `--served-model-name` — the name clients put in the request `model` field; defaults to the model path. Set it to a short alias so requests don't have to repeat the full HF id.
- `--max-num-seqs` — max concurrent sequences in a batch (throughput vs. per-request latency lever).
- `--dtype` — one of `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`.
- `--quantization` / `-q` — e.g. `awq`, `gptq`, `fp8` when serving a quantized checkpoint.
- `--trust-remote-code` — required for models that ship custom modeling code.
- `--chat-template` — path to (or inline) a Jinja chat template, needed when a model's tokenizer has none.

## Chat Completions endpoint

`POST /v1/chat/completions` takes a `messages` list and applies the model's chat template:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

The response is OpenAI-shaped: the reply text is at `choices[0].message.content`. Typical next step: set `"stream": true` to receive incremental `data:` SSE chunks instead of one blob.

## Completions endpoint

`POST /v1/completions` is the legacy text-completion API — a raw `prompt` string, no chat template:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

The generated text is at `choices[0].text`. Use this endpoint for base (non-chat) models or when you want to control the exact prompt string yourself.

## Embeddings endpoint

`/v1/embeddings` is only served when the loaded model is run as a pooling/embedding model. Start the server in embed mode:

```bash
vllm serve intfloat/e5-mistral-7b-instruct --task embed
```

Then request vectors:

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "intfloat/e5-mistral-7b-instruct",
        "input": "Hello world"
    }'
```

Each input string returns a vector at `data[i].embedding`. `input` may be a single string or a list of strings for batched embedding.

## Calling the server with the OpenAI Python client

Because the API is OpenAI-compatible, point the official `openai` client at the local base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

completion = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="San Francisco is a",
)
print(completion.choices[0].text)
```

Chat works the same way:

```python
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)
print(chat_response.choices[0].message.content)
```

When no `--api-key` is set, pass any placeholder (e.g. `"EMPTY"`) for `api_key` — it is ignored.

## vLLM-specific sampling parameters

Parameters that exist in vLLM but not in the OpenAI schema are passed through `extra_body`:

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Classify the sentiment."}],
    extra_body={
        "top_k": 50,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "guided_choice": ["positive", "negative"],
    },
)
```

`top_k`, `min_p`, `repetition_penalty`, `length_penalty`, `use_beam_search`, `stop_token_ids`, and the structured-output controls (`guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`) all travel in `extra_body`. Standard fields (`temperature`, `top_p`, `max_tokens`, `stop`, `n`, `stream`) go at the top level as usual.

## Utility endpoints: models, health, metrics

```bash
curl http://localhost:8000/v1/models      # list served model ids
curl http://localhost:8000/health         # 200 OK when ready to serve
curl http://localhost:8000/metrics        # Prometheus-format engine metrics
```

`/v1/models` returns the id(s) clients must use in the `model` field (i.e. the `--served-model-name` or model path). `/health` is the readiness probe — poll it after launch before sending real traffic. `/metrics` exposes counters and gauges (running/waiting requests, token throughput, KV-cache usage) for scraping.

## Request and stats logging

By default vLLM logs each incoming request (its prompt and sampling parameters) and periodic throughput statistics. Two flags turn these off:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --disable-log-requests \
    --disable-log-stats
```

- `--disable-log-requests` — stop logging per-request lines (prompt text + `SamplingParams`). Use this in production to avoid writing prompt content to logs.
- `--disable-log-stats` — stop the periodic `Avg prompt throughput / Avg generation throughput / Running / Pending / GPU KV cache usage` summary lines.

Control overall verbosity with the uvicorn HTTP log level and the vLLM logger:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --uvicorn-log-level warning
```

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

`--uvicorn-log-level` sets the HTTP access-log level; the `VLLM_LOGGING_LEVEL` environment variable sets the engine's logger level for deeper troubleshooting.

## API-key authentication

Require a bearer token on every request by passing `--api-key`:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --api-key token-abc123
```

Clients must then send `Authorization: Bearer token-abc123`. With the OpenAI client this is just the `api_key` argument:

```python
client = OpenAI(
    api_key="token-abc123",
    base_url="http://localhost:8000/v1",
)
```

Requests without the matching key are rejected with `401 Unauthorized`. This is the minimum control for exposing the server beyond localhost; for anything more, front it with a reverse proxy.
