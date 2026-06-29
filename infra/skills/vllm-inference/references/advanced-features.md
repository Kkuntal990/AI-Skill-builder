# Advanced Features: LoRA, Speculative Decoding, Structured Outputs, Tool Calling, Prefix Caching

How to layer the five high-leverage serving features onto a running vLLM engine — multi-adapter LoRA, draft-model/n-gram speculative decoding, grammar-constrained outputs, OpenAI-style tool calling, and the prefix cache that makes shared-prefix workloads cheap.

## Contents

- LoRA Adapters — Offline (`LLM`)
- LoRA Adapters — Online Server
- Dynamic LoRA Loading at Runtime
- Speculative Decoding — Draft Model
- Speculative Decoding — N-Gram and EAGLE
- Structured Outputs — Offline
- Structured Outputs — Online Server
- Tool Calling
- Automatic Prefix Caching
- Combining Features
- Gotchas

## LoRA Adapters — Offline (`LLM`)

LoRA lets one base model serve many fine-tuned adapters without loading a separate full model per task. Enable it at engine construction, then attach an adapter per request via `LoRARequest`.

```python
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=1,        # max adapters resident in a single batch
    max_lora_rank=16,   # must be >= the rank of any adapter you load
)

sampling_params = SamplingParams(temperature=0, max_tokens=256, stop=["[/assistant]"])

prompts = [
    "[user] Write a SQL query to get the names of users older than 30. [/user] [assistant]",
]

# Third arg is a local path or a downloaded snapshot dir.
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
)
```

- `LoRARequest(name, int_id, path)` — `int_id` must be unique and stable per adapter; vLLM keys its adapter cache on it.
- **Typical next step:** pass `lora_request=None` (or omit it) on some prompts in the same `generate` call to compare adapter vs. base-model behavior in one pass.

## LoRA Adapters — Online Server

Register adapters at launch with `--lora-modules`, then select one by using its **name as the model field** in the request.

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 16 \
    --lora-modules sql-lora=/path/to/sql_lora_adapter another=/path/to/other_adapter
```

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "Write a SQL query for active users:",
        "max_tokens": 128,
        "temperature": 0
    }'
```

- `GET /v1/models` lists both the base model and each registered LoRA name as selectable models.
- **Typical next step:** point your OpenAI client's `model=` at the LoRA name; no other client change is needed.

## Dynamic LoRA Loading at Runtime

To add or drop adapters without restarting, set the env var and call the admin endpoints.

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve meta-llama/Llama-2-7b-hf --enable-lora
```

```bash
# Load
curl http://localhost:8000/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "sql-lora", "lora_path": "/path/to/sql_lora_adapter"}'

# Unload
curl http://localhost:8000/v1/unload_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "sql-lora"}'
```

- **Typical next step:** after a successful load, the adapter name immediately becomes a valid `model` value in completion/chat requests.

## Speculative Decoding — Draft Model

Speculative decoding uses a small "draft" model to propose `num_speculative_tokens` tokens that the large model verifies in a single forward pass, cutting decode latency when acceptance is high. Configure it via `speculative_config`.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_config={
        "model": "facebook/opt-125m",   # the draft model
        "num_speculative_tokens": 5,
    },
)

outputs = llm.generate("The future of AI is", SamplingParams(temperature=0))
```

- The draft model must share a tokenizer/vocabulary with the target model.
- Output text is **identical** to non-speculative greedy decoding — this is a latency optimization, not a quality change.
- **Typical next step:** sweep `num_speculative_tokens` (3–7) and measure tokens/sec; gains fall off when the draft's proposals are frequently rejected.

## Speculative Decoding — N-Gram and EAGLE

N-gram speculation needs no draft model — it proposes tokens by matching recent context against the prompt, which is strong for summarization/RAG where output echoes input.

```python
# N-gram: no draft weights required
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,   # longest n-gram to match
    },
)
```

```python
# EAGLE: a trained lightweight draft head
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)
```

- **Typical next step:** for input-heavy tasks (long-doc summarization), try `ngram` first — it's free to enable and often the biggest win.

## Structured Outputs — Offline

Constrain generation to valid JSON, a regex, a fixed choice set, or a grammar. Offline, attach a guided-decoding spec to `SamplingParams`.

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

# 1. Constrain to a choice set
choice = GuidedDecodingParams(choice=["Positive", "Negative"])
out = llm.generate(
    "Classify the sentiment: 'I loved it.'",
    SamplingParams(guided_decoding=choice),
)

# 2. Constrain to a JSON schema
json_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
guided = GuidedDecodingParams(json=json_schema)
out = llm.generate(
    "Generate a person record.",
    SamplingParams(guided_decoding=guided),
)

# 3. Constrain to a regex
regex = GuidedDecodingParams(regex=r"\w+@\w+\.com")
```

- `GuidedDecodingParams` accepts `json=`, `regex=`, `choice=`, or `grammar=` (EBNF) — use exactly one.
- **Typical next step:** wrap the output in `json.loads(...)`; with a JSON schema the parse is guaranteed to succeed.

## Structured Outputs — Online Server

The OpenAI-compatible server exposes the same constraints through `extra_body` (or the standard `response_format`). The default backend is `xgrammar`; `guidance` is also available.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# JSON schema via extra_body
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Give me a person as JSON."}],
    extra_body={"guided_json": json_schema},
)

# Choice set
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Positive or negative?"}],
    extra_body={"guided_choice": ["Positive", "Negative"]},
)
```

Supported `extra_body` keys: `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`. The standard `response_format={"type": "json_object"}` and `{"type": "json_schema", ...}` are also honored.

Pick the backend at launch:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --structured-outputs-config.backend xgrammar
```

- **Typical next step:** if a complex grammar errors under `xgrammar`, switch the backend to `guidance`, which supports a wider feature set.

## Tool Calling

For automatic tool calling, the server needs a parser matched to the model's tool-call format. Enable it at launch, then send OpenAI-style `tools`.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

Common `--tool-call-parser` values: `hermes` (Qwen/Hermes), `llama3_json` (Llama 3.1+), `mistral`, `pythonic`. The parser must match the model family.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)
tool_calls = resp.choices[0].message.tool_calls
```

- `tool_choice="auto"` lets the model decide; force a specific call with `tool_choice={"type": "function", "function": {"name": "get_weather"}}` (named calling, which uses guided decoding under the hood and does not require `--enable-auto-tool-choice`).
- **Typical next step:** execute the returned call, append a `{"role": "tool", ...}` message with the result, and re-invoke the model to get the final answer.

## Automatic Prefix Caching

APC caches the KV blocks of already-computed prefixes so requests sharing a leading prefix (system prompt, few-shot block, long document) skip recomputation. In vLLM V1 it is **on by default**; enable/disable explicitly elsewhere.

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_prefix_caching=True)
```

```bash
# Online server (disable with the negated flag if needed)
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching
vllm serve meta-llama/Llama-3.1-8B-Instruct --no-enable-prefix-caching
```

- Best wins come from a long, shared, **identical** prefix across requests (e.g., a fixed system prompt + a multi-question pass over the same document). Any byte difference in the prefix breaks the match.
- The cache is transparent — outputs are unchanged; only the time-to-first-token on cache hits drops.
- **Typical next step:** order a batch so requests over the same document are contiguous, maximizing prefix reuse before blocks are evicted.

## Combining Features

These features are largely composable on one engine:

- **LoRA + prefix caching** — cache keys account for the active adapter, so shared prefixes are reused per-adapter.
- **Structured outputs + tool calling** — named tool choice (`tool_choice={"type": "function", ...}`) is implemented via the structured-outputs backend.
- **Speculative decoding** is mutually exclusive with some experimental paths; enable it alone first, confirm a throughput gain, then layer others.

## Gotchas

- **`max_lora_rank` is a hard ceiling** — loading an adapter whose rank exceeds it fails at request time, not launch time.
- **Draft/target tokenizer mismatch** silently degrades speculative acceptance to near zero; verify both models share a vocabulary.
- **Tool parser must match the model** — using `hermes` against a Llama 3.1 model produces unparsed tool calls that surface as plain text.
- **Guided-decoding param name is version-sensitive** — older vLLM uses `SamplingParams(guided_decoding=GuidedDecodingParams(...))`; newer releases expose a `structured_outputs` field. Check the installed version's `vllm.sampling_params` if an import fails.
- **Prefix cache hits require byte-identical prefixes** — a timestamp or per-request ID at the top of the system prompt defeats caching entirely.
