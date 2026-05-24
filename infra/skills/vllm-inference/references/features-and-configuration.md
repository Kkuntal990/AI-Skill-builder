# Features and Configuration

Covers automatic prefix caching, structured outputs, tool calling, multimodal inputs, reasoning outputs, engine arguments, environment variables, memory conservation, and observability/metrics in vLLM.

## Automatic Prefix Caching (APC)

Automatic prefix caching reuses KV cache blocks for shared prompt prefixes, reducing redundant computation across requests.

Enable at server startup:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching
```

Enable in offline inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", enable_prefix_caching=True)
```

**How it works:** vLLM hashes KV cache blocks by their token content. When a new request shares a prefix with a cached block, those blocks are reused instead of recomputed. This is especially effective for:
- System prompts shared across many requests
- Few-shot examples repeated in every prompt
- Multi-turn conversations (each turn reuses prior context)

**Checking cache hit rate:** Monitor via the `gpu_prefix_cache_hit_rate` metric exposed on the `/metrics` Prometheus endpoint.

**Constraints:**
- APC works with chunked prefill enabled by default in V1.
- Sliding window attention models do not support APC.
- The cache is stored in GPU HBM; CPU offload of prefix cache is a separate feature.

---

## Structured Outputs

vLLM supports constrained decoding to guarantee outputs conform to a schema.

### Guided JSON

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

sampling_params = SamplingParams(
    guided_decoding={"json": json_schema}
)

outputs = llm.generate("Give me a person's info.", sampling_params)
print(outputs[0].outputs[0].text)
```

### Guided Choice

```python
sampling_params = SamplingParams(
    guided_decoding={"choice": ["positive", "negative", "neutral"]}
)
```

### Guided Regex

```python
sampling_params = SamplingParams(
    guided_decoding={"regex": "\\d{3}-\\d{4}"}
)
```

### Guided Grammar (EBNF)

```python
sampling_params = SamplingParams(
    guided_decoding={"grammar": "root ::= 'yes' | 'no'"}
)
```

### Via OpenAI-compatible API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Give me a JSON person object."}],
    extra_body={"guided_json": json_schema}
)
```

**Backend selection:** vLLM uses `xgrammar` by default for structured outputs. You can override with `--guided-decoding-backend outlines` at server startup.

---

## Tool Calling

vLLM supports OpenAI-compatible function/tool calling for models that have been trained for it.

### Server startup with tool call parser

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tool-call-parser mistral \
    --enable-auto-tool-choice
```

Common `--tool-call-parser` values: `mistral`, `hermes`, `llama3_json`, `internlm`, `xlam`.

### Offline tool calling

```python
from vllm import LLM
from vllm.entrypoints.openai.tool_parsers import MistralToolParser

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    enable_auto_tool_choice=True,
    tool_call_parser="mistral",
)
```

### Client-side tool call request

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
```

**`tool_choice="required"`** forces the model to always call a tool. Pass `tool_choice={"type": "function", "function": {"name": "get_weather"}}` to force a specific function.

---

## Multimodal Inputs

vLLM supports image, video, and audio inputs for vision-language and audio-language models.

### Image input (offline)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_url},
    },
    SamplingParams(max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

### Multiple images

```python
outputs = llm.generate(
    {
        "prompt": "USER: <image><image>\nCompare these two images.\nASSISTANT:",
        "multi_modal_data": {"image": [image_url_1, image_url_2]},
    },
    SamplingParams(max_tokens=256),
)
```

### Via OpenAI-compatible API

```python
response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ]
)
```

**Limit per-request images** with `--limit-mm-per-prompt 'image=4'` at server startup to cap resource use.

---

## Reasoning Outputs

For models with chain-of-thought or thinking tokens (e.g., DeepSeek-R1, QwQ), vLLM can separate reasoning content from the final answer.

### Server startup

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --reasoning-parser deepseek_r1
```

Common `--reasoning-parser` values: `deepseek_r1`, `qwen3`.

### Accessing reasoning content

```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Solve: 2x + 3 = 11"}],
)

choice = response.choices[0].message
print("Reasoning:", choice.reasoning_content)
print("Answer:", choice.content)
```

### Streaming reasoning

```python
stream = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "What is 15 * 17?"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print("Thinking:", delta.reasoning_content, end="", flush=True)
    elif delta.content:
        print("Answer:", delta.content, end="", flush=True)
```

---

## Engine Arguments

Engine arguments configure the `LLM` class (offline) and `vllm serve` (online). They map 1:1.

### Key arguments

| Argument | Type | Description |
|---|---|---|
| `model` | str | HuggingFace model ID or local path |
| `tokenizer` | str | Tokenizer path (defaults to model) |
| `dtype` | str | Weight dtype: `auto`, `float16`, `bfloat16`, `float32` |
| `max_model_len` | int | Maximum sequence length (prompt + output) |
| `tensor_parallel_size` | int | Number of GPUs for tensor parallelism |
| `pipeline_parallel_size` | int | Number of pipeline stages |
| `gpu_memory_utilization` | float | Fraction of GPU memory to use (default `0.90`) |
| `max_num_seqs` | int | Max concurrent sequences per iteration |
| `enable_prefix_caching` | bool | Enable automatic prefix caching |
| `enable_chunked_prefill` | bool | Enable chunked prefill |
| `max_num_batched_tokens` | int | Max tokens per forward pass |
| `quantization` | str | Quantization method: `awq`, `gptq`, `fp8`, etc. |
| `enforce_eager` | bool | Disable CUDA graphs (use eager mode) |
| `trust_remote_code` | bool | Allow custom model code from HuggingFace |
| `seed` | int | Random seed for reproducibility |

### Offline usage

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,
)
```

### Online usage

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching
```

Retrieve full argument list:

```bash
vllm serve --help
```

---

## Environment Variables

vLLM reads environment variables for low-level control. Set before launching the process.

### Common variables

| Variable | Description |
|---|---|
| `VLLM_HOST_IP` | IP address vLLM binds to for distributed comms |
| `VLLM_PORT` | Port for inter-process communication |
| `VLLM_USE_MODELSCOPE` | Use ModelScope instead of HuggingFace Hub |
| `VLLM_WORKER_MULTIPROC_METHOD` | Multiprocessing start method: `fork` or `spawn` |
| `VLLM_TRACE_FUNCTION` | Enable function-level tracing for debugging |
| `VLLM_ATTENTION_BACKEND` | Override attention backend: `FLASH_ATTN`, `FLASHINFER`, `XFORMERS` |
| `VLLM_USE_V1` | Force V1 engine (`1`) or V0 engine (`0`) |
| `VLLM_LOGGING_LEVEL` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `VLLM_CONFIGURE_LOGGING` | Set to `0` to disable vLLM log configuration |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs are visible |
| `VLLM_NCCL_SO_PATH` | Custom path to NCCL shared library |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow `max_model_len` beyond model config limit |

### Example

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

---

## Memory Conservation

Use these strategies when GPU memory is constrained.

### Reduce `gpu_memory_utilization`

```python
llm = LLM(model="...", gpu_memory_utilization=0.80)
```

Lower values leave more memory for the OS and other processes. Default is `0.90`.

### Limit `max_model_len`

```python
llm = LLM(model="...", max_model_len=4096)
```

Shorter context windows require fewer KV cache blocks.

### Quantization

```python
llm = LLM(model="...", quantization="fp8")
# or load a pre-quantized model
llm = LLM(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
```

### Enforce eager mode (disables CUDA graph memory)

```python
llm = LLM(model="...", enforce_eager=True)
```

CUDA graphs pre-allocate memory for captured graph sizes. Disabling them reduces peak memory at the cost of some throughput.

### Limit `max_num_seqs`

```python
llm = LLM(model="...", max_num_seqs=32)
```

Fewer concurrent sequences means fewer KV cache blocks allocated simultaneously.

### CPU offload fraction

```python
llm = LLM(model="...", cpu_offload_gb=10)
```

Offloads the specified number of GB of model weights to CPU RAM, freeing GPU memory.

### Swap space

```bash
vllm serve ... --swap-space 4
```

Allocates CPU swap space (in GiB) for KV cache blocks that overflow GPU memory.

---

## Observability and Metrics

### Prometheus metrics endpoint

When running `vllm serve`, metrics are exposed at:

```
GET http://localhost:8000/metrics
```

### Key production metrics

| Metric | Description |
|---|---|
| `vllm:num_requests_running` | Requests currently being processed |
| `vllm:num_requests_waiting` | Requests queued waiting for capacity |
| `vllm:gpu_cache_usage_perc` | Fraction of GPU KV cache in use |
| `vllm:cpu_cache_usage_perc` | Fraction of CPU KV cache in use |
| `vllm:gpu_prefix_cache_hit_rate` | Prefix cache hit rate (APC) |
| `vllm:time_to_first_token_seconds` | TTFT histogram |
| `vllm:time_per_output_token_seconds` | TPOT histogram |
| `vllm:e2e_request_latency_seconds` | End-to-end latency histogram |
| `vllm:request_prompt_tokens` | Prompt token count histogram |
| `vllm:request_generation_tokens` | Generated token count histogram |
| `vllm:num_preemptions_total` | Total preemption events |

### OpenTelemetry tracing

Enable OTLP tracing at server startup:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --otlp-traces-endpoint http://localhost:4317 \
    --collect-detailed-traces all
```

`--collect-detailed-traces` options: `model_forward`, `model_execute`, `worker`, `all`.

### Grafana + Prometheus stack

vLLM ships example dashboards. Point Prometheus at the `/metrics` endpoint and import the provided Grafana dashboard JSON from `examples/observability/`.

### Logging request stats

vLLM logs a stats summary every `--log-stats-interval` seconds (default: 5). Disable with `--disable-log-stats`.

```bash
vllm serve ... --log-stats-interval 10
```
