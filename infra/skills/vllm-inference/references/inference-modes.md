# Inference Modes in vLLM

Covers offline inference with the `LLM` class (including async streaming), online serving via the OpenAI-compatible server, and batch inference patterns.

## Offline Inference with the `LLM` Class

The `LLM` class is the primary entry point for synchronous offline inference. Instantiate it with a model identifier, then call `.generate()`.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

prompts = [
    "Hello, my name is",
    "The capital of France is",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

- `LLM(model=...)` loads the model and allocates KV cache.
- `SamplingParams` controls decoding (temperature, top_p, max_tokens, etc.).
- `.generate()` returns a list of `RequestOutput` objects; each has `.prompt` and `.outputs[0].text`.
- Typical next step: iterate outputs and post-process or write to disk.

### Chat-style Offline Inference

Pass a list of message dicts to `.chat()` instead of raw strings:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

outputs = llm.chat(conversation)
print(outputs[0].outputs[0].text)
```

- `.chat()` applies the model's chat template automatically.
- Typical next step: pass multi-turn history by appending assistant and user turns.

### Using the LLM Engine Directly

For lower-level control, use `LLMEngine` with a step loop:

```python
from vllm import LLMEngine, EngineArgs, SamplingParams

engine_args = EngineArgs(model="meta-llama/Llama-3.2-1B-Instruct")
engine = LLMEngine.from_engine_args(engine_args)

engine.add_request("req-0", "Hello, my name is", SamplingParams(max_tokens=50))

while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(output.outputs[0].text)
```

- `engine.add_request()` enqueues a prompt with a unique request ID.
- `engine.step()` runs one decode iteration and returns any finished outputs.
- Typical next step: add more requests dynamically between steps.

---

## Async Streaming with `AsyncLLMEngine`

For async contexts (e.g., servers, async pipelines), use `AsyncLLMEngine` and iterate the async generator returned by `.generate()`.

```python
import asyncio
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

async def main():
    engine_args = AsyncEngineArgs(model="meta-llama/Llama-3.2-1B-Instruct")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    prompt = "What is the meaning of life?"

    async for output in engine.generate(prompt, sampling_params, request_id="req-0"):
        final_output = output

    print(final_output.outputs[0].text)

asyncio.run(main())
```

- `engine.generate()` returns an async generator; each yielded item is a partial `RequestOutput`.
- Stream tokens incrementally by printing inside the `async for` loop.
- Typical next step: integrate into an `asyncio`-based server or use with `aiohttp`.

### Streaming Tokens as They Arrive

```python
async for output in engine.generate(prompt, sampling_params, request_id="req-1"):
    # output.outputs[0].text grows with each iteration
    token = output.outputs[0].text
    print(token, end="", flush=True)
```

- Each iteration appends newly generated tokens to `.outputs[0].text`.
- Typical next step: send each chunk over a WebSocket or SSE stream.

---

## Online Serving: OpenAI-Compatible Server

Start the server from the command line:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct
```

Or with explicit options:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123
```

- The server exposes OpenAI-compatible endpoints at `http://localhost:8000`.
- Typical next step: point any OpenAI SDK client at `base_url="http://localhost:8000/v1"`.

### Chat Completions Endpoint

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ],
)
print(completion.choices[0].message.content)
```

- Uses the standard `POST /v1/chat/completions` endpoint.
- Typical next step: add `stream=True` for token-by-token streaming.

### Chat Completions Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Count to 10."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

- Each `chunk` carries a `delta.content` fragment.
- Typical next step: forward chunks to a frontend via SSE.

### Text Completions Endpoint

```python
completion = client.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.8,
)
print(completion.choices[0].text)
```

- Uses `POST /v1/completions` (legacy text completion format).
- Typical next step: use for non-chat models or raw prompt control.

### Responses Endpoint

vLLM also exposes a `/v1/responses` endpoint compatible with the OpenAI Responses API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

response = client.responses.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    input="Tell me a three-word story.",
)
print(response.output_text)
```

- Useful for stateless single-turn interactions with a simpler request shape.
- Typical next step: use with tool calling via the responses API for agentic workflows.

### Responses Endpoint Streaming

```python
with client.responses.stream(
    model="meta-llama/Llama-3.2-1B-Instruct",
    input="Tell me a three-word story.",
) as stream:
    for event in stream:
        print(event)
```

- Yields server-sent events; iterate to process each streaming event.
- Typical next step: parse event types (`response.output_text.delta`, etc.) for UI rendering.

---

## Batch Inference

### Offline Batch with the `LLM` Class

Pass a list of prompts directly to `.generate()` — vLLM automatically batches them:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

prompts = [f"Question {i}: What is {i} + {i}?" for i in range(20)]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

- All prompts are scheduled together; vLLM fills the batch optimally using PagedAttention.
- Typical next step: write outputs to a JSONL file for evaluation.

### OpenAI Batch File Format (Offline)

vLLM supports the OpenAI batch file format for offline processing:

```bash
vllm run-batch \
    -i input.jsonl \
    -o output.jsonl \
    --model meta-llama/Llama-3.2-1B-Instruct
```

Input JSONL format (one request per line):

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50}}
```

- `vllm run-batch` reads the JSONL, processes all requests, and writes results to the output file.
- Each output line contains the `custom_id` and the response body.
- Typical next step: parse output JSONL and match results by `custom_id`.

### Batched Online Requests

Send multiple concurrent requests to the running server using async HTTP:

```python
import asyncio
import aiohttp

async def send_request(session, prompt):
    payload = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }
    async with session.post(
        "http://localhost:8000/v1/chat/completions", json=payload
    ) as resp:
        return await resp.json()

async def main():
    prompts = ["Tell me a joke.", "What is Python?", "Explain gravity."]
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    for r in results:
        print(r["choices"][0]["message"]["content"])

asyncio.run(main())
```

- Concurrent requests are batched server-side by vLLM's continuous batching scheduler.
- Typical next step: tune concurrency level and monitor throughput via `/metrics`.

---

## Choosing an Inference Mode

| Mode | Entry Point | Use Case |
|---|---|---|
| Offline sync | `LLM.generate()` | Scripts, evaluation, data processing |
| Offline async streaming | `AsyncLLMEngine.generate()` | Async pipelines, custom servers |
| Online chat | `POST /v1/chat/completions` | Chat applications, OpenAI SDK clients |
| Online completion | `POST /v1/completions` | Raw prompt control, legacy integrations |
| Online responses | `POST /v1/responses` | Stateless single-turn, agentic tools |
| Batch file | `vllm run-batch` | Large-scale offline batch jobs |
