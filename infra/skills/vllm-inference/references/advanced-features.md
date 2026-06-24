# Advanced Inference Features

How to drive vLLM's high-leverage serving features — LoRA adapters, speculative decoding, structured outputs, tool calling, automatic prefix caching, and multimodal inputs — from both the offline `LLM` API and the OpenAI-compatible server.

## Contents

- LoRA Adapters
- Speculative Decoding
- Structured Outputs
- Tool Calling
- Automatic Prefix Caching
- Multimodal Inputs

## LoRA Adapters

Serve one base model with many swappable fine-tuned adapters instead of loading a full model per task.

Offline: pass `enable_lora=True`, then route each prompt through a `LoRARequest(name, unique_int_id, path)`:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"],
)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
)
```

The second arg is a unique integer ID — reuse it to address the same adapter, change it for a different one. Next step: batch prompts with different `lora_request` values to multiplex adapters in one call.

Serving: register adapters at startup with `--enable-lora` and `--lora-modules name=path`:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

Clients then pass the adapter `name` as the `model` field. The JSON form also accepts a `base_model_name`:

```bash
vllm serve model \
    --enable-lora \
    --lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

Hot-swap at runtime (requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`):

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'

curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

Add `"load_inplace": true` to the load call to replace an adapter of the same name without unloading first. Set `--max-lora-rank` to the highest rank you serve, and `--lora-target-modules o_proj qkv_proj` to restrict which modules get adapters. For multimodal adapters that should always apply, map modality→adapter via `default_mm_loras` (offline) or `--default-mm-loras '{"audio":"..."}'` (server).

## Speculative Decoding

Cut latency by drafting several tokens cheaply and verifying them in one base-model pass. All variants are configured through one `speculative_config` dict.

Draft-model method — a small model proposes, the big model verifies:

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)
outputs = llm.generate(prompts, sampling_params)
```

The same config goes on the server as a JSON string:

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --seed 42 -tp 1 --max-model-len 2048 --gpu-memory-utilization 0.8 \
    --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}'
```

N-gram method — no draft model; proposals come from matching n-grams already in the prompt (great for tasks with heavy input copying like summarization or code edits):

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
```

EAGLE / EAGLE3 method — a trained lightweight drafter; note `draft_tensor_parallel_size` is set independently of the base model's `tensor_parallel_size`:

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)
```

For EAGLE3, switch `"method": "eagle3"` and point `model` at an eagle3 drafter (e.g. `"RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"`). Next step: tune `num_speculative_tokens` up while watching the acceptance rate — too high wastes verification compute when drafts are rejected.

## Structured Outputs

Constrain generation to a schema so downstream parsing never fails. As of v0.12.0 the legacy `guided_*` params are unified under a `structured_outputs` object (`guided_json`→`{"structured_outputs": {"json": ...}}`, etc.).

Online, pass the constraint via `extra_body` (or `response_format` for JSON schema). Choice restricts output to a fixed set:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
model = client.models.list().data[0].id

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}],
    extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
)
print(completion.choices[0].message.content)
```

Regex forces the output to match a pattern:

```python
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate an example email address for Alan Turing..."}],
    extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
)
```

JSON schema — derive it from a Pydantic model and pass it through `response_format`:

```python
from pydantic import BaseModel

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: str

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate a JSON with the brand, model and car_type..."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "car-description", "schema": CarDescription.model_json_schema()},
    },
)
```

A `grammar` (EBNF) key constrains to a formal grammar (e.g. a SQL subset). Offline, the same constraints live in `StructuredOutputsParams` inside `SamplingParams`:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(choice=["Positive", "Negative"])
)
outputs = llm.generate(prompts="Classify this sentiment: vLLM is wonderful!", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

`StructuredOutputsParams` fields: `json`, `regex`, `choice`, `grammar`, `structural_tag`. Next step: validate the JSON output against the same Pydantic model — with schema enforcement it should always parse.

## Tool Calling

Let the model emit structured function calls. Start the server with auto tool choice plus a model-specific parser:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

`--enable-auto-tool-choice` is mandatory. Pick `--tool-call-parser` to match the model family: `hermes` (Qwen 2.5, Nous Hermes), `mistral`, `llama3_json` (Llama 3.1/3.2), `llama4_pythonic` (Llama 4), `granite`, `deepseek_v3`, `xlam`, `pythonic`, and others.

Call it with the standard OpenAI `tools` + `tool_choice` fields:

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```

`tool_choice` modes: `"auto"` (model decides — arguments may occasionally be malformed), `"required"` (guarantees ≥1 tool call), `{"type": "function", "function": {"name": "get_weather"}}` (forces a named function, backed by the structured-outputs engine), and `"none"` (no calls). Next step: `json.loads(tool_call.arguments)`, run the function, and append the result as a `tool` message for the follow-up turn.

## Automatic Prefix Caching

APC caches the KV cache of past queries so a new query that **shares a prefix** reuses it instead of recomputing — ideal for repeated questions over one long document or multi-round chat.

Enable it offline with `enable_prefix_caching=True`:

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_prefix_caching=True)
```

Or on the server with `--enable-prefix-caching`. Caveat: APC only speeds up the prefill phase, not decoding — gains are large when prompts share long prefixes and small when outputs are long or prefixes diverge. Next step: order your prompts so the shared context (system prompt, document) comes first, maximizing the cacheable prefix.

## Multimodal Inputs

Feed images, video, or audio alongside text. Offline, put a placeholder token in the `prompt` and the media under `multi_modal_data`:

```python
from vllm import LLM
import PIL.Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
image = PIL.Image.open(...)

outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}})
```

The placeholder is model-specific (`<image>` for LLaVA, `<|image_1|>` for Phi-3.5-vision). For multiple images, raise `limit_mm_per_prompt` and pass a list:

```python
llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 2},
)
prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"
outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": [image1, image2]}})
```

Use the `"video"` key (with `limit_mm_per_prompt={"video": 1}`) for video and the `"audio"` key — value `(array, sampling_rate)` — for audio.

On the server, launch with the modality limit, then use the OpenAI chat content blocks (`image_url`, `video_url`, or base64 `input_audio`):

```bash
vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt.image 2
```

```python
chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
)
```

Next step: pass stable `multi_modal_uuids` (offline) so repeated media is cached across requests — you can even send `None` in place of the data once a UUID has been seen.
```

**Note on sourcing:** the documentation block in the original task was only the vLLM nav sidebar + README — no feature pages, no code. Rather than invent APIs (which the hard rules forbid), I fetched the six live feature pages from `docs.vllm.ai` and the vLLM GitHub repo. The spec-decode page had moved, so those snippets come from `docs/features/speculative_decoding/{draft_model,n_gram,eagle}.md` on `main`. Two minor simplifications I made to keep examples self-contained: in the structured-outputs Pydantic example I inlined `car_type: str` instead of the source's `CarType` enum, and I trimmed long prompt strings to `...`. Everything else is verbatim. If this is destined for a specific skill's `references/` dir, tell me the path and I'll write it there.
