# Features and Integrations

Covers automatic prefix caching, LoRA adapters, speculative decoding, structured outputs, tool calling, multimodal inputs, reasoning outputs, and LangChain/LlamaIndex integrations in vLLM.

## Contents

- Automatic Prefix Caching
- LoRA Adapters
- Speculative Decoding
- Structured Outputs
- Tool Calling
- Multimodal Inputs
- Reasoning Outputs
- LangChain Integration
- LlamaIndex Integration

---

## Automatic Prefix Caching

Automatic Prefix Caching (APC) reuses KV cache blocks from previous requests that share a common prompt prefix, eliminating redundant computation.

### Enabling APC

Pass `--enable-prefix-caching` to the server or set it in the `LLM` constructor:

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_prefix_caching=True)
```

Or via the CLI:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching
```

### When APC Helps

APC is most effective when:
- Many requests share a long system prompt (e.g., a RAG context or few-shot examples).
- Multi-turn conversations reuse prior turns.
- The same document is queried repeatedly.

The first request populates the cache; subsequent requests with the same prefix skip prefill for those tokens. No change to request format is required — vLLM detects shared prefixes automatically.

### Typical Next Step

Combine with chunked prefill (`--enable-chunked-prefill`) for best throughput when prefixes are long.

---

## LoRA Adapters

vLLM supports loading and serving multiple LoRA adapters on top of a single base model, including adapters for dense and MoE layers.

### Enabling LoRA at Server Start

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules sql-lora=./sql-lora-adapter
```

`--lora-modules` accepts `name=path` pairs. Multiple adapters can be listed.

### Offline Inference with LoRA

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_lora=True)

sampling_params = SamplingParams(temperature=0, max_tokens=256)

outputs = llm.generate(
    ["Translate to SQL: find all users older than 30"],
    sampling_params,
    lora_request=LoRARequest("sql-lora", 1, "./sql-lora-adapter"),
)
print(outputs[0].outputs[0].text)
```

`LoRARequest` takes `(name, id, local_path)`. The `id` must be unique per adapter.

### Online: Selecting an Adapter per Request

When the server is running with `--enable-lora`, pass `model` as the adapter name in the OpenAI-compatible request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="sql-lora",   # adapter name registered at startup
    messages=[{"role": "user", "content": "List all active users"}],
)
```

### Typical Next Step

Use `--max-loras` and `--max-lora-rank` to control memory allocation when serving many adapters concurrently.

---

## Speculative Decoding

Speculative decoding accelerates generation by having a small draft model (or heuristic) propose tokens that the target model verifies in parallel.

### Draft Model Speculation

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5
```

The draft model proposes `num-speculative-tokens` tokens per step; the target model verifies them in one forward pass.

### N-Gram Speculation (no draft model)

N-gram speculation uses the prompt itself to propose continuations — zero additional model overhead:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model [ngram] \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 4
```

### EAGLE Draft Models

EAGLE uses a trained draft head for higher acceptance rates:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model lmzheng/eagle-llama3.1-70b-instruct \
  --speculative-draft-tensor-parallel-size 1 \
  --num-speculative-tokens 5
```

### MTP (Multi-Token Prediction)

For models with built-in MTP heads (e.g., DeepSeek-V3):

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --speculative-model [mtp] \
  --num-speculative-tokens 1
```

### Offline Inference with Speculative Decoding

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
)
outputs = llm.generate(["The future of AI is"], SamplingParams(max_tokens=100))
```

### Typical Next Step

Speculative decoding is output-equivalent to standard decoding. Measure acceptance rate via production metrics to tune `num-speculative-tokens`.

---

## Structured Outputs

vLLM can constrain generation to valid JSON, a JSON Schema, a regex, a grammar, or a choice list using `xgrammar` or `guidance` backends.

### JSON Mode (online)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Give me a JSON object with name and age"}],
    response_format={"type": "json_object"},
)
```

### JSON Schema (online)

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
    },
    "required": ["name", "age"],
}

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Extract person info"}],
    response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}},
)
```

### Structured Outputs Offline

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

guided = GuidedDecodingParams(json=schema)
params = SamplingParams(guided_decoding=guided, max_tokens=200)

outputs = llm.generate(["Extract: Alice is 30 years old"], params)
print(outputs[0].outputs[0].text)
```

### Regex and Choice Constraints

```python
# Regex
guided = GuidedDecodingParams(regex=r"\d{3}-\d{2}-\d{4}")

# Choice list
guided = GuidedDecodingParams(choice=["positive", "negative", "neutral"])
```

### Grammar (EBNF / GBNF)

```python
grammar = r"""
root  ::= object
object ::= "{" pair ("," pair)* "}"
pair  ::= string ":" value
"""
guided = GuidedDecodingParams(grammar=grammar)
```

### Selecting the Backend

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --guided-decoding-backend xgrammar   # or: guidance
```

### Typical Next Step

Use `json_schema` with Pydantic models by calling `model.model_json_schema()` to generate the schema automatically.

---

## Tool Calling

vLLM supports OpenAI-compatible function/tool calling, including streaming and required-tool-choice modes.

### Server Setup

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

`--tool-call-parser` selects the parser for the model family. Common values: `llama3_json`, `hermes`, `mistral`, `xlam`.

### Defining and Calling Tools (online)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

### Requiring a Specific Tool

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Look up the weather"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
```

### Offline Tool Calling

```python
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import load_chat_template

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_auto_tool_choice=True,
          tool_call_parser="llama3_json")

messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]
outputs = llm.chat(messages, tools=tools, sampling_params=SamplingParams(max_tokens=256))
```

### Streaming Tool Calls

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Get weather for Berlin"}],
    tools=tools,
    tool_choice="auto",
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        print(delta.tool_calls[0].function.arguments, end="", flush=True)
```

### Typical Next Step

After receiving a tool call, append the assistant message and a `tool` role message with the result, then call the API again for the final response.

---

## Multimodal Inputs

vLLM supports image, video, and audio inputs for vision-language and multimodal models.

### Image Input (online)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
)
```

### Image Input (offline)

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct")

image = Image.open("cat.jpg")

outputs = llm.generate(
    {
        "prompt": "<|vision_start|><|image_pad|><|vision_end|>Describe this image.",
        "multi_modal_data": {"image": image},
    },
    SamplingParams(max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

### Multiple Images

```python
outputs = llm.generate(
    {
        "prompt": "<image><image>Compare these two images.",
        "multi_modal_data": {"image": [image1, image2]},
    },
    SamplingParams(max_tokens=256),
)
```

### Video Input (offline)

```python
import numpy as np

# frames: list of PIL Images or numpy arrays
outputs = llm.generate(
    {
        "prompt": "<video>Summarize this video.",
        "multi_modal_data": {"video": frames},
    },
    SamplingParams(max_tokens=256),
)
```

### Audio Input (offline)

```python
import librosa

audio, sr = librosa.load("speech.wav", sr=16000)

outputs = llm.generate(
    {
        "prompt": "<audio>Transcribe the audio.",
        "multi_modal_data": {"audio": (audio, sr)},
    },
    SamplingParams(max_tokens=256),
)
```

### Serving a Multimodal Model

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 32768 \
  --limit-mm-per-prompt image=5
```

`--limit-mm-per-prompt` caps how many media items a single request may include.

### Typical Next Step

Check the supported models list (`docs/models/supported_models`) to confirm which multimodal modalities a given model supports before serving.

---

## Reasoning Outputs

vLLM exposes chain-of-thought / reasoning tokens (e.g., from DeepSeek-R1, QwQ) separately from the final answer.

### Enabling Reasoning (server)

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --reasoning-parser deepseek_r1
```

Common `--reasoning-parser` values: `deepseek_r1`, `qwen3`.

### Accessing Reasoning Content (online)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Solve: 2x + 5 = 13"}],
)

message = response.choices[0].message
print("Reasoning:", message.reasoning_content)
print("Answer:   ", message.content)
```

### Streaming Reasoning

```python
stream = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
    elif delta.content:
        print(delta.content, end="", flush=True)
```

### Reasoning with Tool Calls

Reasoning models can also emit tool calls. Start the server with both flags:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### Typical Next Step

Use `reasoning_content` for logging or display; use `content` as the model's final answer to pass downstream.

---

## LangChain Integration

vLLM's OpenAI-compatible server works directly with LangChain's `ChatOpenAI` and `OpenAI` wrappers.

### Chat Model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="token",
)

response = llm.invoke("Explain PagedAttention in one sentence.")
print(response.content)
```

### Streaming with LangChain

```python
for chunk in llm.stream("Write a haiku about inference speed."):
    print(chunk.content, end="", flush=True)
```

### Using in a Chain

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

chain = prompt | llm
result = chain.invoke({"question": "What is speculative decoding?"})
print(result.content)
```

### Typical Next Step

For RAG pipelines, pair `ChatOpenAI` pointing at vLLM with LangChain's `OpenAIEmbeddings` (or a local embedding model) and a vector store retriever.

---

## LlamaIndex Integration

LlamaIndex connects to vLLM via its OpenAI-compatible client wrappers.

### LLM Setup

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="token",
)

response = llm.complete("The capital of France is")
print(response.text)
```

### Chat Interface

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a concise assistant."),
    ChatMessage(role="user", content="Summarize vLLM in two sentences."),
]

response = llm.chat(messages)
print(response.message.content)
```

### Using as a Query Engine

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

Settings.llm = llm

documents = SimpleDirectoryReader("./docs").load_data()
