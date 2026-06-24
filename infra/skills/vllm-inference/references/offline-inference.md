# Offline Batched Inference

How to run vLLM in-process (no server) for generation, chat, beam search, prompt-embedding inputs, and pooling/embedding/scoring — all driven by the `LLM` class and `SamplingParams`.

## Contents

- The LLM class
- SamplingParams
- Batched generate
- Chat over conversations
- Beam search
- Prompt embedding inputs
- Pooling models: embed, classify, score, reward, encode
- Output object shapes
- Determinism notes

## The LLM class

`LLM` loads a model once and serves many requests in-process. Construct it once, reuse it for every batch — initialization (weight load + CUDA-graph capture) is the expensive step.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
```

Frequently-used constructor arguments:

| Arg | Purpose |
|---|---|
| `model` | HF repo id or local path. |
| `tensor_parallel_size` | Number of GPUs to shard the model across. |
| `dtype` | `"auto"`, `"half"`, `"bfloat16"`, `"float16"`, `"float32"`. |
| `quantization` | `"awq"`, `"gptq"`, `"fp8"`, etc. |
| `max_model_len` | Cap the context length (lowers KV-cache memory). |
| `gpu_memory_utilization` | Fraction of GPU memory for weights + KV cache (default 0.9). |
| `enforce_eager` | Disable CUDA graphs (slower, less memory, faster startup). |
| `trust_remote_code` | Allow custom model code from the HF repo. |
| `seed` | RNG seed for sampling. |
| `task` | Select model role: `"generate"`, `"embed"`, `"classify"`, `"score"`, `"reward"`. |
| `enable_prefix_caching` | Reuse KV across prompts sharing a prefix. |

Typical next step after construction is to build a `SamplingParams` and call `llm.generate(...)`.

## SamplingParams

`SamplingParams` controls decoding. Pass one shared object for the whole batch, or a list aligned 1:1 with the prompts for per-prompt control.

```python
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

Key fields:

| Field | Meaning |
|---|---|
| `n` | Number of output sequences to return per prompt. |
| `temperature` | 0 = greedy/deterministic; higher = more random. |
| `top_p`, `top_k`, `min_p` | Nucleus / top-k / min-probability truncation. |
| `max_tokens` | Max tokens to generate per output. |
| `min_tokens` | Force at least this many tokens before EOS allowed. |
| `stop`, `stop_token_ids` | Stop strings / token ids that end generation. |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Penalize repetition. |
| `seed` | Per-request seed (overrides engine seed for this request). |
| `logprobs` | Return this many top logprobs per generated token. |
| `prompt_logprobs` | Return logprobs for the prompt tokens too. |
| `ignore_eos` | Keep generating past the EOS token. |
| `skip_special_tokens` | Strip special tokens from decoded text (default True). |

For greedy, reproducible decoding set `temperature=0`.

## Batched generate

`llm.generate` takes a single prompt or a list of prompts and runs them through continuous batching automatically — you do not manage batches yourself.

```python
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Output order matches input order. To get multiple completions per prompt, set `SamplingParams(n=k)` and read `output.outputs[0..k-1]`. Pass `use_tqdm=False` to silence the progress bar in scripts.

## Chat over conversations

`llm.chat` applies the model's chat template, so you pass role/content messages instead of a pre-formatted string.

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Write an essay about the importance of higher education."},
]

outputs = llm.chat(conversation, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

To batch many independent conversations, pass a list of conversations (a list of message-lists). Useful keyword arguments:

- `chat_template` — override the model's default template.
- `add_generation_prompt` — append the assistant-turn prompt (default True).
- `use_tqdm` — progress bar toggle.

## Beam search

Beam search is a separate method (not a `SamplingParams` mode) and uses `BeamSearchParams`.

```python
from vllm.sampling_params import BeamSearchParams

outputs = llm.beam_search(
    prompts,
    BeamSearchParams(beam_width=4, max_tokens=50),
)

for output in outputs:
    for seq in output.sequences:
        print(seq.text, seq.cum_logprob)
```

`BeamSearchParams` fields include `beam_width`, `max_tokens`, `ignore_eos`, `temperature`, and `length_penalty`. Each result exposes `.sequences`, ranked best-first, where every sequence has `.text`, `.tokens`, and `.cum_logprob`.

## Prompt embedding inputs

Instead of token ids, you can feed precomputed embeddings (e.g. from an upstream encoder or a soft prompt). Enable the feature on the engine, then pass a `prompt_embeds` tensor of shape `(seq_len, hidden_size)`.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", enable_prompt_embeds=True)

# embeds: torch.Tensor with shape (seq_len, hidden_size), dtype matching the model
outputs = llm.generate(
    {"prompt_embeds": embeds},
    SamplingParams(max_tokens=64),
)
print(outputs[0].outputs[0].text)
```

Pass a list of `{"prompt_embeds": tensor}` dicts to batch multiple embedding prompts. The hidden size must match the model's embedding dimension.

## Pooling models: embed, classify, score, reward, encode

Pooling models return vectors or scalars instead of generated text. Load the model with the matching `task`, then call the corresponding method. These methods are unavailable on a model loaded for plain generation.

**Embeddings** — load with `task="embed"`:

```python
llm = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed")

outputs = llm.embed(["Follow the white rabbit."])
for output in outputs:
    embedding = output.outputs.embedding   # list[float]
    print(len(embedding))
```

**Classification** — load with `task="classify"`; returns per-class probabilities:

```python
llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", task="classify")

outputs = llm.classify(["Hello, my name is"])
for output in outputs:
    probs = output.outputs.probs           # list[float]
    print(probs)
```

**Scoring / reranking** — load with `task="score"`; scores a query against one or more candidate texts:

```python
llm = LLM(model="BAAI/bge-reranker-v2-m3", task="score")

outputs = llm.score(
    "What is the capital of France?",
    [
        "The capital of France is Paris.",
        "The capital of Brazil is Brasilia.",
    ],
)
for output in outputs:
    print(output.outputs.score)            # float
```

**Reward models** — load with `task="reward"`:

```python
llm = LLM(model="<reward-model>", task="reward")
outputs = llm.reward(["The quick brown fox."])
```

**Raw pooler output** — `llm.encode` returns the unprocessed pooled tensor and accepts an optional `PoolingParams`:

```python
outputs = llm.encode(["Hello, my name is"])
for output in outputs:
    data = output.outputs.data             # raw pooled tensor
```

## Output object shapes

- **Generation** (`generate`, `chat`): each result has `.prompt`, `.prompt_token_ids`, and `.outputs` — a list of completions, each with `.text`, `.token_ids`, `.cumulative_logprob`, and `.finish_reason`.
- **Beam search**: each result has `.sequences`, each with `.text`, `.tokens`, `.cum_logprob`.
- **Pooling**: each result has a single `.outputs` object — `.embedding` (embed), `.probs` (classify), `.score` (score), or `.data` (encode).

## Determinism notes

For reproducible runs, set both the engine `seed` (`LLM(..., seed=0)`) and use greedy decoding (`SamplingParams(temperature=0)`). A per-request `SamplingParams(seed=...)` overrides the engine seed for that request only. Note that changing `tensor_parallel_size`, batch composition, or hardware can still shift floating-point results even with a fixed seed.
