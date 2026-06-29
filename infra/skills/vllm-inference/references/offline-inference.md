# Offline Inference with the LLM Class

How to run batched, in-process LLM inference with `vllm.LLM` — sampling configuration, text and chat generation, multimodal and embedding inputs, and beam search.

## Contents

- Creating an LLM instance
- Configuring generation with SamplingParams
- Batch text generation with generate()
- Chat-style generation with chat()
- Multimodal inputs
- Prompt embedding inputs
- Beam search
- Reading outputs

## Creating an LLM instance

The `LLM` class loads model weights once and holds the engine for the lifetime of the process. Construct it a single time and reuse it for every batch — re-instantiating reloads weights and re-allocates the KV cache.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
```

Common constructor arguments:

- `model` — HF repo id or local path to the weights.
- `tensor_parallel_size` — number of GPUs to shard the model across (default `1`).
- `dtype` — `"auto"`, `"float16"`, `"bfloat16"`, etc.
- `gpu_memory_utilization` — fraction of GPU memory reserved for weights + KV cache (default `0.9`).
- `max_model_len` — cap the context length to fit a smaller KV cache.
- `quantization` — e.g. `"awq"`, `"gptq"`, `"fp8"` for quantized checkpoints.
- `trust_remote_code` — required for models that ship custom modeling code.
- `seed` — engine-level seed for reproducibility.
- `enable_prompt_embeds` — must be `True` to pass `prompt_embeds` (see below).

Typical next step: build a `SamplingParams` and call `generate()`.

## Configuring generation with SamplingParams

`SamplingParams` controls decoding. Pass one instance shared across all prompts, or a list aligned per-prompt.

```python
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
```

Frequently used fields:

- `temperature` — `0.0` gives greedy (deterministic) decoding; higher is more random.
- `top_p` / `top_k` — nucleus / top-k truncation of the sampling distribution.
- `n` — number of output sequences to return per prompt (parallel sampling).
- `max_tokens` — maximum number of tokens to generate.
- `min_tokens` — minimum tokens before an EOS / stop string can end generation.
- `stop` / `stop_token_ids` — strings or token ids that halt generation.
- `presence_penalty`, `frequency_penalty`, `repetition_penalty` — repetition controls.
- `seed` — per-request seed.
- `logprobs` — number of top logprobs to return per generated token.

For greedy decoding, set `temperature=0`. For sampling several candidates, set `n>1` and read all of `output.outputs`.

## Batch text generation with generate()

Pass a list of prompts; vLLM batches them internally with continuous batching, so submitting many prompts at once is far faster than looping one at a time.

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`generate()` returns one `RequestOutput` per input prompt, in the same order. Use `use_tqdm=False` to silence the progress bar in scripted runs. Typical next step: extract `output.outputs[0].text` for each result.

## Chat-style generation with chat()

`chat()` applies the model's chat template to a message list before generating — use it for instruction-tuned / chat models instead of hand-formatting the prompt string.

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

To batch multiple independent conversations, pass a list of message lists. Useful keyword arguments:

- `chat_template` — override the model's default template.
- `add_generation_prompt` — append the assistant turn marker (default `True`).
- `use_tqdm` — toggle the progress bar.

## Multimodal inputs

For vision-language models, pass a dict with the templated `prompt` (including the model's image placeholder token) and a `multi_modal_data` entry.

```python
from vllm import LLM
from vllm.assets.image import ImageAsset

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

image = ImageAsset("stop_sign").pil_image
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})

for output in outputs:
    print(output.outputs[0].text)
```

To send several images to a model that supports it, pass a list under the key:

```python
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": [image1, image2]},
})
```

Other modalities use the same shape with keys like `"video"` or `"audio"`, depending on model support. The placeholder token in `prompt` must match the number of media items provided.

## Prompt embedding inputs

Instead of token ids, you can feed precomputed input embeddings. The engine must be created with `enable_prompt_embeds=True`, and each prompt is passed as a dict with a `prompt_embeds` tensor of shape `(sequence_length, hidden_size)`.

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", enable_prompt_embeds=True)

# inputs_embeds: a torch.Tensor produced from the model's embedding layer
outputs = llm.generate({"prompt_embeds": inputs_embeds})

for output in outputs:
    print(output.outputs[0].text)
```

This is for advanced cases (e.g. soft prompts or embeddings produced by a separate module). For ordinary text, prefer plain string prompts.

## Beam search

Beam search is a separate method from `generate()` and takes a `BeamSearchParams` rather than `SamplingParams`.

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

outputs = llm.beam_search(
    ["The capital of France is"],
    BeamSearchParams(beam_width=4, max_tokens=64),
)

for output in outputs:
    for seq in output.sequences:
        print(seq.text)
```

`beam_width` sets the number of beams kept at each step; `max_tokens` caps generation length. Each entry in the returned list carries its beams under `.sequences`.

## Reading outputs

Both `generate()` and `chat()` return a list of `RequestOutput`, one per prompt, in input order. The useful fields:

- `output.prompt` — the original prompt string.
- `output.prompt_token_ids` — token ids of the prompt.
- `output.outputs` — list of completions (length `n` from `SamplingParams`).

Each completion in `output.outputs` exposes:

- `.text` — the generated string.
- `.token_ids` — generated token ids.
- `.cumulative_logprob` — summed logprob of the sequence.
- `.finish_reason` — `"stop"`, `"length"`, etc.

```python
for output in outputs:
    for completion in output.outputs:
        print(completion.finish_reason, completion.text)
```

When `n>1`, iterate all of `output.outputs` rather than only index `0` to capture every sampled candidate.
