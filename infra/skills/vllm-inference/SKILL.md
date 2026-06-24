---
name: vllm-inference
description: "Serve and run large language models for fast, high-throughput inference using vLLM's `LLM` offline class and the OpenAI-compatible `vllm serve` server. Use when deploying a model behind an OpenAI-compatible API, running offline batched generation over a fixed dataset with `SamplingParams`, quantizing a model (AWQ, GPTQ, FP8, GGUF, BitsAndBytes) to fit limited VRAM, or scaling a large model across GPUs with tensor, pipeline, data, or expert parallelism. Reach for this skill whenever the user works with vLLM, `vllm serve`, `LLM.generate`/`LLM.chat`, batched LLM generation, LoRA serving, speculative decoding, or structured/tool-calling outputs — even if they don't explicitly mention vLLM, and even when fast inference is only one step of a larger task (for example, evaluating a fine-tuned checkpoint)."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": [], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://docs.vllm.ai/en/latest/", "repo": "vllm-project/vllm", "fetched_at": "2026-06-24T10:19:27Z", "content_sha256": "62043f6b79ffcaa5b1a926189c017bafbbb3d28934c8d6afba78fa0a7a524f54", "builder_version": "2.0.0"}, "coverage": ["html", "gh-readme", "changelog"]}}
---

# vLLM Inference

Serve and run large language models for fast, high-throughput inference using vLLM's `LLM` offline class and the OpenAI-compatible `vllm serve` server. Use when deploying a model behind an OpenAI-compatible API, running offline batched generation over a fixed dataset with `SamplingParams`, quantizing a model (AWQ, GPTQ, FP8, GGUF, BitsAndBytes) to fit limited VRAM, or scaling a large model across GPUs with tensor, pipeline, data, or expert parallelism. Reach for this skill whenever the user works with vLLM, `vllm serve`, `LLM.generate`/`LLM.chat`, batched LLM generation, LoRA serving, speculative decoding, or structured/tool-calling outputs — even if they don't explicitly mention vLLM, and even when fast inference is only one step of a larger task (for example, evaluating a fine-tuned checkpoint).

## Installation

```bash
pip install vllm
```

## Quick Start

Run offline batched generation against a model in a few lines:

```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

To instead expose an OpenAI-compatible HTTP API, launch a server with `vllm serve <model>` (see the first workflow below).

## Decision Tree

| If the user wants... | Choose... | Then see |
|---|---|---|
| Concurrent/streaming API traffic vs a one-shot batch over a fixed dataset | Online serving (`vllm serve`, OpenAI-compatible endpoints) for concurrent/streaming traffic; offline `LLM` class for fixed batches and pipelines | `references/online-serving.md` vs `references/offline-inference.md` |
| To quantize a model to fit or run faster | FP8 on Hopper+ for near-lossless speedup; AWQ/GPTQ INT4 for broad GPU support and max compression; BitsAndBytes for on-the-fly load without calibration; GGUF for portability/CPU | `references/quantization.md` |
| More throughput or to fit a model that doesn't fit on one GPU | Tensor parallel within one node with fast interconnect; pipeline parallel across nodes; data parallel to replicate for throughput; expert parallel for MoE | `references/distributed-and-tuning.md` |
| To fix CUDA out-of-memory at startup or under load | Lower `--gpu-memory-utilization`, reduce `--max-model-len`, quantize weights/KV cache, or enable prefix caching | `references/distributed-and-tuning.md` |
| To optimize for low latency vs high throughput | Speculative decoding (draft/EAGLE/MTP/n-gram) for latency; larger batches + chunked prefill for throughput | `references/advanced-features.md` |
| JSON/grammar-constrained output or tool/function calling | Structured outputs for schema/regex/grammar constraints; tool calling for function invocation and reasoning | `references/advanced-features.md` |

## Common Workflows

### Deploy an OpenAI-compatible server

Stand up a streaming, concurrent inference API behind the OpenAI protocol. For server/engine arguments and deployment recipes, see `references/online-serving.md`.

Copy this checklist:

- [ ] Install vLLM and confirm GPU/driver visibility
- [ ] Launch with `vllm serve <model>` and set key server args (`--max-model-len`, `--gpu-memory-utilization`)
- [ ] Poll `/health` until ready and list `/v1/models`
- [ ] Send a test request to `/v1/chat/completions` and verify output
- [ ] **MCP fallback**: if a server or engine argument you need isn't in `references/online-serving.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned `libraryId` and `query="vllm serve <arg>"` — skip if references covered your case.

### Offline batched inference

Run a fixed batch of prompts through the `LLM` class and persist results. For `SamplingParams` fields, `generate`/`chat`/beam search, and pooling models, see `references/offline-inference.md`.

Copy this checklist:

- [ ] Instantiate `LLM(model=...)` with dtype and memory settings
- [ ] Build prompts and configure `SamplingParams`
- [ ] Call `llm.generate()` / `llm.chat()` over the batch
- [ ] Collect outputs and persist results
- [ ] **MCP fallback**: if the `SamplingParams` field or `LLM` method you need isn't in `references/offline-inference.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned `libraryId` and `query="SamplingParams <field>"` — skip if references covered your case.

### Serve a quantized model on limited VRAM

Fit a model into a tight VRAM budget by quantizing weights and KV cache. For method recipes and LLM Compressor (W8A8/W4A16/W4A8), see `references/quantization.md`.

Copy this checklist:

- [ ] Pick a quantization method for the GPU and quality budget
- [ ] Load the prequantized checkpoint or apply LLM Compressor
- [ ] Set `--gpu-memory-utilization`, `--max-model-len`, and optional quantized KV cache
- [ ] Probe VRAM headroom and run a smoke request
- [ ] **MCP fallback**: if your quantization format or LLM Compressor recipe isn't in `references/quantization.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned `libraryId` and `query="<quant-method> quantization"` — skip if references covered your case.

### Multi-GPU distributed serving for a large model

Spread a large model across GPUs (and nodes) without OOM. For parallelism modes and multi-node serving via Ray, see `references/distributed-and-tuning.md`.

Copy this checklist:

- [ ] Choose parallelism (tensor vs pipeline vs data vs expert) for the topology
- [ ] Set `--tensor-parallel-size` / `--pipeline-parallel-size` (multi-node via Ray)
- [ ] Launch and confirm all workers join the engine
- [ ] Verify throughput and that the model fits without OOM
- [ ] **MCP fallback**: if your parallelism or multi-node setup isn't covered in `references/distributed-and-tuning.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned `libraryId` and `query="<tensor|pipeline|data|expert> parallel serving"` — skip if references covered your case.

## When to Use

Use this skill whenever the user works with vLLM, `vllm serve`, `LLM.generate`/`LLM.chat`, batched LLM generation, quantized inference, LoRA serving, speculative decoding, or structured/tool-calling outputs — even as ONE STEP of a larger task.

**Use this skill when:**
- Serving a model behind an OpenAI-compatible API for concurrent or streaming traffic.
- Running offline batched generation, chat, beam search, or pooling/embedding/scoring over a fixed dataset.
- Quantizing a model (AWQ, GPTQ, FP8, GGUF, BitsAndBytes, LLM Compressor) to fit limited VRAM or run faster.
- Scaling a large model across GPUs or nodes with tensor/pipeline/data/expert parallelism.
- Needing fast inference to evaluate a checkpoint, generate completions, or constrain output to JSON/grammar or tool calls.

**Reach for a different tool when the task needs a capability this skill does not provide:**
- For training or fine-tuning model weights via gradient updates (SFT, LoRA training, RLHF) — use a trainer like Hugging Face `Trainer`, TRL, or PEFT, then serve the resulting checkpoint here.
- For arbitrary access to model internals (hidden states, custom forward hooks, single-step debugging) beyond vLLM's exposed APIs — use raw `transformers` instead.

## Scripts

Execute these — don't read them as reference. Each runs without consuming context tokens.

- `scripts/wait_for_server.sh` — Poll the `/health` endpoint until the server returns 200, then list `/v1/models`. Run when: right after `vllm serve` is launched, before sending requests.
  ```bash
  bash scripts/wait_for_server.sh
  ```
- `scripts/check_vram.sh` — Probe `nvidia-smi` for free VRAM per GPU and warn if headroom is below a threshold. Run when: before loading a model, to confirm the GPU can hold weights + KV cache.
  ```bash
  bash scripts/check_vram.sh
  ```
- `scripts/estimate_model_vram.py` — Estimate weight + KV-cache memory from parameter count, dtype, and max-model-len and compare against available VRAM. Run when: during capacity planning, before choosing quantization or parallelism.
  ```bash
  python scripts/estimate_model_vram.py
  ```

## Old Patterns

<details>
<summary>Deprecated APIs (kept for historical context)</summary>

- **V0 engine** (`LLMEngine`/`AsyncLLMEngine` legacy path, deprecated in 0.8.0) — use the V1 engine instead; it is the default since 0.8.x and enabled automatically.
- **`SamplingParams(use_beam_search=True)`** (deprecated in 0.6.3) — use the dedicated beam search API / `BeamSearchParams` via `LLM.beam_search()` instead.

</details>

## References

- `references/offline-inference.md` — `LLM` class, `SamplingParams`, batched generate/chat, beam search, prompt embeds, and pooling/embedding/scoring models.
- `references/online-serving.md` — `vllm serve`, OpenAI-compatible server endpoints, server/engine arguments, Docker and Kubernetes deployment.
- `references/quantization.md` — AWQ, GPTQ, GGUF, BitsAndBytes, FP8, LLM Compressor (W8A8/W4A16/W4A8), and quantized KV cache.
- `references/distributed-and-tuning.md` — tensor/pipeline/data/expert parallelism, multi-node serving, conserving memory, and throughput-vs-latency tuning.
- `references/advanced-features.md` — LoRA adapters, speculative decoding, structured outputs, tool calling, automatic prefix caching, and multimodal inputs.

## Looking things up live (MCP fallback)

Per-workflow MCP triggers (above) handle the common cases. For anything else not covered by `references/`:

1. Verify the question is genuinely not in `references/` (grep first).
2. Resolve the libraryId — call `context7__resolve-library-id` with `libraryName="vllm"`. It returns one or more candidate library IDs; pick the one whose docs match the question.
3. Fetch the docs — call `context7__query-docs` with `libraryId="<from step 2>"` and `query="<your topic>"`. Read the returned snippet.
4. Cite the MCP source + libraryId in your answer. If `query-docs` returns nothing useful, say so — do not invent function names or flag values.

Note on naming: `context7__query-docs` and `context7__resolve-library-id` are the OpenClaw native tool names (double underscore prefix). Both are pre-registered for this skill — call them directly, no bash needed.
