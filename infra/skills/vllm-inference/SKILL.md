---
name: vllm-inference
description: "Run fast LLM inference and serving with vLLM. Use when serving a model over an OpenAI-compatible API (`vllm serve`, `/v1/chat/completions`), running offline batch generation with the `LLM` class and `SamplingParams`, fitting a large model on a small GPU via quantization (AWQ, GPTQ, FP8, BitsAndBytes, GGUF), scaling across GPUs with tensor/pipeline parallelism, or producing structured outputs, tool calls, LoRA-adapter serving, or speculative decoding. Use this whenever the user loads, serves, evaluates, or batch-generates from a Hugging Face / open-weights LLM and wants high-throughput inference — even as one step of a larger task, and even if they don't explicitly mention vLLM."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": [], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://docs.vllm.ai/en/latest/", "repo": "vllm-project/vllm", "fetched_at": "2026-06-28T15:02:59Z", "content_sha256": "c85d00862a98d5d5d1b9178e847bfee6aeb71615ace226d985bb9b7864de9654", "builder_version": "2.1.0"}, "coverage": ["html", "gh-readme", "changelog"]}}
---

# vLLM Inference

Run fast LLM inference and serving with vLLM. Use when serving a model over an OpenAI-compatible API (`vllm serve`, `/v1/chat/completions`), running offline batch generation with the `LLM` class and `SamplingParams`, fitting a large model on a small GPU via quantization (AWQ, GPTQ, FP8, BitsAndBytes, GGUF), scaling across GPUs with tensor/pipeline parallelism, or producing structured outputs, tool calls, LoRA-adapter serving, or speculative decoding. Use this whenever the user loads, serves, evaluates, or batch-generates from a Hugging Face / open-weights LLM and wants high-throughput inference — even as one step of a larger task, and even if they don't explicitly mention vLLM.

## Quick Start

Fastest path is an offline batch job in a single Python script:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="<model>")
params = SamplingParams(temperature=0.8, max_tokens=128)
outputs = llm.generate(["Hello, my name is"], params)
print(outputs[0].outputs[0].text)
```

For app code or multiple clients, run a persistent server instead: `vllm serve <model>` (see the serving workflow below).

## Decision Tree

| If the user wants... | Choose... | Then see |
|---|---|---|
| To call the model from app code / multiple clients | Online serving (`vllm serve`) for a persistent API; the offline `LLM` class for a one-shot batch job in a single script | `references/online-serving.md` vs `references/offline-inference.md` |
| To run a model that doesn't fit in VRAM at full precision | FP8 on Hopper+/Ada (fastest, minimal quality loss); pre-quantized AWQ/GPTQ (INT4) for broad GPU support; BitsAndBytes for on-the-fly quantization without a quantized checkpoint; GGUF for CPU/edge | `references/quantization.md` |
| To run a model too large for one GPU | Tensor parallelism within a single node (fast NVLink/PCIe); add pipeline parallelism only to span multiple nodes | `references/parallelism-scaling.md` |
| JSON/grammar-constrained output or function calling | Structured outputs (guided decoding) for schema-constrained text; tool calling for OpenAI-style function calls | `references/advanced-features.md` |
| Lower latency on single-stream decoding | Enable speculative decoding (draft model / n-gram / EAGLE) to cut per-step latency; otherwise keep default scheduling | `references/advanced-features.md` |

## Common Workflows

### Serve a model over an OpenAI-compatible API

Start a persistent server that exposes `/v1` endpoints for app code or multiple clients. For full flag and endpoint detail, see `references/online-serving.md`.

Copy this checklist:

- [ ] Run `vllm serve <model>` (set `--max-model-len` and `--gpu-memory-utilization` as needed)
- [ ] Wait for the server to report healthy on `/health` (run `scripts/wait_for_server.sh`)
- [ ] Confirm the model appears at `GET /v1/models`
- [ ] Send a request to `/v1/chat/completions` or `/v1/completions` (verify end-to-end with `scripts/smoke_completion.py`)
- [ ] **MCP fallback**: if a serve flag or endpoint you need isn't in `references/online-serving.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="vllm serve <flag-or-endpoint>"` — skip if references covered your case.

### Run offline batch inference

One-shot batch generation in a single Python script, no server. See `references/offline-inference.md` for chat messages, multimodal inputs, prompt embeds, and beam search.

Copy this checklist:

- [ ] Instantiate `LLM(model=<model>)`
- [ ] Build a list of prompts (or chat messages)
- [ ] Configure `SamplingParams` (temperature, max_tokens, n)
- [ ] Call `llm.generate(prompts, sampling_params)` and read outputs
- [ ] **MCP fallback**: if a `SamplingParams` field or `LLM` constructor option isn't in `references/offline-inference.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="SamplingParams <field>"` — skip if references covered your case.

### Fit a large model on a small GPU (quantization)

Use this only when the model does not fit in available VRAM at full precision. With memory headroom, load/serve at full precision — quantization adds dequantization overhead and can cost quality for no benefit when memory isn't the constraint. See `references/quantization.md`.

Copy this checklist:

- [ ] Step 1: Confirm the precondition — run `scripts/check_vram.sh`; the full-precision weights genuinely don't fit (otherwise load at full precision)
- [ ] Step 2: Pick a quantization method by checkpoint availability (pre-quantized AWQ/GPTQ/FP8/GGUF vs on-the-fly BitsAndBytes)
- [ ] Step 3: Pass `--quantization <method>` (or load a pre-quantized checkpoint that declares it)
- [ ] Step 4: Optionally enable a quantized KV cache to free more VRAM
- [ ] Step 5: Lower `--gpu-memory-utilization` / `--max-model-len` if it still OOMs
- [ ] **MCP fallback**: if your model's quantization method isn't in `references/quantization.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="<method> quantization support"` — skip if references covered your case.

### Scale across multiple GPUs (tensor/pipeline parallelism)

Use this only when the model is too large for a single GPU. On one GPU, keep the single-GPU path — parallelism adds cross-GPU communication overhead that buys nothing when the model already fits. See `references/parallelism-scaling.md`.

Copy this checklist:

- [ ] Step 1: Confirm the precondition — the model doesn't fit on one GPU (run `scripts/check_vram.sh`); otherwise serve on one GPU
- [ ] Step 2: Set `--tensor-parallel-size` to the GPU count on one node (fits a model too large for one GPU)
- [ ] Step 3: Add `--pipeline-parallel-size` only when spanning multiple nodes
- [ ] Step 4: For multi-node, launch a Ray cluster first, then serve across it
- [ ] Step 5: Verify all shards loaded and the server reports healthy (run `scripts/wait_for_server.sh`)
- [ ] **MCP fallback**: if your distributed/multi-node topology isn't in `references/parallelism-scaling.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="distributed serving <topology>"` — skip if references covered your case.

## When to Use

Use this skill whenever the user loads, serves, batch-generates from, or evaluates an open-weights / Hugging Face LLM, even as ONE STEP of a larger task.

**Use this skill when:**
- Serving a model behind an OpenAI-compatible API for app code or multiple clients
- Running high-throughput offline batch generation in a script
- A model won't fit in VRAM and needs quantization (AWQ/GPTQ/FP8/BitsAndBytes/GGUF) or multi-GPU parallelism
- Generating structured/JSON-constrained output, tool calls, or serving LoRA adapters
- Cutting single-stream decode latency with speculative decoding

**Reach for a different tool when the task needs a capability this skill does not provide:**
- For computing gradient updates / fine-tuning the weights themselves (including training LoRA adapters), use a trainer such as Hugging Face `transformers` `Trainer`, `peft`, or `trl` — then load the trained model or adapter back into vLLM to serve and evaluate it.

## Scripts

Execute these — don't read them as reference. Each runs without consuming context tokens.

- `scripts/check_vram.sh` — Probe `nvidia-smi` for free VRAM per GPU and warn if below a threshold. Run when: before launching `vllm serve` or instantiating `LLM`, to catch OOM early.
  ```bash
  bash scripts/check_vram.sh
  ```
- `scripts/wait_for_server.sh` — Poll `/health` until ready, then `GET /v1/models` to confirm the model loaded. Run when: right after starting `vllm serve`, before sending the first request.
  ```bash
  bash scripts/wait_for_server.sh
  ```
- `scripts/smoke_completion.py` — Send one `/v1/chat/completions` request via the openai client and print the response. Run when: after the server is healthy, to verify end-to-end serving works.
  ```bash
  python scripts/smoke_completion.py
  ```

## Old Patterns

<details>
<summary>Deprecated APIs (kept for historical context)</summary>

- **vLLM V0 engine** (legacy `LLMEngine` path and V0-only block-manager flags; deprecated at V1 GA) — use the V1 engine, now the default (`VLLM_USE_V1=1`). V0-specific flags are no-ops.

</details>

## References

- `references/offline-inference.md` — `LLM` class, `SamplingParams`, batch generate/chat, multimodal inputs, prompt embeds, beam search.
- `references/online-serving.md` — `vllm serve`, OpenAI-compatible endpoints (completions/chat/embeddings), engine + server arguments, request logging.
- `references/quantization.md` — AWQ, GPTQ, FP8, BitsAndBytes, GGUF, quantized KV cache, LLM Compressor (W8A8/W4A16).
- `references/parallelism-scaling.md` — tensor/pipeline/data/expert parallelism, distributed + multi-node serving, conserving memory, optimization and tuning.
- `references/advanced-features.md` — LoRA adapters, speculative decoding, structured outputs, tool calling, automatic prefix caching.

## Looking things up live (MCP fallback)

Per-workflow MCP triggers (above) handle the common cases. For anything else not covered by `references/`:

1. Verify the question is genuinely not in `references/` (grep first).
2. Resolve the libraryId — call `context7__resolve-library-id` with `libraryName="vllm"`. It returns one or more candidate library IDs; pick the one whose docs match the question.
3. Fetch the docs — call `context7__query-docs` with `libraryId="<from step 2>"` and `query="<your topic>"`. Read the returned snippet.
4. Cite the MCP source + libraryId in your answer. If `query-docs` returns nothing useful, say so — do not invent function names or flag values.

Note on naming: `context7__query-docs` and `context7__resolve-library-id` are the OpenClaw native tool names (double underscore prefix). Both are pre-registered for this skill — call them directly, no bash needed.
