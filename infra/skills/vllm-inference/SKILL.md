---
name: vllm-inference
description: "Serve and query large language models at high throughput using vLLM's `LLM` class for offline batch inference or the OpenAI-compatible server (`vllm serve`) for online serving. Invoke when you need PagedAttention-based KV cache management, multi-GPU tensor/pipeline parallelism, quantized model serving (AWQ, GGUF, FP8, GPTQ, BitsAndBytes, INT4/INT8), LoRA adapter hot-swapping, structured outputs, or speculative decoding — use when latency and throughput matter at production scale."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": [], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://docs.vllm.ai/en/latest/", "repo": "", "fetched_at": "2026-05-08T02:18:43Z", "content_sha256": "4da8c0dae223ea3f91e732b88397be1a1b6bf913cb29ea957e22d46ecb70bb41", "builder_version": "1.3.0"}, "coverage": ["html"]}}
---

# vLLM Inference

Serve and query large language models at high throughput using vLLM's `LLM` class for offline batch inference or the OpenAI-compatible server (`vllm serve`) for online serving. Invoke when you need PagedAttention-based KV cache management, multi-GPU tensor/pipeline parallelism, quantized model serving (AWQ, GGUF, FP8, GPTQ, BitsAndBytes, INT4/INT8), LoRA adapter hot-swapping, structured outputs, or speculative decoding — use when latency and throughput matter at production scale.

## Installation

```bash
pip install vllm
```

## Quick Start

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, my name is"], params)
print(outputs[0].outputs[0].text)
```

## Common Workflows

### Offline batch inference

Run inference locally without a server using the `LLM` class and `SamplingParams`. For full API details see `references/inference-modes.md`.

Copy this checklist:

- [ ] Step 1: Install vLLM and load model with `LLM(model=...)`
- [ ] Step 2: Build prompt list and set `SamplingParams`
- [ ] Step 3: Call `llm.generate(prompts, sampling_params)`
- [ ] Step 4: Iterate outputs and extract `output.outputs[0].text`

### Launch OpenAI-compatible server and query it

Start the vLLM server and query it using the standard OpenAI client or curl. For endpoint details and server arguments see `references/inference-modes.md` and `references/features-and-configuration.md`.

Copy this checklist:

- [ ] Step 1: Start server: `vllm serve <model> [--tensor-parallel-size N] [--quantization ...]`
- [ ] Step 2: Wait for server ready (`GET /health`)
- [ ] Step 3: Send requests via `openai` Python client or curl to `/v1/chat/completions`
- [ ] Step 4: Parse streamed or batched response JSON

### Serve a quantized model with LoRA adapters

Reduce VRAM usage with a quantized checkpoint and attach named LoRA adapters at serve time. For supported formats and adapter configuration see `references/models-and-quantization.md`.

Copy this checklist:

- [ ] Step 1: Choose quantization format (e.g. AWQ, FP8) and download quantized checkpoint
- [ ] Step 2: Launch server with `--quantization <method> --enable-lora --lora-modules <name>=<path>`
- [ ] Step 3: Specify model name in client request to select LoRA adapter
- [ ] Step 4: Verify reduced VRAM usage via `/metrics` or `nvidia-smi`

### Enable structured outputs and tool calling

Constrain model outputs to JSON schemas or function call signatures. For parser options and schema formats see `references/features-and-configuration.md`.

Copy this checklist:

- [ ] Step 1: Launch server with `--enable-auto-tool-choice --tool-call-parser <parser>`
- [ ] Step 2: Define JSON schema or tools list in the client request body
- [ ] Step 3: Send request to `/v1/chat/completions` with `response_format` or `tools` field
- [ ] Step 4: Parse structured JSON or `tool_calls` from the response

## When to Use

Invoke this skill when the user mentions vLLM, `vllm serve`, `LLM` class, `SamplingParams`, PagedAttention, tensor parallelism for inference, quantized serving, or OpenAI-compatible self-hosted endpoints.

**Use this skill when:**
- Serving one or more LLMs with high throughput and low latency using PagedAttention
- Running offline batch inference with `LLM.generate()` or async streaming with `AsyncLLMEngine`
- Deploying quantized models (AWQ, GGUF, FP8, GPTQ, BitsAndBytes, INT4/INT8)
- Hot-swapping LoRA adapters on a running server
- Enabling structured outputs, tool calling, or multimodal inputs on a self-hosted endpoint
- Scaling across multiple GPUs with tensor, pipeline, data, or expert parallelism
- Deploying via Docker, Kubernetes, or Ray Serve

**NOT for (use alternatives instead):**
- Fine-tuning or RLHF training — use `trl` or `transformers.Trainer`
- YAML-driven training pipelines — use Axolotl
- Lightweight LoRA fine-tuning — use Unsloth
- Simple single-request inference without throughput requirements — use `transformers` pipeline directly

## References

- `references/inference-modes.md` — `LLM` class, async streaming, OpenAI-compatible server endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`), and batch inference
- `references/models-and-quantization.md` — Supported generative and pooling models, LoRA adapters, quantization methods (AWQ, GGUF, FP8, BitsAndBytes, GPTQ, INT4/INT8), and speculative decoding
- `references/deployment-and-scaling.md` — Docker, Kubernetes, Nginx, Ray Serve, tensor/pipeline/data/expert parallelism, disaggregated prefill, and cloud framework integrations
- `references/features-and-configuration.md` — Automatic prefix caching, structured outputs, tool calling, multimodal inputs, reasoning outputs, engine arguments, environment variables, memory conservation, and observability/metrics

## Looking things up live (MCP fallback)

This skill ships pre-distilled `references/`. If a question is **not** answered by the bundled references, fall back to live docs via MCP. The frontmatter lists preferred MCPs under `metadata.openclaw.mcps`.

Use this routing:

1. First, search the bundled `references/` (including `community-gotchas.md` and `troubleshooting.md` if present).
2. If still missing, call Context7 via the OpenClaw `mcporter` CLI:
   ```bash
   mcporter call context7.resolve-library-id query="<question>" libraryName=vllm
   # then, with the returned /org/project ID:
   mcporter call context7.query-docs libraryId="/<org>/<project>" query="<topic>"
   ```
   (If you're unsure of a tool name, run `mcporter list-tools context7` to discover.)
3. For HF-hosted packages, prefer `hf-mcp/doc_search` if it's registered (`openclaw mcp list`).
4. Cite the MCP source in your answer. Do not invent function names — if Context7 doesn't return the answer, say so.

Skip MCP entirely if the question IS covered by `references/`. Loading MCP tool definitions is expensive; the references are zero-cost progressive disclosure.
