---
name: vllm-inference
description: "Serve and query large language models using vLLM's `LLM` class for offline batch inference or the OpenAI-compatible `vllm serve` CLI for online serving. Use when you need high-throughput token generation, multi-GPU tensor/pipeline parallelism, quantized inference (FP8, AWQ, GPTQ, BitsAndBytes), speculative decoding, automatic prefix caching, or structured outputs — and you want an OpenAI SDK-compatible endpoint without writing a custom server."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": [], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://docs.vllm.ai/en/latest/", "repo": "vllm-project/vllm", "fetched_at": "2026-05-25T05:32:18Z", "content_sha256": "f7f76f38df036e77f2a1c582ad0b97ccd631e181620889c8d2f8b584dfa744ae", "builder_version": "1.4.0"}, "coverage": ["html", "gh-readme", "gh-issues-bug-closed", "changelog"]}}
---

# vllm-inference

Serve and query large language models using vLLM's `LLM` class for offline batch inference or the OpenAI-compatible `vllm serve` CLI for online serving. Use when you need high-throughput token generation, multi-GPU tensor/pipeline parallelism, quantized inference (FP8, AWQ, GPTQ, BitsAndBytes), speculative decoding, automatic prefix caching, or structured outputs — and you want an OpenAI SDK-compatible endpoint without writing a custom server.

## Installation

```bash
pip install vllm
```

## Quick Start

```bash
# Online: start an OpenAI-compatible server
vllm serve meta-llama/Llama-3-8B-Instruct --tensor-parallel-size 1

# Offline: batch inference in Python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.8, max_tokens=128))
print(outputs[0].outputs[0].text)
```

## Decision Tree

| If the user wants... | Choose... | Then see |
|---|---|---|
| Online API vs batch script | Online (`vllm serve`) if latency/streaming matters or clients use OpenAI SDK; offline (`LLM.generate`) if processing a fixed dataset or running evals | `references/serving-modes.md` |
| Model too large for available VRAM | FP8 W8A8 for NVIDIA Hopper+ (best quality/speed); AWQ/GPTQ INT4 for older GPUs or extreme compression; BitsAndBytes for quick prototyping without pre-quantized weights | `references/quantization-and-memory.md` |
| Multi-GPU deployment strategy | Tensor parallelism (TP) within a node for latency; pipeline parallelism (PP) across nodes for very large models; expert parallelism (EP) for MoE models; data parallelism (DP) for throughput scaling with smaller models | `references/parallelism-and-scaling.md` |
| Faster decoding / lower latency per token | Speculative decoding with draft model (EAGLE/MLP/N-gram) if draft overhead is acceptable; prefix caching if prompts share long common prefixes; torch.compile + CUDA graphs for pure kernel throughput | `references/features-and-integrations.md` |
| Disaggregated vs co-located prefill/decode | Disaggregated prefill (experimental) when prefill latency dominates and you have spare nodes; co-located (default) for simpler ops and moderate traffic | `references/serving-modes.md` |
| Observability and production readiness | Prometheus + Grafana for metrics dashboards; OpenTelemetry for distributed tracing; benchmark CLI (`vllm benchmark serve/throughput/latency`) before go-live | `references/observability-and-tuning.md` |

## Common Workflows

### Launch OpenAI-compatible online server

Start a production-ready OpenAI-compatible HTTP server with optional quantization and multi-GPU parallelism. For detailed argument reference, see `references/serving-modes.md`.

Copy this checklist:

- [ ] Choose model and confirm VRAM budget; select quantization if needed (see `references/quantization-and-memory.md`)
- [ ] Set parallelism flags (`--tensor-parallel-size`, `--pipeline-parallel-size`) based on GPU count
- [ ] Run `vllm serve <model> [--quantization ...] [--tensor-parallel-size N]` and wait for `Application startup complete`
- [ ] Validate with a curl POST to `/v1/chat/completions` and check `/metrics` endpoint for Prometheus scrape
- [ ] **MCP fallback**: if the specific `--served-model-name` or server argument you need is not in `references/serving-modes.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="online serving server arguments"` — skip if references covered your case.

---

### Offline batch inference

Process a fixed prompt list with `LLM.generate` and collect `RequestOutput` objects. For engine arguments and quantization options, see `references/serving-modes.md` and `references/quantization-and-memory.md`.

Copy this checklist:

- [ ] Instantiate `LLM(model=..., quantization=..., tensor_parallel_size=N)` with desired engine arguments
- [ ] Build a list of prompt strings or token-ID lists
- [ ] Call `llm.generate(prompts, SamplingParams(...))` and collect `RequestOutput` objects
- [ ] Post-process outputs and optionally save results; profile throughput with `vllm benchmark throughput`
- [ ] **MCP fallback**: if the model architecture or quantization format you need is not in `references/quantization-and-memory.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="offline inference quantization <format>"` — skip if references covered your case.

---

### Multi-node distributed serving with Ray

Deploy vLLM across multiple nodes using a Ray cluster for models that exceed single-node GPU capacity. For topology options and KubeRay setup, see `references/parallelism-and-scaling.md`.

Copy this checklist:

- [ ] Start Ray cluster: `ray start --head` on head node, `ray start --address=<head>` on workers
- [ ] Confirm all nodes are visible via `ray status`
- [ ] Launch vLLM with `--tensor-parallel-size` and `--pipeline-parallel-size` matching total GPU topology
- [ ] Run a smoke-test request and verify load is distributed across nodes via `nvidia-smi` on each host
- [ ] **MCP fallback**: if your Ray cluster topology or multi-node config is not covered in `references/parallelism-and-scaling.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="multi-node Ray distributed serving"` — skip if references covered your case.

---

### Enable and validate automatic prefix caching (APC)

Reduce time-to-first-token for requests sharing a common system prompt by enabling the KV-cache prefix reuse. For eviction tuning and IndexCache variants, see `references/features-and-integrations.md`.

Copy this checklist:

- [ ] Add `--enable-prefix-caching` to the serve command (or `enable_prefix_caching=True` in `LLM` constructor)
- [ ] Send repeated requests sharing a common system prompt to warm the cache
- [ ] Check `vllm:cache_hit_rate` in Prometheus or logs for non-zero hit rate
- [ ] Tune `--max-num-seqs` and `--gpu-memory-utilization` if cache eviction is too aggressive
- [ ] **MCP fallback**: if the prefix caching eviction behavior or IndexCache variant you need is not in `references/features-and-integrations.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="automatic prefix caching configuration"` — skip if references covered your case.

## When to Use

Invoke this skill when the user mentions serving LLMs, running inference at scale, OpenAI-compatible endpoints, quantized model deployment, multi-GPU parallelism, speculative decoding, or vLLM specifically.

**Use this skill when:**
- Deploying an OpenAI-compatible inference server (`vllm serve`) for production or development
- Running offline batch inference over a dataset using `LLM.generate` and `SamplingParams`
- Fitting a large model into available VRAM via FP8, AWQ, GPTQ, or BitsAndBytes quantization
- Scaling across multiple GPUs or nodes with tensor, pipeline, expert, or data parallelism
- Enabling speculative decoding (EAGLE, MLP Speculator, N-gram), prefix caching, or structured outputs
- Integrating with LangChain, LlamaIndex, or Prometheus/OpenTelemetry observability stacks

**NOT for (use alternatives instead):**
- Supervised fine-tuning or LoRA training — use `transformers.Trainer` or Axolotl
- Fast LoRA fine-tuning with minimal setup — use Unsloth
- YAML-driven training configuration — use Axolotl
- Embedding-only workloads without generation — use sentence-transformers

## Scripts

Execute these — don't read them as reference. Each runs without consuming context tokens.

- `scripts/check_vram.sh` — Query nvidia-smi for free VRAM on all GPUs and warn if any device has less than 16 GB free. Run when: before launching `vllm serve` or `LLM()` to confirm memory headroom.
  ```bash
  bash scripts/check_vram.sh
  ```

- `scripts/health_check.sh` — Poll the vLLM `/health` endpoint in a loop until the server is ready or timeout is reached, then print latency. Run when: after starting `vllm serve`, before sending the first real request.
  ```bash
  bash scripts/health_check.sh
  ```

- `scripts/validate_openai_endpoint.py` — Send a minimal chat-completion request to a running vLLM server and assert a non-empty response, printing token throughput. Run when: as a smoke test after server startup or after config changes.
  ```bash
  python scripts/validate_openai_endpoint.py
  ```

## Old Patterns

<details>
<summary>Deprecated APIs (kept for historical context)</summary>

- **`AsyncLLMEngine` direct instantiation** (deprecated in 0.4.x) — use `vllm serve` CLI or `AsyncLLM` via the v1 engine path instead; direct `AsyncLLMEngine` construction is superseded by the new engine entrypoints.
- **`LLMEngine` (synchronous) for production serving** (deprecated in 0.5.x) — use the OpenAI-compatible server (`vllm serve`) or `AsyncLLM` for async workloads; synchronous `LLMEngine` is for offline/testing only.
- **`--worker-use-ray` flag** (deprecated in 0.3.x) — Ray workers are now selected automatically based on distributed config; remove the flag and rely on `--tensor-parallel-size` with Ray cluster active.

</details>

## References

- `references/serving-modes.md` — Offline `LLM` class, online OpenAI-compatible server, disaggregated prefill, data-parallel and context-parallel deployment
- `references/quantization-and-memory.md` — FP8, AWQ, GPTQ, BitsAndBytes, INT4/INT8, quantized KV cache, VRAM conservation strategies, engine memory arguments
- `references/parallelism-and-scaling.md` — Tensor, pipeline, expert, data, and context parallelism; multi-node serving; Ray and KubeRay cluster setup
- `references/features-and-integrations.md` — Automatic prefix caching, LoRA adapters, speculative decoding, structured outputs, tool calling, multimodal inputs, reasoning outputs, LangChain/LlamaIndex integrations
- `references/observability-and-tuning.md` — Prometheus metrics, OpenTelemetry, Grafana dashboards, benchmarking CLI, optimization and tuning knobs, environment variables

## Looking things up live (MCP fallback)

Per-workflow MCP triggers (above) handle the common cases. For anything else not covered by `references/`:

1. Verify the question is genuinely not in `references/` (grep first).
2. Resolve the libraryId — call `context7__resolve-library-id` with `libraryName="vllm"`. It returns one or more candidate library IDs; pick the one whose docs match the question.
3. Fetch the docs — call `context7__query-docs` with `libraryId="<from step 2>"` and `query="<your topic>"`. Read the returned snippet.
4. Cite the MCP source + libraryId in your answer. If `query-docs` returns nothing useful, say so — do not invent function names or flag values.

Note on naming: `context7__query-docs` and `context7__resolve-library-id` are the OpenClaw native tool names (double underscore prefix). Both are pre-registered for this skill — call them directly, no bash needed.
