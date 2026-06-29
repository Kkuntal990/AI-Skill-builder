---
name: peft-tuning
description: "Adapt large pretrained models with parameter-efficient fine-tuning — train a small set of extra weights instead of all parameters, yielding performance comparable to full fine-tuning at a fraction of the compute, memory, and storage cost. Use this whenever the user fine-tunes, adapts, or customizes an LLM or diffusion model with LoRA, DoRA, AdaLoRA, QLoRA, IA3, or soft-prompt methods (prefix tuning, P-tuning, prompt tuning), wraps a base model with `get_peft_model` / `LoraConfig`, loads or merges an adapter with `PeftModel`, or needs to fit a model on a single consumer GPU. Invoke it even when the user only says \"fine-tune this model\" or \"train an adapter\" without naming PEFT — it is the default for any sub-task that trains, saves, or runs inference with adapter weights."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": ["hf-mcp/doc_search", "hf-mcp/doc_fetch"], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://huggingface.co/docs/peft/index", "repo": "huggingface/peft", "fetched_at": "2026-06-28T14:51:24Z", "content_sha256": "5373ec162f1fbfc72aff74bc96d27a344135826dc0fc282799b15f4d1f67400e", "builder_version": "2.1.0"}, "coverage": ["html", "gh-readme", "changelog"]}}
---

# PEFT Tuning

Adapt large pretrained models with parameter-efficient fine-tuning — train a small set of extra weights instead of all parameters, yielding performance comparable to full fine-tuning at a fraction of the compute, memory, and storage cost. Use this whenever the user fine-tunes, adapts, or customizes an LLM or diffusion model with LoRA, DoRA, AdaLoRA, QLoRA, IA3, or soft-prompt methods (prefix tuning, P-tuning, prompt tuning), wraps a base model with `get_peft_model` / `LoraConfig`, loads or merges an adapter with `PeftModel`, or needs to fit a model on a single consumer GPU. Invoke it even when the user only says "fine-tune this model" or "train an adapter" without naming PEFT — it is the default for any sub-task that trains, saves, or runs inference with adapter weights.

## Quick Start

The fastest path is plain LoRA: load a base model with `AutoModelForCausalLM.from_pretrained`, build a `LoraConfig` (`task_type`, `r`, `lora_alpha`, `target_modules`), wrap it with `get_peft_model`, train with the Transformers `Trainer` (or TRL `SFTTrainer`), and save with `model.save_pretrained` — the adapter is only ~MBs. See **Fine-tune a model with LoRA** below for the full checklist.

## Decision Tree

| If the user wants... | Choose... | Then see |
|---|---|---|
| parameter-efficient fine-tuning but hasn't picked a method | LoRA as the default; DoRA/AdaLoRA for higher quality at slightly more cost; IA3 or soft prompts (prefix/P-tuning) when you need the fewest trainable params | `references/lora-methods.md` vs `references/prompt-and-other-methods.md` |
| to train when the base model does not fit in VRAM (single GPU <24GB) | QLoRA — load the base in 4-bit and train a LoRA adapter on top | **Fit a large model on a small GPU (QLoRA)** below |
| to train a model too large for one GPU even quantized, with multi-GPU available | DeepSpeed ZeRO-3 for the largest models; FSDP for medium models | `references/integrations-and-scaling.md` |
| to deploy the tuned model | `merge_and_unload` into one model for zero-overhead inference (LoRA-family only, not on a quantized base); keep the adapter separate and swap with `set_adapter` when serving multiple tasks from one base | `references/integrations-and-scaling.md` |

## Common Workflows

### Fine-tune a model with LoRA

The default for parameter-efficient fine-tuning — train low-rank adapters and save only the adapter weights. For per-method parameters (DoRA, AdaLoRA, VeRA, target_modules selection), see `references/lora-methods.md`.

Copy this checklist:

- [ ] Step 1: Load the base model and tokenizer (e.g. `AutoModelForCausalLM.from_pretrained`)
- [ ] Step 2: Build a `LoraConfig` (`task_type`, `r`, `lora_alpha`, `target_modules`) and wrap with `get_peft_model`
- [ ] Step 3: Train with the Transformers `Trainer` (or TRL `SFTTrainer`)
- [ ] Step 4: Save the adapter with `model.save_pretrained` (adapter weights only, ~MBs)
- [ ] **MCP fallback**: if the PEFT method or `LoraConfig` parameter you need isn't in `references/lora-methods.md` or `references/prompt-and-other-methods.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="<method or LoraConfig param>"` — skip if references covered your case.

### Fit a large model on a small GPU (QLoRA)

Use this only when the full-precision / bf16 base model does not fit in GPU memory (e.g. a single <24GB GPU). With memory headroom, use the plain LoRA workflow above — 4-bit adds dequantization overhead and a more fragile k-bit grad path, for no benefit when memory isn't the constraint. For backend specifics (nf4 vs GPTQ vs AWQ bases), see `references/quantization.md`.

Copy this checklist:

- [ ] Step 1: Confirm the precondition — the bf16 base + LoRA + optimizer state won't fit (otherwise use plain LoRA)
- [ ] Step 2: Load the base in 4-bit via `BitsAndBytesConfig` (`load_in_4bit`, nf4, double-quant)
- [ ] Step 3: Call `prepare_model_for_kbit_training` on the loaded model
- [ ] Step 4: Attach a `LoraConfig` with `get_peft_model`
- [ ] Step 5: Train, then save the adapter (keep the quantized base separate)
- [ ] **MCP fallback**: if your quantization backend (bitsandbytes nf4 variant, GPTQ, or AWQ) isn't in `references/quantization.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="QLoRA <backend> quantization"` — skip if references covered your case.

### Load and run inference with a trained adapter

Attach a saved adapter to its base model and generate. For `AutoPeftModel`, merging, and serving multiple adapters from one base, see `references/integrations-and-scaling.md`.

Copy this checklist:

- [ ] Step 1: Load the base model (or use `AutoPeftModelForCausalLM.from_pretrained` for one-call loading)
- [ ] Step 2: Attach the adapter with `PeftModel.from_pretrained(base, adapter_path)`
- [ ] Step 3: Optionally call `merge_and_unload()` to fold weights in for zero inference overhead (LoRA-family only, not on a quantized base)
- [ ] Step 4: Run `generate()`
- [ ] **MCP fallback**: if the adapter loading or merging API you need isn't in `references/integrations-and-scaling.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="PeftModel <load or merge topic>"` — skip if references covered your case.

## When to Use

Use this skill whenever the user works with LoRA, QLoRA, DoRA, AdaLoRA, IA3, or soft-prompt tuning, or wraps a model with `get_peft_model` / loads an adapter with `PeftModel` — even as ONE STEP of a larger task (e.g. fine-tuning a model that will then be evaluated or served).

**Use this skill when:**
- Fine-tuning an LLM or diffusion model on a downstream task without touching all weights.
- Training is memory-constrained and needs 4-bit/8-bit quantized base loading (QLoRA).
- Loading, merging, or swapping adapter checkpoints for inference or multi-task serving.

**Reach for a different tool when the task needs a capability this skill does not provide:**
- For the RL/preference-optimization loop itself (PPO, DPO, reward models) — use `trl` instead; PEFT supplies the adapter that trl trains.
- For high-throughput production serving of the merged model — use a dedicated inference engine (e.g. vLLM) instead.

## Hardware Requirements

- **GPU**: Runs on consumer GPUs; an A100 80GB (with >64GB CPU RAM) is referenced for larger models and Stable Diffusion + LoRA.
- **VRAM**: Llama-2-7B with QLoRA + TRL fits on a 16GB GPU; 20B-parameter RLHF with PEFT + TRL fits on a 24GB consumer GPU; a 3B model trained with PEFT is comparable to a fully fine-tuned model at a fraction of the GPU memory.
- **Checkpoint size**: Adapters are tiny — e.g. a 19MB final checkpoint vs 11GB for the full `bigscience/T0_3B` model.
- **Multi-GPU**: Supported via DeepSpeed ZeRO and FSDP (see `references/integrations-and-scaling.md`).

## Old Patterns

<details>
<summary>Deprecated APIs (kept for historical context)</summary>

- **`prepare_model_for_int8_training`** (deprecated in 0.4.0) — use `prepare_model_for_kbit_training` instead.
- **`load_in_8bit` / `load_in_4bit` kwargs passed directly to `from_pretrained`** (deprecated in transformers 4.30) — use `BitsAndBytesConfig` passed via `quantization_config` instead.

</details>

## References

- `references/lora-methods.md` — LoRA and its variants (DoRA, AdaLoRA, LoHa, LoKr, VeRA, X-LoRA) with `LoraConfig` parameters (`r`, `lora_alpha`, `target_modules`, `init`).
- `references/prompt-and-other-methods.md` — soft-prompt methods (prompt tuning, prefix tuning, P-tuning, multitask prompt tuning), IA3, and OFT/BOFT orthogonal fine-tuning.
- `references/quantization.md` — QLoRA: 4-bit/8-bit bitsandbytes base loading, `prepare_model_for_kbit_training`, and training adapters on GPTQ/AWQ quantized bases.
- `references/integrations-and-scaling.md` — DeepSpeed ZeRO and FSDP multi-GPU training, adapter save/load (`PeftModel`, `AutoPeftModel`), model merging, and Transformers/Diffusers integration.

## Looking things up live (MCP fallback)

Per-workflow MCP triggers (above) handle the common cases. For anything else not covered by `references/`:

1. Verify the question is genuinely not in `references/` (grep first).
2. Resolve the libraryId — call `context7__resolve-library-id` with `libraryName="peft"`. It returns one or more candidate library IDs; pick the one whose docs match the question.
3. Fetch the docs — call `context7__query-docs` with `libraryId="<from step 2>"` and `query="<your topic>"`. Read the returned snippet.
4. Cite the MCP source + libraryId in your answer. If `query-docs` returns nothing useful, say so — do not invent function names or flag values.

Note on naming: `context7__query-docs` and `context7__resolve-library-id` are the OpenClaw native tool names (double underscore prefix). Both are pre-registered for this skill — call them directly, no bash needed.
