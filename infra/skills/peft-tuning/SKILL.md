---
name: peft-tuning
description: "Fine-tune large pretrained models parameter-efficiently with Hugging Face PEFT, adapting LLMs, vision, and diffusion models by training a small set of extra parameters instead of all weights. Use this whenever a task involves LoRA, QLoRA, AdaLoRA, DoRA, soft prompts (prompt/prefix/P-tuning), or IA3 — building a `LoraConfig` and calling `get_peft_model`, running memory-constrained 4-bit training via `prepare_model_for_kbit_training`, loading or merging trained adapters with `AutoPeftModel` and `merge_and_unload`, or serving several adapters on one base model with `set_adapter`, `add_weighted_adapter`, or X-LoRA routing. Invoke this whenever the user fine-tunes, adapts, or customizes a pretrained model on limited GPU memory, even if they never say \"PEFT\" or \"LoRA\" — it is the default approach for efficient fine-tuning on consumer or single-GPU hardware."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": ["hf-mcp/doc_search", "hf-mcp/doc_fetch"], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://huggingface.co/docs/peft/index", "repo": "huggingface/peft", "fetched_at": "2026-06-24T09:56:39Z", "content_sha256": "7abab03572bbbf6a03419874470ec8043cc6c975b8572df61001eaa072b37726", "builder_version": "2.0.0"}, "coverage": ["html", "gh-readme"]}}
---

# peft-tuning

Fine-tune large pretrained models parameter-efficiently with Hugging Face PEFT, adapting LLMs, vision, and diffusion models by training a small set of extra parameters instead of all weights. Use this whenever a task involves LoRA, QLoRA, AdaLoRA, DoRA, soft prompts (prompt/prefix/P-tuning), or IA3 — building a `LoraConfig` and calling `get_peft_model`, running memory-constrained 4-bit training via `prepare_model_for_kbit_training`, loading or merging trained adapters with `AutoPeftModel` and `merge_and_unload`, or serving several adapters on one base model with `set_adapter`, `add_weighted_adapter`, or X-LoRA routing. Invoke this whenever the user fine-tunes, adapts, or customizes a pretrained model on limited GPU memory, even if they never say "PEFT" or "LoRA" — it is the default approach for efficient fine-tuning on consumer or single-GPU hardware.

## Installation

```bash
pip install peft
```

## Quick Start

Fastest path: attach a LoRA adapter to a causal LM, confirm the trainable-parameter count is tiny, then save just the adapter.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()   # confirm only a small fraction is trainable
# ... train with Trainer / SFTTrainer ...
model.save_pretrained("./my-lora-adapter")
```

## Decision Tree

| If the user wants... | Choose... | Then see |
|---|---|---|
| to fine-tune but isn't sure which method | LoRA as the default for most LLM/vision tasks; soft prompts (prompt/prefix tuning) for very low-parameter task adaptation; IA3 when you want fewer params than LoRA | `references/lora-methods.md` vs `references/prompt-and-other-methods.md` |
| to train on a memory-constrained GPU (e.g. <24GB single GPU) | load the base model in 4-bit and use QLoRA via `prepare_model_for_kbit_training` | `references/quantization-and-distributed.md` |
| lowest inference latency in production | `merge_and_unload` the adapter into base weights for zero overhead; keep the adapter separate if you need to swap tasks | `references/model-lifecycle.md` |
| to serve multiple tasks/adapters on one base model | `set_adapter` for one-at-a-time switching, `add_weighted_adapter` to blend, X-LoRA for learned routing, hotswapping for fast runtime swaps | `references/model-lifecycle.md` |
| multi-GPU / large-model distributed training | DeepSpeed ZeRO-3 for very large models; FSDP for medium models | `references/quantization-and-distributed.md` |

## Common Workflows

### Fine-tune a model with LoRA

Standard adapter training from a base model. For per-method `target_modules`/rank/alpha settings, see `references/lora-methods.md`.

Copy this checklist:

- [ ] Load base model and tokenizer
- [ ] Build `LoraConfig` (`task_type`, `r`, `lora_alpha`, `target_modules`)
- [ ] Wrap with `get_peft_model` and inspect trainable params
- [ ] Train with `Trainer`/`SFTTrainer`
- [ ] Save adapter with `save_pretrained`
- [ ] **MCP fallback**: if the PEFT method or `LoraConfig` parameter you need isn't in `references/lora-methods.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="<method> LoraConfig target_modules"` — skip if references covered your case.

### Memory-efficient QLoRA fine-tuning

Fit large models on small GPUs by quantizing the frozen base to 4-bit and training a LoRA adapter on top. Details in `references/quantization-and-distributed.md`.

Copy this checklist:

- [ ] Load base model in 4-bit (`BitsAndBytesConfig`)
- [ ] Call `prepare_model_for_kbit_training`
- [ ] Attach `LoraConfig` and `get_peft_model`
- [ ] Train and save the adapter
- [ ] **MCP fallback**: if a quantization or k-bit training detail isn't in `references/quantization-and-distributed.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="QLoRA prepare_model_for_kbit_training"` — skip if references covered your case.

### Load a trained adapter and run inference

Bring an adapter back for generation, optionally merging it into the base for deployment. Details in `references/model-lifecycle.md`.

Copy this checklist:

- [ ] Load adapter with `AutoPeftModelForCausalLM.from_pretrained`
- [ ] Optionally `merge_and_unload` for zero-overhead inference
- [ ] Run `generate`
- [ ] Save merged model if deploying standalone
- [ ] **MCP fallback**: if your adapter loading or merging case isn't in `references/model-lifecycle.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="AutoPeftModel merge_and_unload"` — skip if references covered your case.

### Manage and combine multiple adapters

Host several adapters on one base model and switch, blend, or route between them. Details in `references/model-lifecycle.md`.

Copy this checklist:

- [ ] Load base model and add multiple named adapters
- [ ] Switch active adapter with `set_adapter`
- [ ] Combine with `add_weighted_adapter` or route via X-LoRA
- [ ] Hotswap adapters at inference time
- [ ] **MCP fallback**: if your multi-adapter routing or combining case isn't in `references/model-lifecycle.md`, call `context7__resolve-library-id` with `libraryName="peft"`, then `context7__query-docs` with the returned libraryId and `query="add_weighted_adapter set_adapter hotswap"` — skip if references covered your case.

## When to Use

Use this skill whenever the user works with LoRA, QLoRA, adapters, `LoraConfig`, `get_peft_model`, soft prompts, IA3, or efficient fine-tuning on limited GPU memory — even as one step of a larger task (for example, adapting a model before evaluating or serving it).

**Use this skill when:**
- Fine-tuning an LLM, vision, or diffusion model without updating all of its weights.
- Memory is tight and the base model must be loaded in 4-bit (QLoRA) to fit.
- A trained adapter needs to be loaded, merged, or swapped at inference time.
- Multiple adapters must coexist on one base model and be switched, blended, or routed.

**Reach for a different tool when the task needs a capability this skill does not provide:**
- For full-parameter fine-tuning where every weight is updated — use the `transformers` `Trainer` directly.
- For the RLHF/alignment training loop itself (reward modeling, PPO, DPO) — use `trl`'s `SFTTrainer`/`PPOTrainer`; PEFT supplies the adapter config they wrap.
- For high-throughput production serving — use `vllm` after `merge_and_unload` folds the adapter into the base weights.

## Hardware Requirements

- **GPU**: NVIDIA, including consumer GPUs — PEFT is designed to keep large-model training accessible on a single card.
- **VRAM**: QLoRA fine-tuning of Llama-2-7B fits on a 16GB GPU (with TRL); 20B-parameter RLHF runs on a 24GB consumer GPU; larger models and Stable Diffusion LoRA were profiled on an A100 80GB with >64GB CPU RAM.
- **Memory savings**: a 3B-parameter PEFT model performs comparably to full fine-tuning at a fraction of the GPU memory; the saved adapter checkpoint is ~19MB versus ~11GB for the full `bigscience/T0_3B` model.
- **4-bit / k-bit**: load the frozen base in 4-bit (QLoRA) to fit large models on small GPUs — see `references/quantization-and-distributed.md`.
- **Multi-GPU**: via DeepSpeed ZeRO-3 (very large models) or FSDP (medium models).

## References

- `references/lora-methods.md` — the LoRA family and `LoraConfig`: LoRA, QLoRA, AdaLoRA, LoHa, LoKr, VeRA, X-LoRA, DoRA, RandLora, GraLoRA, and their `target_modules`/rank/alpha settings.
- `references/prompt-and-other-methods.md` — soft-prompt methods (prompt tuning, prefix tuning, P-tuning, multitask prompt tuning) plus IA3 and OFT/BOFT reparameterization tuners.
- `references/quantization-and-distributed.md` — k-bit/QLoRA quantization with bitsandbytes, `prepare_model_for_kbit_training`, and DeepSpeed/FSDP distributed training.
- `references/model-lifecycle.md` — `AutoPeftModel` loading/saving, adapter injection, model merging, `merge_and_unload`, mixed adapter types, hotswapping, and checkpoint format.

## Looking things up live (MCP fallback)

Per-workflow MCP triggers (above) handle the common cases. For anything else not covered by `references/`:

1. Verify the question is genuinely not in `references/` (grep first).
2. Resolve the libraryId — call `context7__resolve-library-id` with `libraryName="peft"`. It returns one or more candidate library IDs; pick the one whose docs match the question.
3. Fetch the docs — call `context7__query-docs` with `libraryId="<from step 2>"` and `query="<your topic>"`. Read the returned snippet.
4. Cite the MCP source + libraryId in your answer. If `query-docs` returns nothing useful, say so — do not invent function names or flag values.

Note on naming: `context7__query-docs` and `context7__resolve-library-id` are the OpenClaw native tool names (double underscore prefix). Both are pre-registered for this skill — call them directly, no bash needed.
