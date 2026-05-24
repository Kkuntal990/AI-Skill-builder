---
name: peft-tuning
description: "Train, fine-tune, and run inference with parameter-efficient adapters using the Hugging Face `peft` library. Use when you need to adapt a large pretrained model (LLM, diffusion model, or multimodal model) to a downstream task without updating all weights — invoke when the user mentions LoRA, QLoRA, AdaLoRA, IA3, prefix tuning, prompt tuning, adapter injection, or any PEFT-family method. Supports `PeftModel`, `LoraConfig`, `get_peft_model()`, `PeftModel.from_pretrained()`, and adapter merging via `merge_and_unload()`."
metadata: {"openclaw": {"emoji": "🤖", "requires": {"bins": ["python3"]}, "mcps": {"preferred": ["hf-mcp/doc_search", "hf-mcp/doc_fetch"], "fallback": ["context7/get-library-docs"]}, "source": {"url": "https://huggingface.co/docs/peft/index", "repo": "huggingface/peft", "fetched_at": "2026-05-08T02:43:22Z", "content_sha256": "4b6af7036e60a834fb4e500f0011b7cdc123c2f9b96743d303cf47a020defaca", "builder_version": "1.3.0"}, "coverage": ["html", "gh-readme", "gh-issues-open", "stackexchange", "gh-issues-question-closed"]}}
---

# peft-tuning

Train, fine-tune, and run inference with parameter-efficient adapters using the Hugging Face `peft` library. Use when you need to adapt a large pretrained model (LLM, diffusion model, or multimodal model) to a downstream task without updating all weights — invoke when the user mentions LoRA, QLoRA, AdaLoRA, IA3, prefix tuning, prompt tuning, adapter injection, or any PEFT-family method. Supports `PeftModel`, `LoraConfig`, `get_peft_model()`, `PeftModel.from_pretrained()`, and adapter merging via `merge_and_unload()`.

## Installation

```bash
pip install peft
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~4M || all params: ~7B || trainable%: ~0.06%
```

## Common Workflows

### Fine-tune a model with LoRA

Apply a LoRA adapter to a pretrained model and train only the low-rank weights. For adapter variants (AdaLoRA, LoHa, VeRA, etc.), see `references/lora-variants.md`.

Copy this checklist:

- [ ] Step 1: Load base model and tokenizer from Hugging Face Hub
- [ ] Step 2: Define `LoraConfig` (`r`, `lora_alpha`, `target_modules`, `task_type`)
- [ ] Step 3: Wrap model with `get_peft_model()` and verify trainable parameter count
- [ ] Step 4: Train with Transformers `Trainer` or custom loop, then save adapter with `model.save_pretrained()`

### Load and run inference with a saved PEFT adapter

Load a previously saved adapter onto a base model for inference without retraining. See `references/developer-guides.md` for checkpoint format details.

Copy this checklist:

- [ ] Step 1: Load base model from Hub
- [ ] Step 2: Load adapter weights with `PeftModel.from_pretrained(base_model, adapter_path)`
- [ ] Step 3: Optionally merge adapter into base weights with `model.merge_and_unload()` for faster inference
- [ ] Step 4: Run inference with standard `model.generate()` or `pipeline()`

### Apply PEFT with quantization for consumer hardware

Combine 4-bit or 8-bit quantization (bitsandbytes) with LoRA to train large models on limited VRAM. See `references/developer-guides.md` (quantization) and `references/integrations.md`.

Copy this checklist:

- [ ] Step 1: Install bitsandbytes and configure `BitsAndBytesConfig` (4-bit or 8-bit)
- [ ] Step 2: Load quantized base model with `from_pretrained(..., quantization_config=...)`
- [ ] Step 3: Apply `LoraConfig` targeting appropriate linear layers (use `prepare_model_for_kbit_training` first)
- [ ] Step 4: Train and save adapter separately from the frozen quantized base

### Merge multiple adapters or switch adapters at runtime

Manage multiple adapters on one base model — hotswap at inference time or merge into a single set of weights. See `references/developer-guides.md` (hotswapping, mixed adapter types).

Copy this checklist:

- [ ] Step 1: Load base model and add first adapter with `load_adapter()`
- [ ] Step 2: Add additional adapters with `load_adapter()` under distinct names
- [ ] Step 3: Use `set_adapter()` to hotswap between adapters or `add_weighted_adapter()` to merge
- [ ] Step 4: Export final merged model with `merge_and_unload()` and push to Hub

## When to Use

Invoke this skill when the user mentions LoRA, QLoRA, adapter fine-tuning, PEFT, prompt tuning, prefix tuning, IA3, parameter-efficient training, or adapter merging.

**Use this skill when:**
- Fine-tuning LLMs or diffusion models with limited GPU memory using LoRA or QLoRA
- Applying non-LoRA PEFT methods: IA3, OFT, BOFT, FourierFT, prefix tuning, prompt tuning
- Managing multiple task-specific adapters on a single base model
- Merging, hotswapping, or exporting PEFT adapters
- Integrating PEFT with Accelerate, DeepSpeed, or FSDP for distributed training

**NOT for (use alternatives instead):**
- Full fine-tuning of all model weights — use `transformers.Trainer` directly
- YAML-driven training configuration — use Axolotl
- Fastest possible LoRA training with fused kernels — use Unsloth
- RLHF / reward model training — use TRL (which integrates PEFT internally)

## Hardware Requirements

- **GPU**: A100 80GB (reference benchmarks); consumer training demonstrated on 16–24GB GPUs
- **VRAM**: Llama-2-7B with QLoRA: ~16GB; 20B model RLHF fine-tuning: ~24GB consumer GPU; T0-3B full fine-tune ~11GB vs. PEFT adapter ~19MB on disk
- **Multi-GPU**: Supported via Accelerate, DeepSpeed, and FSDP — see `references/integrations.md`
- **Mixed precision**: BF16 recommended on A100/H100; FP16 supported for older GPUs

## Templates

- `templates/lora_finetune.py` — Load a causal LM, apply `LoraConfig`, train on a text dataset with Transformers `Trainer`, and save the adapter
- `templates/qlora_4bit_finetune.py` — 4-bit quantized base model with bitsandbytes, LoRA adapter via PEFT, training loop, and adapter-only checkpoint save
- `templates/load_and_inference.py` — Load a saved PEFT adapter onto a base model, merge weights, and run generation inference

## References

- `references/lora-variants.md` — LoRA, AdaLoRA, AdaMSS, LoHa, LoKr, LyCORIS, X-LoRA, VeRA, GraLoRA, VB-LoRA, RandLoRA, SHiRA, TinyLoRA, DeLoRA, LilyPEA, WaveFT and other LoRA-family adapters
- `references/prompt-and-soft-methods.md` — Prompt tuning, prefix tuning, P-tuning, multitask prompt tuning, Llama-Adapter, soft prompts
- `references/orthogonal-and-sparse-methods.md` — IA3, OFT, BOFT, PSOFT, Polytropon, FourierFT, HRA, CPT, C3A, MiSS, RoAd, Nu, Trainable Tokens, Cartridges, Layernorm tuning
- `references/developer-guides.md` — Model merging, quantization, custom models, adapter injection, mixed adapter types, `torch.compile`, hotswapping, PEFT checkpoint format
- `references/integrations.md` — Transformers, Diffusers, Accelerate, DeepSpeed, FSDP integrations and PEFT integration functions

## Looking things up live (MCP fallback)

This skill ships pre-distilled `references/`. If a question is **not** answered by the bundled references, fall back to live docs via MCP. The frontmatter lists preferred MCPs under `metadata.openclaw.mcps`.

Use this routing:

1. First, search the bundled `references/` (including `community-gotchas.md` and `troubleshooting.md` if present).
2. If still missing, call Context7 via the OpenClaw `mcporter` CLI:
   ```bash
   mcporter call context7.resolve-library-id query="<question>" libraryName=peft
   # then, with the returned /org/project ID:
   mcporter call context7.query-docs libraryId="/<org>/<project>" query="<topic>"
   ```
   (If you're unsure of a tool name, run `mcporter list-tools context7` to discover.)
3. For HF-hosted packages, prefer `hf-mcp/doc_search` if it's registered (`openclaw mcp list`).
4. Cite the MCP source in your answer. Do not invent function names — if Context7 doesn't return the answer, say so.

Skip MCP entirely if the question IS covered by `references/`. Loading MCP tool definitions is expensive; the references are zero-cost progressive disclosure.
