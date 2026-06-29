# LoRA and Reparameterization Methods

How to configure LoRA and its low-rank reparameterization variants in PEFT — LoRA, DoRA, AdaLoRA, LoHa, LoKr, VeRA, and X-LoRA — including the `LoraConfig` parameters (`r`, `lora_alpha`, `target_modules`, weight init) that govern every run.

## Contents

- The shared workflow
- LoraConfig core parameters
- DoRA — weight-decomposed LoRA
- AdaLoRA — budget-allocated rank
- LoHa — Hadamard-product low rank
- LoKr — Kronecker-product low rank
- VeRA — vector-based shared random matrices
- X-LoRA — mixture of LoRA experts
- Choosing target_modules
- Picking a method

## The shared workflow

Every method in this file is a *reparameterization* adapter: it freezes the base weights and injects a small set of trainable parameters into selected linear layers. They all plug into the same three-step flow — build a config, wrap the base model with `get_peft_model`, then train and print the trainable-parameter count.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

This trains only a fraction of the model's parameters (PEFT cites ~0.19% for `bigscience/mt0-large`). The variants below swap `LoraConfig` for a variant-specific config class but keep `get_peft_model(model, peft_config)` and the same training loop. After training, save with `model.save_pretrained(out_dir)`; reload the adapter onto the base model with `PeftModel.from_pretrained(base_model, out_dir)`.

`task_type` must match the head: `TaskType.CAUSAL_LM`, `TaskType.SEQ_2_SEQ_LM`, `TaskType.SEQ_CLS`, `TaskType.TOKEN_CLS`, `TaskType.QUESTION_ANS`, or `TaskType.FEATURE_EXTRACTION`.

## LoraConfig core parameters

These four knobs determine capacity, scaling, placement, and initialization. They are the parameters you tune first.

- **`r`** (int) — the rank of the update matrices `A` (down-projection) and `B` (up-projection). Trainable parameter count grows roughly linearly with `r`. Common values: 8, 16, 32, 64. Higher `r` adds capacity at the cost of more parameters; start low and raise only if the adapter underfits.
- **`lora_alpha`** (int) — the scaling factor for the LoRA update. The injected delta is scaled by `lora_alpha / r`. A common heuristic is `lora_alpha = 2 * r` (e.g. `r=16`, `lora_alpha=32`, as in the snippet above), which keeps the effective scale at 2.
- **`target_modules`** — which submodules receive adapters. Pass a list of module name suffixes (e.g. `["q_proj", "v_proj"]`) or a regex string. If omitted, PEFT applies a model-type default (often the attention query/value projections). See *Choosing target_modules*.
- **`init_lora_weights`** — controls how `A`/`B` are initialized. Default `True` uses Kaiming-uniform on `A` and zeros on `B`, so the initial update is zero (training starts from the unmodified base model). Set to `"gaussian"` for normal init, or to PiSSA/OLoRA/LoftQ-style strings to initialize from a decomposition of the base weights for faster convergence.

Other frequently-used `LoraConfig` fields:

- **`lora_dropout`** (float) — dropout applied to the LoRA input; regularizes the adapter.
- **`bias`** — `"none"` (default), `"all"`, or `"lora_only"`; whether and which bias terms are made trainable.
- **`modules_to_save`** — extra modules (e.g. a freshly-initialized classifier head) to train fully and save alongside the adapter.
- **`use_rslora`** (bool) — use rank-stabilized scaling (`lora_alpha / sqrt(r)` instead of `lora_alpha / r`), which stabilizes training at higher ranks.
- **`use_dora`** (bool) — enable DoRA (see below).

## DoRA — weight-decomposed LoRA

DoRA decomposes each pretrained weight into a magnitude vector and a direction matrix, then applies LoRA only to the direction while training the magnitude separately. It typically improves over plain LoRA at low ranks. It is not a separate config class — it is a flag on `LoraConfig`:

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,
    task_type="CAUSAL_LM",
)
```

Notes: DoRA adds the magnitude parameters, so it is somewhat slower and uses more memory than plain LoRA at the same `r`. For inference speed you can merge the adapter back into the base weights with `model.merge_and_unload()`, which collapses the DoRA update into the frozen weights.

## AdaLoRA — budget-allocated rank

AdaLoRA starts every target module at a higher rank, then prunes ranks during training so the parameter budget is spent where it helps most (importance-scored singular values). Use `AdaLoraConfig`; it requires knowing the total number of training steps so the schedule can be planned.

```python
from peft import AdaLoraConfig, get_peft_model

peft_config = AdaLoraConfig(
    init_r=12,        # starting rank per module
    target_r=8,       # average rank after pruning
    tinit=200,        # steps before pruning begins
    tfinal=1000,      # step at which pruning stops
    total_step=3000,  # total training steps (required)
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
```

Set `total_step` to the actual number of optimizer steps in your run; an incorrect value desynchronizes the rank schedule. AdaLoRA is most useful when you want LoRA-level quality under a tight parameter budget and are willing to let the method decide the per-layer allocation.

## LoHa — Hadamard-product low rank

LoHa (from the LyCORIS family) factors the update as the element-wise (Hadamard) product of two low-rank decompositions, giving more expressive updates than a single low-rank product at the same rank. Use `LoHaConfig`:

```python
from peft import LoHaConfig, get_peft_model

peft_config = LoHaConfig(
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
```

LoHa was developed for diffusion/image models but works on transformers. `r` and `alpha` play the same capacity/scaling roles as in LoRA.

## LoKr — Kronecker-product low rank

LoKr (also LyCORIS) builds the update from a Kronecker product, which can represent large weight deltas with very few parameters. Use `LoKrConfig`:

```python
from peft import LoKrConfig, get_peft_model

peft_config = LoKrConfig(
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
```

LoKr is the most parameter-frugal of the reparameterization methods here; reach for it when storage/parameter count is the binding constraint.

## VeRA — vector-based shared random matrices

VeRA freezes a single pair of random low-rank matrices shared across all adapted layers and trains only small per-layer scaling vectors. This makes the trainable footprint dramatically smaller than LoRA's, since the large `A`/`B` matrices are not learned. Use `VeraConfig`:

```python
from peft import VeraConfig, get_peft_model

peft_config = VeraConfig(
    r=256,            # VeRA typically uses a much larger r than LoRA
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
```

Because only the scaling vectors are trained, VeRA tolerates (and benefits from) a much larger `r` than LoRA while keeping the checkpoint tiny — useful when you need to serve many adapters.

## X-LoRA — mixture of LoRA experts

X-LoRA layers a learned, token/layer-wise gating ("dense gate") over a set of pre-trained LoRA adapters, mixing them per input rather than committing to one. It is built on top of existing LoRA adapters via `XLoraConfig` (you supply the paths/identifiers of the constituent adapters and the base hidden size). Use it when you have several specialized LoRA adapters and want the model to blend them dynamically at inference time instead of swapping a single adapter in and out.

## Choosing target_modules

- **Start with attention projections.** Leaving `target_modules` unset uses the model-type default, usually the query/value projections. Naming them explicitly — `["q_proj", "v_proj"]` for Llama/Qwen-style models — is the conservative, well-tested baseline.
- **Add more modules for more capacity.** Including key/output projections (`k_proj`, `o_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`) increases adapter capacity and parameter count. `"all-linear"` targets every linear layer.
- **Match the architecture's actual names.** Module suffixes differ across model families (`c_attn` for GPT-2, `query`/`value` for BERT). Inspect `model.named_modules()` to confirm the suffixes before setting `target_modules`, or pass a regex string to match a pattern.

## Picking a method

- **Default / unsure** → plain `LoraConfig` (`r=16`, `lora_alpha=32`). It is the most-tested and best-supported path.
- **Want better low-rank quality** → `use_dora=True`, or `use_rslora=True` for stability at higher `r`.
- **Tight parameter budget, let the method allocate** → AdaLoRA.
- **Smallest possible checkpoint** → LoKr or VeRA.
- **More expressive update at fixed rank** → LoHa.
- **Several specialized adapters to blend at inference** → X-LoRA.

Faster convergence with any of these comes from initializing from a base-weight decomposition via `init_lora_weights` (PiSSA/OLoRA/LoftQ) rather than from zero.
