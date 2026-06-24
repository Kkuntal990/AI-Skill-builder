# LoRA Methods and LoraConfig

Configuration and parameter reference for PEFT's LoRA family — standard LoRA plus QLoRA, DoRA, rsLoRA, AdaLoRA, LoHa, LoKr, VeRA, X-LoRA, RandLora, and GraLoRA — focused on which config class to instantiate and how to set `r`, `alpha`, and `target_modules`.

## Contents

- The shared LoRA mechanism
- LoraConfig core parameters
- Choosing target_modules
- r and lora_alpha
- Standard LoRA (quickstart)
- QLoRA (4-bit base + LoRA)
- DoRA (use_dora)
- rsLoRA (use_rslora)
- AdaLoRA (AdaLoraConfig)
- LoHa (LoHaConfig)
- LoKr (LoKrConfig)
- VeRA (VeraConfig)
- X-LoRA (XLoraConfig)
- RandLora (RandLoraConfig)
- GraLoRA
- init_lora_weights initialization schemes
- Picking a method

## The shared LoRA mechanism

Every method here freezes the base weights and trains small added parameters, then wraps the base model with `get_peft_model(model, peft_config)`. They differ in *how* the trainable update is parameterized (two low-rank matrices, a Hadamard product, a Kronecker product, shared random bases, a router over experts, etc.). The wrapping call is identical across all of them — only the config class and its fields change. After wrapping, call `model.print_trainable_parameters()` to confirm how few parameters you are actually training.

## LoraConfig core parameters

`LoraConfig` is the base for standard LoRA, DoRA, rsLoRA, and (by inheritance) AdaLoRA. The fields you will set most often:

- `r` — rank of the update matrices. Higher rank = more capacity and more trainable params.
- `lora_alpha` — scaling numerator; the update is scaled by `lora_alpha / r` (or `lora_alpha / sqrt(r)` with rsLoRA).
- `lora_dropout` — dropout applied on the LoRA path.
- `target_modules` — which submodules to adapt (list of names, a regex string, or `"all-linear"`).
- `bias` — `"none"` (default), `"all"`, or `"lora_only"`.
- `task_type` — e.g. `TaskType.CAUSAL_LM`, `"SEQ_CLS"`, `"SEQ_2_SEQ_LM"`, `"TOKEN_CLS"`. Set this so PEFT attaches the right head behavior.
- `modules_to_save` — extra modules (e.g. a new classifier head) to train fully and save alongside the adapter.
- `use_rslora`, `use_dora` — toggle the variants below.
- `rank_pattern`, `alpha_pattern` — per-module overrides of `r`/`alpha` keyed by module name.
- `init_lora_weights` — initialization scheme (see the final section).

## Choosing target_modules

`target_modules` is the single highest-impact knob. Options:

- An explicit list of submodule name suffixes, e.g. `["q_proj", "v_proj"]` for attention-only adaptation, or `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` to also cover the MLP.
- The string `"all-linear"` to target every linear layer (excluding the output head). This is the common QLoRA default and usually the strongest single choice.
- A regex string matched against module names for fine control.

If you omit `target_modules`, PEFT falls back to architecture-specific defaults for many known model types, but being explicit is safer for unfamiliar architectures.

## r and lora_alpha

`r` sets capacity; `lora_alpha` sets how strongly the update is applied. A widely used heuristic is `lora_alpha = 2 * r` (e.g. `r=16, lora_alpha=32`). With `use_rslora=True` the scaling becomes `alpha / sqrt(r)`, which keeps the effective scale stable as you raise `r`, so larger ranks stay trainable without exploding the update.

Use `rank_pattern` / `alpha_pattern` to give specific modules a different rank:

```python
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    rank_pattern={"v_proj": 16},     # v_proj gets r=16, q_proj stays r=8
    alpha_pattern={"v_proj": 32},
)
```

## Standard LoRA (quickstart)

The canonical setup — wrap a causal LM and train ~0.1–1% of parameters:

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj"]  # optionally indicate target modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Next step: hand `model` to a standard `Trainer`/`SFTTrainer` and train as usual. Save with `model.save_pretrained(path)` — only the adapter is written.

## QLoRA (4-bit base + LoRA)

QLoRA is not a separate config; it is LoRA applied on top of a 4-bit-quantized base model. Quantize with `BitsAndBytesConfig`, prepare the model, then attach a normal `LoraConfig`:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
```

This is the recipe for fitting large models on a single consumer GPU. `prepare_model_for_kbit_training` casts layernorms, enables gradient checkpointing compatibility, and makes the quantized base trainable through the LoRA path.

## DoRA (use_dora)

DoRA (Weight-Decomposed Low-Rank Adaptation) splits each weight into magnitude and direction and applies LoRA to the direction. It often improves on LoRA at low rank. Enable it with one flag:

```python
config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
)
```

DoRA adds a small amount of overhead per step versus plain LoRA. It is compatible with quantization and the other `LoraConfig` fields.

## rsLoRA (use_rslora)

Rank-stabilized LoRA changes only the scaling to `alpha / sqrt(r)`, which lets you increase `r` for more capacity without the update shrinking too much:

```python
config = LoraConfig(r=64, lora_alpha=16, use_rslora=True, target_modules="all-linear")
```

Reach for this when you want higher rank than usual and find standard scaling underperforms at that rank.

## AdaLoRA (AdaLoraConfig)

AdaLoRA starts at a higher rank and prunes it down to a target budget during training, allocating more rank to important modules. It needs a training-step schedule and a call inside your loop.

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    init_r=12,          # starting rank
    target_r=8,         # final average rank budget
    tinit=200,          # steps before pruning starts
    tfinal=1000,        # steps after which rank is fixed
    deltaT=10,          # prune every deltaT steps
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    total_step=2000,    # total optimizer steps (required)
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

In the training loop, after `optimizer.step()` call `model.update_and_allocate(global_step)` so the rank schedule advances. Set `total_step` to your real step count.

## LoHa (LoHaConfig)

LoHa (from the LyCORIS family) parameterizes the update as a Hadamard product of two low-rank pairs, giving higher effective rank for the same budget. Note the field is `alpha`, not `lora_alpha`:

```python
from peft import LoHaConfig, get_peft_model

config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "v_proj"],
    rank_dropout=0.0,
    module_dropout=0.0,
)
model = get_peft_model(model, config)
```

`use_effective_conv2d=True` enables the conv-specific decomposition for vision/diffusion models.

## LoKr (LoKrConfig)

LoKr (also LyCORIS) uses a Kronecker product, which can express large updates very parameter-efficiently and is popular for diffusion models:

```python
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["q_proj", "v_proj"],
    decompose_both=True,    # decompose both Kronecker factors
    decompose_factor=-1,    # -1 lets PEFT choose the factorization
)
model = get_peft_model(model, config)
```

## VeRA (VeraConfig)

VeRA shares a single pair of *frozen random* matrices across all adapted layers and only trains tiny per-layer scaling vectors, so its adapter is far smaller than LoRA's. Because the random matrices are shared, the targeted modules must have compatible shapes.

```python
from peft import VeraConfig, get_peft_model

config = VeraConfig(
    r=256,                              # VeRA uses a much higher rank than LoRA
    target_modules=["q_proj", "v_proj"],
    d_initial=0.1,                      # init value for trainable scaling vector
    save_projection=True,               # store the shared random matrices in the checkpoint
)
model = get_peft_model(model, config)
```

Use a high `r` (256 is typical) — the trained parameter count stays small regardless because only the scaling vectors are learned.

## X-LoRA (XLoraConfig)

X-LoRA is a mixture-of-experts router *over already-trained LoRA adapters*: a learned gate produces per-token, per-layer scalings that blend the experts. You first train several LoRA adapters, then configure X-LoRA to combine them.

```python
from peft import XLoraConfig, get_peft_model

config = XLoraConfig(
    hidden_size=model.config.hidden_size,
    adapters={
        "adapter_1": "/path/to/lora_adapter_1",
        "adapter_2": "/path/to/lora_adapter_2",
    },
    xlora_depth=8,
    layerwise_scalings=True,
)
model = get_peft_model(model, config)
```

Only the router is trained at this stage; the underlying expert adapters stay frozen.

## RandLora (RandLora)

RandLora trains coefficients over a set of fixed random bases to recover full-rank-like updates while keeping the trainable count low — conceptually a full-rank cousin of VeRA. Instantiate `RandLoraConfig` and set `r` and `target_modules` as with the others; consult the RandLora entry in the PEFT Adapters API reference for the exact dropout/alpha and sparsity fields, which differ from `LoraConfig`:

```python
from peft import RandLoraConfig, get_peft_model

config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
```

## GraLoRA

GraLoRA ("granular" LoRA) partitions each weight into a grid of blocks and gives each block its own small low-rank adapter, increasing expressivity at fixed rank and improving locality. It is configured via its own config class (see the GraLoRA entry in the PEFT Adapters API reference for the block-granularity field and exact parameter names) and then wrapped with `get_peft_model` like every other method. Prefer it when standard LoRA underfits at a given rank but you do not want to raise the global rank.

## init_lora_weights initialization schemes

`init_lora_weights` controls how the adapter is initialized and is available on `LoraConfig`:

- `True` (default) — LoRA-paper init (one matrix random, the other zero, so the model starts unchanged).
- `False` — random init for both (the model output changes immediately; mainly for testing).
- `"gaussian"` — Gaussian init scaled by rank.
- `"pissa"` — initialize from the principal singular components of the base weight; can converge faster.
- `"olora"`, `"eva"`, `"loftq"` — data- or decomposition-aware schemes; `"loftq"` pairs with quantization and needs `loftq_config`.

Example:

```python
config = LoraConfig(r=16, lora_alpha=32, init_lora_weights="pissa", target_modules="all-linear")
```

## Picking a method

- General fine-tuning, single GPU, limited memory → **QLoRA** (4-bit base + LoRA, `target_modules="all-linear"`).
- Want a bit more quality than LoRA at the same rank → **DoRA** (`use_dora=True`) or **rsLoRA** at higher rank.
- Want the optimizer to spend rank where it matters → **AdaLoRA**.
- Smallest possible adapter / many tasks to store → **VeRA** (or RandLora).
- Diffusion / vision models → **LoKr** or **LoHa**.
- Combine several specialist adapters at inference → **X-LoRA**.

For exact field lists and defaults of each config class, see the corresponding entries (LoRA, AdaLoRA, LoHa, LoKr, VeRA, X-LoRA, RandLora, GraLoRA) in the PEFT Adapters API reference.
