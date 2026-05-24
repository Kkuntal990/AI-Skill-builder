# LoRA-Family Adapter Variants in PEFT

Covers the full landscape of LoRA-family adapters available in PEFT: their configs, key hyperparameters, and how they differ from standard LoRA.

---

## Standard LoRA

The baseline for all variants. Injects trainable low-rank matrices `A` and `B` into target modules so that the weight update is `ΔW = B @ A`, scaled by `lora_alpha / r`.

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Key parameters:
- `r` — rank of the decomposition; lower = fewer parameters
- `lora_alpha` — scaling factor; effective scale = `lora_alpha / r`
- `target_modules` — list of module name substrings or exact names to inject
- `lora_dropout` — dropout applied to the LoRA path
- `bias` — whether to train bias terms (`"none"`, `"all"`, `"lora_only"`)

Typical next step: pass to a `Trainer` or call `model.train()` directly.

---

## AdaLoRA

Adaptively allocates the rank budget across weight matrices using singular value decomposition. Matrices with more important singular values receive higher rank; less important ones are pruned.

```python
from peft import AdaLoraConfig, TaskType, get_peft_model

peft_config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    deltaT=10,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
```

Key parameters beyond LoRA:
- `init_r` — starting rank before pruning
- `target_r` — final average rank after budget allocation
- `deltaT` — steps between rank reallocations
- `beta1`, `beta2` — EMA coefficients for importance scoring

Use when you want automatic rank distribution rather than a fixed rank for all layers.

---

## AdaMSS

Adaptive Mixed-Sparsity and Structure adapter. Combines structured and unstructured sparsity with low-rank adaptation for fine-grained parameter efficiency.

```python
from peft import AdaMSSConfig, get_peft_model

peft_config = AdaMSSConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

See `examples/adamss_finetuning/` for a complete training script.

---

## LoHa (Low-rank Hadamard Product)

Represents weight updates as a Hadamard (element-wise) product of two pairs of low-rank matrices: `W1_a @ W1_b ⊙ W2_a @ W2_b`. Designed for diffusion model fine-tuning where expressiveness per parameter matters.

```python
from peft import LoHaConfig, get_peft_model

peft_config = LoHaConfig(
    r=4,
    alpha=8,
    target_modules=["to_q", "to_v"],
    module_dropout=0.1,
)
model = get_peft_model(model, peft_config)
```

Key parameters:
- `r` — rank of each of the four factor matrices
- `alpha` — scaling (analogous to `lora_alpha`)
- `module_dropout` — drops entire LoHa modules during training

Typical use: UNet layers in Stable Diffusion fine-tuning.

---

## LoKr (Low-rank Kronecker Product)

Represents weight updates via Kronecker product of two matrices, which can capture structured correlations more compactly than a plain outer product.

```python
from peft import LoKrConfig, get_peft_model

peft_config = LoKrConfig(
    r=4,
    alpha=8,
    decompose_both=False,
    target_modules=["to_q", "to_v"],
)
model = get_peft_model(model, peft_config)
```

Key parameters:
- `decompose_both` — whether to decompose both Kronecker factors with low-rank matrices
- `decompose_factor` — controls the split of the Kronecker factorization

---

## LyCORIS

An umbrella term for LoHa, LoKr, and related structured adapters. In PEFT, `LyCORISConfig` provides a unified entry point.

```python
from peft import LyCORISConfig, get_peft_model

peft_config = LyCORISConfig(
    r=4,
    algo="loha",          # or "lokr", "lora", etc.
    target_modules=["to_q", "to_v"],
)
model = get_peft_model(model, peft_config)
```

The `algo` field selects the underlying decomposition strategy.

---

## X-LoRA

Mixture-of-experts style LoRA: multiple LoRA adapters are loaded simultaneously and a learned router assigns per-layer, per-token mixing weights at inference time.

```python
from peft import XLoraConfig, get_peft_model

peft_config = XLoraConfig(
    hidden_size=model.config.hidden_size,
    adapters={
        "adapter1": "path/to/adapter1",
        "adapter2": "path/to/adapter2",
    },
    xlora_depth=8,
    layerwise_scalings=True,
)
model = get_peft_model(model, peft_config)
```

Key parameters:
- `adapters` — dict mapping names to saved adapter paths
- `xlora_depth` — depth of the router MLP
- `layerwise_scalings` — whether scalings differ per transformer layer

---

## VeRA (Vector-based Random Matrix Adaptation)

Freezes a shared pair of random matrices `A` and `B` (not trained) and learns only small per-layer scaling vectors `d_A` and `d_B`. Drastically reduces trainable parameters.

```python
from peft import VeraConfig, get_peft_model

peft_config = VeraConfig(
    r=256,
    projection_prng_key=42,
    vera_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    save_projection=True,
)
model = get_peft_model(model, peft_config)
```

Key parameters:
- `r` — rank of the shared random projection (can be large since matrices aren't trained)
- `projection_prng_key` — seed for reproducible random matrix generation
- `save_projection` — whether to persist the random matrices in the checkpoint

---

## GraLoRA

Gradient-based rank allocation for LoRA. Uses gradient statistics to determine which layers benefit from higher rank during training.

```python
from peft import GraloraConfig, get_peft_model

peft_config = GraloraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

---

## VB-LoRA (Vector Bank LoRA)

Shares a global bank of vectors across all adapter matrices. Each adapter layer selects and combines vectors from the bank, reducing total storage when many adapters are deployed.

```python
from peft import VBLoRAConfig, get_peft_model

peft_config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    num_vectors_for_each_weight=2,
    target_modules=["q_proj", "v_proj"],
    vblora_dropout=0.1,
    save_only_topk_weights=True,
)
model = get_peft_model(model, peft_config)
```

Key parameters:
- `num_vectors` — size of the shared vector bank
- `vector_length` — dimensionality of each bank vector
- `num_vectors_for_each_weight` — how many bank vectors each weight matrix uses
- `save_only_topk_weights` — saves only the selection logits, not the full bank, for compact checkpoints

---

## RandLoRA

Uses random, frozen projection matrices (similar in spirit to VeRA) but with a different parameterization focused on random feature maps.

```python
from peft import RandLoraConfig, get_peft_model

peft_config = RandLoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

---

## SHiRA (Sparse High-Rank Adapter)

Trains a sparse set of individual weights directly (not a low-rank factorization). Achieves high effective rank with very few parameters by selecting the most impactful weight positions.

```python
from peft import ShiraConfig, get_peft_model

peft_config = ShiraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

---

## TinyLoRA

An extremely compact LoRA variant targeting minimal memory footprint, useful for on-device or edge fine-tuning scenarios.

```python
from peft import TinyLoraConfig, get_peft_model

peft_config = TinyLoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

---

## DeLoRA (Decoupled LoRA)

Decouples the magnitude and direction of the LoRA update to improve training stability and prevent the adapter from collapsing into the base weight scale.

```python
from peft import DeloraConfig, get_peft_model

peft_config = DeloraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

See `examples/delora_finetuning/` for a worked example.

---

## LilyPEA

A lightweight adapter that applies small perturbations to selected weight subspaces, designed for fast convergence with minimal hyperparameter tuning.

```python
from peft import LilyPEAConfig, get_peft_model

peft_config = LilyPEAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

---

## WaveFT

Applies updates in the frequency domain using wavelet or Fourier-like transforms over weight matrices, related to FourierFT but with a different basis.

```python
from peft import WaveftConfig, get_peft_model

peft_config = WaveftConfig(
    n_frequency=1000,
    scaling=150.0,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
```

Key parameters:
- `n_frequency` — number of frequency components to train
- `scaling` — amplitude scaling of the frequency updates

---

## Choosing Between Variants

| Variant | Core idea | Best for |
|---|---|---|
| LoRA | Low-rank `B @ A` | General-purpose baseline |
| AdaLoRA | SVD + rank pruning | When rank distribution matters |
| AdaMSS | Mixed sparsity + low-rank | Structured + unstructured sparsity |
| LoHa | Hadamard of two rank pairs | Diffusion model fine-tuning |
| LoKr | Kronecker product | Structured weight correlations |
| X-LoRA | Router over multiple adapters | Multi-task / mixture-of-experts |
| VeRA | Shared random matrices + scale vectors | Extreme parameter reduction |
| GraLoRA | Gradient-guided rank allocation | Dynamic rank during training |
| VB-LoRA | Shared vector bank | Multi-adapter deployment efficiency |
| RandLoRA | Random projections | Low-storage adaptation |
| SHiRA | Sparse direct weight updates | High-rank with few parameters |
| TinyLoRA | Minimal footprint | Edge / on-device fine-tuning |
| DeLoRA | Decoupled magnitude/direction | Training stability |
| LilyPEA | Subspace perturbations | Fast convergence |
| WaveFT | Frequency-domain updates | Alternative to FourierFT |

---

## Shared Workflow Across All Variants

Every variant follows the same three-step pattern:

```python
# 1. Build config
from peft import <VariantConfig>, get_peft_model
peft_config = <VariantConfig>(r=8, target_modules=["q_proj", "v_proj"], ...)

# 2. Wrap model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # verify parameter count

# 3. Save / load
model.save_pretrained("output_dir/")
# Later:
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "output_dir/")
```

`print_trainable_parameters()` is the fastest way to confirm a variant is behaving as expected before committing to a full training run.
