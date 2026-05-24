# Orthogonal and Sparse PEFT Methods

Covers IA3, OFT, BOFT, PSOFT, Polytropon, FourierFT, HRA, CPT, C3A, MiSS, RoAd, Nu, Trainable Tokens, Cartridges, and Layernorm tuning — their configs, key hyperparameters, and usage patterns in PEFT.

## IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

IA3 rescales inner activations with learned vectors rather than adding low-rank matrices. It is extremely parameter-efficient and well-suited for multi-task inference.

```python
from peft import IA3Config, TaskType, get_peft_model

config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],  # modules where IA3 acts on input (not output)
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~0.01% of total
```

- `target_modules`: which linear layers receive IA3 scaling vectors.
- `feedforward_modules`: subset of `target_modules` that are feedforward (scaling applied to input activations).
- Typical next step: train with a standard `Trainer` or custom loop; merge with `model.merge_adapter()` for zero-overhead inference.

## OFT (Orthogonal Fine-Tuning)

OFT constrains weight updates to orthogonal transformations, preserving the hyperspherical energy of pretrained representations.

```python
from peft import OFTConfig, TaskType, get_peft_model

config = OFTConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # block size / rank of orthogonal matrix
    target_modules=["q_proj", "v_proj"],
    oft_block_size=8,           # size of each orthogonal block
    module_dropout=0.0,
    init_weights=True,
)
model = get_peft_model(model, config)
```

- `r` and `oft_block_size` together control the granularity of the orthogonal decomposition.
- `module_dropout`: randomly drops OFT modules during training for regularization.
- Typical next step: use for image-generation fine-tuning (see `examples/boft_dreambooth`).

## BOFT (Block-diagonal Orthogonal Fine-Tuning)

BOFT generalizes OFT with a butterfly-structured block-diagonal orthogonal matrix, giving more expressive capacity at similar parameter cost.

```python
from peft import BOFTConfig, TaskType, get_peft_model

config = BOFTConfig(
    task_type=TaskType.CAUSAL_LM,
    boft_block_size=8,          # size of each butterfly block
    boft_block_num=0,           # number of blocks; 0 = infer from block_size
    boft_n_butterfly_factor=2,  # butterfly factor controlling depth of decomposition
    target_modules=["q_proj", "v_proj", "out_proj"],
    boft_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, config)
```

- Either `boft_block_size` or `boft_block_num` must be non-zero (not both).
- `boft_n_butterfly_factor`: higher values increase expressiveness but also parameter count.
- See `examples/boft_controlnet` and `examples/boft_dreambooth` for diffusion model usage.

## PSOFT (Partially Orthogonal Fine-Tuning)

PSOFT relaxes the strict orthogonality constraint of OFT, allowing partial orthogonality for a better accuracy–efficiency trade-off.

```python
from peft import PSOFTConfig, TaskType, get_peft_model

config = PSOFTConfig(
    task_type=TaskType.CAUSAL_LM,
    num_soft_prompt_tokens=10,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
```

- `num_soft_prompt_tokens`: controls the degree of partial relaxation from full orthogonality.
- Useful when strict OFT is too constrained for a given task.

## Polytropon (Multi-task Adapter Routing)

Polytropon learns a shared inventory of adapters and a per-task routing mechanism, enabling efficient multi-task fine-tuning.

```python
from peft import PolyConfig, TaskType, get_peft_model

config = PolyConfig(
    task_type=TaskType.CAUSAL_LM,
    poly_type="poly",           # routing type
    r=8,                        # rank of each adapter in the inventory
    n_tasks=16,                 # number of tasks
    n_skills=4,                 # number of adapter modules in the inventory
    n_splits=1,                 # number of heads for routing
)
model = get_peft_model(model, config)
```

- `n_skills`: size of the shared adapter inventory.
- `n_tasks`: total tasks the routing module must handle.
- Typical next step: pass `task_ids` tensor alongside inputs during training so the router can select the right adapter combination.

## FourierFT (Fourier Transform Fine-Tuning)

FourierFT reparameterizes weight updates in the frequency domain using a sparse set of Fourier coefficients, achieving very low parameter counts.

```python
from peft import FourierFTConfig, TaskType, get_peft_model

config = FourierFTConfig(
    task_type=TaskType.CAUSAL_LM,
    n_frequency=1000,           # number of Fourier coefficients to learn
    target_modules=["q_proj", "v_proj"],
    scaling=150.0,              # analogous to lora_alpha / r scaling
    random_loc_seed=777,        # seed for reproducible frequency location sampling
    init_weights=True,
)
model = get_peft_model(model, config)
```

- `n_frequency`: primary knob — more frequencies = more capacity but more parameters.
- `scaling`: controls the magnitude of the learned update applied to the base weight.
- Typical next step: train normally; the inverse FFT reconstructs the weight delta at each forward pass.

## HRA (Householder Reflection Adaptation)

HRA uses a chain of Householder reflections to parameterize orthogonal updates, offering an alternative to OFT with different numerical properties.

```python
from peft import HRAConfig, TaskType, get_peft_model

config = HRAConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # number of Householder reflections
    target_modules=["q_proj", "v_proj"],
    apply_GS=False,             # apply Gram-Schmidt orthogonalization
    init_weights=True,
)
model = get_peft_model(model, config)
```

- `r`: number of reflections; higher values increase expressiveness.
- `apply_GS`: enables Gram-Schmidt re-orthogonalization for numerical stability.

## CPT (Contrastive Preference Tuning / Context-aware Prompt Tuning)

CPT adapts models using context-aware soft prompts optimized with a contrastive objective.

```python
from peft import CPTConfig, TaskType, get_peft_model

config = CPTConfig(
    task_type=TaskType.CAUSAL_LM,
    cpt_token_ids=[1, 2, 3],    # token ids used to initialize prompt embeddings
    cpt_mask=[1, 1, 0],         # which positions are trainable
    cpt_tokens_type_mask=[1, 2, 1],  # semantic role of each token position
    num_virtual_tokens=8,
)
model = get_peft_model(model, config)
```

- `cpt_mask`: binary mask controlling which virtual token positions are updated.
- See `examples/cpt_finetuning` for a complete training script.

## C3A (Cross-layer Cross-head Compact Adaptation)

C3A shares adapter parameters across layers and attention heads to maximize parameter reuse.

```python
from peft import C3AConfig, TaskType, get_peft_model

config = C3AConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    num_heads=4,                # number of attention heads to couple
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
```

- `num_heads`: controls cross-head coupling granularity.
- Reduces total adapter parameters compared to per-head LoRA at similar rank.

## MiSS (Mixed Sparse Structures)

MiSS combines multiple sparse update structures within a single adapter, allowing the model to learn which structure is most effective per layer.

```python
from peft import MiSSConfig, TaskType, get_peft_model

config = MiSSConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "down_proj"],
    r=16,
)
model = get_peft_model(model, config)
```

- Internally selects among sparse update patterns; `r` controls the shared rank budget.

## RoAd (Rotation-based Adaptation)

RoAd applies learned rotation matrices to adapt pretrained weights while preserving their spectral properties.

```python
from peft import RoAdConfig, TaskType, get_peft_model

config = RoAdConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
)
model = get_peft_model(model, config)
```

- `r`: rank of the rotation parameterization; lower values are more constrained.

## Nu (Null-space Update)

Nu constrains weight updates to lie in the null space of the pretrained weight matrix, ensuring the pretrained output subspace is not disturbed.

```python
from peft import NuConfig, TaskType, get_peft_model

config = NuConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
```

- Particularly useful when preserving pretrained knowledge is critical (e.g., continual learning).

## Trainable Tokens

Trainable Tokens prepends or appends a small number of learnable token embeddings to the input sequence without modifying any model weights.

```python
from peft import TrainableTokensConfig, TaskType, get_peft_model

config = TrainableTokensConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,      # number of learnable tokens prepended to input
    target_modules=["embed_tokens"],
)
model = get_peft_model(model, config)
```

- `num_virtual_tokens`: the only major hyperparameter; 8–32 is typical.
- Lighter than prefix tuning because no key/value caches are modified.
- Typical next step: freeze all parameters except the token embeddings and train.

## Cartridges

Cartridges are self-contained adapter modules that can be loaded, swapped, and composed at inference time, designed for rapid task switching.

```python
from peft import CartridgeConfig, TaskType, get_peft_model

config = CartridgeConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
```

- See `examples/cartridge_self_study` for a self-study training workflow.
- Cartridges are saved and loaded like any PEFT adapter:

```python
model.save_pretrained("./my_cartridge")
# later:
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./my_cartridge")
```

## Layernorm Tuning

Layernorm tuning fine-tunes only the scale and bias parameters of LayerNorm layers, leaving all other weights frozen.

```python
from peft import LNTuningConfig, TaskType, get_peft_model

config = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["input_layernorm", "post_attention_layernorm"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~0.004% of total (only gamma/beta of each LayerNorm)
```

- `target_modules`: names of LayerNorm modules to tune; use `"all-linear"` equivalent patterns or explicit names.
- Extremely cheap; often combined with other PEFT methods via `get_peft_model` with multiple configs using mixed adapter types.
- Typical next step: combine with LoRA or IA3 for slightly higher capacity at minimal extra cost.

## Shared Patterns Across These Methods

### Saving and Loading

All methods follow the same PEFT checkpoint interface:

```python
# Save
model.save_pretrained("./adapter_dir")

# Load onto a fresh base model
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./adapter_dir")
```

### Merging Adapters into Base Weights

Where the method supports it (OFT, BOFT, IA3, FourierFT, HRA, Layernorm tuning):

```python
model = model.merge_and_unload()
# Returns a plain nn.Module with adapter weights baked in — no PEFT overhead at inference
```

### Inspecting Trainable Parameters

```python
model.print_trainable_parameters()
# Example output:
# trainable params: 1,572,864 || all params: 7,241,732,096 || trainable%: 0.0217
```

### Disabling / Enabling Adapters

```python
with model.disable_adapter():
    # runs base model only
    output = model(**inputs)
```

### Mixed Adapter Types

These methods can be combined with LoRA or each other using `inject_adapter_in_model` or by loading multiple adapters and setting active adapters:

```python
model.set_adapter("oft_adapter")   # activate a specific named adapter
model.set_adapter("ia3_adapter")   # switch to another
```
