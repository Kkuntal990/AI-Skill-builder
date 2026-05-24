# Developer Guides: Advanced PEFT Usage

Covers model merging, quantization, custom models, adapter injection, mixed adapter types, torch.compile, hotswapping, and the PEFT checkpoint format.

---

## Model Merging

Model merging combines multiple fine-tuned adapters (or full models) into one without additional training. PEFT exposes this through `add_weighted_adapter` and dedicated merge utilities.

### Merging LoRA Adapters into the Base Model

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "path/to/lora")
merged_model = model.merge_and_unload()
```

`merge_and_unload()` folds the LoRA weights into the base model and returns a plain `nn.Module` with no adapter overhead. Use this before exporting or serving.

### Combining Multiple Adapters with `add_weighted_adapter`

```python
model.add_weighted_adapter(
    adapters=["adapter_a", "adapter_b"],
    weights=[0.5, 0.5],
    adapter_name="merged",
    combination_type="linear",
)
model.set_adapter("merged")
```

`combination_type` options include `"linear"`, `"ties"`, `"dare_linear"`, `"dare_ties"`, `"magnitude_prune"`. Each applies a different merging strategy to the delta weights before combining.

### TIES and DARE Merging

TIES resolves sign conflicts across adapters before averaging. DARE randomly drops delta parameters before merging to reduce interference.

```python
model.add_weighted_adapter(
    adapters=["adapter_a", "adapter_b"],
    weights=[0.6, 0.4],
    adapter_name="ties_merged",
    combination_type="ties",
    density=0.5,          # fraction of parameters to keep (TIES/DARE)
)
```

`density` controls sparsity: lower values prune more aggressively.

---

## Quantization

PEFT adapters can be trained and loaded on top of quantized base models, reducing GPU memory requirements significantly.

### Loading a Quantized Base Model with bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
```

`prepare_model_for_kbit_training` enables gradient checkpointing and casts layer norms to float32 so training is stable.

### Attaching LoRA on Top of a Quantized Model

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,504,607,232 || trainable%: 0.12
```

Only the LoRA parameters are trained; the 4-bit base weights remain frozen.

---

## Custom Models

PEFT can wrap arbitrary `nn.Module` objects, not just Transformers models.

### Wrapping a Custom PyTorch Model

```python
from peft import LoraConfig, get_peft_model
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.linear(x))

model = MyModel()
config = LoraConfig(target_modules=["linear"])
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
```

`target_modules` accepts module names as strings; PEFT replaces matching `nn.Linear` layers with LoRA-wrapped equivalents.

### Saving and Loading Custom Model Adapters

```python
peft_model.save_pretrained("my_custom_adapter/")
# Later:
from peft import PeftModel
restored = PeftModel.from_pretrained(model, "my_custom_adapter/")
```

The base model class is not saved — you must reconstruct it before calling `from_pretrained`.

---

## Adapter Injection

Adapter injection lets you add LoRA (or other) layers to a model that was **not** wrapped with `get_peft_model`, useful when you need full control over the model object.

### Injecting LoRA Weights Directly

```python
from peft import inject_adapter_in_model, LoraConfig

config = LoraConfig(target_modules=["q_proj", "v_proj"])
model = inject_adapter_in_model(config, model, adapter_name="default")
```

After injection the model is a plain `nn.Module` with LoRA layers inserted. You manage saving/loading manually.

### Marking Only Adapter Parameters as Trainable

```python
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

# Freeze everything, then unfreeze adapter params
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

This pattern is necessary when using `inject_adapter_in_model` because the model is not a `PeftModel` instance.

---

## Mixed Adapter Types

A single model can host multiple adapters of **different types** simultaneously (e.g., LoRA + IA³).

### Loading Multiple Adapter Types

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "path/to/lora", adapter_name="lora_adapter")
model.load_adapter("path/to/ia3", adapter_name="ia3_adapter")
```

### Activating a Specific Adapter

```python
model.set_adapter("lora_adapter")
output_lora = model(**inputs)

model.set_adapter("ia3_adapter")
output_ia3 = model(**inputs)
```

### Using `MixedModel` for Simultaneous Activation

```python
from peft import PeftMixedModel

mixed_model = PeftMixedModel.from_pretrained(base_model, "path/to/lora", adapter_name="lora")
mixed_model.load_adapter("path/to/ia3", adapter_name="ia3")
# Both adapters are active at inference
```

`PeftMixedModel` applies all loaded adapters in a single forward pass. Adapter types must be compatible (not all combinations are supported).

---

## torch.compile

`torch.compile` can be applied to PEFT models for inference speedups, with a few caveats.

### Compiling a PEFT Model

```python
import torch
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "path/to/adapter")
model.eval()
compiled_model = torch.compile(model)

with torch.no_grad():
    output = compiled_model(**inputs)
```

### Compiling Only the Base Model

If full-model compilation causes graph breaks, compile only the base component:

```python
model.base_model = torch.compile(model.base_model)
```

This avoids recompilation when switching adapters while still gaining speedups on the heavy transformer layers.

### Known Limitations with torch.compile

- Adapter switching (`set_adapter`, `disable_adapter`) after compilation triggers recompilation.
- `merge_and_unload()` should be called **before** `torch.compile` for static deployments.
- Dynamic shapes from variable-length inputs may reduce compile effectiveness; use `torch.compile(model, dynamic=True)` to handle them.

---

## Hotswapping Adapters

Hotswapping replaces adapter weights at runtime without reloading the model, enabling fast adapter switching in serving scenarios.

### Basic Hotswap

```python
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter

model = PeftModel.from_pretrained(base_model, "adapter_v1/", adapter_name="default")

# Replace weights in-place with adapter_v2
hotswap_adapter(model, "adapter_v2/", adapter_name="default")
```

`hotswap_adapter` overwrites the existing adapter tensors with those from the new checkpoint. The model structure is unchanged; only weights are updated.

### Requirements for Hotswapping

- Both adapters must share the same config (same `r`, `target_modules`, adapter type).
- The model must already have the adapter loaded under the target `adapter_name`.
- Works with `torch.compile`; no recompilation is triggered if the model graph is unchanged.

### Hotswapping with Multiple Named Adapters

```python
model.load_adapter("adapter_a/", adapter_name="a")
model.load_adapter("adapter_b/", adapter_name="b")

# Later, replace adapter "a" weights without touching "b"
hotswap_adapter(model, "adapter_a_v2/", adapter_name="a")
```

---

## PEFT Checkpoint Format

Understanding the checkpoint layout is essential for interoperability and custom tooling.

### What Gets Saved

`save_pretrained` writes two files:

```
my_adapter/
├── adapter_config.json   # LoraConfig (or other) serialized
└── adapter_model.safetensors  # adapter weights only (no base model weights)
```

The base model weights are **not** included. The checkpoint is portable across any compatible base model.

### adapter_config.json Structure

```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "bias": "none",
  "base_model_name_or_path": "meta-llama/Llama-2-7b-hf"
}
```

`base_model_name_or_path` is recorded for reference but not enforced at load time.

### Weight Key Naming Convention

Adapter weights follow the pattern:

```
base_model.model.<original_layer_path>.lora_A.weight
base_model.model.<original_layer_path>.lora_B.weight
```

For example:
```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
```

The `default` segment is the `adapter_name`.

### Loading Adapter Weights Manually

```python
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

weights = load_file("my_adapter/adapter_model.safetensors")
set_peft_model_state_dict(peft_model, weights)
```

Use `set_peft_model_state_dict` instead of `load_state_dict` directly to handle key remapping correctly.

### Saving Multiple Adapters Separately

```python
model.save_pretrained("output/", selected_adapters=["adapter_a"])
model.save_pretrained("output/", selected_adapters=["adapter_b"])
```

Each call writes an independent checkpoint directory for the specified adapter.
