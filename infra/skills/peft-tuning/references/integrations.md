# Integrations: Transformers, Diffusers, Accelerate, DeepSpeed, FSDP, and PEFT Integration Functions

Covers how PEFT integrates with Transformers, Diffusers, Accelerate, DeepSpeed, and FSDP, plus the utility functions exposed for external PEFT integration.

## Transformers Integration

PEFT works directly with `transformers` models. Load any `AutoModel` from Transformers, wrap it with `get_peft_model`, and the result is a `PeftModel` that is fully compatible with the Transformers training API.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="cuda")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

What it does: freezes the base model weights, injects adapter layers, and exposes only adapter parameters to the optimizer. Typical next step: pass `model` directly to a `transformers.Trainer` or a custom training loop.

### Loading PEFT Adapters via Transformers

Transformers' `from_pretrained` natively understands PEFT adapter repos on the Hub. You can load a base model and attach a saved adapter in one call:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "username/my-lora-adapter",  # Hub repo containing adapter_config.json
    device_map="auto",
)
```

Alternatively, use `PeftModel.from_pretrained` explicitly:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "username/my-lora-adapter")
```

Typical next step: call `model.merge_and_unload()` to fold adapter weights into the base model for pure-Transformers inference.

## Diffusers Integration

PEFT is the adapter backend for Diffusers. Diffusers pipelines call PEFT internally to load, switch, and fuse LoRA weights for UNet and text encoder components.

### Loading LoRA weights into a Diffusers pipeline

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe.load_lora_weights("username/my-sdxl-lora")
pipe.to("cuda")
```

What it does: Diffusers delegates adapter injection to PEFT under the hood. Typical next step: call `pipe.fuse_lora()` to merge weights for faster inference, or `pipe.unload_lora_weights()` to remove them.

### Managing multiple adapters in Diffusers

```python
pipe.load_lora_weights("adapter-one", adapter_name="style")
pipe.load_lora_weights("adapter-two", adapter_name="subject")
pipe.set_adapters(["style", "subject"], adapter_weights=[0.7, 0.3])
```

Typical next step: generate images; PEFT's `add_weighted_adapter` logic handles the blending.

## Accelerate Integration

PEFT models are compatible with `accelerate` out of the box. Wrap your `PeftModel` with `Accelerator.prepare` exactly as you would any other model.

```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32))
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

What it does: Accelerate handles device placement, mixed precision, and gradient accumulation while PEFT keeps only adapter parameters trainable. Typical next step: run the training loop; save with `accelerator.unwrap_model(model).save_pretrained(...)`.

### Saving and loading with Accelerate

```python
# After training
unwrapped = accelerator.unwrap_model(model)
unwrapped.save_pretrained(
    "output_dir",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
```

## DeepSpeed Integration

PEFT supports DeepSpeed ZeRO stages via Accelerate's DeepSpeed plugin. The key requirement is that only adapter parameters are trainable, which keeps the ZeRO optimizer state small.

### Launching with DeepSpeed

Use an Accelerate config with DeepSpeed enabled, then run normally:

```bash
accelerate launch --config_file deepspeed_config.yaml train.py
```

Inside `train.py`:

```python
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32)
model = get_peft_model(model, peft_config)
model = accelerator.prepare(model)
```

What it does: DeepSpeed shards the frozen base model parameters across GPUs while PEFT adapter parameters remain as a small trainable set. Typical next step: after training, consolidate shards with `accelerator.unwrap_model(model).save_pretrained(...)`.

### ZeRO-3 considerations

With ZeRO-3, base model weights are sharded. When saving, gather all shards first:

```python
from peft import get_peft_model_state_dict

state_dict = get_peft_model_state_dict(
    model,
    state_dict=accelerator.get_state_dict(model),
)
```

`get_peft_model_state_dict` filters the full state dict down to adapter-only parameters, which is what you want to checkpoint.

## FSDP (Fully Sharded Data Parallel) Integration

PEFT works with PyTorch FSDP via Accelerate's FSDP plugin. The adapter modules must be excluded from sharding or wrapped separately so their parameters stay accessible.

### FSDP launch pattern

```bash
accelerate launch --config_file fsdp_config.yaml train.py
```

Inside `train.py`:

```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32))
model = accelerator.prepare(model)
```

### Saving under FSDP

```python
from peft import get_peft_model_state_dict

if accelerator.is_main_process:
    peft_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(model))
    accelerator.unwrap_model(model).save_pretrained("output_dir", state_dict=peft_state_dict)
```

Typical next step: reload with `PeftModel.from_pretrained(base_model, "output_dir")` on a single GPU for inference.

## Functions for PEFT Integration

These utility functions are the public API surface for integrating PEFT into external libraries (e.g., Transformers, Diffusers, or custom trainers).

### `get_peft_model`

Wraps a base model with a PEFT adapter configuration.

```python
from peft import get_peft_model, LoraConfig, TaskType

model = get_peft_model(base_model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16))
model.print_trainable_parameters()
# e.g. trainable params: 3,407,872 || all params: 1,800,000,000 || trainable%: 0.19
```

### `get_peft_model_state_dict`

Extracts only the adapter parameters from a model's state dict. Essential for saving checkpoints under distributed strategies.

```python
from peft import get_peft_model_state_dict

adapter_state = get_peft_model_state_dict(model)
torch.save(adapter_state, "adapter_weights.bin")
```

### `set_peft_model_state_dict`

Loads adapter parameters back into a `PeftModel` from a state dict.

```python
from peft import set_peft_model_state_dict

adapter_state = torch.load("adapter_weights.bin")
set_peft_model_state_dict(model, adapter_state)
```

Typical next step: call `model.eval()` and run inference.

### `inject_adapter_in_model`

Low-level function to inject an adapter configuration directly into a model without creating a full `PeftModel` wrapper. Used by library integrations that manage the model object themselves.

```python
from peft import inject_adapter_in_model, LoraConfig

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = inject_adapter_in_model(config, model)
```

### `get_layer_status` / `get_model_status`

Inspect which layers have active adapters and their current state.

```python
from peft import get_model_status

status = get_model_status(model)
print(status)
```

### `add_hook_to_module` / `remove_hook_from_module`

Utility hooks used internally by PEFT for offloading and device management. Exposed for integrators who need fine-grained control over module execution.

```python
from peft.utils import add_hook_to_module, remove_hook_from_module

add_hook_to_module(module, hook)
# ... custom logic ...
remove_hook_from_module(module)
```

## Adapter Enable/Disable Helpers

These functions are useful when building multi-adapter inference pipelines:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base, "adapter-a", adapter_name="a")
model.load_adapter("adapter-b", adapter_name="b")

model.set_adapter("a")          # activate adapter "a"
model.disable_adapter()         # run base model only (context manager)
model.enable_adapter_layers()   # re-enable all adapter layers
model.disable_adapter_layers()  # disable all adapter layers globally
```

Typical next step: benchmark base model vs. adapted model latency, or implement dynamic adapter routing in a serving loop.
