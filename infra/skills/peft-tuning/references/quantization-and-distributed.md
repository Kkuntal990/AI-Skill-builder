# Quantization & Distributed Training

How to fine-tune k-bit / QLoRA quantized models with bitsandbytes and `prepare_model_for_kbit_training`, and how to scale PEFT runs across GPUs with DeepSpeed ZeRO-3 and Fully Sharded Data Parallel (FSDP).

## Contents

- 4-bit Quantized Loading (QLoRA)
- 8-bit Quantized Loading
- prepare_model_for_kbit_training
- Full QLoRA Recipe
- Saving and Loading Quantized Adapters
- DeepSpeed ZeRO-3 Integration
- FSDP Integration
- QLoRA + FSDP (quant storage dtype)
- Common Pitfalls

## 4-bit Quantized Loading (QLoRA)

Quantize the frozen base model to 4 bits at load time with a `BitsAndBytesConfig`. The adapter weights stay in higher precision; only the base weights are quantized.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
```

- `nf4` (4-bit NormalFloat) is the QLoRA default and is preferred over `fp4`.
- `bnb_4bit_use_double_quant=True` adds a second quantization of the quantization constants, saving ~0.4 bits/param.
- `bnb_4bit_compute_dtype` is the dtype used for the matmul; `bfloat16` is the usual choice on Ampere+ GPUs.

**Next step:** call `prepare_model_for_kbit_training`, then attach a LoRA adapter.

## 8-bit Quantized Loading

For an 8-bit base (LLM.int8()), set `load_in_8bit` instead. This uses more memory than 4-bit but can be more stable.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Next step:** same as 4-bit — prepare for k-bit training, then `get_peft_model`.

## prepare_model_for_kbit_training

A quantized base model is not directly trainable. This helper makes it ready: it casts layernorms and the LM head to fp32, enables gradient checkpointing, and makes the input embeddings require grad so gradients flow back through the adapter.

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

Pass through gradient-checkpointing options if needed:

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

- Call this **before** `get_peft_model`.
- `use_reentrant=False` is recommended when combining gradient checkpointing with distributed training.

## Full QLoRA Recipe

The canonical end-to-end pattern: 4-bit base + k-bit prep + LoRA adapter.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

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

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Hand `model` to a `Trainer`/`SFTTrainer` as usual. Only the LoRA parameters are updated; the 4-bit base stays frozen.

## Saving and Loading Quantized Adapters

`save_pretrained` writes only the adapter (a few MB), not the quantized base.

```python
model.save_pretrained("qlora-out")
```

Reload by quantizing the base again, then attaching the saved adapter:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "qlora-out")
```

Note: you cannot `merge_and_unload` directly into a 4-bit base. To produce a merged checkpoint, reload the base in fp16/bf16 (un-quantized), attach the adapter, then merge.

## DeepSpeed ZeRO-3 Integration

PEFT works with 🤗 Accelerate's DeepSpeed launcher. ZeRO stage 3 shards optimizer state, gradients, and parameters across GPUs. Configure via an Accelerate config file.

`deepspeed_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 4
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
use_cpu: false
```

Launch the training script with this config:

```bash
accelerate launch --config_file deepspeed_config.yaml train.py
```

- `zero3_init_flag: true` enables partitioned model init so the full model never materializes on one rank.
- For CPU/NVMe offload of optimizer or params, set `offload_optimizer_device` / `offload_param_device`.

## FSDP Integration

FSDP shards parameters, gradients, and optimizer state via PyTorch-native sharding. Use a `distributed_type: FSDP` Accelerate config.

`fsdp_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
use_cpu: false
```

```bash
accelerate launch --config_file fsdp_config.yaml train.py
```

- `TRANSFORMER_BASED_WRAP` wraps each transformer block as an FSDP unit.
- `fsdp_cpu_ram_efficient_loading: true` + `fsdp_sync_module_states: true` load the model on rank 0 only and broadcast, avoiding N× host-RAM blowup.

## QLoRA + FSDP (quant storage dtype)

To shard a 4-bit base across GPUs with FSDP, the bitsandbytes quantized weights must be stored in a dtype FSDP can flatten. Set `bnb_4bit_quant_storage` to match the compute/mixed-precision dtype.

```python
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)
```

- `bnb_4bit_quant_storage` must equal the FSDP mixed-precision dtype (here `bfloat16`); a mismatch raises an FSDP flattening error.
- Keep `fsdp_use_orig_params: false` for QLoRA + FSDP.

## Common Pitfalls

- **Skipping `prepare_model_for_kbit_training`** → no gradients flow and loss stays flat; quantized bases need the prep step before `get_peft_model`.
- **Merging into a quantized base** → `merge_and_unload` on a 4-bit/8-bit model fails or degrades; reload the base in fp16/bf16 to merge.
- **`device_map="auto"` with multi-GPU distributed launchers** → let Accelerate/FSDP/DeepSpeed place the model instead of `device_map` when launching with `accelerate launch`.
- **QLoRA + FSDP dtype mismatch** → set `bnb_4bit_quant_storage` equal to the mixed-precision dtype.
- **OOM during load on multi-node** → enable `fsdp_cpu_ram_efficient_loading` (FSDP) or `zero3_init_flag` (DeepSpeed) so the full model never lands on a single rank.
