# QLoRA and Quantized Base Models

How to fine-tune LoRA adapters on top of 4-bit/8-bit (bitsandbytes) and GPTQ/AWQ quantized base models with PEFT, including the `prepare_model_for_kbit_training` step that makes a quantized base trainable.

## Contents

- Why quantize the base
- 4-bit base loading (QLoRA)
- 8-bit base loading
- prepare_model_for_kbit_training
- Attaching the LoRA adapter
- End-to-end QLoRA recipe
- Training on a GPTQ base
- Training on an AWQ base
- Inference and merging
- Common pitfalls

## Why quantize the base

QLoRA loads the frozen base model in 4-bit (or 8-bit) to cut the memory footprint of the weights, then trains full-precision LoRA adapters on top. Only the adapter parameters carry gradients, so the large quantized base stays read-only and the optimizer state is tiny. The base is dequantized on the fly during the forward/backward pass; the stored weights remain low-bit.

Two distinct families are supported:

- **bitsandbytes** (`load_in_4bit` / `load_in_8bit`) — quantization happens at load time, in-memory. This is the canonical "QLoRA" path.
- **Pre-quantized checkpoints** (GPTQ, AWQ) — the weights are already quantized on disk; PEFT attaches adapters to the quantized linear layers.

## 4-bit base loading (QLoRA)

Configure 4-bit quantization with `BitsAndBytesConfig`, then pass it to `from_pretrained`. The canonical QLoRA settings use the NF4 datatype, nested (double) quantization, and a bf16/fp16 compute dtype.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NF4 is the QLoRA default
    bnb_4bit_use_double_quant=True,       # nested quantization, extra memory saving
    bnb_4bit_compute_dtype=torch.bfloat16 # matmul compute precision
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

What it does: weights load as 4-bit NF4 blocks; matmuls run in `bnb_4bit_compute_dtype`. Typical next step: call `prepare_model_for_kbit_training`, then attach LoRA.

Notes on the fields:
- `bnb_4bit_quant_type` accepts `"nf4"` (normal-float, the QLoRA recommendation) or `"fp4"`.
- `bnb_4bit_use_double_quant=True` quantizes the quantization constants for a further ~0.4 bits/param saving.
- `bnb_4bit_compute_dtype` should be `torch.bfloat16` on Ampere+ GPUs (A6000, A100), or `torch.float16` on older hardware. Keeping this at float32 silently slows training.

## 8-bit base loading

For an 8-bit base, set `load_in_8bit` instead. This is heavier than 4-bit but closer to full precision.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

The rest of the recipe (prepare → LoRA) is identical to the 4-bit path.

## prepare_model_for_kbit_training

A quantized base loaded straight from `from_pretrained` is **not** ready to train. `prepare_model_for_kbit_training` performs the setup that makes gradients flow into the LoRA adapters:

```python
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()   # optional but recommended for memory
model = prepare_model_for_kbit_training(model)
```

What it does:
- Casts layernorms (and other small modules) to fp32 for numerical stability.
- Casts the output embedding / `lm_head` to fp32.
- Enables gradient flow through the frozen quantized weights by registering a hook that makes the input to the model require gradients (so backprop reaches the adapters even though every base parameter has `requires_grad=False`).
- Wires up gradient checkpointing compatibility.

`use_gradient_checkpointing` is enabled by default inside the call; if you also call `model.gradient_checkpointing_enable()` yourself, do it **before** `prepare_model_for_kbit_training`.

**This step is mandatory.** Skipping it is the single most common QLoRA failure: the model "trains" but the loss is flat because no gradient reaches the adapters. Call it once, immediately after loading the quantized base and before `get_peft_model`.

## Attaching the LoRA adapter

After preparation, attach LoRA exactly as with a full-precision base — wrap the model with `get_peft_model` and a `LoraConfig`.

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # adjust per architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

`print_trainable_parameters()` reports the number and percentage of trainable params — for a 7B base with this config it should be well under 1%. If it reports 0 trainable params or an unexpectedly large count, the config or preparation step is wrong.

## End-to-end QLoRA recipe

The full minimal sequence, in order:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Order matters: **load quantized → (gradient checkpointing) → prepare_model_for_kbit_training → get_peft_model**. Hand this model to a `Trainer`/`SFTTrainer` as usual. Typical next step: build the training loop and pass `model` to the trainer.

## Training on a GPTQ base

GPTQ checkpoints are already quantized on disk. Load the GPTQ config and pass it to `from_pretrained`, then follow the same prepare → LoRA flow.

```python
from transformers import AutoModelForCausalLM, GPTQConfig
from peft import prepare_model_for_kbit_training

gptq_config = GPTQConfig(bits=4, use_exllama=False)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    quantization_config=gptq_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
# then get_peft_model(model, lora_config) as above
```

Set `use_exllama=False` (or `disable_exllama=True` on older versions) when training — the ExLlama kernel is inference-only and does not support backprop through the quantized layers.

## Training on an AWQ base

AWQ-quantized checkpoints follow the same pattern: the model is loaded already-quantized, prepared, then wrapped with LoRA.

```python
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
# then get_peft_model(model, lora_config) as above
```

The AWQ quantization metadata travels with the checkpoint, so no explicit quantization config is required on load. Adapters train in full precision on top of the AWQ-quantized base.

## Inference and merging

For a trained QLoRA adapter, load the quantized base the same way you trained it, then load the adapter on top:

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "path/to/adapter")
```

**Merging caveat:** `merge_and_unload()` folds the adapter weights into the base. This requires dequantizing the base first — you cannot losslessly merge a full-precision adapter into 4-bit weights and keep them 4-bit. To deploy a single merged checkpoint, reload the base in fp16/bf16 (not quantized), attach the adapter, then merge.

## Common pitfalls

- **Forgetting `prepare_model_for_kbit_training`** — gradients never reach the adapters; loss stays flat. This step is non-optional for any quantized base. *(Recurring failure mode in our trajectories.)*
- **Calling `prepare_*` after `get_peft_model`** — the preparation must wrap the bare quantized base before the adapter is attached.
- **Wrong compute dtype** — leaving `bnb_4bit_compute_dtype` at the default float32 negates much of the speed benefit; use `bfloat16` on Ampere+.
- **ExLlama kernel during GPTQ training** — disable it; it is inference-only and blocks backprop.
- **Tokenizer right-padding for causal LM** — set `tokenizer.padding_side = "left"` (or pad correctly for the loss) to avoid silently corrupting generation/eval on a QLoRA-trained model.
- **Expecting to keep 4-bit after merge** — merging dequantizes; plan to re-quantize the merged checkpoint if a low-bit deployment artifact is needed.
