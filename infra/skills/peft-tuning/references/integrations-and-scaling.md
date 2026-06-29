# Integrations and Scaling

How PEFT scales to multiple GPUs via Accelerate (DeepSpeed ZeRO and FSDP), how to save/load adapters, switch and merge them, and how Transformers and Diffusers consume PEFT adapters natively.

## Contents

- Multi-GPU training with DeepSpeed ZeRO-3
- DeepSpeed ZeRO-3 + QLoRA
- DeepSpeed ZeRO-3 + CPU offload (single GPU)
- Multi-GPU training with FSDP
- FSDP + QLoRA
- Distributed training caveats
- Saving an adapter
- Loading an adapter for inference (AutoPeftModel)
- Transformers native PEFT integration
- Managing multiple adapters
- Hotswapping adapters
- Merging adapters (TIES / DARE / linear)
- Merging and unmerging in-place
- Merging in distributed (ZeRO-3) settings
- Diffusers integration

## Multi-GPU training with DeepSpeed ZeRO-3

DeepSpeed's ZeRO shards optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data-parallel processes. All three stages are compatible with PEFT LoRA + `bitsandbytes`. It is driven through Accelerate — you write a config file once, then `accelerate launch` your training script.

Generate the config interactively (pick ZeRO-3 when prompted):

```bash
accelerate config --config_file deepspeed_config.yaml
```

The resulting config for ZeRO-3 on 8 GPUs (single machine):

```yml
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
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Launch the PEFT SFT script with the config (LoRA r=8, alpha=16, all linear layers; SFT hyperparameters abbreviated):

```bash
accelerate launch --config_file "configs/deepspeed_config.yaml" train.py \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False \
# ... + standard SFT flags (dataset, seq len, lr, epochs, batch size, gradient_checkpointing, etc.)
```

`SFTTrainer` builds the `PeftModel` from `peft_config` and, on `trainer.train()`, lets Accelerate wrap it into a DeepSpeed engine. The salient trainer call:

```python
# trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)
trainer.accelerator.print(f"{trainer.model}")

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
trainer.save_model()
```

Typical next step: confirm `zero3_save_16bit_model: true` so the saved adapter weights are 16-bit and reloadable with `from_pretrained`.

## DeepSpeed ZeRO-3 + QLoRA

Combines 4-bit quantization with ZeRO-3 to fit a 70B model on 2×40GB GPUs. Requires `bitsandbytes>=0.43.3`, `accelerate>=1.0.1`, `transformers>4.44.2`, `trl>0.11.4`, `peft>0.13.0`, and `zero3_init_flag: true`. The config drops `gradient_accumulation_steps` from the DeepSpeed block and sets `num_processes: 2`; otherwise it matches the ZeRO-3 config above.

The training-code change is the 4-bit storage dtype, which must match the model load dtype:

```diff
bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit_quantization,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
+   bnb_4bit_quant_storage=quant_storage_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
+   dtype=quant_storage_dtype or torch.float32,
)
```

Launch with the extra quantization flags:

```bash
accelerate launch --config_file "configs/deepspeed_config_z3_qlora.yaml" train.py \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16" \
# ... + the same LoRA + SFT flags as above
```

`bnb_4bit_quant_storage_dtype` denotes the dtype used to pack the 4-bit parameters (e.g. with `bfloat16`, 4-bit params are packed together post-quantization). Memory: ~36.6 GB/GPU vs the 8×80GB needed for ZeRO-3 + LoRA.

## DeepSpeed ZeRO-3 + CPU offload (single GPU)

To train a large model on one GPU, offload optimizer and parameter state to CPU. The key fields are `zero_stage: 3` with `offload_optimizer_device: cpu` and `offload_param_device: cpu`:

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
mixed_precision: 'no'
num_machines: 1
num_processes: 1
use_cpu: false
```

In a custom (non-Trainer) loop, detect ZeRO-3 to gate `generate()` during inference, and use Accelerate's `backward`:

```py
is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
```

```bash
accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
```

## Multi-GPU training with FSDP

Fully Sharded Data Parallel shards parameters, gradients, and optimizer states across data-parallel processes and can offload sharded params to CPU. Generate the config with `accelerate config --config_file fsdp_config.yaml`. The LoRA FSDP config for 8 GPUs:

```yml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
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
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
use_cpu: false
```

With PEFT + FSDP, `use_orig_params` must be `False` to realize GPU memory savings, which means trainable and non-trainable params must be wrapped separately. Use PEFT's `fsdp_auto_wrap_policy`, and switch the state-dict type to `FULL_STATE_DICT` before saving so the adapter reloads normally with `from_pretrained`:

```python
trainer.accelerator.print(f"{trainer.model}")
if model_args.use_peft_lora:
    # handle PEFT+FSDP case
    trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
```

Launch is identical in shape to the DeepSpeed launch, pointing at the FSDP config.

## FSDP + QLoRA

Trains a 70B model on 2×24GB GPUs (Answer.AI + bitsandbytes + HF). Requires the same versions as ZeRO-3+QLoRA, plus `fsdp_cpu_ram_efficient_loading=true`, `fsdp_use_orig_params=false`, and `fsdp_offload_params=true` (CPU offload). Without the accelerate launcher, instead `export FSDP_CPU_RAM_EFFICIENT_LOADING=true`. The config sets `fsdp_offload_params: true`, `mixed_precision: 'no'`, `num_processes: 2`. The training-code change and the `bnb_4bit_quant_storage_dtype` launch flag are identical to the ZeRO-3+QLoRA section. Memory: ~19.6 GB/GPU (with CPU offload) or ~35.6 GB/GPU (without).

## Distributed training caveats

From the FSDP guide:

1. Merging when using PEFT and FSDP is currently unsupported and will raise an error.
2. Passing the `modules_to_save` config parameter is untested at present.
3. GPU memory saving when using CPU offloading is untested at present.
4. When using FSDP+QLoRA, `paged_adamw_8bit` currently errors when saving a checkpoint.
5. DoRA training with FSDP works (slower than LoRA); QDoRA at 4-bit works, but 8-bit has known issues and is not recommended.

Transformers' Trainer handles the distributed plumbing automatically: for ZeRO-3 it passes `exclude_frozen_parameters=True` when saving (only adapter params are written); for FSDP it updates the auto-wrap policy and adjusts the QLoRA mixed-precision policy to the quantization storage dtype.

## Saving an adapter

Both methods save *only* the trained PEFT weights — typically just `adapter_config.json` + `adapter_model.safetensors` (often a few MB).

```py
model.save_pretrained("output_dir")
```

```python
from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("your-name/bigscience/mt0-large-lora")
```

Next step: the saved directory's `adapter_config.json` records `base_model_name_or_path`, which is what lets loaders re-attach the adapter to the right base model.

## Loading an adapter for inference (AutoPeftModel)

`AutoPeftModel*` classes read the adapter config and pull in the correct base model in one line. Pick the class matching the task:

- `AutoPeftModel` — base class, for any task (e.g. ASR) not covered below
- `AutoPeftModelForCausalLM`
- `AutoPeftModelForSeq2SeqLM`
- `AutoPeftModelForSequenceClassification`
- `AutoPeftModelForTokenClassification`
- `AutoPeftModelForQuestionAnswering`
- `AutoPeftModelForFeatureExtraction`

```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model = model.to("cuda")
model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
```

For an unsupported task type, fall back to the base class:

```py
from peft import AutoPeftModel

model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")
```

`from_pretrained` also accepts `adapter_name`, `is_trainable`, `config`, and `revision`.

## Transformers native PEFT integration

Every `PreTrainedModel` mixes in `PeftAdapterMixin`, so you can load/add/train/switch/delete adapters without wrapping in a `PeftModel`. Requires `peft >= 0.19.0`. All non-prompt-learning methods (LoRA, IA3, AdaLoRA) are supported; prompt-based methods require the PEFT library directly.

Attach a fresh adapter to a base model:

```py
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model.add_adapter(lora_config, adapter_name="my_adapter")
```

`from_pretrained` auto-detects an adapter (it reads `adapter_config.json` → `base_model_name_or_path`):

```py
from transformers import AutoModelForCausalLM

# Automatically loads the base model and attaches the adapter
model = AutoModelForCausalLM.from_pretrained("klcsp/gemma7b-lora-alpaca-11-v1")
```

Load an adapter onto an already-loaded model:

```py
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
model.load_adapter("klcsp/gemma7b-lora-alpaca-11-v1")
```

Load a quantized base (QLoRA-style inference) by passing a `BitsAndBytesConfig`:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "klcsp/gemma7b-lora-alpaca-11-v1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

## Managing multiple adapters

A model can hold several adapters at once; only the active one(s) contribute to the forward pass.

```py
from peft import LoraConfig

model.add_adapter(LoraConfig(r=8, lora_alpha=32), adapter_name="adapter_1")
model.add_adapter(LoraConfig(r=16, lora_alpha=64), adapter_name="adapter_2")
```

```py
model.set_adapter("adapter_2")          # activate one (others stay in memory, disabled)
model.disable_adapters()                # base-model inference
model.enable_adapters()                 # re-enable all attached adapters
model.active_adapters()                 # -> ["adapter_1"]
model.delete_adapter("adapter_1")       # free memory
```

## Hotswapping adapters

Loading a new adapter per request allocates memory and (under `torch.compile`) triggers recompilation. Hotswapping replaces weights in-place. LoRA only.

```py
model = AutoModel.from_pretrained(...)
model.load_adapter(adapter_path_1)
# ... generate with adapter 1 ...
model.load_adapter(adapter_path_2, hotswap=True, adapter_name="default")
# ... generate with adapter 2 ...
```

For compiled models, call `enable_peft_hotswap` *before* loading the first adapter and *before* compiling; `target_rank` must be the max rank across all LoRAs you'll load:

```py
model = AutoModel.from_pretrained(...)
max_rank = ...  # highest rank among all LoRAs you'll load
model.enable_peft_hotswap(target_rank=max_rank)
model.load_adapter(adapter_path_1, adapter_name="default")
model = torch.compile(model, ...)
output_1 = model(...)

model.load_adapter(adapter_path_2, adapter_name="default")  # no recompilation
output_2 = model(...)
```

Load the adapter targeting the most layers first to avoid recompilation.

## Merging adapters (TIES / DARE / linear)

Combine several trained LoRA adapters into one new adapter with `add_weighted_adapter()`. The merge algorithm is chosen via `combination_type`, one of: `svd` (default), `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`. TIES/DARE variants also take a `density` (0–1, fraction of weights kept).

Load a base model, then load each adapter under a name:

```py
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

config = PeftConfig.from_pretrained("smangrul/tinyllama_lora_norobots")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots")

model.config.vocab_size = 32005
model.resize_token_embeddings(32005)

model = PeftModel.from_pretrained(model, "smangrul/tinyllama_lora_norobots", adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")
```

Merge with TIES (weights ≥ 1.0 preserve scale; 1.0 is a good default):

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
```

Or DARE-TIES:

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
```

Activate the merged adapter, then run inference:

```py
model.set_adapter("merge")
```

Notes: with `combination_type="cat"` the merged rank = sum of all adapter ranks (risk of OOM). When merging fully-trained models with TIES, watch for special tokens added at the same embedding index — use `resize_token_embeddings` to avoid collisions (not an issue for LoRA adapters from the same base model). (IA)³ models merge the same way but `add_weighted_adapter` takes no `combination_type`, and weights should sum to 1.0.

## Merging and unmerging in-place

To bake the active adapter into the base weights for a forward pass (and optionally undo it):

```python
model.merge_adapter()
# ... do whatever is needed ...
model.unmerge_adapter()
```

## Merging in distributed (ZeRO-3) settings

Under DeepSpeed ZeRO-3, parameters are sharded, so gather them with `deepspeed.zero.GatheredParameters` before merging:

```python
import deepspeed

is_ds_zero_3 = ... # check if Zero-3

with deepspeed.zero.GatheredParameters(list(model.parameters()), enabled= is_ds_zero_3):
    model.merge_adapter()
    # do whatever is needed, then unmerge in the same context if unmerging is required
    ...
    model.unmerge_adapter()
```

## Diffusers integration

Diffusers pipelines consume PEFT LoRA adapters directly. Load named adapters with `load_lora_weights`:

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
pipeline("bears, pizza bites").images[0]
```

Activate/blend multiple LoRAs by concatenating weighted matrices via `set_adapters`:

```py
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
```

Manage adapters:

```py
pipeline.set_adapters("feng")          # switch active LoRA
pipeline.disable_lora()                # disable all (kept on pipeline)
pipeline.get_active_adapters()         # -> ["cereal", "ikea"]
pipeline.get_list_adapters()           # -> {"unet": [...], "text_encoder_2": [...]}
pipeline.delete_adapters("ikea")       # remove a LoRA and its layers
```

For `torch.compile` or fastest inference, fuse the LoRA into the base weights and unload, then compile the UNet:

```py
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
pipeline.unload_lora_weights()
pipeline.save_pretrained("path/to/fused-pipeline")
```

`unfuse_lora()` restores base weights, but only when a single LoRA is fused (otherwise reload the model). For efficient merging across LoRAs of identical rank, Diffusers also exposes the PEFT `add_weighted_adapter` path (`combination_type` such as `"dare_linear"`); see the Model merging guide.
```

I grounded every code block in the actual PEFT / Transformers / Diffusers docs (fetched live) rather than the empty documentation block I was handed — the scrape you passed in contained only the PEFT landing-page navigation and the README quickstart, with no source for DeepSpeed/FSDP/save-load/merge/Diffusers. One deviation worth flagging: I did **not** include a `merge_and_unload()` example because no fetched page gave a verbatim one; I used `merge_adapter()`/`unmerge_adapter()` (which the DeepSpeed page documents) and `add_weighted_adapter` (the model-merging guide) instead.
