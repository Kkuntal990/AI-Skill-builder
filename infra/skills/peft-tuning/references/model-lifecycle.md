# Model Lifecycle: Loading, Saving, Merging, and Swapping Adapters

How to load and save PEFT adapters, inject them into arbitrary `torch` modules, merge multiple adapters or fold them into base weights, combine mixed adapter types for inference, and hotswap adapter weights in place — plus the on-disk checkpoint format that underpins all of it.

## Contents

- Loading a trained adapter for inference (AutoPeftModel)
- Saving an adapter checkpoint
- Loading onto an explicit base model (PeftModel.from_pretrained)
- Adapter injection into any torch module
- Saving and loading injected adapters via state_dict
- Merging an adapter into the base weights (merge_and_unload)
- Merging multiple LoRA adapters (add_weighted_adapter, TIES/DARE)
- Mixed adapter types (PeftMixedModel)
- Hotswapping adapters
- Hotswapping under torch.compile
- Checkpoint format on disk
- Storing the whole model

## Loading a trained adapter for inference (AutoPeftModel)

`AutoPeftModelFor*` infers the task type from `adapter_config.json` and loads base + adapter in one line. Use it whenever you have a saved/hub adapter and just want to run inference.

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

For a task with no dedicated `AutoPeftModelFor*` class (e.g. ASR), fall back to the generic `AutoPeftModel`:

```py
from peft import AutoPeftModel

model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")
```

Available classes: `AutoPeftModel`, `AutoPeftModelForCausalLM`, `AutoPeftModelForSeq2SeqLM`, `AutoPeftModelForSequenceClassification`, `AutoPeftModelForTokenClassification`, `AutoPeftModelForQuestionAnswering`, `AutoPeftModelForFeatureExtraction`. The `from_pretrained` signature accepts `adapter_name="default"`, `is_trainable=False`, `config=None`, `revision=None`, and forwards remaining `**kwargs` (e.g. `device_map`) to the config and Hub methods.

## Saving an adapter checkpoint

After training a `PeftModel`, `save_pretrained` writes only the trained adapter weights — typically a few MB, not the full model. Next step is usually reloading with `AutoPeftModelFor*` (above) or pushing to the Hub.

```py
model.save_pretrained("output_dir")
```

```python
from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("your-name/bigscience/mt0-large-lora")
```

Both methods save only the extra PEFT weights, so storing/transferring is cheap — a LoRA adapter on `facebook/opt-350m` is just `adapter_config.json` + a ~6.3MB `adapter_model.safetensors`.

## Loading onto an explicit base model (PeftModel.from_pretrained)

When you already hold a base model in memory (e.g. quantized, custom `device_map`), attach the adapter explicitly instead of letting `AutoPeftModel` reload the base. The first adapter loaded is named `"default"`; load more with `load_adapter` and a distinct `adapter_name`.

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

With multiple adapters loaded, switch the active one with `model.set_adapter("sql")` before generating. The `resize_token_embeddings` call above is only needed when adapters added vocabulary tokens.

## Adapter injection into any torch module

Use `inject_adapter_in_model` when you want adapters on a plain `torch.nn.Module` without the `PeftModel` wrapper — it modifies the model in place, preserving its original attributes and methods. Trade-off: you lose `PeftModel` conveniences (disable/merge) and must write your own save/load. Works for all adapters except prompt-learning methods.

```python
import torch
from peft import inject_adapter_in_model, LoraConfig

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear"],
)

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_outputs = model(dummy_inputs)
```

Call `inject_adapter_in_model` again with a different `adapter_name` to attach more adapters. After injection, `print(model)` shows the `lora_A`/`lora_B` `ModuleDict`s nested under the targeted `linear` layer.

When you have a checkpoint but not its config, pass the loaded `state_dict` so PEFT infers which layers to target — leave `target_modules=None`:

```python
from safetensors.torch import load_file

model = ...
state_dict = load_file()
lora_config = LoraConfig(...)
model = inject_adapter_in_model(lora_config, model, state_dict=state_dict)
```

This only creates the (uninitialized) layers; weights are populated separately via `set_peft_model_state_dict` (below). Injecting from a `state_dict` does **not** work if the original adapter used `target_parameters` — supply the correct config in that case.

## Saving and loading injected adapters via state_dict

Because injected models have no `save_pretrained`, extract just the adapter weights with `get_peft_model_state_dict` (vs. `model.state_dict()`, which returns everything).

```python
from peft import get_peft_model_state_dict

peft_state_dict = get_peft_model_state_dict(model)
print(peft_state_dict)
```

Re-apply the saved weights after re-injecting the layers; check `outcome.unexpected_keys` to confirm a clean load.

```python
from peft import set_peft_model_state_dict

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)
outcome = set_peft_model_state_dict(model, peft_state_dict)
# check that there were no wrong keys
print(outcome.unexpected_keys)
```

For large or numerous adapters, build empty weights on the meta device and fill them only at load time by passing `low_cpu_mem_usage=True` to both calls:

```python
model = DummyModel()
model = inject_adapter_in_model(lora_config, model, low_cpu_mem_usage=True)

print(model.linear.lora_A["default"].weight.device.type == "meta")  # should be True
set_peft_model_state_dict(model, peft_state_dict, low_cpu_mem_usage=True)
print(model.linear.lora_A["default"].weight.device.type == "cpu")  # should be True
```

## Merging an adapter into the base weights (merge_and_unload)

To bake a single adapter into the base model for faster, dependency-free inference, call `merge_and_unload`. The result is a plain model — next step is `save_pretrained`.

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained(...)
```

Caveats: you lose all PEFT functionality (cannot unmerge, disable, or hold multiple adapters), the saved model is full-size, and not every method/quantization setting supports merging. If you need to keep merging reversible, use the non-destructive `merge_adapter()` / `unmerge_adapter()` pair instead.

## Merging multiple LoRA adapters (add_weighted_adapter, TIES/DARE)

To combine several trained LoRA adapters into one new adapter, load each (see `load_adapter` above) then call `add_weighted_adapter`. The merge algorithm is chosen by `combination_type`; `density` controls how many weights to keep for the pruning-based methods (TIES, DARE).

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
```

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
```

Then activate the merged adapter before generating:

```py
model.set_adapter("merge")
```

`combination_type` accepts `svd` (default), `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`. Notes:
- Weights may be positive or negative (negative subtracts an adapter's effect); values `> 1.0` often preserve scale better, and `1.0` is a good default.
- `cat` makes the merged rank the **sum** of all input ranks — it can OOM.
- `density` (0–1) applies only to the `ties`/`dare`/`magnitude_prune` family.
- `majority_sign_method` (`"total"` or `"frequency"`) applies to the sign-based variants.

(IA)³ models also expose `add_weighted_adapter`, but **without** `combination_type` (linear only); weights should sum to 1.0:

```py
adapters = ["adapter1", "adapter2", "adapter3"]
weights = [0.4, 0.3, 0.3]
adapter_name = "merge"
model.add_weighted_adapter(adapters, weights, adapter_name)
```

## Mixed adapter types (PeftMixedModel)

Normal `PeftModel` cannot combine *different* adapter types (e.g. LoRA + LoHa). `PeftMixedModel` can, for inference. Activate all desired adapters with a list passed to `set_adapter` — otherwise only the first is active.

```py
from peft import PeftMixedModel

base_model = ...  # load the base model, e.g. from transformers
# load first adapter, which will be called "default"
peft_model = PeftMixedModel.from_pretrained(base_model, )
peft_model.load_adapter(, adapter_name="other")
peft_model.set_adapter(["default", "other"])
```

Caveats: `PeftMixedModel` does **not** support saving/loading the mixed combination — adapters must already be trained, and you re-run this script each time. Only compatible types can be combined (see `peft.tuners.mixed.COMPATIBLE_TUNER_TYPES`; incompatible combinations raise). For many adapters, add same-type adapters consecutively (e.g. LoRA1, LoRA2, LoHa1, LoHa2) for best performance.

## Hotswapping adapters

`hotswap_adapter` replaces a loaded adapter's weights **in place**, keeping the adapter name. It's faster than delete-then-load and avoids recompilation when the model is `torch.compile`d.

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter

model_id = ...
inputs = ...
device = ...
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, )
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, , adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

Caveats: only LoRA is supported, both adapters must be the same PEFT method, and the incoming adapter must target the same layers (or a subset) as the current one — so load the widest-targeting adapter first. For a pre-mapped low-level swap, `hotswap_adapter_from_state_dict(model, state_dict, adapter_name, config, parameter_prefix="lora_")` is available.

## Hotswapping under torch.compile

When the model is compiled and the incoming adapter has a different rank/scaling, call `prepare_model_for_compiled_hotswap` with the maximum rank **before** compiling so the swap doesn't trigger recompilation. Skip it if all ranks/scalings are identical.

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap

model_id = ...
inputs = ...
device = ...
max_rank = ...  # maximum rank among all LoRA adapters that will be used
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, )
# Prepare the model to allow hotswapping even if ranks/scalings of 2nd adapter differ.
# You can skip this step if all ranks and scalings are identical.
prepare_model_for_compiled_hotswap(model, target_rank=max_rank)
model = torch.compile(model)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, , adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

## Checkpoint format on disk

`save_pretrained` writes three files: `adapter_model.safetensors` (or `.bin`), `adapter_config.json`, and a `README.md` model card (the card is not required to load). The `state_dict` holds **only** adapter parameters, not the base model — so you always need the base model available to load. An IA³ adapter on BERT is ~260KB vs. the ~420MB base.

`adapter_config.json` records the method type, base model, and config needed to rebuild the layers. Minimal LoRA example:

```json
{
  "target_modules": ["query", "value"],
  "peft_type": "LORA"
}
```

Weight keys are prefixed `base_model.model.` (the `PeftModel` → tuner-model wrapping), and LoRA stores its low-rank matrices as `...lora_A.weight` / `...lora_B.weight`:

- `base_model.model.encoder.layer.0.attention.self.query.lora_A.weight`
- `base_model.model.encoder.layer.0.attention.self.query.lora_B.weight`
- `base_model.model.encoder.layer.0.attention.self.value.lora_A.weight`

In a *live* model the keys also carry the adapter name (`...lora_A.default.weight`); `save_pretrained` strips that name, since it's arbitrary on reload. If the adapter name isn't `"default"`, it is saved into a subdirectory of that name (e.g. `some/path/other`). Prefix-tuning methods are the exception — their embeddings are stored without the `base_model.model.` prefix.

## Storing the whole model

When the base model won't be available to whoever loads the adapter, persist everything. Easiest path is to merge (see `merge_and_unload` above). Alternatively, for LoRA on a Transformers base, inject the weights and trick Transformers into saving the full model:

```python
model = ...  # the PEFT model
...
# after you finish training the model, save it in a temporary location
model.save_pretrained()
# now load this model directly into a transformers model, without the PEFT wrapper
# the PEFT weights are directly injected into the base model
model_loaded = AutoModel.from_pretrained()
# now make the loaded model believe that it is _not_ a PEFT model
model_loaded._hf_peft_config_loaded = False
# now when we save it, it will save the whole model
model_loaded.save_pretrained()
# or upload to Hugging Face Hub
model_loaded.push_to_hub()
```

This only works with LoRA (other adapter types aren't implemented inside Transformers). The result is a standard, full-size model with no PEFT functionality.
