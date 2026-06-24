# Prompt-Based and Reparameterization PEFT Methods

The soft-prompt family (prompt tuning, prefix tuning, P-tuning, multitask prompt tuning) plus the IA3 activation-rescaling tuner and the OFT/BOFT orthogonal-transformation tuners — what each one trains, the exact config to instantiate, and when to reach for it.

## Contents

- When to reach for each method
- Shared workflow
- Prompt tuning
- Prefix tuning
- P-tuning
- Multitask prompt tuning
- IA3
- OFT (Orthogonal Finetuning)
- BOFT (Butterfly OFT)
- Choosing block size for OFT/BOFT
- Loading and inference

## When to reach for each method

| Method | Config class | What it trains | Designed for |
|---|---|---|---|
| Prompt tuning | `PromptTuningConfig` | Soft prompt added to **input embeddings only** | Text classification cast as generation; scales with model size |
| Prefix tuning | `PrefixTuningConfig` | Prefix vectors in **all model layers**, via a discarded FFN | Natural language **generation** (NLG) on GPT-style models |
| P-tuning | `PromptEncoderConfig` | Soft prompt (insertable anywhere) optimized by an LSTM/MLP encoder | Natural language **understanding** (NLU), all LMs |
| Multitask prompt tuning | `MultitaskPromptTuningConfig` | One shared transferable prompt + per-task low-rank updates | Transfer across many tasks |
| IA3 | `IA3Config` | Three learned vectors that **rescale** K, V, and FFN activations | Few-shot; tiny param count, supports weight merge |
| OFT | `OFTConfig` | An orthogonal matrix that **multiplicatively** transforms weights | Preserving pretraining knowledge; mergeable |
| BOFT | `BOFTConfig` | Orthogonal transform factorized via butterfly matrices | More parameter-efficient generalization of OFT |

The three soft-prompt methods differ only in *where* the learnable tensor is injected and *how* it is reparameterized. The trainable-parameter counts below (all on `bigscience/bloomz-560m`) make the trade-off concrete:

```
prompt tuning : trainable params: 8,192       || trainable%: 0.0015%
P-tuning      : trainable params: 300,288     || trainable%: 0.0537%
prefix tuning : trainable params: 983,040     || trainable%: 0.1755%
```

## Shared workflow

Every method here follows the same four steps (illustrated with OFT, but identical in shape for all):

1. Instantiate a base model.
2. Create a configuration (`OFTConfig`, `IA3Config`, `PromptTuningConfig`, …) where you define the method-specific parameters.
3. Wrap the base model with `get_peft_model()` to get a trainable `PeftModel`.
4. Train the `PeftModel` as you normally would train the base model.

After wrapping, always confirm how little you are training:

```py
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

The standard next step is an ordinary optimizer + training loop on the frozen-base `PeftModel` — no special handling required:

```py
from transformers import get_linear_schedule_with_warmup

lr = 3e-2
num_epochs = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

## Prompt tuning

Prompt tuning casts all tasks as *generation* and adds a task-specific soft prompt to the input that is updated independently of the frozen base. It is the cheapest method here (only the prompt embeddings train).

Initialize the prompt from real text for the best results — set `num_virtual_tokens` to the token count of the init text so the prompt has the same length as what should be predicted:

```py
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

What it does: learns 8,192 prompt-embedding parameters; the base model stays frozen.
Typical next step: run the shared optimizer + training loop above.

`prompt_tuning_init` controls initialization:
- `TEXT` — initialize with `prompt_tuning_init_text` (requires `tokenizer_name_or_path`; pass extra args via `tokenizer_kwargs`).
- `SAMPLE_VOCAB` — initialize with tokens randomly sampled from the model's vocabulary.
- `RANDOM` — initialize with random continuous soft tokens (these may fall outside the embedding manifold).

For seq2seq bases, additional shape fields are available:

```py
>>> from peft import PromptEmbedding, PromptTuningConfig

>>> config = PromptTuningConfig(
...     peft_type="PROMPT_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     prompt_tuning_init="TEXT",
...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
...     tokenizer_name_or_path="t5-base",
... )
```

## Prefix tuning

Prefix tuning is like prompt tuning but inserts task-specific vectors into **all** model layers, not just the input embeddings. The prefix is optimized through a separate feed-forward network (FFN) — training the soft prompts directly causes instability — and the FFN is discarded once the prompts are updated. It matches full finetuning with ~1000× fewer parameters and excels in low-data settings.

```py
from peft import PrefixTuningConfig, get_peft_model

peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"
```

What it does: adds 20 trainable prefix vectors per layer (983,040 params here — more than prompt tuning because every layer gets a prefix).
Typical next step: train; prefer prefix tuning over prompt tuning for generation tasks and small datasets.

## P-tuning

P-tuning adds a trainable embedding tensor that can be inserted **anywhere** in the input sequence (not just the front) and optimizes it with a prompt encoder (a bidirectional LSTM, or MLP). The prompt is added only to the input, not to every layer. Anchor tokens can further improve results. It lets GPT-like models compete with BERT-like models on NLU tasks.

Set `encoder_hidden_size` to size the encoder that learns the prompt parameters:

```py
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```

What it does: trains a 128-hidden-size prompt encoder producing 20 virtual tokens (300,288 params).
Typical next step: train; reach for P-tuning on classification/NLU rather than free-form generation.

## Multitask prompt tuning

Multitask prompt tuning (MPT) decomposes the soft prompts of many tasks into a **single transferable prompt**, then adapts it per target task via multiplicative low-rank updates. Source training distills task-specific source prompts into one shared prompt; target adaptation initializes a target prompt as the Hadamard product of the shared matrix and a task-specific low-rank matrix.

`MultitaskPromptTuningConfig` fields:

| Field | Type | Default |
|---|---|---|
| `prompt_tuning_init` | `Union[MultitaskPromptTuningInit, str]` | `MultitaskPromptTuningInit.RANDOM` |
| `prompt_tuning_init_state_dict_path` | `Optional[str]` | `None` |
| `prompt_tuning_init_task` | `Optional[int]` | `0` |
| `num_ranks` | `Optional[int]` | `1` |
| `num_tasks` | `Optional[int]` | `1` |

`MultitaskPromptTuningInit` controls how the shared prompt is seeded, and is the lever that links the two MPT stages:

- `TEXT`
- `RANDOM`
- `AVERAGE_SOURCE_TASKS`
- `EXACT_SOURCE_TASK`
- `ONLY_SOURCE_SHARED`

Typical use: source-train with `RANDOM`/`TEXT` across `num_tasks`, save the prompt state dict, then for target adaptation load it via `prompt_tuning_init_state_dict_path` with one of the `*_SOURCE_*` init modes plus `prompt_tuning_init_task`.

## IA3

IA3 ("Infused Adapter by Inhibiting and Amplifying Inner Activations") adds three learned vectors that **rescale** the keys and values of self-attention / encoder-decoder attention, and the intermediate activation of the position-wise FFN. It introduces a very small number of parameters and, unlike soft prompts, modifies activations directly — so weights remain mergeable.

```py
>>> from peft import IA3Model, IA3Config

>>> config = IA3Config(
...     peft_type="IA3",
...     task_type="SEQ_2_SEQ_LM",
...     target_modules=["k", "v", "w0"],
...     feedforward_modules=["w0"],
... )
```

Keep in mind when building an `IA3Config`:
- `target_modules` — modules to adapt. Accepts a regex string, a list of names (exact or suffix match), `"all-linear"` (all linear/Conv1D except the output layer), or `None` to infer from the architecture (errors on unknown architectures).
- `feedforward_modules` — modules treated as feedforward; their IA3 vectors multiply the module **input** instead of its output. Must be a subset of `target_modules`.
- `exclude_modules` — names to skip (regex or list).
- `fan_in_fan_out` — set `True` for layers storing weights as `(fan_in, fan_out)`, e.g. GPT-2's `Conv1D`.
- `modules_to_save` — extra modules (e.g. a randomly-initialized head) to train and save.
- `init_ia3_weights` — defaults to `True`; setting `False` is discouraged.

What it does: wraps `k`, `v`, and `w0` with rescaling vectors, treating `w0` as feedforward.
Typical next step: `model = get_peft_model(model, config)` (or pass `config` to `IA3Model(config, model)`), then train.

## OFT (Orthogonal Finetuning)

OFT represents the weight update as a multiplicative **orthogonal** transformation of the frozen pretrained weights. Because the transform is orthogonal, it keeps the model's hyperspherical energy unchanged during finetuning, which reduces forgetting of pretraining knowledge. Like LoRA, OFT weights can be folded back with `merge_and_unload()`.

Configure OFT by its block structure (works directly, and composes with quantized TRL training):

```py
from peft import OFTConfig

# Configure OFT
peft_config = OFTConfig(
    oft_block_size=32,
    use_cayley_neumann=True,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM"
)
```

`OFTConfig` parameters:
- `r` — OFT rank (number of blocks per layer). **Bigger `r` → sparser updates, fewer trainable params.** Specify either `r` or `oft_block_size`, never both (`r × oft_block_size = layer dimension`). Default `r = 0`; prefer setting `oft_block_size`.
- `oft_block_size` — block size. **Bigger → denser updates, more params.** Choose a value dividing `in_features` (e.g. 4, 8, 16). Default `32`.
- `use_cayley_neumann` — `True` uses the efficient approximate Cayley-Neumann parameterization; `False` uses the exact (but matrix-inverse-costly) vanilla Cayley. Default `False`; test both.
- `module_dropout` — multiplicative dropout that sets OFT blocks to identity during training.
- `bias` — `"none"`, `"all"`, or `"oft_only"`.
- `target_modules` — modules (e.g. attention blocks) to inject OFT matrices into.
- `modules_to_save` — extra trainable/saved modules (e.g. a custom head).

What it does: applies a 32-sized orthogonal block transform to all linear layers with the efficient Cayley-Neumann parameterization.
Typical next step: `get_peft_model(model, peft_config)`, train, then `merge_and_unload()` for a standalone merged model. For quantized SFT/PPO/DPO, pass `peft_config` straight to a TRL `SFTTrainer`.

## BOFT (Butterfly OFT)

BOFT generalizes OFT with a **butterfly factorization** of the orthogonal matrix, giving a more compact yet expressive learning space and better parameter efficiency. OFT is the special case `boft_n_butterfly_factor=1`.

Initialize for, e.g., a DinoV2 image-classification backbone:

```py
from peft import BOFTConfig, get_peft_model

config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

boft_model = get_peft_model(model, config)
```

`BOFTConfig` parameters:
- `boft_block_size` — matrix block size. **Bigger → denser updates, more params.** Choose divisible by most layers' `in_features` (4, 8, 16).
- `boft_block_num` — number of blocks. **Bigger → sparser updates, fewer params.** Specify either `boft_block_size` or `boft_block_num`, never both and never both zero (`boft_block_size × boft_block_num = in_features`).
- `boft_n_butterfly_factor` — number of butterfly factors. `1` = vanilla OFT; `2` doubles the effective block size and halves the block count.
- `bias` — `"none"`, `"all"`, or `"boft_only"`.
- `boft_dropout` — multiplicative dropout probability.
- `target_modules` — modules to inject BOFT matrices into.
- `modules_to_save` — extra trainable/saved modules (here, the randomly-initialized `classifier` head).

What it does: injects block-size-4 butterfly-factorized orthogonal transforms into the listed attention/MLP modules while keeping the classifier head trainable.
Typical next step: train the `boft_model`; merge with `merge_and_unload()` when done.

## Choosing block size for OFT/BOFT

The block knobs are the main tuning surface and trade density against parameter count:

- For OFT, set exactly one of `r` or `oft_block_size`; the other is inferred since `r × oft_block_size = layer dimension`.
- For BOFT, set exactly one of `boft_block_size` or `boft_block_num` (neither both nor both-zero), since `boft_block_size × boft_block_num = in_features`.
- Bigger block **size** ⇒ denser update matrix ⇒ more trainable parameters; bigger block **count/rank** ⇒ sparser ⇒ fewer parameters.
- Pick block sizes that divide the layer's `in_features` (e.g. 4, 8, 16).

## Loading and inference

Every method above saves a small adapter that loads with the matching `AutoPeftModel*` class — load the adapter, attach the original tokenizer, and call `generate` as usual:

```py
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("peft_model_id").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
```

Then run generation normally:

```py
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
```

For OFT/BOFT (and IA3) you can instead call `merge_and_unload()` before inference to fold the adapter into the base weights and serve a single standalone model; soft-prompt methods cannot be merged because they add virtual tokens rather than transform existing weights.
```

A note on provenance: the documentation block passed in was empty (only HF site navigation chrome), so I fetched the real PEFT v0.19.0 docs — the prompt-based-methods task guide, the soft-prompts / IA3 / OFT-BOFT conceptual guides, and the per-method config API references. Every code block above is verbatim from those sources; no APIs were invented.
