# Prompt Tuning, Prefix Tuning, P-Tuning, and Soft Prompt Methods

Covers the conceptual foundation and practical usage of soft-prompt-based PEFT methods: prompt tuning, prefix tuning, P-tuning, multitask prompt tuning, and Llama-Adapter.

## What Are Soft Prompts

Hard prompts are hand-crafted text strings prepended to input — brittle and require manual effort. Soft prompts replace or augment them with **learnable continuous vectors** (embeddings) that are optimized during training while the base model weights remain frozen.

Key properties:
- Soft prompt tensors are the only trainable parameters
- They live in the model's embedding space but are not tied to real vocabulary tokens
- The same frozen base model can serve multiple tasks by swapping soft prompt weights

## Prompt Tuning

### Concept

Prompt tuning prepends a small set of trainable tokens to the **input embeddings** of the model. Only these tokens are updated during training; all transformer layers are frozen.

### Configuration

```python
from peft import PromptTuningConfig, PromptTuningInit, TaskType

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
```

- `num_virtual_tokens`: how many soft tokens to prepend
- `prompt_tuning_init=TEXT`: initializes soft tokens from real text embeddings (better than random)
- `prompt_tuning_init=RANDOM`: random initialization

### Applying and Training

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0015
```

Typical next step: pass `model` to a `Trainer` or custom training loop. Only the soft prompt embeddings accumulate gradients.

### Saving and Loading

```python
model.save_pretrained("my_prompt_tuning_weights")

# Reload
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
model = PeftModel.from_pretrained(model, "my_prompt_tuning_weights")
```

The saved directory contains only the soft prompt tensor, not the full model.

---

## Prefix Tuning

### Concept

Prefix tuning prepends trainable vectors to the **keys and values of every attention layer**, not just the input embeddings. This gives the soft prompt influence over every transformer layer's attention computation.

A small feed-forward reparameterization network is used during training to stabilize optimization; it is discarded at inference.

### Configuration

```python
from peft import PrefixTuningConfig, TaskType

config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30,
    prefix_projection=True,   # enables the reparameterization MLP
)
```

- `num_virtual_tokens`: prefix length prepended to each layer's K/V
- `prefix_projection=True`: recommended for training stability; set `False` to skip the MLP

### Applying

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("bigscience/mt0-large")
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Typical next step: use with sequence-to-sequence or causal LM training. Prefix tuning generally outperforms prompt tuning on harder tasks because it conditions every layer.

---

## P-Tuning

### Concept

P-tuning uses a small **LSTM or MLP encoder** to produce soft prompt embeddings that are inserted at arbitrary positions in the input sequence (not just the front). This allows more flexible prompt placement and is especially effective for NLU tasks with encoder models.

### Configuration

```python
from peft import PromptEncoderConfig, PromptEncoderReparameterizationType, TaskType

config = PromptEncoderConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1,
    encoder_num_layers=2,
    encoder_hidden_size=128,
)
```

- `encoder_reparameterization_type`: `MLP` or `LSTM`
- `encoder_hidden_size`: hidden size of the reparameterization network
- `num_virtual_tokens`: number of soft tokens produced

### Applying

```python
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Typical next step: fine-tune on a classification dataset. P-tuning is well-suited to GPT-style models on NLU benchmarks where prompt position matters.

---

## Multitask Prompt Tuning

### Concept

Multitask Prompt Tuning (MPT) learns a **single shared soft prompt** across multiple tasks, then derives task-specific prompts via multiplicative decomposition. This enables knowledge transfer between tasks while keeping per-task overhead minimal.

### Configuration

```python
from peft import MultitaskPromptTuningConfig, MultitaskPromptTuningInit, TaskType

config = MultitaskPromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=50,
    num_tasks=4,
    num_ranks=4,
    prompt_tuning_init=MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
)
```

- `num_tasks`: total number of tasks sharing the prompt
- `num_ranks`: rank of the task-specific decomposition matrices
- `prompt_tuning_init`: how to initialize — `RANDOM`, `TEXT`, `ONLY_SOURCE_SHARED`, `AVERAGE_SOURCE_TASKS`, `EXACT_SOURCE_TASK`, `TRANSFER_SOURCE_TASK`

### Applying

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("bigscience/mt0-large")
model = get_peft_model(model, config)
```

At training time, pass `task_ids` as an additional input so the model selects the correct task-specific prompt decomposition.

Typical next step: construct a multi-task dataloader that includes `task_ids` per batch, then train normally.

---

## Llama-Adapter

### Concept

Llama-Adapter injects **learnable adaptation prompts** into the upper transformer layers only (not all layers). A **zero-initialized attention gating mechanism** starts with zero contribution and gradually learns to incorporate the adapter signal — preventing early training instability.

This method was designed for Llama but applies to other causal LMs.

### Configuration

```python
from peft import AdaptionPromptConfig, TaskType

config = AdaptionPromptConfig(
    task_type=TaskType.CAUSAL_LM,
    adapter_len=10,       # number of adapter tokens per adapted layer
    adapter_layers=30,    # how many of the top layers receive adapters
)
```

- `adapter_len`: length of the learnable prompt inserted per layer
- `adapter_layers`: number of layers (counting from the top) that receive adaptation prompts

### Applying

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Typical next step: instruction fine-tuning on a small dataset. The zero-init gating means the model starts as the original frozen LLM and smoothly learns to use the adapters.

---

## Comparing the Methods

| Method | Where prompt lives | Layers affected | Reparameterization | Best for |
|---|---|---|---|---|
| Prompt Tuning | Input embeddings | First layer only | None | Simple classification, low resource |
| Prefix Tuning | K/V of attention | All layers | Optional MLP | Seq2seq, generation tasks |
| P-Tuning | Input embeddings (flexible position) | First layer only | MLP or LSTM | NLU with encoder models |
| Multitask Prompt Tuning | Input embeddings | First layer only | Decomposition matrices | Multi-task transfer learning |
| Llama-Adapter | Top-N attention layers | Top N layers only | Zero-init gating | Instruction tuning of LLMs |

---

## Shared Workflow Across All Soft Prompt Methods

All methods follow the same PEFT interface:

```python
# 1. Define config
config = PromptTuningConfig(...)   # or PrefixTuningConfig, etc.

# 2. Wrap model
model = get_peft_model(base_model, config)

# 3. Train — only soft prompt params have requires_grad=True
trainer.train()

# 4. Save only the soft prompt weights
model.save_pretrained("output_dir/")

# 5. Reload for inference
model = PeftModel.from_pretrained(base_model, "output_dir/")
```

Calling `model.print_trainable_parameters()` after `get_peft_model` confirms that only the soft prompt tensors are trainable — typically well under 1% of total parameters.

---

## Inference with Soft Prompts

```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("username/my-soft-prompt-model")
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "username/my-soft-prompt-model")
model.eval()

inputs = tokenizer("Is this a complaint?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

The soft prompt is automatically prepended to the input during the forward pass — no manual prompt engineering needed at inference time.
