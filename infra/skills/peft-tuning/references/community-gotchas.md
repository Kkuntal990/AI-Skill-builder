# Community Gotchas

Curated user-reported issues with fixes, sourced from Stack Overflow (CC BY-SA) and the package's GitHub issues. Each item links back to the original; consult the source for context not summarized here.

## LoRA configuration / target modules

- **Symptom:** `ValueError: Target modules ['query_key_value', 'dense', ...] not found in the base model` when applying LoRA to a model like Llama-2.
  **Fix:** Target module names are architecture-specific. Print the loaded model (`print(model)`) to inspect layer names, then pass the correct names (e.g. `["q_proj", "v_proj"]` for Llama) to `LoraConfig(target_modules=...)`.
  **Source:** [Target modules for applying PEFT / LoRA on different models (Stack Overflow Q#76768226)](https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models) — CC BY-SA

- **Symptom:** `ValueError: Target modules ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'] not found in the base model` when loading Llama-2-7b with QLoRA.
  **Fix:** Those module names belong to Falcon/BLOOM architectures, not Llama. Load the model and inspect it with `print(model)` to find the correct linear layer names before setting `target_modules`.
  **Source:** [Llama QLora error: Target modules not found in the base model (Stack Overflow Q#76736361)](https://stackoverflow.com/questions/76736361/llama-qlora-error-target-modules-query-key-value-dense-dense-h-to-4h) — CC BY-SA

- **Symptom:** `ValueError: You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided ['label']` when training a LoRA model for sequence classification.
  **Fix:** PEFT's LoRA model expects the column name `labels` (plural), not `label`. Rename the column with `dataset.rename_column('label', 'labels')` and ensure `task_type=TaskType.SEQ_CLS` is set in `LoraConfig`.
  **Source:** [How to resolve ValueError: You should supply an encoding... (Stack Overflow Q#78031519)](https://stackoverflow.com/questions/78031519/how-to-resolve-valueerror-you-should-supply-an-encoding-or-a-list-of-encodings) — CC BY-SA

- **Symptom:** Newly added task-specific heads (e.g. classification heads) are unexpectedly frozen or unexpectedly trained when using PEFT, causing confusion about which parameters are updated.
  **Fix:** PEFT freezes the base model backbone but does not freeze newly added layers such as classification heads; those remain trainable by default. Verify with `for name, param in model.named_parameters(): print(name, param.requires_grad)`.
  **Source:** [Does peft train newly initialized weights? (Stack Overflow Q#79221146)](https://stackoverflow.com/questions/79221146/does-peft-train-newly-initialized-weights) — CC BY-SA

## Model loading and merging

- **Symptom:** Loading a fine-tuned PEFT/LoRA model with `AutoModelForCausalLM.from_pretrained("finetuned_model")` results in the process being killed (OOM) or raises errors.
  **Fix:** Load the base model separately, then wrap it with `PeftModel`: load the base with `AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map={"": 0})`, then call `PeftModel.from_pretrained(m, adapter_name)`.
  **Source:** [How to load a fine-tuned peft/lora model based on llama with Huggingface transformers? (Stack Overflow Q#76459034)](https://stackoverflow.com/questions/76459034/how-to-load-a-fine-tuned-peft-lora-model-based-on-llama-with-huggingface-transfo) — CC BY-SA

- **Symptom:** Attempting to merge a LoRA adapter into the base model using `AutoModel` from `transformers` fails or produces incorrect results.
  **Fix:** Use `AutoPeftModelForCausalLM.from_pretrained(adapter_path)` from `peft`, then call `.merge_and_unload()` to obtain a standard `transformers` model with weights merged.
  **Source:** [I want to merge my PEFT adapter model with the base model (Stack Overflow Q#77596271)](https://stackoverflow.com/questions/77596271/i-want-to-merge-my-peft-adapter-model-with-the-base-model-and-make-a-fully-new-m) — CC BY-SA

## API breakage / version mismatch

- **Symptom:** `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` after upgrading `trl` to 0.12.0 or later.
  **Fix:** Replace the `tokenizer` argument with `processing_class` in the `SFTTrainer` constructor: `SFTTrainer(..., processing_class=tokenizer)`.
  **Source:** [TypeError in SFTTrainer Initialization: Unexpected Keyword Argument 'tokenizer' (Stack Overflow Q#79546910)](https://stackoverflow.com/questions/79546910/typeerror-in-sfttrainer-initialization-unexpected-keyword-argument-tokenizer) — CC BY-SA

- **Symptom:** `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'` when passing training arguments to `SFTTrainer`.
  **Fix:** Move `dataset_text_field`, `packing`, and `max_seq_length` out of `TrainingArguments` and into `SFTConfig` instead. Pass an `SFTConfig` instance as the `args` argument to `SFTTrainer`.
  **Source:** [TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field' (Stack Overflow Q#79509805)](https://stackoverflow.com/questions/79509805/typeerror-sfttrainer-init-got-an-unexpected-keyword-argument-dataset-tex) — CC BY-SA

- **Symptom:** `AttributeError: 'TrainingArguments' object has no attribute 'model_init_kwargs'` when using `SFTTrainer` with a standard `TrainingArguments` object.
  **Fix:** Replace `TrainingArguments` with `SFTConfig` from `trl` (`from trl import SFTConfig`) and pass it to `SFTTrainer(args=training_arguments, ...)`.
  **Source:** [AttributeError: 'TrainingArguments' object has no attribute 'model_init_kwargs' (Stack Overflow Q#78575305)](https://stackoverflow.com/questions/78575305/attributeerror-trainingarguments-object-has-no-attribute-model-init-kwargs) — CC BY-SA

- **Symptom:** `AttributeError: 'torch.dtype' object has no attribute 'itemsize'` during PEFT/LoRA training with gradient checkpointing enabled.
  **Fix:** Upgrade `transformers`, `torch`, and `accelerate` to their latest versions (`pip install --upgrade transformers torch accelerate`). The error is caused by a version incompatibility between these libraries.
  **Source:** [PyTorch: AttributeError: 'torch.dtype' object has no attribute 'itemsize' (Stack Overflow Q#78211526)](https://stackoverflow.com/questions/78211526/pytorch-attributeerror-torch-dtype-object-has-no-attribute-itemsize) — CC BY-SA

## Multi-GPU / training behavior

- **Symptom:** `trainer.predict` only uses a single GPU despite `Seq2SeqTrainingArguments` reporting 8 GPUs; the PEFT model device shows `cpu` after loading.
  **Status:** Unresolved
  **Source:** [peft#607](https://github.com/huggingface/peft/issues/607)

- **Symptom:** LoRA fine-tuning consistently yields 4–6% lower task performance compared to full fine-tuning on instruction-tuning datasets across 6B–40B parameter models.
  **Status:** Unresolved
  **Source:** [peft#622](https://github.com/huggingface/peft/issues/622)
