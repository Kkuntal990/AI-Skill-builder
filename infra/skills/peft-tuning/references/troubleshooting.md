# Troubleshooting

Known failure modes, error messages, and fixes — distilled from closed bug
reports, open issues with community traction, and stack traces from real user
reports.

## Installation & Environment

- **Symptom:** `AttributeError: 'PromptEncoderConfig' object has no attribute 'modules_to_save'` when calling `get_peft_model` with `PromptEncoderConfig(task_type="SEQ_CLS")`.
  **Fix:** Downgrade to `peft<=0.14.0` or upgrade past the regression fix; the attribute was accidentally dropped in 0.15.0. ([#2477](https://github.com/huggingface/peft/issues/2477))

- **Symptom:** `copy.deepcopy` of a `LoraModel` produces a copy whose `PeftConfig` fields revert to defaults (e.g. `r=8`) instead of the configured values.
  **Fix:** Re-apply the config explicitly after deepcopy: `model_copy = get_peft_model(model_copy, peft_config)`. This was a missing `__deepcopy__` implementation, fixed in later releases. ([#424](https://github.com/huggingface/peft/issues/424))

## CUDA / GPU Memory

- **Symptom:** BOFT forward pass and merging produce all-zero outputs when running on a CUDA device with the `fbd_cuda` kernel.
  **Fix:** Disable the CUDA fast-butterfly kernel by falling back to the pure-PyTorch BOFT implementation until the upstream fix is applied. ([#2219](https://github.com/huggingface/peft/issues/2219))

- **Symptom:** `torch._dynamo.exc.Unsupported: Data-dependent branching` during `model.generate()` with Gemma models on GPU.
  **Fix:** Pin `torch<=2.6` or disable `torch.compile` / dynamo for the generate call; the branching in Gemma's attention is not yet supported by the dynamo tracer used in torch 2.7.x. ([#2627](https://github.com/huggingface/peft/issues/2627))

- **Symptom:** P-tuning / prefix-tuning with FSDP and CPU offload raises:
  ```
  RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cpu and cuda:0!
  (when checking argument for argument tensors in method wrapper_CUDA_cat)
  ```
  **Fix:** Disable CPU offloading in the FSDP config when using `p_tuning`; prefix attention masks are not automatically moved to the model device in this path. ([#499](https://github.com/huggingface/peft/issues/499))

## Data & Tokenization

- **Symptom:** Prefix tuning on a quantized (8-bit) model (e.g. `codellama-7b`) fails at runtime with a dtype or device mismatch error.
  **Fix:** Use LoRA, IA³, or prompt tuning instead; prefix tuning is not compatible with bitsandbytes 8-bit quantization. ([#2035](https://github.com/huggingface/peft/issues/2035))

## Training Errors

- **Symptom:** LoRA merge with DeepSpeed ZeRO-3 raises:
  ```
  RuntimeError: The size of tensor a (0) must match the size of tensor b (2048)
  at non-singleton dimension 1
  ```
  **Fix:** Gather all sharded parameters before merging — call `model = accelerator.unwrap_model(model)` and ensure ZeRO-3 parameter consolidation is complete before invoking `merge_and_unload()`. ZeRO-2 is unaffected. ([#297](https://github.com/huggingface/peft/issues/297))

- **Symptom:** `AdaLoRA` training raises `TypeError: peft.tuners.adalora.AdaLoraLayer.update_layer.<locals>.lora_dropout_layer is not a Module subclass` when `lora_dropout=0`.
  **Fix:** Set `lora_dropout=0.0` (float) instead of `0` (int), or use any small positive float. The zero-integer path skipped wrapping in `nn.Identity`, producing a non-Module callable. ([#730](https://github.com/huggingface/peft/issues/730))

- **Symptom:** PiSSA or OLoRA initialization silently updates base model weights when used alongside quantization, causing unexpected model behaviour.
  **Fix:** Do not combine PiSSA/OLoRA init with `load_in_4bit` or `load_in_8bit`; dequantization during SVD decomposition writes back to the base weights. Use full-precision base models for these init methods. ([#1999](https://github.com/huggingface/peft/issues/1999))

- **Symptom:** PiSSA adapter saves and appears to train correctly, but the base model weights are mutated in-place, breaking the separation between adapter and base.
  **Fix:** Call `peft_model.save_pretrained()` with `save_embedding_layers=True` and verify base model weights are frozen before training; upgrade to peft>=0.14.0 where the silent mutation is guarded. ([#2184](https://github.com/huggingface/peft/issues/2184))

- **Symptom:** Applying two `LoraConfig` instances with `target_modules='all-linear'` to the same model produces nested LoRA layers (LoRA wrapping LoRA).
  **Fix:** Explicitly list target module names instead of using `'all-linear'` when composing multiple adapters, or add the second adapter via `model.add_adapter()` after the first is fully initialised. ([#2390](https://github.com/huggingface/peft/issues/2390))

- **Symptom:** Deleting an adapter on a model that has `modules_to_save` leaves the model in a broken state (wrong active module or KeyError).
  **Fix:** Upgrade to the patched release; as a workaround, manually reset `module.active_adapter` on each `ModulesToSaveWrapper` after deletion. ([#2381](https://github.com/huggingface/peft/issues/2381))

- **Symptom:** `merge_and_unload()` produces a checkpoint far smaller than the base model; reloading raises `Cannot copy out of meta tensor`.
  **Fix:** Ensure the base model is fully loaded (not on the `meta` device) before merging. Avoid calling `merge_and_unload()` on models loaded with `device_map='auto'` and mixed meta shards; load with `device_map='cpu'` first. ([#868](https://github.com/huggingface/peft/issues/868))

- **Symptom:** `past_key_values` passed to a `PeftModelForCausalLM` forward call raises a `KeyError` or is silently ignored during prefix tuning inference.
  **Fix:** Upgrade to peft>=0.13.0 where the `past_key_values` presence check was added to `PeftModelForCausalLM.forward()`; on older versions, pass `past_key_values` only through the base model directly. ([#1938](https://github.com/huggingface/peft/issues/1938))

- **Symptom:** `disable_adapter_layers()` does not restore original (pre-LoRA) logits when `modules_to_save` is also configured; the saved module remains active.
  **Fix:** After calling `disable_adapter_layers()`, also iterate `modules_to_save` wrappers and set `module.enable_adapters(False)` explicitly. Fixed in later peft releases. ([#493](https://github.com/huggingface/peft/issues/493))

- **Symptom:** Loading a model with `modules_to_save` and then disabling adapters raises an error or returns incorrect outputs because the original module is not properly restored.
  **Fix:** Upgrade to peft>=0.8.0 where `ModulesToSaveWrapper` correctly tracks and restores the original module on adapter disable. ([#850](https://github.com/huggingface/peft/issues/850))

- **Symptom:** `modules_to_save` layers raise errors or behave incorrectly when the base model is loaded with `load_in_4bit=True` or `load_in_8bit=True`.
  **Fix:** Exclude quantized layers from `modules_to_save`; bitsandbytes quantized layers cannot be copied and re-wrapped by PEFT's save mechanism. Use LoRA on those layers instead. ([#602](https://github.com/huggingface/peft/issues/602))

- **Symptom:** LoHa merge tests fail on Windows with CPU-only PyTorch builds (assertion or shape errors in the merge path).
  **Fix:** Run LoHa merging on Linux or with a CUDA-enabled build; the Windows/CPU code path has known numerical issues that were deprioritised. ([#1194](https://github.com/huggingface/peft/issues/1194))

## API / Version

- **Symptom:** After wrapping a model in `PeftModel`, `model.state_dict()` returns keys prefixed with `base_model.model.*` rather than the original model's key names, breaking checkpoint compatibility.
  **Known issue:** No built-in utility exists to recover original-key state dicts from a `PeftModel`. Track the upstream feature request for `get_base_model_state_dict()`. ([#2945](https://github.com/huggingface/peft/issues/2945) — open)

- **Symptom:** LoRA fine-tuning on MPS (Apple Silicon) is not possible with quantized models because no supported quantization backend targets MPS.
  **Known issue:** `optimum-quanto` MPS support is not yet integrated into PEFT. ([#1997](https://github.com/huggingface/peft/issues/1997) — open)
