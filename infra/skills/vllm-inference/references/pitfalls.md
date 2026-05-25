# Common Pitfalls

## Installation & Environment

- **Symptom:** Installing vllm with `cu129` torch backend attempts to open `libcudart.so.13` (a CUDA 13 library), causing startup failure. **Fix:** Match the vllm CUDA build variant exactly to your installed CUDA version (e.g., use `cu128` index for CUDA 12.x).
- **Symptom:** `vllm/vllm-openai:nightly` Docker image fails to start due to `pytest` being imported transitively through `humming`/`cupy`. **Fix:** Pin to a stable tagged image (e.g., `vllm/vllm-openai:v0.x.y`) instead of `nightly` for production deployments.
- **Symptom:** Distributed inference with InfiniBand/mlx5 fails with `undefined symbol: mlx5dv_get_data_direct_sysfs_path` in `libmlx5.so`. **Fix:** Upgrade the host `libibverbs`/`libmlx5` userspace drivers to match the kernel RDMA module version.

## Deadlocks & Crashes

- **Symptom:** `EngineDeadError` when serving Qwen3.5-4B under 2-way concurrent image requests. **Fix:** Upgrade to the patched vllm release that fixes multimodal concurrent request handling in the V1 engine.
- **Symptom:** `EngineCore` deadlock when serving Kimi-K2.5 on 8×H800. **Fix:** Upgrade to the vllm release that resolves the EngineCore scheduling deadlock for large MoE models on multi-GPU setups.
- **Symptom:** `AssertionError` on image reuse after calling sleep/wake on a V1 P0/P1 disaggregated setup due to desynced multimodal sender cache. **Fix:** Avoid reusing images across sleep/wake cycles until the cache-reset fix is applied; upgrade to the patched release.

## Model Loading & Runner Configuration

- **Symptom:** `--runner pooling --convert classify` crashes with `ValueError: missing score.weight` on Qwen3-Reranker-8B. **Fix:** Use `--runner pooling` without `--convert classify` for reranker models, or upgrade to the release that adds proper weight mapping for classify heads.
- **Symptom:** `--runner draft` or `--runner generate` is silently accepted for embedding models but crashes with an opaque `ValueError` during weight loading. **Fix:** Use `--runner pooling` for embedding/reranker models; do not pass generation-oriented runner flags to embedding models.
- **Symptom:** Gemma-4 fails to start on GPUs with less than 70 GB memory with an error about `max_num_batched_tokens < multimodal token size`. **Fix:** Pass `--max-num-batched-tokens <N>` explicitly with a value large enough to accommodate Gemma-4's vision token count (e.g., `--max-num-batched-tokens 16384`).

## Quantization

- **Symptom:** FP8 quantization on L20 GPUs with Qwen2-7B-Instruct produces garbled/corrupted output in vllm ≥0.19.0. **Fix:** Disable FP8 quantization on L20 GPUs until the per-tensor scale calibration bug is resolved, or pin to vllm 0.18.0.

## Streaming & Inference Correctness

- **Symptom:** Streaming reasoning tokens are truncated when `</think>` and `<tool_call>` appear in the same delta with Qwen3.5 + `--reasoning-parser qwen3` + tool calling. **Fix:** Use non-streaming inference as a workaround, or upgrade to the release that fixes the streaming reasoning parser flush logic.
- **Symptom:** Native Triton top-k/top-p kernel produces incorrect results when logits tensor is non-contiguous (e.g., on GH200). **Fix:** Call `.contiguous()` on logits before sampling, or upgrade to the release that adds a contiguity check in `apply_top_k_top_p_triton`.

## Performance

- **Symptom:** MoE models (e.g., Mixtral, DeepSeek) show significant throughput regression at low batch sizes since vllm v0.20.0. **Fix:** Pass `--use-v1` explicitly or upgrade to the patched release that restores the fused MoE kernel selection for small batches.
- **Symptom:** NIXL + FlashInfer fails with Qwen3 MRV2 when using `--block-size 128`. **Fix:** Use `--block-size 16` or `--block-size 32` with NIXL+FlashInfer until block-size-128 support is fixed.

## KV Cache & Connectors

- **Symptom:** `ExampleConnector.start_load_kv` hardcodes `.cuda()`, crashing on CPU-only deployments. **Fix:** Override `start_load_kv` in your connector subclass to use the appropriate device, or upgrade to the release that makes the device configurable.
- **Symptom:** `ExampleConnector` fails to inject KV cache on CPU with a device mismatch error. **Fix:** Ensure the connector's tensor device matches the executor device; upgrade to the patched release that fixes CPU KV injection in `ExampleConnector`.

## API & Configuration

- **Symptom:** `input_audio` content blocks with a UUID reference are parsed incorrectly, causing audio not to be resolved. **Fix:** Pass raw audio bytes/base64 directly in the content block instead of a UUID reference until the resolver bug is patched.
- **Symptom:** `renderer_num_workers` setting is silently ignored when using the offline `LLM` class (only takes effect in async serving). **Fix:** Use the async `AsyncLLMEngine` / `vllm serve` path if parallel rendering workers are required; do not rely on this parameter in offline `LLM` usage.
