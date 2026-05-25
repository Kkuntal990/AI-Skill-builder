# Quantization and Memory Conservation

Covers FP8, AWQ, GPTQ, BitsAndBytes, INT4/INT8, quantized KV cache, and engine memory arguments for reducing VRAM usage in vLLM.

## Contents

- FP8 Weight and Activation Quantization (W8A8)
- INT8 W8A8 Quantization
- INT4 W4A16 Quantization
- AWQ (AutoAWQ)
- GPTQ (GPTQModel)
- BitsAndBytes
- Quantized KV Cache
- Online Quantization
- Engine Memory Arguments
- Conserving Memory: Practical Strategies

---

## FP8 Weight and Activation Quantization (W8A8)

FP8 quantizes both weights and activations to 8-bit floating point, reducing VRAM and increasing throughput on supported hardware (NVIDIA H100, AMD MI300X).

**Loading a pre-quantized FP8 model:**

```python
from vllm import LLM
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
```

**On-the-fly FP8 quantization of a non-quantized model:**

```python
from vllm import LLM
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", quantization="fp8")
```

This performs dynamic per-tensor activation quantization and static per-channel weight quantization at load time. No pre-quantized checkpoint is required, but a pre-quantized model is preferred for accuracy.

**Via CLI:**

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --quantization fp8
```

**Typical next step:** For best accuracy, use a checkpoint quantized offline with `llm-compressor` or `nvidia-modelopt` rather than on-the-fly quantization.

---

## INT8 W8A8 Quantization

INT8 W8A8 quantizes weights and activations to 8-bit integers. Supported via `compressed-tensors` or `bitsandbytes`.

**Loading a pre-quantized INT8 model:**

```python
from vllm import LLM
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8")
```

**Explicit quantization flag:**

```python
llm = LLM(
    model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8",
    quantization="compressed-tensors",
)
```

**Via CLI:**

```bash
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8 \
    --quantization compressed-tensors
```

**Typical next step:** Benchmark with `vllm bench throughput` to confirm throughput improvement vs. BF16 baseline.

---

## INT4 W4A16 Quantization

INT4 W4A16 quantizes weights to 4-bit integers while keeping activations in 16-bit. Achieves the highest compression ratio with acceptable accuracy loss on most models.

**Loading a pre-quantized W4A16 model:**

```python
from vllm import LLM
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w4a16")
```

**Via CLI:**

```bash
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w4a16 \
    --quantization compressed-tensors
```

**Typical next step:** W4A16 models fit on smaller GPUs; combine with `--max-model-len` reduction if KV cache is still tight.

---

## AWQ (AutoAWQ)

AWQ (Activation-aware Weight Quantization) is a popular INT4 weight-only quantization format. vLLM supports AWQ checkpoints natively.

**Loading an AWQ model:**

```python
from vllm import LLM
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq",
)
```

**Via CLI:**

```bash
vllm serve TheBloke/Llama-2-7B-Chat-AWQ --quantization awq
```

**AWQ with Marlin kernel (faster on Ampere/Hopper):**

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq_marlin",
)
```

`awq_marlin` uses the Marlin GEMM kernel for higher throughput. vLLM will automatically select it when the hardware supports it if you pass `awq_marlin`.

**Typical next step:** Check `quantization` field in the model's `config.json`; if it already says `awq`, you can omit the flag.

---

## GPTQ (GPTQModel)

GPTQ is a widely-used post-training quantization format. vLLM supports GPTQ and its Marlin-accelerated variant.

**Loading a GPTQ model:**

```python
from vllm import LLM
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq",
)
```

**GPTQ with Marlin kernel:**

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq_marlin",
)
```

**Via CLI:**

```bash
vllm serve TheBloke/Llama-2-7B-Chat-GPTQ --quantization gptq_marlin
```

**Typical next step:** `gptq_marlin` requires the model to have been quantized with symmetric quantization and group size 128 or -1; verify with the model card before switching.

---

## BitsAndBytes

BitsAndBytes enables on-the-fly INT4 (`load_in_4bit`) and INT8 (`load_in_8bit`) quantization without a pre-quantized checkpoint. Useful for quick memory reduction without offline quantization.

**INT4 loading (NF4):**

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

**INT8 loading:**

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

**Via CLI:**

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes
```

**Important:** Both `quantization` and `load_format` must be set to `bitsandbytes` together. BitsAndBytes quantization happens at load time and does not require a separate quantization step.

**Typical next step:** BitsAndBytes is convenient but slower than pre-quantized AWQ/GPTQ at inference time; use it for experimentation, then switch to a dedicated format for production.

---

## Quantized KV Cache

The KV cache can be quantized independently of model weights, reducing memory pressure during long-context inference.

**FP8 KV cache:**

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    kv_cache_dtype="fp8",
)
```

**Via CLI:**

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --kv-cache-dtype fp8
```

**Supported values for `kv_cache_dtype`:**

| Value | Description |
|---|---|
| `auto` | Matches model dtype (default) |
| `fp8` | FP8 E4M3, requires H100 / MI300X |
| `fp8_e5m2` | FP8 E5M2 variant |

**FP8 KV cache with a scaling factor override:**

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    kv_cache_dtype="fp8",
    quantization_param_path="/path/to/kv_cache_scales.json",
)
```

`quantization_param_path` points to a JSON file containing per-layer KV cache scaling factors produced by calibration tools. Without it, vLLM uses a default scale of 1.0, which may reduce accuracy.

**Typical next step:** Combine FP8 KV cache with FP8 weights for maximum memory savings on H100.

---

## Online Quantization

vLLM supports activating quantization at serve time without a pre-quantized checkpoint for select formats.

**FP8 dynamic quantization at runtime:**

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    quantization="fp8",
)
```

This applies:
- Static per-channel weight quantization (computed once at load)
- Dynamic per-token activation quantization (computed each forward pass)

**When to use online vs. offline quantization:**

| Scenario | Recommendation |
|---|---|
| Production, accuracy-sensitive | Offline quantized checkpoint |
| Quick VRAM reduction, prototyping | Online (`quantization="fp8"`) |
| No calibration data available | Online quantization acceptable |

---

## Engine Memory Arguments

These `LLM()` constructor / `vllm serve` arguments directly control VRAM allocation.

### `gpu_memory_utilization`

Controls the fraction of GPU memory vLLM reserves for the model and KV cache.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.90,  # default: 0.90
)
```

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --gpu-memory-utilization 0.85
```

Lower this value if other processes share the GPU or if you see OOM errors.

### `max_model_len`

Caps the maximum sequence length (prompt + output). Shorter sequences require less KV cache memory.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_model_len=8192,  # override model's default context length
)
```

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 8192
```

### `max_num_seqs`

Limits the number of sequences processed in a single batch, capping peak KV cache usage.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_num_seqs=64,
)
```

### `enforce_eager`

Disables CUDA graph capture, which itself consumes VRAM. Useful on memory-constrained GPUs.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    enforce_eager=True,
)
```

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --enforce-eager
```

**Trade-off:** Disabling CUDA graphs reduces throughput; use only when necessary.

### `swap_space`

Amount of CPU RAM (in GiB) to use as swap space for KV cache blocks when GPU memory is exhausted.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    swap_space=4,  # GiB, default: 4
)
```

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --swap-space 8
```

### `cpu_offload_gb`

Offloads a portion of model weights to CPU RAM, freeing GPU VRAM.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    cpu_offload_gb=10,  # offload 10 GiB of weights to CPU
)
```

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct --cpu-offload-gb 10
```

**Typical next step:** Increase `cpu_offload_gb` incrementally until the model fits; expect latency increase proportional to offloaded layers.

---

## Conserving Memory: Practical Strategies

Combine the following techniques in order of impact:

### 1. Choose the right quantization format

| Format | Weight bits | Activation bits | VRAM reduction | Speed impact |
|---|---|---|---|---|
| FP8 W8A8 | 8 | 8 | ~50% vs BF16 | Minimal on H100 |
| INT8 W8A8 | 8 | 8 | ~50% vs BF16 | Minimal |
| AWQ / GPTQ W4A16 | 4 | 16 | ~75% vs BF16 | Small |
| BitsAndBytes NF4 | 4 | 16 | ~75% vs BF16 | Moderate |

### 2. Quantize the KV cache

```python
llm = LLM(
    model="...",
    quantization="fp8",
    kv_cache_dtype="fp8",
)
```

Combining weight FP8 + KV cache FP8 gives the maximum memory reduction on H100/MI300X.

### 3. Reduce `max_model_len`

KV cache size scales linearly with `max_model_len`. Halving the context length roughly halves KV cache VRAM.

```python
llm = LLM(model="...", max_model_len=4096)
```

### 4. Use tensor parallelism to spread across GPUs

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,
)
```

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 4
```

### 5. Lower `gpu_memory_utilization`

If OOM errors occur during KV cache allocation (not weight loading), lower this value:

```bash
vllm serve ... --gpu-memory-utilization 0.80
```

### 6. Enable CPU offload as a last resort

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --quantization fp8 \
    --cpu-offload-gb 20 \
    --gpu-memory-utilization 0.85
```

### Decision flowchart

```
Model fits on GPU?
├── Yes → Use BF16 or FP8 for best throughput
└── No  → Apply W4A16 (AWQ/GPTQ) quantization
           Still doesn't fit?
           ├── Add tensor parallelism
           ├── Reduce max_model_len
           └── Enable cpu_offload_gb
```
