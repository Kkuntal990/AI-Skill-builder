# Quantization in vLLM

How to load and serve quantized checkpoints (AWQ, GPTQ, FP8, BitsAndBytes, GGUF), quantize a model yourself with LLM Compressor, and enable a quantized KV cache — to cut VRAM and raise throughput on a fixed GPU.

## Contents

- Choosing a method
- Loading a pre-quantized checkpoint
- AWQ
- GPTQ
- FP8 (online and offline)
- BitsAndBytes
- GGUF
- Quantized KV cache
- LLM Compressor: W4A16, W8A8, W4A8
- Marlin kernels and method aliases
- Quick reference

## Choosing a method

Weight-only 4-bit (AWQ, GPTQ, W4A16) shrinks weights ~4× and is the best fit when you are **VRAM-bound** and want to fit a larger model on one card. Weight+activation 8-bit (FP8, INT8 W8A8) keeps more accuracy and is best when you are **throughput-bound** on a card with FP8/INT8 tensor-core support (Ada/Hopper for FP8, Ampere+ for INT8). BitsAndBytes is the fastest path to *any* 4-bit run because it quantizes in-flight (no pre-quantized checkpoint needed) but is slower at serve time. GGUF is for importing llama.cpp checkpoints. Reach for these only after the model in full precision does not fit — quantization always costs some accuracy.

## Loading a pre-quantized checkpoint

For most pre-quantized models on the Hub, vLLM auto-detects the format from the checkpoint config and you do not need to pass `quantization` at all:

```python
from vllm import LLM

llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ")
out = llm.generate("Hello, my name is")
```

Pass `quantization=` explicitly only to force a specific method/kernel (e.g. `"awq"` vs `"awq_marlin"`). On the server it is the `--quantization` flag. Next step: confirm the kernel actually selected by reading the startup log line that reports the quantization method.

## AWQ

Serve an AWQ checkpoint:

```python
from vllm import LLM

llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="awq")
```

```bash
vllm serve TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```

Quantize your own model with AutoAWQ first, then point vLLM at the output dir:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

Next step: load `quant_path` with `quantization="awq"`. Note AWQ in vLLM is described as under-optimized vs FP8/INT8 — prefer it for VRAM savings, not peak throughput.

## GPTQ

Serving is identical in shape to AWQ; pass `quantization="gptq"` (or let auto-detect handle it):

```python
from vllm import LLM

llm = LLM(model="TheBloke/Llama-2-7B-Chat-GPTQ", quantization="gptq")
```

Quantize with GPTQModel, which needs calibration data:

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save(quant_path)
```

Next step: serve `quant_path`; vLLM will pick the GPTQ Marlin kernel automatically on supported GPUs (see Marlin section).

## FP8 (online and offline)

**Online (dynamic) FP8** quantizes a full-precision checkpoint to FP8 at load time — no pre-quantized files needed. Weights go to 8-bit and you reclaim ~half the weight memory:

```python
from vllm import LLM

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", quantization="fp8")
```

This is convenient but the docs note dynamic FP8 may have a small accuracy cost vs an offline-calibrated checkpoint, so for production prefer an offline FP8 checkpoint produced by LLM Compressor:

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)
oneshot(model=model, recipe=recipe, output_dir="Meta-Llama-3-8B-Instruct-FP8-Dynamic")
```

FP8 requires a GPU with FP8 tensor cores (compute capability 8.9+/Hopper). Next step: load the `output_dir` directly — the produced checkpoint is in `compressed-tensors` format and auto-detected.

## BitsAndBytes

BitsAndBytes can quantize **in-flight**, so you can 4-bit any HF model without a pre-quantized checkpoint:

```python
from vllm import LLM

llm = LLM(
    model="huggyllama/llama-7b",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

```bash
vllm serve huggyllama/llama-7b --quantization bitsandbytes --load-format bitsandbytes
```

For an already-quantized bnb checkpoint, the same two flags load it:

```python
llm = LLM(
    model="unsloth/tinyllama-bnb-4bit",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

Use this when you need a quick fit on a small card and serve-time speed is secondary. Next step: if throughput matters more than convenience, re-quantize the same model to AWQ/GPTQ/FP8 instead.

## GGUF

GGUF support is experimental. Point vLLM at the `.gguf` file and supply the matching HF tokenizer (GGUF files do not carry a vLLM-compatible tokenizer):

```bash
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

```python
from vllm import LLM

llm = LLM(
    model="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)
```

For a model split into multiple GGUF shards, merge them with `gguf-split --merge` first, then pass the single merged file. Next step: verify generations look sane — quantization type strings vary across GGUF producers and not all are supported.

## Quantized KV cache

The KV cache often dominates memory at long context, so quantizing it to FP8 frees room for more concurrent sequences. This is orthogonal to weight quantization — combine it with any of the above:

```python
from vllm import LLM

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", kv_cache_dtype="fp8")
```

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --kv-cache-dtype fp8
```

`fp8` defaults to the `e4m3` format on CUDA. By default vLLM uses dynamic per-tensor scaling; for better accuracy, load a checkpoint that carries calibrated KV scales or enable scale calculation:

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,
)
```

Next step: measure accuracy on your eval — FP8 KV cache is a memory/throughput win but can degrade quality on long-context tasks, so confirm before shipping.

## LLM Compressor: W4A16, W8A8, W4A8

LLM Compressor is the recommended tool for producing `compressed-tensors` checkpoints that vLLM loads natively. The pattern is always: build a `recipe`, run `oneshot`, save.

**W4A16** (4-bit weights, 16-bit activations) — VRAM savings, uses GPTQ-style weight quantization:

```python
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
```

**W8A8 INT8** (8-bit weights and activations) — throughput on Ampere+; pair GPTQ with SmoothQuant to protect activation outliers:

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

**W4A8 INT8** — 4-bit weights with 8-bit activations, balancing the memory of W4A16 against the activation-quant speedup. Build the recipe with the corresponding `W4A8` scheme, then run the same oneshot flow.

Drive any of these recipes through `oneshot` with a calibration dataset, then serve the output dir:

```python
from llmcompressor.transformers import oneshot

oneshot(
    model=model,
    dataset="open_platypus",
    recipe=recipe,
    output_dir="Llama-3-8B-W4A16",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

Next step: `LLM(model="Llama-3-8B-W4A16")` — `compressed-tensors` is auto-detected, no `quantization=` flag required. Always keep `lm_head` in `ignore` unless you have measured it is safe to quantize.

## Marlin kernels and method aliases

On Ampere+ GPUs vLLM upgrades AWQ/GPTQ to fast Marlin kernels automatically, reported as `awq_marlin` / `gptq_marlin` in the startup log. You can force a path explicitly:

```bash
vllm serve TheBloke/Llama-2-7B-Chat-GPTQ --quantization gptq_marlin
```

Force the plain (non-Marlin) kernel with `--quantization gptq` if you hit a Marlin compatibility issue (e.g. an unsupported group size). The `compressed-tensors` format produced by LLM Compressor likewise dispatches to Marlin kernels where available.

## Quick reference

| Method | `quantization=` | Need pre-quantized file? | Best for |
|---|---|---|---|
| AWQ | `"awq"` / `"awq_marlin"` | yes (AutoAWQ) | 4-bit VRAM savings |
| GPTQ | `"gptq"` / `"gptq_marlin"` | yes (GPTQModel) | 4-bit VRAM savings |
| FP8 (online) | `"fp8"` | no | quick FP8 on Hopper/Ada |
| FP8 (offline) | auto (compressed-tensors) | yes (LLM Compressor) | production FP8 |
| BitsAndBytes | `"bitsandbytes"` (+ `load_format`) | no (in-flight) | fastest 4-bit fit |
| GGUF | auto | yes (+ `--tokenizer`) | importing llama.cpp |
| W4A16 / W8A8 / W4A8 | auto (compressed-tensors) | yes (LLM Compressor) | tuned weight/act trade-offs |
| KV cache FP8 | `kv_cache_dtype="fp8"` | no | long-context memory |

Verify the actual method and kernel selected by reading vLLM's startup log rather than assuming — auto-detection and Marlin upgrades can override what you passed.
