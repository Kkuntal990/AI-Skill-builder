# Quantization in vLLM

How to run weight- and activation-quantized models in vLLM — producing checkpoints with AutoAWQ, GPTQModel, LLM Compressor (FP8/INT8/INT4), or BitsAndBytes, loading GGUF files, and enabling an FP8 KV cache.

## Contents

- Choosing a method
- AWQ (4-bit weights, AutoAWQ)
- GPTQ (4-bit weights, GPTQModel)
- LLM Compressor: FP8 W8A8 (dynamic, no calibration)
- LLM Compressor: INT8 W8A8 (SmoothQuant + GPTQ)
- LLM Compressor: INT4 W4A16
- BitsAndBytes (4-bit, in-flight or pre-quantized)
- GGUF (experimental, out-of-tree plugin)
- Quantized KV cache (FP8)
- Gotchas

## Choosing a method

All methods below either (a) load an already-quantized checkpoint from the Hub, or (b) quantize a full-precision model yourself first, then load the result. vLLM auto-detects most formats from the checkpoint's config — the explicit `quantization=` argument is only needed for AWQ and in-flight BitsAndBytes.

| Goal | Method | Calibration data? | Load arg |
|---|---|---|---|
| 4-bit weights, ready-made checkpoints | AWQ | yes (in author's script) | `quantization="auto_awq"` |
| 4-bit weights, broad model coverage | GPTQ (GPTQModel) | yes | auto-detected |
| FP8 weights+activations, fastest setup | LLM Compressor `FP8_DYNAMIC` | **no** | auto-detected |
| INT8 weights+activations | LLM Compressor `W8A8` | yes | auto-detected |
| INT4 weights only | LLM Compressor `W4A16` | yes | auto-detected |
| Quick 4-bit, no separate quant step | BitsAndBytes in-flight | no | `quantization="bitsandbytes"` |
| llama.cpp-style single-file weights | GGUF (plugin) | no | `--tokenizer` required |

FP8 KV cache is orthogonal — it can be combined with any weight method to shrink the cache.

## AWQ (4-bit weights, AutoAWQ)

Quantize a model with AutoAWQ, then serve the saved directory.

```bash
pip install autoawq
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "mistralai/Mistral-7B-Instruct-v0.2"
quant_path = "mistral-instruct-v0.2-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

This writes a 4-bit AWQ checkpoint to `quant_path`. Next, load it in vLLM (works on both freshly-quantized dirs and Hub checkpoints like `TheBloke/Llama-2-7b-Chat-AWQ`):

```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="auto_awq")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

## GPTQ (4-bit weights, GPTQModel)

GPTQModel uses a calibration dataset to fit the 4-bit weights. Note `--no-build-isolation` on install.

```bash
pip install -U gptqmodel --no-build-isolation -v
```

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

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)
```

Loading needs no `quantization=` argument — vLLM detects GPTQ from the checkpoint config:

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")
outputs = llm.generate(["The future of AI is"], sampling_params)
for output in outputs:
    print(f"{output.prompt!r}\n{output.outputs[0].text!r}")
```

## LLM Compressor: FP8 W8A8 (dynamic, no calibration)

The fastest path — `FP8_DYNAMIC` quantizes weights statically and activations dynamically at runtime, so **no calibration dataset is needed**.

```bash
pip install llmcompressor
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

Then load the saved directory directly — the format is auto-detected:

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
model.generate("Hello my name is")
```

## LLM Compressor: INT8 W8A8 (SmoothQuant + GPTQ)

INT8 activations need calibration. The recipe is a list: SmoothQuant first to redistribute outliers, then GPTQ for `W8A8`.

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms to run.
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Load in vLLM, then optionally check accuracy with `lm_eval`:

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

## LLM Compressor: INT4 W4A16

Weights to 4-bit, activations kept at 16-bit. Reuses the same calibration `ds` from the INT8 section. GPTQ alone — no SmoothQuant.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Configure the quantization algorithm to run.
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model, dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128")
```

## BitsAndBytes (4-bit, in-flight or pre-quantized)

Lowest-friction 4-bit: no separate quantize-and-save step. Either load a pre-quantized `bnb` checkpoint, or quantize on the fly at load time.

```bash
pip install bitsandbytes>=0.49.2
```

Reading a pre-quantized checkpoint (no `quantization=` arg needed):

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

In-flight quantization of a full-precision model — pass `quantization="bitsandbytes"`:

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
)
```

For the OpenAI-compatible server, pass the same flag:

```
--quantization bitsandbytes
```

## GGUF (experimental, out-of-tree plugin)

GGUF support in vLLM is highly experimental and under-optimized, and has moved to the out-of-tree `vllm-gguf-plugin`. Always pass `--tokenizer` pointing at the base model — the base tokenizer avoids slow/unreliable conversion.

```
uv pip install vllm-gguf-plugin
```

Serve a GGUF quant directly from a Hub repo (`repo:QUANT` syntax):

```
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

Or download the single file and serve it locally:

```
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

Add `--tensor-parallel-size 2` for multi-GPU, or `--hf-config-path Qwen/Qwen3-0.6B` if the config can't be inferred. The Python entrypoint mirrors the CLI — `tokenizer=` is mandatory:

```python
from vllm import LLM, SamplingParams

conversation = [
   {"role": "system", "content": "You are a helpful assistant"},
   {"role": "user", "content": "Write an essay about the importance of higher education."},
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
   model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
   tokenizer="Qwen/Qwen3-0.6B",
)
outputs = llm.chat(conversation, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

## Quantized KV cache (FP8)

Shrinks the KV cache to FP8, independent of how the weights are quantized. Set `kv_cache_dtype="fp8"`. The two FP8 layouts: `fp8_e5m2` (CUDA 11.8+) and `fp8_e4m3` (CUDA 11.8+ and ROCm).

Simplest form — fixed scales (`calculate_kv_scales=False`):

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

Set `calculate_kv_scales=True` to estimate scales from random tokens during warmup. For best accuracy, calibrate scales from real data with LLM Compressor and bake them into the checkpoint via `kv_cache_scheme`:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048

def process_and_tokenize(example, tokenizer: AutoTokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,
    )

def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
```

The resulting checkpoint carries its calibrated KV scales, so loading it in vLLM needs no extra `calculate_kv_scales` flag.

## Gotchas

- **`quantization=` is only needed for AWQ (`"auto_awq"`) and in-flight BitsAndBytes (`"bitsandbytes"`).** GPTQ and all LLM Compressor outputs (FP8/INT8/INT4) are auto-detected from the checkpoint config — passing the wrong value can cause a load failure.
- **`save_compressed=True` matters for INT8/INT4.** The W8A8/W4A16 recipes save compressed; the FP8 dynamic example saves uncompressed — match the documented call for each scheme.
- **`ignore=["lm_head"]`** is in every LLM Compressor recipe; quantizing the LM head degrades quality.
- **GGUF requires `--tokenizer` / `tokenizer=`** pointing at the base model, and is experimental — prefer AWQ/GPTQ/LLM-Compressor for production.
- **FP8 KV cache layout depends on hardware**: `fp8_e4m3` is the one available on ROCm; `fp8_e5m2` is CUDA-only-style. Pick based on your GPU.
