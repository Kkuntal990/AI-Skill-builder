# Models and Quantization in vLLM

Covers supported generative and pooling model types, LoRA adapter usage, quantization methods (AWQ, GGUF, FP8, BitsAndBytes, GPTQ, INT4/INT8), and speculative decoding strategies available in vLLM.

## Generative Models

Generative models produce token sequences and are the primary use case for vLLM's `LLM` class and OpenAI-compatible server.

### Loading a Generative Model (Offline)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(["Hello, my name is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Typical next step: swap `model=` for any HuggingFace-compatible checkpoint; vLLM resolves the architecture automatically.

### Loading via OpenAI-Compatible Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then query with any OpenAI client pointed at `http://localhost:8000`.

---

## Pooling Models

Pooling models produce fixed-size representations (embeddings, classifications, rewards, scores) rather than token sequences. They use `task=` to select the pooling head.

### Embedding

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed")
outputs = llm.embed(["Hello, my name is", "What is your name?"])
for output in outputs:
    print(output.outputs.embedding)  # list of floats
```

### Classification

```python
llm = LLM(model="jason9693/Qwen2.5-1.5B-aisafe", task="classify")
outputs = llm.classify(["This is a safe text.", "This text is harmful."])
for output in outputs:
    print(output.outputs.probs)  # class probabilities
```

### Reward Modeling

```python
llm = LLM(model="Skywork/Skywork-Reward-Llama-3.1-8B", task="reward")
# Expects conversation-formatted inputs
```

### Scoring (Cross-Encoder)

```python
llm = LLM(model="BAAI/bge-reranker-v2-m3", task="score")
outputs = llm.score(
    ["What is the capital of France?"],
    ["Paris is the capital of France."]
)
for output in outputs:
    print(output.outputs.score)
```

Typical next step: use `task="embed"` for retrieval pipelines; `task="score"` for reranking.

---

## LoRA Adapters

vLLM supports loading LoRA adapters at serve time, including multiple adapters per server instance.

### Enabling LoRA at Engine Level

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_lora=True)
```

### Passing a LoRA Request per Inference Call

```python
outputs = llm.generate(
    ["Hello, my name is"],
    sampling_params,
    lora_request=LoRARequest(
        lora_name="my-adapter",
        lora_int_id=1,
        lora_path="/path/to/lora/adapter",
    )
)
```

- `lora_int_id`: unique integer identifier used for caching; different adapters must have different IDs.
- `lora_path`: local directory or HuggingFace repo containing adapter weights.

### Serving Multiple LoRA Adapters via API Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules my-adapter=/path/to/adapter1 other-adapter=/path/to/adapter2 \
    --max-lora-rank 64
```

Clients specify the adapter by passing `model: "my-adapter"` in the request body.

### Key LoRA Configuration Options

| Argument | Purpose |
|---|---|
| `--enable-lora` | Activate LoRA support |
| `--max-lora-rank` | Maximum rank supported (default 16) |
| `--max-loras` | Max adapters loaded simultaneously |
| `--lora-dtype` | Dtype for adapter weights |

---

## Quantization Methods

### AWQ (Activation-aware Weight Quantization)

Load a pre-quantized AWQ model directly:

```python
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="awq")
```

Or let vLLM detect quantization from the model config automatically (no explicit `quantization=` needed for HF-hosted AWQ models).

Server:

```bash
vllm serve TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```

### GGUF

Load GGUF files directly:

```python
llm = LLM(model="bartowski/Llama-3.2-1B-Instruct-GGUF",
           tokenizer="meta-llama/Llama-3.2-1B-Instruct")
```

Or specify a single GGUF file:

```python
llm = LLM(
    model="bartowski/Llama-3.2-1B-Instruct-GGUF",
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
    quantization="gguf",
)
```

GGUF support requires the model file to be accessible locally or via HuggingFace Hub.

### FP8 (W8A8)

Run FP8 quantized inference (requires Hopper+ or Ada Lovelace GPUs for hardware FP8):

```python
llm = LLM(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
           quantization="fp8")
```

Online (dynamic) FP8 quantization without a pre-quantized model:

```python
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
           quantization="fp8",
           kv_cache_dtype="fp8")
```

Server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --kv-cache-dtype fp8
```

### BitsAndBytes (INT4/INT8 via bitsandbytes)

Load with 4-bit quantization on the fly:

```python
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
           quantization="bitsandbytes",
           load_format="bitsandbytes")
```

- `quantization="bitsandbytes"` and `load_format="bitsandbytes"` must both be set.
- Quantizes weights at load time; no pre-quantized checkpoint required.

### GPTQ (GPTQModel)

```python
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ", quantization="gptq")
```

For models quantized with GPTQModel and Marlin kernel acceleration:

```python
llm = LLM(model="ModelCloud/Llama-3.2-1B-Instruct-gptq-4bit",
           quantization="gptq_marlin")
```

### INT4 W4A16

```python
llm = LLM(model="nm-testing/Llama-3.2-1B-Instruct-W4A16-Compressed-tensors-test",
           quantization="compressed-tensors")
```

### INT8 W8A8

```python
llm = LLM(model="neuralmagic/Llama-3.1-8B-Instruct-quantized.w8a8",
           quantization="compressed-tensors")
```

### Quantized KV Cache

Reduce KV cache memory footprint independently of weight quantization:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype fp8
```

```python
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", kv_cache_dtype="fp8")
```

Supported values: `"auto"` (default, matches model dtype), `"fp8"`.

---

## Speculative Decoding

Speculative decoding uses a smaller draft model to propose tokens, verified in parallel by the target model, increasing throughput without changing output distribution.

### Draft Model Speculation

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "num_speculative_tokens": 5,
    },
)
```

Server:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5
```

### EAGLE Draft Models

EAGLE uses a fine-tuned draft head for higher acceptance rates:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)
```

### MLP Speculator (MLP Draft Models)

```python
llm = LLM(
    model="ibm-granite/granite-3.1-8b-instruct",
    speculative_config={
        "model": "ibm-granite/granite-3.1-8b-instruct-accelerator",
        "num_speculative_tokens": 5,
    },
)
```

### N-Gram Speculation (Prompt Lookup)

No draft model required; reuses n-grams from the prompt as draft tokens. Effective for tasks where output echoes input (e.g., summarization, code editing):

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
```

Server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4
```

### MTP (Multi-Token Prediction)

For models with built-in MTP heads (e.g., DeepSeek-V3):

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
```

### Key Speculative Decoding Parameters

| Parameter | Description |
|---|---|
| `num_speculative_tokens` | Draft tokens proposed per step |
| `speculative_model` | Draft model path or `"[ngram]"` |
| `speculative_draft_tensor_parallel_size` | TP degree for draft model (can differ from target) |
| `typical_acceptance_sampler_posterior_threshold` | Acceptance threshold for typical acceptance |

---

## Combining Quantization and Other Features

Quantization and LoRA can be combined:

```python
llm = LLM(
    model="TheBloke/Llama-2-7b-Chat-AWQ",
    quantization="awq",
    enable_lora=True,
)
```

Speculative decoding and quantization can be combined; apply quantization to the target model independently of the draft model:

```bash
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --quantization fp8 \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5
```

---

## Model Architecture Resolution

vLLM resolves model architecture from the HuggingFace `config.json`. If a model uses a non-standard architecture name, override with:

```python
llm = LLM(model="/path/to/model", trust_remote_code=True)
```

Or via server:

```bash
vllm serve /path/to/model --trust-remote-code
```

Use `--trust-remote-code` only for checkpoints you control or trust.
