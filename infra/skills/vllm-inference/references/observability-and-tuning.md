# Observability and Tuning

Covers Prometheus metrics, OpenTelemetry tracing, Grafana dashboards, the benchmarking CLI, and environment variables and knobs for optimizing vLLM performance.

## Contents

- Prometheus Metrics
- Production Metrics Reference
- OpenTelemetry Setup
- Grafana Dashboards
- Benchmark CLI
- Parameter Sweeps
- Optimization and Tuning Knobs
- Environment Variables

---

## Prometheus Metrics

vLLM exposes a `/metrics` endpoint on the OpenAI-compatible server that Prometheus can scrape directly.

Start the server and verify the endpoint:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
curl http://localhost:8000/metrics
```

Key metric families emitted:

| Metric | Type | Description |
|---|---|---|
| `vllm:num_requests_running` | Gauge | Requests currently being processed |
| `vllm:num_requests_waiting` | Gauge | Requests queued, waiting for capacity |
| `vllm:num_requests_swapped` | Gauge | Requests swapped to CPU |
| `vllm:gpu_cache_usage_perc` | Gauge | Fraction of GPU KV-cache blocks in use |
| `vllm:cpu_cache_usage_perc` | Gauge | Fraction of CPU KV-cache blocks in use |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT distribution |
| `vllm:time_per_output_token_seconds` | Histogram | TPOT distribution |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency |
| `vllm:request_prompt_tokens` | Histogram | Prompt token counts per request |
| `vllm:request_generation_tokens` | Histogram | Generation token counts per request |
| `vllm:request_success_total` | Counter | Successful requests, labeled by finish reason |
| `vllm:num_preemptions_total` | Counter | Cumulative preemption events |
| `vllm:prompt_tokens_total` | Counter | Total prompt tokens processed |
| `vllm:generation_tokens_total` | Counter | Total generation tokens produced |

Enable or disable the metrics endpoint with `--disable-log-stats` (disables internal stats logging, not the `/metrics` endpoint itself).

Configure the Prometheus multiprocess directory when running multiple workers:

```bash
export PROMETHEUS_MULTIPROC_DIR=/tmp/vllm_prometheus
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2
```

---

## Production Metrics Reference

The **Production Metrics** guide (`docs/source/general/production_metrics.md`) groups metrics into three tiers for alerting:

**Tier 1 — Latency SLOs**
- `vllm:time_to_first_token_seconds` — alert when p99 exceeds your TTFT budget
- `vllm:e2e_request_latency_seconds` — alert when p99 exceeds your end-to-end budget

**Tier 2 — Saturation**
- `vllm:gpu_cache_usage_perc` — sustained values above 0.90 indicate KV-cache pressure; consider reducing `--max-num-seqs` or enabling chunked prefill
- `vllm:num_requests_waiting` — a growing queue signals the server cannot keep up

**Tier 3 — Errors**
- `vllm:request_success_total{finished_reason="abort"}` — aborted requests (client disconnects, timeouts)

Typical Prometheus alert rule skeleton:

```yaml
groups:
  - name: vllm
    rules:
      - alert: HighTTFT
        expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 2.0
        for: 2m
        labels:
          severity: warning
      - alert: KVCacheSaturation
        expr: vllm:gpu_cache_usage_perc > 0.90
        for: 1m
        labels:
          severity: critical
```

---

## OpenTelemetry Setup

vLLM supports OpenTelemetry (OTel) tracing for distributed request tracing. The observability example lives in `examples/observability/`.

Install the required extras:

```bash
pip install vllm[otel]
# or manually:
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

Start an OTel collector (e.g., the OTLP gRPC endpoint) and point vLLM at it:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces all
```

`--collect-detailed-traces` accepts:
- `all` — trace every component
- `model` — model forward pass only
- `worker` — worker-level spans
- `scheduler` — scheduler decision spans

Environment variable equivalents (set before launching):

```bash
export VLLM_TRACE_FUNCTION=1          # enable function-level tracing hooks
export OTEL_SERVICE_NAME=vllm-server  # service name shown in Jaeger/Tempo
```

Spans produced per request include:
- `vllm.request` — top-level span with prompt/output token counts as attributes
- `vllm.scheduler` — time spent in the scheduler
- `vllm.model_forward` — GPU forward pass duration
- `vllm.worker_execute_model` — full worker execution including sampling

Propagate trace context from an upstream caller by passing the `traceparent` header in HTTP requests to the vLLM server; vLLM will attach child spans automatically.

---

## Grafana Dashboards

vLLM ships a pre-built Grafana dashboard JSON in `examples/observability/` (see `Prometheus and Grafana` example).

**Quick setup with Docker Compose** (from the observability example):

```bash
cd examples/observability
docker compose up -d   # starts Prometheus + Grafana
```

The compose file wires:
- Prometheus scraping `host.docker.internal:8000/metrics` every 15 s
- Grafana at `http://localhost:3000` (default credentials `admin/admin`)
- The vLLM dashboard imported automatically via provisioning

**Manual import**: download the dashboard JSON from the repo and import it via Grafana UI → Dashboards → Import.

Dashboard panels include:
- Request rate and queue depth over time
- TTFT / TPOT / E2E latency percentiles (p50, p95, p99)
- GPU KV-cache utilization heatmap
- Token throughput (prompt + generation tokens/s)
- Preemption rate

Customize the `scrape_interval` in `prometheus.yml` for higher resolution during load tests:

```yaml
scrape_configs:
  - job_name: vllm
    scrape_interval: 5s
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

---

## Benchmark CLI

vLLM provides first-class benchmarking scripts under `benchmarks/` and exposes them via the `vllm bench` CLI group.

### Throughput benchmark

```bash
python benchmarks/benchmark_throughput.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 1000 \
  --input-len 512 \
  --output-len 128
```

Reports: total throughput (tokens/s), requests/s, and per-request latency stats.

### Latency benchmark

```bash
python benchmarks/benchmark_latency.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --num-iters 100 \
  --batch-size 1
```

Reports: mean, median, p99 TTFT and TPOT in isolation (no concurrent load).

### Online serving benchmark

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --request-rate 10 \
  --num-prompts 500
```

Reports: request throughput, TTFT, TPOT, ITL (inter-token latency), and E2E latency at multiple percentiles.

Key flags shared across benchmark scripts:

| Flag | Purpose |
|---|---|
| `--tokenizer` | Override tokenizer (defaults to `--model`) |
| `--quantization` | Match the quantization used by the server |
| `--tensor-parallel-size` | TP degree for offline benchmarks |
| `--max-model-len` | Cap context length |
| `--dtype` | `float16`, `bfloat16`, `float32`, `auto` |
| `--seed` | Fix random seed for reproducibility |
| `--output-json` | Write structured results to a JSON file |

### Startup benchmark

```bash
python benchmarks/benchmark_startup.py \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Measures cold-start time from process launch to first token ready.

---

## Parameter Sweeps

The `benchmarks/` directory includes a sweep harness for automated multi-dimensional benchmarking.

```bash
python -m vllm.benchmarks.sweep.cli \
  --config sweep_config.yaml \
  --output-dir results/
```

A minimal `sweep_config.yaml`:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
request_rates: [1, 5, 10, 20, 50]
input_lens: [128, 512, 1024]
output_lens: [128]
num_prompts: 200
```

After the sweep completes, plot a Pareto frontier of throughput vs. latency:

```bash
python -m vllm.benchmarks.sweep.plot_pareto \
  --results-dir results/ \
  --output pareto.png
```

The performance dashboard (`Benchmarking → Performance Dashboard` in the docs) aggregates nightly sweep results across commits for regression tracking.

---

## Optimization and Tuning Knobs

### Chunked prefill

Splits long prefill sequences into chunks to reduce TTFT variance and allow decode batches to interleave:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048
```

- Increase `--max-num-batched-tokens` for higher throughput at the cost of longer individual prefill steps.
- Decrease it to reduce TTFT jitter under mixed workloads.

### Prefix caching (APC)

Caches KV blocks for repeated prompt prefixes (system prompts, few-shot examples):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching
```

Monitor effectiveness via `vllm:gpu_prefix_cache_hit_rate` (Gauge, 0–1).

### Scheduler and batching

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --scheduler-delay-factor 0.0
```

- `--max-num-seqs`: maximum concurrent sequences; lower values reduce memory pressure.
- `--scheduler-delay-factor`: artificial delay (seconds × TTFT) before scheduling; non-zero values allow more requests to batch together.

### KV cache and memory

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.90 \
  --swap-space 4 \
  --kv-cache-dtype fp8
```

- `--gpu-memory-utilization` (default `0.90`): fraction of GPU VRAM reserved for KV cache after model weights load. Raise toward `0.95` on dedicated inference nodes; lower if OOM.
- `--swap-space`: CPU swap space in GiB for preempted sequences.
- `--kv-cache-dtype`: `auto`, `fp8`, `fp8_e5m2`, `fp8_e4m3` — FP8 KV cache halves KV memory at minor quality cost.

### Speculative decoding

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5
```

Reduces TPOT for latency-sensitive workloads. Monitor acceptance rate via `vllm:spec_decode_draft_acceptance_rate`.

### Quantization

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --dtype auto
```

Quantization reduces memory and often increases throughput; benchmark with `benchmark_throughput.py` before and after to confirm gains for your workload.

### Compilation and CUDA graphs

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --compilation-config '{"level": 3}'
```

Compilation levels (from `Optimization Levels` design doc):
- `0` — no compilation
- `1` — basic torch.compile
- `2` — piecewise CUDA graphs
- `3` — full graph with fusion passes (default for most models)

Disable CUDA graphs entirely for debugging:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enforce-eager
```

---

## Environment Variables

vLLM reads a large set of environment variables (defined in `vllm/envs.py`). The most operationally relevant ones:

### Logging and debugging

| Variable | Default | Effect |
|---|---|---|
| `VLLM_LOGGING_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `VLLM_TRACE_FUNCTION` | `0` | Set to `1` to enable function-level tracing hooks |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `0` | Set to `1` to bypass the max-model-len safety cap |

### Performance

| Variable | Default | Effect |
|---|---|---|
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | `fork` or `spawn` for worker process creation |
| `VLLM_USE_TRITON_FLASH_ATTN` | `1` | Use Triton FlashAttention kernel; set to `0` to fall back |
| `VLLM_ATTENTION_BACKEND` | *(auto)* | Override attention backend: `FLASH_ATTN`, `FLASHINFER`, `XFORMERS` |
| `VLLM_USE_V1` | `1` | Use the V1 engine architecture (default in recent releases) |
| `VLLM_TORCH_COMPILE_LEVEL` | *(from config)* | Override compilation level without changing server args |
| `CUDA_VISIBLE_DEVICES` | *(all)* | Restrict which GPUs vLLM uses |

### Multiprocessing and distributed

| Variable | Default | Effect |
|---|---|---|
| `VLLM_HOST_IP` | *(auto)* | Bind IP for inter-worker communication |
| `VLLM_PORT` | *(auto)* | Port for inter-worker RPC |
| `NCCL_DEBUG` | *(unset)* | Set to `INFO` or `WARN` to debug NCCL collective issues |
| `PROMETHEUS_MULTIPROC_DIR` | *(unset)* | Required when running multiple workers; set to a shared writable directory |

### Caching

| Variable | Default | Effect |
|---|---|---|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Root directory for compiled artifacts and model cache |
| `VLLM_CONFIG_ROOT` | `~/.config/vllm` | Directory for vLLM config files |

Set variables inline for a one-off test:

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Or export them in your container/systemd environment for persistent configuration.
