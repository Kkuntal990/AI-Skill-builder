# Deployment and Scaling

Covers Docker, Kubernetes, Nginx, Ray Serve, parallelism strategies (tensor/pipeline/data/expert), disaggregated prefill, and cloud/framework integrations for production vLLM deployments.

---

## Docker

### Official Image

vLLM publishes a pre-built Docker image. Pull and run it directly:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

- `--ipc=host` is required for shared memory used by tensor parallelism.
- Mount the HuggingFace cache to avoid re-downloading weights on each container start.

### Building a Custom Image

Use the provided `Dockerfile` at the repo root:

```bash
# Build from source
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    -t my-vllm:latest \
    --build-arg max_jobs=8
```

Typical next step: push to a private registry and reference it in Kubernetes manifests.

---

## Kubernetes

### Basic Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - --model
            - meta-llama/Meta-Llama-3-8B-Instruct
          resources:
            limits:
              nvidia.com/gpu: "1"
          ports:
            - containerPort: 8000
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: token
```

### Service Manifest

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

### Helm Charts

vLLM ships Helm chart support. Example usage from the online serving examples:

```bash
helm install vllm ./charts/vllm \
    --set model=meta-llama/Meta-Llama-3-8B-Instruct \
    --set replicaCount=2
```

Typical next step: configure a `HorizontalPodAutoscaler` keyed on GPU utilization or request queue depth.

---

## Nginx

Nginx acts as a reverse proxy and load balancer in front of multiple vLLM instances.

### Minimal Nginx Config

```nginx
upstream vllm_backend {
    server vllm-instance-1:8000;
    server vllm-instance-2:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;   # long timeout for generation
        proxy_send_timeout 300s;
    }
}
```

- `proxy_read_timeout` must be long enough for slow completions or streaming responses.
- Combine with `least_conn` directive for better load distribution across replicas.

---

## Ray Serve

### Single-Node Serving

```python
import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

@serve.deployment(ray_actor_options={"num_gpus": 1})
class VLLMDeployment:
    def __init__(self, **kwargs):
        args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def __call__(self, request):
        ...

deployment = VLLMDeployment.bind(model="meta-llama/Meta-Llama-3-8B-Instruct")
serve.run(deployment)
```

### Multi-Node Serving with Ray

For models that span multiple nodes, start a Ray cluster first:

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address=<head-node-ip>:6379
```

Then launch vLLM pointing at the cluster:

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```

### Ray Serve with DeepSeek / Large MoE Models

```python
# From examples/ray_serving/ray_serve_deepseek.py pattern
@serve.deployment(
    ray_actor_options={"num_gpus": 8},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4},
)
class DeepSeekDeployment:
    def __init__(self):
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model="deepseek-ai/DeepSeek-R1",
                tensor_parallel_size=8,
                enable_expert_parallel=True,
            )
        )
```

### Elastic Expert Parallelism with Ray

```bash
# Launch with elastic EP support
vllm serve <model> \
    --enable-expert-parallel \
    --enable-elastic-ep
```

Ray manages worker lifecycle; elastic EP allows the number of expert-parallel workers to scale dynamically.

---

## Parallelism Strategies

### Tensor Parallelism

Splits individual weight matrices across GPUs. Use when a single model layer does not fit on one GPU.

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4
```

- `--tensor-parallel-size` must divide evenly into the number of attention heads.
- All GPUs must be on the same node (NVLink preferred) unless using a fast interconnect.

### Pipeline Parallelism

Splits model layers across nodes/GPUs sequentially.

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

- Total GPUs = `tensor_parallel_size × pipeline_parallel_size`.
- Adds inter-stage communication latency; best for very large models that cannot fit with TP alone.

### Data Parallelism

Runs independent replicas, each handling a shard of the request stream.

```bash
# Via vLLM CLI
vllm serve <model> --data-parallel-size 4

# Or launch multiple independent vllm serve processes behind Nginx/Ray Serve
```

From the data parallel deployment docs, you can also use `--data-parallel-size` with a single launch command when using the v1 engine.

### Expert Parallelism (MoE Models)

Distributes MoE expert layers across GPUs independently of tensor parallelism.

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

- Expert parallelism is orthogonal to TP: TP splits attention, EP distributes experts.
- Requires `--enable-expert-parallel` flag; supported on models with MoE layers.

### Context Parallelism

Splits the KV cache and attention computation along the sequence length dimension.

```bash
vllm serve <model> \
    --tensor-parallel-size 4 \
    --context-parallel-size 2
```

- Useful for very long context workloads where KV cache memory is the bottleneck.
- See the Context Parallel Deployment guide for hardware requirements.

### Combining Parallelism

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --enable-expert-parallel \
    --data-parallel-size 2
```

Total GPU count = `tp × pp × dp`. Plan cluster topology before combining strategies.

---

## Disaggregated Prefill

Disaggregated prefill separates the prefill (prompt processing) and decode (token generation) phases onto different worker pools, improving throughput under mixed workloads.

### Basic Setup

```python
# From examples/disaggregated/disaggregated_prefill.py pattern
# Prefill instance — processes prompts, transfers KV cache
prefill_engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        kv_transfer_config=KVTransferConfig(
            kv_connector="PyNcclConnector",
            kv_role="kv_producer",
            kv_rank=0,
            kv_parallel_size=2,
        ),
    )
)

# Decode instance — receives KV cache, generates tokens
decode_engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        kv_transfer_config=KVTransferConfig(
            kv_connector="PyNcclConnector",
            kv_role="kv_consumer",
            kv_rank=1,
            kv_parallel_size=2,
        ),
    )
)
```

### KV Transfer Connectors

| Connector | Transport | Use Case |
|---|---|---|
| `PyNcclConnector` | NCCL (GPU-GPU) | Same-node or NVLink clusters |
| `MooncakeConnector` | RDMA / CXL | Cross-node, low-latency fabric |
| `LMCacheConnector` | CPU offload + network | Cost-sensitive multi-node |
| `NixlConnector` | NIXL (NVIDIA) | High-performance disaggregation |

### Disaggregated Prefill V1 (vLLM v1 Engine)

```bash
# Prefill node
VLLM_DISAGG_PREFILL_ROLE=prefill \
vllm serve <model> \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer"}'

# Decode node
VLLM_DISAGG_PREFILL_ROLE=decode \
vllm serve <model> \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer"}'
```

### KV Load Failure Recovery

vLLM supports fallback when KV transfer fails mid-request — the decode instance re-runs prefill locally rather than failing the request. Enable via the `KVTransferConfig` `allow_kv_load_failure` option.

---

## Cloud Framework Integrations

### SkyPilot

```yaml
# sky.yaml
resources:
  accelerators: A100:8
  cloud: aws

run: |
  pip install vllm
  vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
      --tensor-parallel-size 8 \
      --host 0.0.0.0
```

```bash
sky launch sky.yaml --cluster vllm-cluster
```

### BentoML

```python
import bentoml

@bentoml.service(resources={"gpu": 1})
class VLLMService:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

    @bentoml.api
    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate(prompt)
        return outputs[0].outputs[0].text
```

### KubeRay (Kubernetes + Ray)

KubeRay manages Ray clusters on Kubernetes. Deploy a `RayCluster` CRD, then use the Ray Serve integration above. vLLM's multi-node serving examples include a `run_cluster.sh` helper:

```bash
# From examples/ray_serving/run_cluster.sh pattern
ray up cluster.yaml --no-config-cache
ray submit cluster.yaml serve_script.py
```

### KServe

KServe wraps vLLM as an `InferenceService`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vllm-llama
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      storageUri: hf://meta-llama/Meta-Llama-3-8B-Instruct
      resources:
        limits:
          nvidia.com/gpu: "1"
```

### LiteLLM

LiteLLM proxies requests to a running vLLM OpenAI-compatible server:

```yaml
# litellm_config.yaml
model_list:
  - model_name: llama3
    litellm_params:
      model: openai/meta-llama/Meta-Llama-3-8B-Instruct
      api_base: http://localhost:8000/v1
      api_key: token-abc123
```

```bash
litellm --config litellm_config.yaml
```

### Modal

```python
import modal

app = modal.App("vllm-server")
image = modal.Image.debian_slim().pip_install("vllm")

@app.function(gpu="A100", image=image)
@modal.web_endpoint()
def serve():
    from vllm import LLM
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    ...
```

### RunPod

Deploy via RunPod's serverless template pointing to `vllm/vllm-openai` with environment variables:

```bash
SERVED_MODEL_NAME=llama3
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
```

### Hugging Face Inference Endpoints

Select the `vllm` container type when creating an endpoint. Pass engine arguments via the `VLLM_ARGS` environment variable:

```bash
VLLM_ARGS="--max-model-len 8192 --tensor-parallel-size 2"
```

### dstack

```yaml
# dstack.yml
type: service
image: vllm/vllm-openai:latest
env:
  - HUGGING_FACE_HUB_TOKEN
commands:
  - vllm serve meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0
port: 8000
resources:
  gpu: 1
```

---

## SageMaker

vLLM provides a `sagemaker-entrypoint` compatible script. The container entrypoint reads SageMaker environment variables (`SM_MODEL_DIR`, `SM_NUM_GPUS`) and maps them to vLLM arguments:

```bash
# Dockerfile entrypoint for SageMaker
ENTRYPOINT ["python", "-m", "vllm.entrypoints.sagemaker_entrypoint"]
```

---

## Troubleshooting Distributed Deployments

- **NCCL timeout**: Increase `NCCL_TIMEOUT` env var; check firewall rules between nodes.
- **OOM on large TP**: Reduce `--gpu-memory-utilization` (default `0.90`) or enable `--enable-chunked-prefill`.
- **Pipeline stall**: Ensure all pipeline stages have symmetric GPU memory; mixed GPU types cause imbalance.
- **Ray actor crash**: Check `ray status` and worker logs; ensure `--ipc=host` in Docker or equivalent shared memory.
- **KV transfer failure**: Set `allow_kv_load_failure=True` in `KVTransferConfig` to fall back to local prefill.
