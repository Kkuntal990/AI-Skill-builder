# MLEvolve agent runtime (spike)

Scaffolding for the **MLEvolve smoke spike** that replaced AIDE on the
`mlevolve-smoke` branch. Goal: validate the four spike claims documented
in `docs/eval/stage2.md` (subprocess isolation, OpenAI-compat wrapper,
`use_grading_server: false` escape hatch, small adapter).

## Layout

| Path | Purpose |
|---|---|
| `upstream/` | Vendored [InternScience/MLEvolve](https://github.com/InternScience/MLEvolve) at sha `26bde89` (git submodule). |
| `Dockerfile` | Image built on `vllm/vllm-openai:v0.9.2`. Installs MLEvolve's three requirement files; explicitly verifies `mlebench` is **not** importable. |
| `entrypoint.sh` | Honors the universal entrypoint contract (env vars in, artifacts to `MLEVAL_OUTPUT_DIR`). Builds MLEvolve's expected dataset layout (`<dataset_dir>/<exp_id>/prepared/public/`) from our `MLEVAL_TASK_DATA_DIR`. Runs the memory sampler. Runs `adapter_mlevolve` post-mortem. |
| `run_mlevolve.py` | Thin Python launcher — imports the sidecar BEFORE MLEvolve, then calls `run.run()`. |
| `mlevolve_sidecar/` | Monkey-patches applied at import: seed pin, OpenAI api-key env fallback, prompt logger. |
| `config.yaml` | Spike config with `use_grading_server: false`, `use_global_memory: false`, `use_coldstart: false`, sequential search. `envsubst`-rendered per trajectory. |

## Sidecar order (matters!)

```
1. seed                — pin RNGs before any nondeterministic import
2. openai_apikey_env   — coerce "" → None so OpenAI client falls back to env
3. prompt_logger       — wraps llm.openai.{query,generate} to write prompts.jsonl
```

Each module patches at import time. The `prompt_logger` patch must run
before any `from llm import query` in the agent modules — that's why
`run_mlevolve.py` does `import mlevolve_sidecar` before `from run import run`.

## Decisions explained

**Why vendor as a submodule instead of `pip install`**:
- MLEvolve has no PyPI release; install via git would pull HEAD on every build.
- Pinning the SHA in our submodule means image rebuilds reproduce.
- Also: we want to be able to patch MLEvolve source files if a critical bug emerges, without forking the repo. (We don't do this in the spike.)

**Why a separate image (`mleval-agent-mlevolve:dev`)**:
- AIDE image had monkey-patches for `aide.backend`, `aide.interpreter`, `aide.utils.config` — all dead code on the MLEvolve path.
- Keeping them in one image bloats the layer graph and risks accidental coupling. Clean separation lets us delete AIDE entirely if the spike succeeds.

**Why the grading server is never launched**:
- The mle-bench coupling lives entirely in `engine/validation/format_server.py`. As long as we never `bash launch_server.sh` (we don't), the import never fires.
- The agent loop's `_validate_submission_with_retry` short-circuits when `cfg.use_grading_server is False` ([source](https://github.com/InternScience/MLEvolve/blob/26bde89/engine/validation/quality_check.py#L219)), treating every submission as format-valid. Our per-task `predicates.py` does the real scoring.

## How to run locally (helper pod path)

```bash
# 1. Build + push image (on amusing)
ssh amusing 'cd ~/AI-Skill-builder && make docker-mlevolve && make docker-mlevolve-push'

# 2. Redeploy helper pod with the new image
make k8s-helper-apply

# 3. Exec in and run the entrypoint directly (no Job yet — spike phase)
kubectl -n ecepxie exec -it mleval-jupyter-1gpu -- bash -c '
  export MLEVAL_RUN_ID=mlevolve-spike-001 \
         MLEVAL_TRAJECTORY_ID=spike-001-llama-without-skill-s0 \
         TASK=llama-inference CELL=without_skill SEED=0 \
         TIME_LIMIT_SECONDS=1800 STEP_LIMIT=5 \
         MLEVAL_LLM_MODEL=deepseek/deepseek-v4-flash \
         MLEVAL_LLM_TIMEOUT_SEC=120 \
         MLEVAL_OUTPUT_DIR=/results/mlevolve-spike-001/spike-001-llama-without-skill-s0 \
         MLEVAL_TASK_INSTRUCTION_PATH=/results/data/llama-inference/instruction.md \
         MLEVAL_TASK_DATA_DIR=/results/data/llama-inference/data \
         OPENAI_BASE_URL=https://openrouter.ai/api/v1
  /workspace/entrypoint.sh
'

# 4. Inspect results
kubectl -n ecepxie exec mleval-jupyter-1gpu -- ls /results/mlevolve-spike-001/spike-001-llama-without-skill-s0/
```

## Spike success criteria

See `docs/eval/stage2.md` → "MLEvolve spike" section. Quick recap:

- **C1**: trajectory completes ≥5 nodes at 64 GiB without OOMKill
- **C2**: at least one `prompts.jsonl` row per LLM stage (`agent.code`, `agent.feedback`)
- **C3**: `pip freeze | grep mlebench` empty; no import in any log
- **C4**: `adapter_mlevolve` ≤200 LoC (raised from 150 after seeing the prompt-bucketing heuristic) and produces a valid `trajectory.jsonl`

## Known data-quality gaps (intentional, not bugs)

- **Token counts for `generate` calls are `null`.** MLEvolve has two
  LLM entry points: `llm.openai.query()` (function-calling, returns a
  5-tuple with token counts) and `llm.openai.generate()` (streaming,
  returns a bare string). The sidecar wraps both, but the `generate`
  return shape provides no token counts so `in_tokens`/`out_tokens` are
  `null` for those rows. Affects most upstream agent modules (`draft`,
  `improve`, `evolution`) — only `code_review` / `result_parse` /
  `data_leakage` use `query`. C2 (prompt-count claim) is unaffected.
- **Prompt → node assignment is heuristic.** MLEvolve doesn't tag
  prompts with the node id they belong to. The adapter buckets prompts
  to nodes by ctime window between parent.ctime and node.ctime. Edge
  cases (parallel branches, fusion nodes spanning multiple parents)
  may misattribute; for the spike's single-trajectory sequential search
  this is fine.
