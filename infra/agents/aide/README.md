# `infra/agents/aide/` — AIDE agent plugin

Per-agent plugin that wraps [WecoAI/aideml](https://github.com/WecoAI/aideml) (the AIDE MLE agent) so the rest of our framework (PVC, secrets, k8s helper pod, trajectory schema) stays agent-agnostic.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Lean build — pytorch base + AIDE harness deps + curated PEFT stack. |
| `entrypoint.sh` | Job-mode entrypoint. Helper pod overrides `command:` to run Jupyter directly. |
| `run_aide.py` | Shim that loads sidecar patches BEFORE `aide.run` imports. |
| `aide_sidecar/backend_wrapper.py` | Monkey-patches `aide.backend.query` → captures (prompt, response, tokens) per LLM call. |
| `aide_sidecar/seed.py` | Pins `random` / `numpy` / `torch` RNGs from `$SEED`. |
| `aide_sidecar/interpreter_patch.py` | Placeholder for working_dir preservation (task #63). |

## Build + push (on amusing)

```bash
cd ~/AI-Skill-builder && git pull
make docker-agent     # buildx build, ~5-10 min native amd64
make docker-push      # ghcr.io login required (see CLAUDE.md)
```

Defaults from `.env`: `AIDE_REPO=https://github.com/WecoAI/aideml.git`, `AIDE_REF=main`. Pin to a SHA before any real A/B run.

## Run (inside a pod)

The image's helper-pod env exports `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
and `OPENAI_API_KEY=$OPENROUTER_API_KEY` so AIDE's openai backend uses
chat.completions.create (OpenRouter-compatible) instead of responses.create
(OpenAI-only). Models matching `gpt-*/o*/codex` will still go to api.openai.com.

```bash
python /workspace/run_aide.py \
    data_dir=/path/to/task/data \
    desc_file=/path/to/task/instruction.md \
    agent.code.model=$MLEVAL_LLM_MODEL \
    agent.code.temp=0 \
    agent.feedback.model=$MLEVAL_LLM_MODEL \
    agent.feedback.temp=0 \
    agent.steps=20 \
    generate_report=false \
    log_dir=/results/$MLEVAL_RUN_ID/$MLEVAL_TRAJECTORY_ID/aide_logs \
    workspace_dir=/results/$MLEVAL_RUN_ID/$MLEVAL_TRAJECTORY_ID/aide_workspace \
    exp_name=$MLEVAL_TRAJECTORY_ID
```

`generate_report=false` skips AIDE's journal2report step, which would call
the `report.model=gpt-4.1` default through OpenAI's `responses.create` and
401 against OpenRouter. To enable, override `report.model` to a non-openai
slug (e.g. `deepseek/deepseek-v4-flash`) as well.

The entrypoint script wires all of this from env vars in a Job pod.

## What the sidecar produces

- `$MLEVAL_PROMPTS_LOG` (default `./prompts.jsonl`): one JSON line per LLM call with `model`, `provider`, `system_message`, `user_message`, `output`, `in_tokens`, `out_tokens`, `req_time_sec`, `ts`, `func_spec_name`.
- Standard AIDE outputs at `log_dir/exp_name/`: `journal.json`, `config.yaml`, `tree_plot.html`, `best_solution.py`, `report.md`.

## Notes on AIDE's deps

AIDE's `requirements.txt` is split with a comment:

```
# AIDE requirements
... harness deps the AIDE code actually imports ...

# agent requirements (packages that the agent might need)
... kitchen-sink of "the agent's generated code might want this" ...
```

The Dockerfile installs only the harness block (`awk '/^# agent requirements/{exit} {print}'`). The kitchen-sink contains broken wheels (`tensorpack`, `bayespy`) and UI-only deps (`streamlit`). For PEFT trajectories we install a curated subset (`transformers`, `peft`, `trl`, `datasets`, `accelerate`) explicitly. Add to that list if a real trajectory hits a missing import.
