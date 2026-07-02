# Agent contract

The Stage 2 framework runs any MLE agent in an A/B eval (with-skill vs without-skill) across a configurable task pool. Each agent lives under `infra/agents/<name>/` and conforms to the contract in this document.

**Adding a new agent = drop a new dir under `infra/agents/`** with the four required files below. The orchestrator, k8s manifests, analyzer, and report generator do not change.

**Schema version:** 1.0 (2026-05-19).

---

## 1. Files every agent dir must provide

Two files are mandatory; everything else is implementation freedom. What
the contract requires is the env-var inputs (§2) and the output files (§3)
— how the plugin organizes its internals is up to the plugin author.

```
infra/agents/<name>/
├── Dockerfile     # required — builds the runtime image
├── entrypoint.sh  # required — reads env (§2), produces outputs (§3)
└── ...            # whatever the plugin needs: sidecars, shims, patches, helpers
```

The image's entrypoint is always `/workspace/entrypoint.sh` (copied into
the image during build). The orchestrator only knows about env vars and
output files — never invokes the agent directly.

The post-trajectory work (`adapter`, `stage_classifier`, `state_predicates`)
lives in `src/mleval/analyzer/` and runs in the container after the agent
terminates — the entrypoint invokes `python -m mleval.analyzer.adapter_<name>`
etc. Plugins for new agents add a sibling module when their native log
format differs enough to need a separate adapter.

The skill-injection mechanism is plugin-specific. MLEvolve, having no
clean per-call Python hook (`load_task_desc` reads `desc_file` as one
blob), splices the skill into `description.md` from
`mlevolve_sidecar/skill_inject.py` invoked from `entrypoint.sh` BEFORE
the agent starts. A future agent with a runtime hook could monkey-patch
its config loader instead. Either is acceptable as long as the
`MLEVAL_SKILL_PATH` env var contract from §2 is honored.

The reference plugin on this branch is MLEvolve (see [`infra/agents/mlevolve/`](mlevolve/)):

```
infra/agents/mlevolve/
├── Dockerfile, entrypoint.sh    # contract requirements
├── run_mlevolve.py              # shim: loads sidecar patches, calls MLEvolve's run()
├── job.yaml.tmpl                # envsubst-rendered per-trajectory Job
├── config.yaml                  # spike profile (use_grading_server: false, no FAISS)
├── upstream/                    # git submodule pinned to InternScience/MLEvolve@26bde89
├── mlevolve_sidecar/            # monkey-patches: seed pin, OpenAI api_key env, prompt logger
└── README.md
```

History: an earlier reference plugin was removed when the harness
standardized on MLEvolve during the `mlevolve-smoke` branch — see
`docs/eval/stage2.md` for why.

---

## 2. Environment variables the entrypoint reads

Set by the orchestrator (`scripts/run_ab.py`) per trajectory. All required unless marked optional.

| Var | Example | Purpose |
|---|---|---|
| `MLEVAL_RUN_ID` | `ab-2026-05-19-001` | Identifies the full A/B sweep |
| `MLEVAL_TRAJECTORY_ID` | `jigsaw-cell1-seed42` | Unique per (task, cell, seed) |
| `TASK` | `jigsaw-toxic-comment-classification` | Must match `infra/tasks/<TASK>/` |
| `CELL` | `with_skill` \| `without_skill` | Which arm of the A/B |
| `SEED` | `42` | Pinned at entrypoint: `torch`, `numpy`, `random` |
| `TIME_LIMIT_SECONDS` | `43200` | Hard wall-clock cap; entrypoint must enforce |
| `MLEVAL_OUTPUT_DIR` | `/pvc/<run_id>/<trajectory_id>` | Where adapter writes outputs |
| `MLEVAL_SKILL_PATH` | `/results/skills/vllm-inference/SKILL.md` | Path to the skill's entry-point file. Empty/unset in `without_skill` cell. Sibling `references/*.md` (deep-dive markdown) and `scripts/*` (executables the skill instructs the agent to run) are auto-discovered relative to this path. |
| `MLEVAL_SKILL_SHA256` | `4b6af703...` | Pinned skill version; recorded in `manifest.json` |
| `MLEVAL_TASK_INSTRUCTION_PATH` | `/workspace/task/instruction.md` | Task description (the agent reads this as its task description) |
| `MLEVAL_TASK_DATA_DIR` | `/workspace/task/data` | Task data; read-only mount |
| `MLEVAL_PLAYGROUND_DIR` | `/workspace/playground` | Where the agent writes scratch code & checkpoints |
| `MLEVAL_LLM_API_KEY` | (secret) | API key for whatever LLM the agent uses |
| `MLEVAL_LLM_MODEL` | `deepseek/deepseek-v4-flash` | Optional override |

The contract: **if `MLEVAL_SKILL_PATH` is set and the file exists, the agent must inject the skill content into its prompt(s); otherwise it must run as if no skill exists.** How that wiring happens is `inject_skill.{patch|py}`'s job.

---

## 3. Output contract

After the agent terminates (success or failure), `entrypoint.sh` invokes `adapter.py`. The adapter writes these files under `$MLEVAL_OUTPUT_DIR`:

```
$MLEVAL_OUTPUT_DIR/
├── manifest.json              # run-level metadata (required)
├── trajectory.jsonl           # per-operator-call records (required)
├── usage.json                 # Layer 3 cost/effort summary (required)
├── submission.<ext>           # task-specific submission, copied from playground (if present)
├── code/                      # emitted code per operator call (required if agent emits code)
│   ├── op_001.py
│   ├── op_002.py
│   └── ...
├── playground_snapshots/      # optional: per-checkpoint snapshots of agent's playground
│   └── op_NNN/
└── agent_native_logs/         # optional: raw agent output for debugging
    └── ...
```

All files are required to exist before the pod terminates. Missing required files → trajectory marked failed by the orchestrator.

---

## 4. `trajectory.jsonl` schema

JSON Lines: one JSON object per operator call. Order is execution order (chronological). Records are immutable once written.

### Required fields

```json
{
  "schema_version": "1.0",
  "record_id": "op_001",
  "run_id": "ab-2026-05-19-001",
  "trajectory_id": "jigsaw-cell1-seed42",

  "agent": {
    "name": "mlevolve",
    "version": "git-sha-abc123",
    "operator_native": "improve"
  },

  "stage": {
    "top_level": "3",
    "sub_stage": "3c",
    "label": "adapter_config",
    "classifier_source": "ast_choice_extractor",
    "classifier_confidence": 0.95
  },

  "code": {
    "emitted_path": "code/op_001.py",
    "emitted_lines": 87,
    "imports_top": ["torch", "peft", "transformers"]
  },

  "execution": {
    "ran": true,
    "exit_code": 0,
    "wall_clock_sec": 142.3,
    "stdout_tail_sha": "...",
    "stderr_tail_sha": ""
  },

  "usage": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "wall_clock_sec": 144.1
  },

  "state_snapshot": {
    "playground_files": ["model.py", "checkpoint.pt"],
    "predicate_results": {
      "data_loaded": true,
      "model_defined": true,
      "training_started": false
    }
  },

  "timestamp": {
    "started_at": "2026-05-19T15:23:11Z",
    "ended_at": "2026-05-19T15:25:33Z"
  }
}
```

### Optional fields

```json
{
  "input": {
    "prompt_hash": "sha256:...",
    "predecessor_record_id": "op_000"
  },

  "output": {
    "completion_hash": "sha256:...",
    "errors": ["NameError: foo not defined"]
  },

  "skill_citation": {
    "cited": true,
    "matches": [
      { "skill_section": "SKILL.md#L24", "agent_code_line": "code/op_001.py:42" }
    ]
  },

  "tags": ["debug-attempt", "retry-1"]
}
```

### Field semantics

- **`record_id`** — `op_NNN` where NNN is zero-padded sequence within trajectory. Used as filename in `code/op_NNN.py`.
- **`agent.operator_native`** — the agent's own term for this operation, e.g. a search-loop role such as `draft|debug|improve`. Each plugin defines its own vocabulary; the adapter does not normalize, downstream analysis treats it as opaque metadata.
- **`stage.top_level`** — one of `1|2|3|4|5|6` (the 6 top-level stages from the taxonomy).
- **`stage.sub_stage`** — one of `1a|1b|2a|2b|2c|3a|3b|3c|4a|4b|4c|5a|5b|6a|6b|6c` (the 16 sub-stages).
- **`stage.classifier_source`** — `agent_native` (mapped directly from `operator_native`), `ast_choice_extractor` (inferred from emitted code by PyCG-Extended + AST rules), `manual` (human-labeled during debugging), or `unknown`.
- **`stage.classifier_confidence`** — float in [0, 1]. For `agent_native`, always 1.0. For `ast_choice_extractor`, the classifier's confidence.
- **`code.emitted_path`** — relative to `$MLEVAL_OUTPUT_DIR`. `null` if this operator did not emit code (e.g., a tool call without code).
- **`execution.ran`** — `true` if the emitted code was executed; `false` if it was emitted but not run (e.g., agent decided to discard).
- **`execution.stdout_tail_sha` / `stderr_tail_sha`** — sha256 of last 4KB; full text optionally archived under `agent_native_logs/`. Privacy-safe deduplication.
- **`state_snapshot.predicate_results`** — keys defined per-task by `infra/tasks/<task>/predicates.py`. Free-form dict; adapter passes through.
- **`skill_citation.matches`** — populated by AST citation extractor (see Layer 2c notes in `docs/eval/stage2.md`). Empty array if no citations found.

### Validation

```bash
python -m infra.analyzer.validate_trajectory $MLEVAL_OUTPUT_DIR/trajectory.jsonl
```

Validator checks: schema version, required fields present, record_ids sequential, file references exist, predicate_results keys match task spec.

---

## 5. `manifest.json` schema

One per trajectory. Captures run-level metadata so each trajectory is self-describing.

```json
{
  "schema_version": "1.0",
  "run_id": "ab-2026-05-19-001",
  "trajectory_id": "jigsaw-cell1-seed42",

  "task": {
    "name": "jigsaw-toxic-comment-classification",
    "benchmark": "mle-bench",
    "metric": "auc_roc",
    "time_budget_sec": 43200,
    "instruction_path": "infra/tasks/jigsaw-toxic-comment-classification/instruction.md",
    "instruction_sha256": "..."
  },

  "agent": {
    "name": "mlevolve",
    "version": "git-sha-abc123",
    "container_image": "ghcr.io/kkuntal990/mleval-agent:dev",
    "llm_model": "deepseek/deepseek-v4-flash"
  },

  "cell": {
    "with_skill": true,
    "skill_name": "peft-tuning",
    "skill_sha256": "4b6af703...",
    "skill_path_in_container": "/workspace/skill/SKILL.md"
  },

  "seed": 42,

  "pod": {
    "namespace": "<your-ns>",
    "pod_name": "mleval-jigsaw-cell1-seed42-xyz",
    "node": "node-gpu-12",
    "gpu_type": "A100-80GB"
  },

  "timestamps": {
    "started_at": "2026-05-19T14:00:00Z",
    "ended_at": "2026-05-20T02:00:00Z",
    "wall_clock_sec": 41230
  },

  "result": {
    "status": "completed",
    "submission_present": true,
    "submission_path": "submission.csv",
    "score": { "auc_roc": 0.937 },
    "graded_by": "infra/tasks/jigsaw-toxic-comment-classification/grader.py"
  }
}
```

**`result.status`** must be one of: `completed`, `crashed`, `time_capped`, `preempted`, `validation_failed`.

---

## 6. `usage.json` schema

Layer 3 (cost/effort) summary. Aggregated from `trajectory.jsonl` records.

```json
{
  "schema_version": "1.0",

  "wall_clock_sec": 41230,
  "wall_clock_breakdown": {
    "agent_thinking_sec": 8400,
    "code_execution_sec": 32800,
    "other_sec": 30
  },

  "tokens": {
    "input": 2300000,
    "output": 240000,
    "total": 2540000
  },
  "tokens_by_stage": {
    "1a": 23000, "1b": 12000,
    "2a": 45000, "2b": 8000, "2c": 67000,
    "3a": 134000, "3b": 22000, "3c": 89000,
    "4a": 19000, "4b": 78000, "4c": 0,
    "5a": 56000, "5b": 11000,
    "6a": 14000, "6b": 6000, "6c": 4000
  },

  "operator_calls": {
    "total": 47,
    "by_native_operator": {
      "draft": 1, "improve": 28, "debug": 12, "evolution": 4, "fusion": 2
    }
  },

  "execution": {
    "calls": 23,
    "errors": 4,
    "timeouts": 1
  },

  "skill_citations": {
    "total": 12,
    "by_stage": { "3c": 8, "2c": 3, "1a": 1 }
  }
}
```

---

## 7. Stage classification

Every record carries a sub-stage label from the 16-bucket taxonomy. Three classifier sources:

### `agent_native`
The agent's own operator type maps to one of our stages. Used when the agent's vocabulary aligns cleanly. Note: an agent whose operator types (e.g. `draft|debug|improve`) describe search-loop role rather than ML-pipeline stage falls back to `ast_choice_extractor` instead. Confidence is always 1.0 when this source applies.

The mapping table per agent lives in `infra/agents/<name>/stage_map.json`.

### `ast_choice_extractor`
Default classifier. PyCG-Extended + a rule book under `infra/analyzer/stage_classifier.py` reads emitted code and infers the sub-stage from imports + API calls. Example: code that calls `peft.LoraConfig(...)` and `get_peft_model(...)` → `3c (adapter_config)`. Validated against the Ramasamy 470-notebook corpus (task #70).

### `manual`
Reserved for debug/replay; analyst overrides a misclassification by editing the JSONL.

### `unknown`
The adapter could not classify. Records with `unknown` are flagged in the report; if >5% of records in a trajectory are `unknown`, the trajectory is marked as needing review.

### The taxonomy enum

```
1. Data understanding
   1a. data_loading
   1b. eda

2. Data engineering
   2a. cleaning
   2b. split_and_validation
   2c. feature_engineering

3. Model design
   3a. architecture
   3b. loss
   3c. adapter_config            [LLM-relevant]

4. Training execution
   4a. optimizer
   4b. training_loop
   4c. preference_optimization   [LLM-relevant]

5. Tuning & ablation
   5a. hpo
   5b. ablation

6. Evaluation & delivery
   6a. held_out_eval
   6b. inference_or_merge        [LLM-relevant]
   6c. submission
```

---

## 8. State predicates

Per-task. Defined in `infra/tasks/<task>/predicates.py`. The adapter imports the task's module and calls `evaluate(playground_dir, record) -> dict[str, bool]` after each operator.

Example skeleton (`infra/tasks/jigsaw-toxic-comment-classification/predicates.py`):

```python
from pathlib import Path

def evaluate(playground_dir: Path, record: dict) -> dict[str, bool]:
    return {
        "data_loaded": (playground_dir / "data_loaded.flag").exists(),
        "train_val_split_made": (playground_dir / "val.csv").exists(),
        "model_defined": any(f.endswith("_model.py") for f in os.listdir(playground_dir)),
        "training_started": (playground_dir / "checkpoints").is_dir(),
        "submission_written": (playground_dir / "submission.csv").exists(),
        # ... task-specific predicates
    }
```

Predicates are **boolean and cheap** — they run after every operator. Anything expensive belongs offline in `infra/analyzer/`.

The set of valid predicate keys per task is declared in `infra/tasks/<task>/predicates_schema.json`. The validator (§4) checks that `state_snapshot.predicate_results` keys are a subset of that schema.

---

## 9. Skill citation extraction

For records in the `with_skill` cell, the adapter optionally populates `skill_citation`. Two extraction sources:

- **Textual quote match**: agent's prompt or output literally contains a phrase from `SKILL.md` (substring or fuzzy threshold).
- **API-choice match**: agent's emitted code uses an API combination that matches a workflow recipe from `SKILL.md` (e.g., `LoraConfig + get_peft_model + Trainer` matches the "Fine-tune a model with LoRA" workflow).

The extractor lives in `infra/analyzer/skill_citation.py` and is agent-agnostic. Adapters call it; they don't implement it.

In the `without_skill` cell, this section is omitted (or set to `{ "cited": false, "matches": [] }`).

---

## 10. How to add a new agent

Concrete steps to wire up agent `<myagent>`:

1. **Create the dir.** `mkdir infra/agents/myagent`
2. **Write `Dockerfile`.** Base image: usually `nvidia/cuda:12.4.0-runtime-ubuntu22.04` or `pytorch/pytorch:2.x-cuda12.4`. Install the agent + analyzer deps. Copy `entrypoint.sh`, `adapter.py`, and `inject_skill.patch` into `/workspace/`.
3. **Write `entrypoint.sh`.** Pseudocode:
   ```bash
   #!/usr/bin/env bash
   set -e
   set_seeds_via_pyhook "$SEED"   # provided helper
   apply_skill_injection           # no-op if MLEVAL_SKILL_PATH unset
   timeout "$TIME_LIMIT_SECONDS" run_myagent
   python3 /workspace/adapter.py   # writes trajectory.jsonl, manifest.json, usage.json
   ```
4. **Design `inject_skill.patch`** (or `.py` for runtime injection). The patch must:
   - Be a no-op when `MLEVAL_SKILL_PATH` is empty/unset
   - When set, read the file and prepend its content to every operator/agent prompt
   - Preserve the agent's existing prompt structure (don't break baseline behavior in the `without_skill` cell)
5. **Write `adapter.py`.** Implements the agent-specific event → universal record mapping. Imports:
   - `infra.analyzer.stage_classifier` for code-based classification
   - `infra.analyzer.skill_citation` for citation extraction
   - `infra.tasks.<TASK>.predicates` for state predicates
6. **Build & test locally** before pushing to the cluster:
   ```bash
   docker build -t mleval-myagent:dev infra/agents/myagent/
   docker run --rm \
     -e TASK=debug-trl-grpo -e CELL=without_skill -e SEED=0 \
     -e MLEVAL_OUTPUT_DIR=/out -v $(pwd)/out:/out \
     mleval-myagent:dev
   python -m infra.analyzer.validate_trajectory ./out/trajectory.jsonl
   ```
7. **Push to `gitlab-registry.nrp-nautilus.io/.../mleval-myagent:vX`.**
8. **Add `agent_image` row to `infra/orchestrator/agents.csv`.** Orchestrator picks it up on next run.

---

## 11. Versioning

- `schema_version` field is required on every JSON file emitted by the adapter.
- Breaking changes bump major version (1.0 → 2.0). Adapters must declare which version they emit; orchestrator and analyzer support reading any version ≥ 1.0 (forward-compatible reads).
- This document's revision history lives in the git log for `infra/agents/_interface.md`.

---

## 12. Non-requirements (explicit)

The contract intentionally does **not** specify:

- **Agent's internal architecture.** Iterative, evolutionary, tree-search, single-shot — all acceptable.
- **LLM provider.** OpenAI, Anthropic, OpenRouter, local — adapter just records `agent.llm_model`.
- **Prompt format inside the agent.** Skill injection is the only mandated input; everything else is the agent's choice.
- **Code execution environment.** Agent may run code in-process, subprocess, or remote — as long as `code/op_NNN.py` and `execution.exit_code` are populated.
- **Trajectory length.** No min/max operator-call count.

These are deliberate non-requirements so the contract works for as wide a range of agent designs as possible.
