# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project scope

This repo houses three loosely-coupled bodies of work, all centered on **OpenClaw skills for ML-engineering agents**:

| Area | Where | What |
|---|---|---|
| **Skill-builder** | `agents/ai-skill-builder/`, `docs/skill-builder/` | OpenClaw agent that turns a Python package URL into a progressive-disclosure `SKILL.md` |
| **Skill-scout** | `agents/ai-skill-scout/`, `docs/skill-scout/` | OpenClaw agent that searches GitHub for existing OpenClaw skills and installs them safely |
| **Skill evaluation framework** | `agents/skill-tester/`, `docs/eval/`, `infra/`, `src/mleval/` | Two-stage A/B framework measuring whether a skill makes an MLE agent measurably better (Stage 1 local, Stage 2 plug-in MLE-agent A/B on Nautilus NRP — primary agent: AIDE) |

Repo was forked from [aibuildai/AI-Build-AI](https://github.com/aibuildai/AI-Build-AI) in 2026-04, then diverged completely — fork connection severed 2026-05.

## Repo layout

```
agents/
├── ai-skill-builder/             OpenClaw builder agent + bundled skills
├── ai-skill-scout/               OpenClaw discovery agent + bundled skills
└── skill-tester/                 Stage-1 eval harness agent

docs/
├── eval/
│   ├── overview.md               2-stage pipeline summary
│   ├── stage1.md                 local CI-style skill eval (locked, in production)
│   ├── stage2.md                 MLE-agent A/B framework (v0.3 AIDE pivot)
│   └── ops.md                    Stage 2 runtime playbook (Nautilus)
├── skill-scout/                  hld.md + plan.md
└── skill-builder/                hld.md + plan.md

src/mleval/                       harness Python package (pip-installable)
├── _cli/                         `mleval` CLI (stub)
└── analyzer/                     post-trajectory analyzer chain
    ├── adapter_aide.py             AIDE journal.json + prompts.jsonl -> trajectory.jsonl
    ├── stage_classifier.py         AST -> 6x16 sub-stage labels (MVP; PyCG upgrade is task #62)
    ├── state_predicates.py         generic + per-task assertions over outputs
    └── aggregate.py                cross-trajectory L1/L3 + paired Lift rollup

infra/                            Stage-2 runtime, plugin layout
├── agents/                       per-agent plugins (one subdir each)
│   ├── _interface.md             contract: env vars, output files, schema
│   └── aide/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       ├── run_aide.py             shim — loads sidecar patches before AIDE
│       ├── job.yaml.tmpl           envsubst Job template
│       ├── aide_sidecar/           monkey-patches (seed, backend, skill, interpreter)
│       └── README.md
├── tasks/                        per-task scaffolds (instruction.md + predicates.py)
│   └── _template/
├── skills/                       skills under eval (SKILL.md files)
│   └── _template/
└── orchestrator/
    └── run_ab.py                 spawn N Jobs per (task × cell × seed), wait, collect

deploy/k8s/                       agent-agnostic k8s manifests
├── pvc.yaml                      1Ti CephFS RWX shared trajectory storage
├── helper-jupyter-1gpu.yaml      interactive 1×rtxa6000 dev pod
├── secret.template.yaml          reference for the secret keys; real Secret via `make k8s-secret`
└── README.md
```

`infra/agents/<name>/` and `src/mleval/analyzer/<adapter_X>.py` together form the plugin axis: adding a second agent means a new agent dir + a new adapter module. Tasks and skills are also plugin-shaped.

## Skill evaluation framework — `docs/eval/`

Current research focus. Two-stage pipeline; index at `docs/eval/overview.md`.

| Stage | Question | Doc | Status |
|---|---|---|---|
| 1 | Does the skill fire on the right prompts and surface the right facts? | `docs/eval/stage1.md` | locked, in production |
| 2 | Does the MLE agent build measurably better pipelines *with* the skill, and *where* does the help land? | `docs/eval/stage2.md` | v0.3 AIDE pivot complete; pre-pilot |

**Stage 2 in one sentence:** paired with-skill / without-skill A/B on a freeform PEFT-relevant task on a pluggable MLE agent (primary: AIDE), decomposing trajectories into a 6 × 16 pipeline-stage taxonomy and attributing improvements via AST imports/calls + state predicates (LLM judge default OFF).

**Runtime infrastructure:** UCSD Nautilus NRP k8s cluster; one container image per agent plugin; one Pod per (task × cell × seed) trajectory. Full playbook in `docs/eval/ops.md`.

## Architecture diagram (Stage 2)

```
                      .env (source of truth)
                            |
        +-------------------+---------------------+
        |                                          |
   amusing.ucsd.edu                          this Mac (kubectl)
        |                                          |
   docker buildx                               envsubst + kubectl apply
        |                                          |
   ghcr.io/.../mleval-agent:dev   ---->   Nautilus pod (ecepxie ns)
                                                   |
                                       +-----------+-----------+
                                       |                       |
                                  helper pod                Job (per trajectory)
                                  (jupyter)                    |
                                                       /workspace/run_aide.py
                                                              |
                                                aide_sidecar.{seed,backend,skill,interp}
                                                              |
                                                       aide.run.run()
                                                              |
                                                 +------------+------------+
                                                 |                         |
                                       journal.json + tree.html      prompts.jsonl
                                                 |                         |
                                                 +------------+------------+
                                                              |
                                          python -m mleval.analyzer.{adapter,classifier,preds}
                                                              |
                                            trajectory.jsonl + code/ + state.json + manifest.json
                                                              |
                                                       (CephFS PVC)
                                                              |
                                                local kubectl cp + mleval.analyzer.aggregate
                                                              |
                                                  report.json + report.md
```

## Container image — `infra/agents/aide/`, `deploy/k8s/`

- **Image tag:** `ghcr.io/kkuntal990/mleval-agent:dev` (overwrites on every push)
- **Base:** `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` — shared across plugins for layer reuse.
- **Build/push host:** `amusing.ucsd.edu` (32-core Linux amd64, native BuildKit via per-user `~/.docker/cli-plugins/docker-buildx` v0.18.0). Mac builds via QEMU emulation take ~30 min vs ~5–10 min native. **Don't build locally on Mac unless amusing is down.**
- **All other ops happen from this Mac:** `kubectl` against `ecepxie`, `make k8s-*`, helper-pod / Job lifecycle, secret creation, log tailing, orchestrator runs.
- **SSH to amusing:** `ssh ad-kkokate@amusing.ucsd.edu`. Key passphrase is the literal string `"amusing"`.
- **Build workflow on amusing:**
  ```
  cd ~/AI-Skill-builder && git pull
  make docker-agent     # buildx build, ~5-10 min from scratch, ~1 min cache-hot
  make docker-push      # ghcr.io login must already be done (write:packages PAT)
  ```
- **Config:** everything is `.env`-driven (gitignored; template at `.env.example`). Variables: `IMAGE_*`, `K8S_NAMESPACE=ecepxie`, `GPU_TYPE=nvidia.com/rtxa6000`, `OPENROUTER_API_KEY`, `MLEVAL_LLM_MODEL`, `MLEVAL_RUN_ID`, `DEFAULT_SEED`, `GHCR_READ_TOKEN`, `AIDE_REPO`, `AIDE_REF`. The Makefile `-include`s it and `envsubst`s it into the k8s manifests at apply time.
- **AIDE pinning:** `AIDE_REF=main` is OK for smoke; pin to a SHA before the pilot.
- **OpenRouter routing:** Image sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1` so AIDE's openai backend's `use_chat_api=true` path is taken (supports function-calling). The openrouter backend is also patched but unused — it hardcodes `provider.order=[Fireworks]` which breaks DeepSeek.
- **Manifests** in `deploy/k8s/` (agent-agnostic): `pvc.yaml` (1Ti CephFS RWX), `helper-jupyter-1gpu.yaml` (1× rtxa6000 interactive pod), `secret.template.yaml` (reference; real Secret created via `make k8s-secret`). Per-agent Job manifests live under `infra/agents/<name>/job.yaml.tmpl` and are envsubst-rendered by the orchestrator.
- **Hard rule (still in effect):** do **not** apply any trajectory Job to Nautilus until the user explicitly approves a live run.

## Sidecar architecture (AIDE plugin)

The AIDE plugin runs four monkey-patches that load at import time via `run_aide.py`:

| Patch | What it does | Why |
|---|---|---|
| `aide_sidecar.seed` | Seeds `random`, `numpy.random` (legacy + `default_rng`), `torch`. Sets `PYTHONHASHSEED`. | AIDE never seeds anything; paired seeds require this. |
| `aide_sidecar.backend_wrapper` | Wraps each entry in `aide.backend.provider_to_query_func` to log `(prompt, response, in/out tokens, req_time)` to `$MLEVAL_PROMPTS_LOG`. | AIDE discards token counts; prompts are never persisted. |
| `aide_sidecar.skill_inject` | If `$MLEVAL_SKILL_PATH` is set + exists, splices the file content into `aide.utils.config.load_task_desc`'s return value. | Makes the skill visible to every code-gen and judge call. |
| `aide_sidecar.interpreter_patch` | Wraps `Interpreter.run` to snapshot `working_dir` to `$MLEVAL_OUTPUT_DIR/working_dirs/op_<step>/` per step. Skips heavy globs (`*.bin`, `*.safetensors`, etc.). | State predicates need to inspect submission CSVs / checkpoints; AIDE deletes them. |

Patch ordering matters: seed first (so any subsequent randomness is determined), then backend (must run before `aide.agent` does `from .backend import query`), then skill_inject (before `Agent.__init__` calls `load_task_desc`).

The `aide.agent.query` capture-at-import problem is solved by patching the dict `provider_to_query_func` (which the top-level `query` reads at call time) instead of `aide.backend.query` itself. The dispatcher reads the dict fresh on every call — bypass-proof.

## Known issues & deferred work

Logged in the task list (use `TaskList` to inspect). Highlights:

| # | Issue | Why deferred |
|---|---|---|
| #62 | Stage classifier is a flat AST rule table (MVP); real PyCG-Extended integration is task #62 + #70 validation gate | Pilot uses MVP; PyCG upgrade gates the full A/B (#66) |
| #71 | Layer-2c LLM judge is unimplemented | Default OFF; only flipped on if pilot's L2a/L2b can't discriminate cells |
| Reproducibility | `random.shuffle` of package list in AIDE prompts depends on retry-count nondeterminism (network) | Documented in `aide_sidecar/seed.py`; bounded within a trajectory, not across |
| Token undercount | Backoff-layer retries inside provider backends are not counted by our wrapper | Acceptable bias for L3 cost; documented in `backend_wrapper.py` |
| Secret leak surface | AIDE's `logger.info` includes raw completion text in `run.log` on the PVC | API key isn't in payload; mitigated by namespace ACLs |

## Stage 1 — local skill eval (`docs/eval/stage1.md`)

CI-style; runs on every `build-skill-from-docs` invocation. ~5 min, ~$1 per skill. Tests whether the skill fires on the right prompts and surfaces the right facts using a separate test agent (the `main` OpenClaw agent acts as baseline, MCP sidecar captures prompts).

## OpenClaw agents — `agents/`

Three agents live under `agents/`. Each follows the standard openclaw layout (`AGENTS.md` + `IDENTITY.md` + `SOUL.md` + `TOOLS.md` + `HEARTBEAT.md` + `USER.md` + `skills/` + optional `data/`).

| Agent | Doc | Role |
|---|---|---|
| `ai-skill-builder` | `docs/skill-builder/{hld,plan}.md` | Builds a `SKILL.md` from a Python package URL |
| `ai-skill-scout` | `docs/skill-scout/{hld,plan}.md` | Searches GitHub for existing OpenClaw skills |
| `skill-tester` | referenced from `docs/eval/stage1.md` | Test-harness agent (Stage 1 actually pins the `main` agent as baseline, not this one) |

These are tracked-in-repo canonical copies. The live agents run under `~/.openclaw/agents/` on the user's machine; this repo holds reproducible snapshots.

## Conventions

- **Filenames:** kebab-case (`build-skill-from-docs/`, `find-ai-skill/`). Markdown in subdirs uses short lowercase names (`hld.md`, `plan.md`, `stage1.md`).
- **Docs structure:** each multi-doc area gets a subdir under `docs/`. New agents/projects get their own subdir.
- **Skill-evaluation work always lives in `docs/eval/`** — don't scatter eval changes across stage1/stage2/agent docs.
- **Absolute paths in code/docs:** the repo lives at `/Users/kuntalkokate/pengtao-lab/ai-builds-ai-project/AI-Skill-builder/` after the 2026-05 rename. Old `/AI-Build-AI/` paths are stale.
- **Plugin code lives in `infra/`; harness code lives in `src/mleval/`.** Harness modules are pip-installable; plugin modules are file-copied into the container image during build.
- **`.env` is the single source of truth.** Never hard-code config in YAML or Python; always thread via `.env` -> Makefile -> envsubst / orchestrator.
- **Don't `kubectl apply` agent-agnostic YAMLs directly** — they contain `${...}` envsubst markers. Always go through `make k8s-apply-*` targets.
