# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project scope

This repo houses three loosely-coupled bodies of work, all centered on **OpenClaw skills for ML-engineering agents**:

| Area | Where | What |
|---|---|---|
| **Skill-builder** | `agents/ai-skill-builder/`, `docs/skill-builder/` | OpenClaw agent that turns a Python package URL into a progressive-disclosure `SKILL.md` |
| **Skill-scout** | `agents/ai-skill-scout/`, `docs/skill-scout/` | OpenClaw agent that searches GitHub for existing OpenClaw skills and installs them safely |
| **Skill evaluation framework** | `agents/skill-tester/`, `docs/eval/`, `infra/`, `src/mleval/` | Two-stage A/B framework measuring whether a skill makes an MLE agent measurably better (Stage 1 local, Stage 2 plug-in MLE-agent A/B on Nautilus NRP — agent: MLEvolve) |

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
│   ├── stage2.md                 MLE-agent A/B framework (MLEvolve)
│   └── ops.md                    Stage 2 runtime playbook (Nautilus)
├── skill-scout/                  hld.md + plan.md
└── skill-builder/                hld.md + plan.md

src/mleval/                       harness Python package (pip-installable)
├── _cli/                         `mleval` CLI (stub)
└── analyzer/                     post-trajectory analyzer chain
    ├── adapter_mlevolve.py         MLEvolve journal.json + prompts.jsonl -> trajectory.jsonl
    ├── stage_classifier.py         AST -> 6x16 sub-stage labels; multi-label (all_sub_stages) + parse_error (MVP; PyCG upgrade #62)
    ├── stage_metrics.py            3 per-sub-stage metrics: clean-reach · rework · failure-modes (co-location-proof)
    ├── state_predicates.py         generic + per-task assertions over outputs
    ├── pricing.py                  OpenRouter $/1M-token table (per-model in/out)
    ├── metrics.py                  per-trajectory derivations + cost-norm Lift + stage chi-sq
    └── aggregate.py                cross-trajectory L1/L3 + paired Lift rollup

infra/                            Stage-2 runtime, plugin layout
├── agents/                       per-agent plugins (one subdir each)
│   ├── _interface.md             contract: env vars, output files, schema
│   └── mlevolve/
│       ├── Dockerfile              vllm/vllm-openai:v0.9.2 base + curated requirements.txt (~12 pkgs; MLE-Bench pattern)
│       ├── requirements.txt        hand-curated pins on top of vllm-openai's universe (NOT upstream's 665-line lockfile)
│       ├── upstream/               vendored InternScience/MLEvolve (git submodule @26bde89), COPYed to /workspace/mlevolve
│       ├── entrypoint.sh           setsid + wall-clock watchdog; runs MLEvolve, then analyzers + held-out grader
│       ├── run_mlevolve.py         shim — imports sidecar patches before MLEvolve's run.run()
│       ├── config.yaml             spike config (use_grading_server:false, sequential search); envsubst-rendered per trajectory
│       ├── job.yaml.tmpl           GPU profile (1×rtxa6000), envsubst-rendered by orchestrator
│       ├── patches/                build-time source patches (de_kaggle.py: bypass LLM clean_task_desc + hardcoded caps)
│       ├── mlevolve_sidecar/       import-time monkey-patches (seed, prompt_logger, skill_injector, …)
│       └── README.md
├── tasks/                        per-task scaffolds (instruction.md + predicates.py + [empty] requirements.txt [+ data/])
│   ├── _template/
│   ├── house-prices/               tabular Kaggle pilot — mvp-003 (without_skill: 6c overmatch fixed)
│   └── llama-inference/            MLAgentBench port — primary GPU pilot, paired with vllm-inference
├── skills/                       skills under eval (SKILL.md [+ references/*.md] [+ scripts/*] + [empty] requirements.txt)
│   ├── _template/
│   ├── tabular-baseline/           tested in mvp-003 on house-prices
│   └── vllm-inference/             162-line SKILL.md + 6 references (~2,300 LOC) + 3 scripts — next pilot
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
| 2 | Does the MLE agent build measurably better pipelines *with* the skill, and *where* does the help land? | `docs/eval/stage2.md` | mvp-003 dry-run on house-prices × tabular-baseline; next: llama-inference × vllm-inference on GPU |

**Stage 2 in one sentence:** paired with-skill / without-skill A/B on a freeform PEFT-relevant task on a pluggable MLE agent (MLEvolve), decomposing trajectories into a 6 × 16 pipeline-stage taxonomy and attributing improvements via AST imports/calls + state predicates (LLM judge default OFF).

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
                                                    /workspace/run_mlevolve.py
                                                              |
                                     mlevolve_sidecar.{seed,prompt_logger,skill_injector,…}
                                                              |
                                                          run.run()
                                                              |
                                                 +------------+------------+
                                                 |                         |
                                        mlevolve_runs/journal.json     prompts.jsonl
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

## Container image — `infra/agents/mlevolve/`, `deploy/k8s/`

- **Image tag:** `ghcr.io/kkuntal990/mleval-agent:dev` (overwrites on every push)
- **Base:** `vllm/vllm-openai:v0.9.2` — vllm owns the torch ABI (torch 2.7.0+cu128, vllm 0.9.2, transformers 4.53.1 pinned). Replaces the prior `pytorch/pytorch:2.7.1` base, which broke when per-task `pip install vllm` forced torch 2.7→2.5 downgrade. Switched 2026-05-25.
- **Dep stack:** single `pip install -r requirements.txt` in the Dockerfile (MLE-Bench atomic-install pattern). `infra/agents/mlevolve/requirements.txt` is a hand-curated ~12-package list resolved once inside the same `vllm/vllm-openai:v0.9.2` base — NOT MLEvolve upstream's 665-line lockfile (internally inconsistent; rationale in `docs/env-build-details.md`). MLEvolve source is vendored (git submodule at `infra/agents/mlevolve/upstream/`, COPYed with its relative imports intact, not pip-installed). Per-task `requirements.txt` files are intentionally empty (kept for backward compat; tasks ship data + prompts, not deps).
- **Build/push host:** `amusing.ucsd.edu` (32-core Linux amd64, native BuildKit via per-user `~/.docker/cli-plugins/docker-buildx` v0.18.0). Mac builds via QEMU emulation take ~30 min vs ~5–10 min native. **Don't build locally on Mac unless amusing is down.**
- **All other ops happen from this Mac:** `kubectl` against `ecepxie`, `make k8s-*`, helper-pod / Job lifecycle, secret creation, log tailing, orchestrator runs.
- **SSH to amusing:** `ssh ad-kkokate@amusing.ucsd.edu`. Key passphrase is the literal string `"amusing"`.
- **Build workflow on amusing:**
  ```
  cd ~/AI-Skill-builder && git pull
  make docker-mlevolve       # inits the upstream submodule, then builds; ~5-10 min from scratch, ~1 min cache-hot
  make docker-mlevolve-push  # ghcr.io login must already be done (write:packages PAT)
  ```
- **Config:** everything is `.env`-driven (gitignored; template at `.env.example`). Variables: `IMAGE_*`, `K8S_NAMESPACE=ecepxie`, `GPU_TYPE=nvidia.com/rtxa6000`, `OPENROUTER_API_KEY`, `MLEVAL_LLM_MODEL`, `MLEVAL_RUN_ID`, `DEFAULT_SEED`, `GHCR_READ_TOKEN`. The Makefile `-include`s it and `envsubst`s it into the k8s manifests at apply time.
- **MLEvolve pinning:** vendored as a git submodule at `infra/agents/mlevolve/upstream/` pinned to SHA `26bde89` (InternScience/MLEvolve). No `.env` build-arg — `make docker-mlevolve` runs `git submodule update --init` then COPYs the tree into the image. Bump the submodule to upgrade; the manifest records `agent.version = vendored-26bde89` for provenance.
- **OpenRouter routing:** Image sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1`; MLEvolve talks to it through the stock OpenAI-compatible client (function-calling `query()` + streaming `generate()`). The API key flows via `OPENAI_API_KEY` — the sidecar's `openai_apikey_env` coerces the config's empty `api_key` so the client falls back to the env var.
- **Manifests** in `deploy/k8s/` (agent-agnostic): `pvc.yaml` (1Ti CephFS RWX), `helper-jupyter-1gpu.yaml` (1× rtxa6000 interactive pod), `pip-warm.yaml` (ephemeral pod that pre-downloads task/skill wheels into `/results/.pip-cache`), `secret.template.yaml` (reference; real Secret created via `make k8s-secret`). Per-agent Job manifests live under `infra/agents/<name>/{job,job_cpu}.yaml.tmpl` and are envsubst-rendered by the orchestrator. Pick GPU vs CPU profile via `make ab-apply PROFILE=cpu|gpu` (default `gpu`).
- **HF cache on PVC:** GPU Jobs set `HF_HOME`/`TRANSFORMERS_CACHE`/`HF_DATASETS_CACHE`/`TORCH_HOME` to `/results/.hf-cache/*` so model weights download once per sweep, not once per trajectory. `hf-warm.yaml` + `make hf-warm MODEL=<name>` to pre-populate this cache is still pending — without it, the first cohort of A/B pods can race-download large checkpoints (huggyllama/llama-7b is ~13 GB).
- **Hard rule (still in effect):** do **not** apply any trajectory Job to Nautilus until the user explicitly approves a live run.

## Sidecar architecture (MLEvolve plugin)

The MLEvolve plugin applies monkey-patches at import time via `run_mlevolve.py` (re-exported in order from `mlevolve_sidecar/__init__.py`), so they take effect before MLEvolve loads any agent module:

| Patch | What it does | Why |
|---|---|---|
| `seed` | Seeds `random`, `numpy.random` (legacy + `default_rng`), `torch`; sets `PYTHONHASHSEED` from `$SEED`. | MLEvolve doesn't seed; paired A/B seeds require it. |
| `openai_apikey_env` | Coerces the config's empty `api_key` → `None` so the OpenAI client falls back to `$OPENAI_API_KEY`. | Lets the key flow via env instead of the committed config. |
| `prompt_logger` | Wraps `llm.openai.{query,generate}` to log `(system, user, output, tokens, req_time)` per call to `$MLEVAL_PROMPTS_LOG`. | MLEvolve never persists prompts; `generate` returns a bare string so its token counts are `null`. |
| `token_budget` | Raises the default `max_tokens` (anti-truncation). | Truncated completions corrupt SEARCH/REPLACE diffs mid-patch. |
| `diff_guard` | Hardens the SEARCH/REPLACE patcher against `=======`-marker corruption (AST-guard + divider normalize). | Truncated/garbled diffs otherwise write broken source. |
| `metric_direction` | Pins maximize/minimize from `$MLEVAL_METRIC_MAXIMIZE` (default maximize). | MLEvolve's LLM `determine_metric_direction` flips nondeterministically (inverted a search once). |
| `skill_retriever` | LOADER: reads a skill library from `$MLEVAL_SKILL_LIBRARY` (dir of `*/SKILL.md`); exposes `loaded_skills()` + `catalog_text()`. No patching. | Populates the library the injector routes over. |
| `eval_harness` | RULES: task-agnostic benchmark rules (held-out test, validate tool, submission-is-the-score, `num_workers` fix) appended to `impl_guideline`. | Reaches BOTH cells identically — a harness constant, not a skill effect. |
| `skill_injector` | PATCHER (imports LAST): a `sys.meta_path` hook rebinds `run` + `get_impl_guideline_from_agent` on the 4 codegen agents (draft/improve/debug/evolution) — Tier-0 catalog into every node + a per-node temp-0 selector loading only the relevant skill(s)+references (Anthropic progressive disclosure). | The agent can't navigate the filesystem from prompts; this makes the skill bundle visible to every code-gen call. |

Order matters: `prompt_logger` must wrap the LLM call site before agent modules cache a reference to it, and `skill_injector`'s import hook must register before `draft_agent.py` et al. load — hence it imports last, after the library is populated. `run_mlevolve.py` imports the sidecar package before `from run import run`.

## Per-task / per-skill Python dependencies

**As of 2026-05-25: per-task and per-skill `requirements.txt` are deprecated.** All ML/agent deps are baked into the base image's curated `infra/agents/mlevolve/requirements.txt` (resolved once at image-build time inside the same `vllm/vllm-openai:v0.9.2` base — MLE-Bench's atomic-install pattern). The earlier "per-task pip install at trajectory startup" architecture caused a cascade where `vllm==0.6.6` in a task's reqs forced torch 2.7→2.5 downgrade, broke torchaudio + transformers ABI, and crashed the agent; see the git log around commit 72cb6bd / 5d1c5d6 for the post-mortem.

- `infra/tasks/<task>/requirements.txt`  → kept as **empty files** for backward compat (entrypoint.sh skips on no-non-comment-lines). Add to these ONLY if a task needs a niche package not in the image — and never re-pin major libs (torch, transformers, vllm), that goes in `infra/agents/mlevolve/requirements.txt` + Dockerfile rebuild.
- `infra/skills/<skill>/requirements.txt` → same. Empty by design; the image owns the dep universe so methodological isolation is preserved by-default across all cells.

`make pip-warm` is now a no-op for the canonical tasks (their requirements.txt are empty) but still works for any task that does declare extras. The PVC pip cache at `/results/.pip-cache/` is still mounted.

Skill bundles may include a `scripts/` dir (executable helpers the skill markdown references) and a build-tool-output `evals/` dir (Stage 1 grading artifacts; `.gitignore`'d, belongs in `~/.openclaw/skills/<name>/evals/` on the author's machine). Note: MLEvolve surfaces skills by splicing SKILL.md + references into the code-gen prompt (`skill_injector`), not by copying files into a workspace — so a script the skill only points at by path is not auto-materialized.

## Trajectory lifecycle (entrypoint signal handling)

`infra/agents/mlevolve/entrypoint.sh` is the contract between k8s and MLEvolve; getting it wrong loses the entire trajectory's analyzer output. Key invariants:

- MLEvolve (`run_mlevolve.py`) is launched under `setsid … &` so it gets its own session/PGID. The entrypoint `wait`s on the PID in a loop (the builtin is signal-interruptible; it retries `wait` while the agent is still alive so a stray SIGTERM can't run the analyzers mid-run — spike-012). A parallel **wall-clock watchdog** SIGTERMs the agent PGID after `TIME_LIMIT_SECONDS` (MLEvolve's own `time_limit` doesn't gate the main search loop), then SIGKILLs after a 30s grace.
- On exit, the analyzer chain runs (`adapter_mlevolve` → `stage_classifier` → `state_predicates`), then the **held-out grader** (`mleval.grader`, run in the entrypoint parent so it can read gold answers the agent never sees), then the manifest write — each idempotent and non-fatal, so `manifest.json` + `trajectory.jsonl` + `state.json` + `held_out_score.json` always land on the PVC.
- A watchdog-triggered stop is an **expected time-budget harvest**: the entrypoint exits 0 (status `completed`) so the Job doesn't `backoffLimit`-retry from scratch and discard the graded best node (spike-018).
- Pod spec sets `terminationGracePeriodSeconds: 90` (analyzer headroom before SIGKILL). Orchestrator sets `ACTIVE_DEADLINE_SECONDS = time_limit_sec + 1200` to cover image pull (~10 min cold) + analyzers + grader.

## Known issues & deferred work

Logged in the task list (use `TaskList` to inspect). Highlights:

| # | Issue | Why deferred |
|---|---|---|
| #62 | Stage classifier is a flat AST rule table (MVP); real PyCG-Extended integration is task #62 + #70 validation gate | Pilot uses MVP; PyCG upgrade gates the full A/B (#66). 6c/6b overmatch (#104) was fixed in-place by demoting their priority — fallback-only labels now. |
| #71 | Layer-2c LLM judge is unimplemented | Default OFF; only flipped on if pilot's L2a/L2b can't discriminate cells |
| transformers pin | `transformers==4.53.1` hard-pinned in `infra/agents/mlevolve/requirements.txt` because `vllm==0.9.2` + `transformers>=5` conflict on the `aimv2` config registration | Bumping `VLLM_TAG` to a release supporting transformers 5.x (e.g., the post-July-2025 nightlies) would unlock newer transformers — but those don't have stable Docker tags, only `nightly-<sha>`. Deferred until methodologically necessary. |
| trl excluded | `trl` is intentionally NOT in `requirements.txt` because trl 1.4+ requires transformers>=4.56, which conflicts with the vllm pin | PEFT tasks needing trl must pin to a pre-1.4 release in per-task requirements.txt or bump VLLM_TAG. |
| hf-warm | Pre-populating `/results/.hf-cache/` is not yet automated (mirrors pip-warm pattern) | Workaround: first GPU trajectory pays a one-time download; subsequent ones hit the PVC. Acceptable for smoke; will hurt parallel A/B if both pods race the download. |
| Reproducibility | Seeds are pinned (`mlevolve_sidecar/seed.py`), but LLM sampling + network-retry counts still vary across paired cells | Bounded within a trajectory; temperature-0 responses can still differ between cells |
| Token undercount | Backoff retries aren't counted, and MLEvolve's streaming `generate()` calls report `null` token counts | Acceptable bias for L3 cost; documented in `mlevolve_sidecar/prompt_logger.py` |
| Secret leak surface | MLEvolve logs raw completion text to `agent_logs/mlevolve_stdout.log` on the PVC | API key isn't in payload; mitigated by namespace ACLs |

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
- **Bump `MLEVAL_RUN_ID` per sweep** (last shipped: `mvp-003` on house-prices CPU; next: `mvp-004-smoke` for llama-inference GPU smoke, then `mvp-005-ab` for the paired A/B). The orchestrator treats an existing manifest at that run_id as "already done" and won't re-run — fresh sweeps require a fresh ID so PVC paths stay disjoint.
- **k8s Job names are DNS-1123 labels** — no underscores. The orchestrator does `cell.replace("_", "-")` to render `with_skill` → `with-skill` in the Job name. Don't undo this.
- **Limit CLAUDE.md to 250 lines, use references to other docs**
