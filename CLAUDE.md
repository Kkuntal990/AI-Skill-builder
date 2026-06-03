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
    ├── adapter_mlevolve.py         MLEvolve journal.json + prompts.jsonl -> trajectory.jsonl (current agent)
    ├── stage_classifier.py         AST -> 6x16 sub-stage labels; multi-label (all_sub_stages) + parse_error (MVP; PyCG upgrade #62)
    ├── stage_metrics.py            3 per-sub-stage metrics: clean-reach · rework · failure-modes (co-location-proof)
    ├── state_predicates.py         generic + per-task assertions over outputs
    ├── pricing.py                  OpenRouter $/1M-token table (per-model in/out)
    ├── metrics.py                  per-trajectory derivations + cost-norm Lift + stage chi-sq
    └── aggregate.py                cross-trajectory L1/L3 + paired Lift rollup

infra/                            Stage-2 runtime, plugin layout
├── agents/                       per-agent plugins (one subdir each)
│   ├── _interface.md             contract: env vars, output files, schema
│   └── aide/
│       ├── Dockerfile              vllm/vllm-openai:v0.9.2 base + lockfile install (MLE-Bench pattern)
│       ├── requirements.in         top-level pins added on top of vllm-openai's universe
│       ├── requirements.lock       278-package frozen lockfile (do not hand-edit; regen recipe in .in)
│       ├── entrypoint.sh           setsid+PGID trap; runs (now empty) pip install then AIDE then analyzers
│       ├── run_aide.py             shim — loads sidecar patches before AIDE
│       ├── job.yaml.tmpl           GPU profile (1×rtxa6000 + 2cpu/24Gi)
│       ├── job_cpu.yaml.tmpl       CPU profile (1cpu/2Gi NRP-exempt, tabular pilots)
│       ├── aide_sidecar/           monkey-patches (seed, openai_timeout, backend, skill, interpreter)
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
- **Base:** `vllm/vllm-openai:v0.9.2` — vllm owns the torch ABI (torch 2.7.0+cu128, vllm 0.9.2, transformers 4.53.1 pinned). Replaces the prior `pytorch/pytorch:2.7.1` base, which broke when per-task `pip install vllm` forced torch 2.7→2.5 downgrade. Switched 2026-05-25.
- **Dep stack:** single `pip install --no-deps -r requirements.lock` in the Dockerfile (MLE-Bench pattern). Lockfile is 280 packages frozen from `infra/agents/aide/requirements.in` resolved INSIDE the same `vllm/vllm-openai:v0.9.2` base (recipe in `requirements.in` header). pip sees the full dep graph once at image-build time — no runtime cascades. Per-task `requirements.txt` files are intentionally empty (kept for backward compat; tasks ship data + prompts, not deps).
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
- **AIDE pinning:** pinned to a SHA (`AIDE_REF=40dcf28fc3a39e93c7192acec0c9e2e9bffa973d`). The Dockerfile uses `git init + git fetch --depth 1 + git checkout FETCH_HEAD` rather than `git clone --branch` so SHAs (which are not branch names) work. Bump only when intentionally upgrading AIDE; the SHA is also written to `/opt/aide/.aide_sha` inside the image for provenance.
- **OpenRouter routing:** Image sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1` so AIDE's openai backend's `use_chat_api=true` path is taken (supports function-calling). The openrouter backend is also patched but unused — it hardcodes `provider.order=[Fireworks]` which breaks DeepSeek.
- **Manifests** in `deploy/k8s/` (agent-agnostic): `pvc.yaml` (1Ti CephFS RWX), `helper-jupyter-1gpu.yaml` (1× rtxa6000 interactive pod), `pip-warm.yaml` (ephemeral pod that pre-downloads task/skill wheels into `/results/.pip-cache`), `secret.template.yaml` (reference; real Secret created via `make k8s-secret`). Per-agent Job manifests live under `infra/agents/<name>/{job,job_cpu}.yaml.tmpl` and are envsubst-rendered by the orchestrator. Pick GPU vs CPU profile via `make ab-apply PROFILE=cpu|gpu` (default `gpu`).
- **HF cache on PVC:** GPU Jobs set `HF_HOME`/`TRANSFORMERS_CACHE`/`HF_DATASETS_CACHE`/`TORCH_HOME` to `/results/.hf-cache/*` so model weights download once per sweep, not once per trajectory. `hf-warm.yaml` + `make hf-warm MODEL=<name>` to pre-populate this cache is still pending — without it, the first cohort of A/B pods can race-download large checkpoints (huggyllama/llama-7b is ~13 GB).
- **Hard rule (still in effect):** do **not** apply any trajectory Job to Nautilus until the user explicitly approves a live run.

## Sidecar architecture (AIDE plugin)

The AIDE plugin runs five monkey-patches that load at import time via `run_aide.py` (and re-exported in order from `aide_sidecar/__init__.py`):

| Patch | What it does | Why |
|---|---|---|
| `aide_sidecar.seed` | Seeds `random`, `numpy.random` (legacy + `default_rng`), `torch`. Sets `PYTHONHASHSEED`. | AIDE never seeds anything; paired seeds require this. |
| `aide_sidecar.openai_timeout` | Injects `httpx.Timeout(read=$MLEVAL_LLM_TIMEOUT_SEC, connect=$MLEVAL_LLM_CONNECT_TIMEOUT_SEC)` (defaults 120s / 10s) into `openai.OpenAI.__init__` and `openai.AsyncOpenAI.__init__`. | OpenAI client default is no read timeout — mvp-001 hung 24+ min on a stalled OpenRouter call before this was added. |
| `aide_sidecar.backend_wrapper` | Wraps each entry in `aide.backend.provider_to_query_func` to log `(prompt, response, in/out tokens, req_time)` to `$MLEVAL_PROMPTS_LOG`. | AIDE discards token counts; prompts are never persisted. |
| `aide_sidecar.skill_inject` | `$MLEVAL_SKILL_PATH` may be a SKILL.md file *or* a skill directory; if a sibling `references/` dir exists, all `*.md` are concatenated in deterministic filename order with `## references/<name>.md` headers. Result spliced into `aide.utils.config.load_task_desc`'s return value. Also exports `get_skill_dir()` used by `interpreter_patch` to locate `scripts/`. | Makes the full progressive-disclosure bundle visible to every code-gen and judge call (AIDE cannot navigate the filesystem from prompts). |
| `aide_sidecar.interpreter_patch` | (1) **Before** `Interpreter.run`: idempotently copies the skill's `scripts/` dir into `working_dir/scripts/` so SKILL.md instructions like `bash scripts/check_vram.sh` actually find the file. (2) **After**: snapshots `working_dir` to `$MLEVAL_OUTPUT_DIR/working_dirs/op_<step>/` per step. Skips heavy globs (`*.bin`, `*.safetensors`, etc.). | Skill scripts are dead pointers without (1) — markdown gets spliced but executables don't follow. State predicates need (2) because AIDE deletes intermediates. |

Patch ordering matters: seed first (so any subsequent randomness is determined), then openai_timeout (must run before any openai client is constructed), then backend (must run before `aide.agent` does `from .backend import query`), then skill_inject (before `Agent.__init__` calls `load_task_desc`).

The `aide.agent.query` capture-at-import problem is solved by patching the dict `provider_to_query_func` (which the top-level `query` reads at call time) instead of `aide.backend.query` itself. The dispatcher reads the dict fresh on every call — bypass-proof.

## Per-task / per-skill Python dependencies

**As of 2026-05-25: per-task and per-skill `requirements.txt` are deprecated.** All ML/agent deps are now in the base image's `infra/agents/aide/requirements.lock` (resolved once at image-build time inside the same `vllm/vllm-openai:v0.9.2` base — MLE-Bench's frozen-lockfile pattern). The earlier "per-task pip install at trajectory startup" architecture caused a cascade where `vllm==0.6.6` in a task's reqs forced torch 2.7→2.5 downgrade, broke torchaudio + transformers ABI, and crashed AIDE; see the git log around commit 72cb6bd / 5d1c5d6 for the post-mortem.

- `infra/tasks/<task>/requirements.txt`  → kept as **empty files** for backward compat (entrypoint.sh skips on no-non-comment-lines). Add to these ONLY if a task needs a niche package not in the lockfile — and never re-pin major libs (torch, transformers, vllm), that goes in `infra/agents/aide/requirements.in` + Dockerfile rebuild.
- `infra/skills/<skill>/requirements.txt` → same. Empty by design; the lockfile owns the dep universe so methodological isolation is preserved by-default across all cells.

`make pip-warm` is now a no-op for the canonical tasks (their requirements.txt are empty) but still works for any task that does declare extras. The PVC pip cache at `/results/.pip-cache/` is still mounted and used by the entrypoint's pip-install step (which now finds nothing to install).

Skill bundles may include a `scripts/` dir (executable helpers the skill markdown instructs the agent to run) — copied into AIDE's working_dir by `interpreter_patch` — and a build-tool-output `evals/` dir (Stage 1 grading artifacts; `.gitignore`'d, belongs in `~/.openclaw/skills/<name>/evals/` on the author's machine).

## Trajectory lifecycle (entrypoint signal handling)

`infra/agents/aide/entrypoint.sh` is the contract between k8s and AIDE; getting it wrong loses the entire trajectory's analyzer output. Key invariants:

- AIDE is launched inside `setsid bash -c '...timeout ... python ... | tee ...' &` so it gets its own PGID. The entrypoint waits on the background PID via the `wait` builtin (signal-interruptible). On SIGTERM, the trap `kill -TERM -- -$PGID` tears down the whole `timeout|python|tee` pipeline, then `finalize()` runs the analyzer chain (adapter → classifier → predicates) idempotently so `manifest.json` + `trajectory.jsonl` + `state.json` always land on the PVC.
- Without `setsid`, a foreground pipeline blocks the trap from firing until the pipeline completes — observed in mvp-001/002.
- Pod spec sets `terminationGracePeriodSeconds: 90` (analyzer headroom before SIGKILL).
- Orchestrator sets `ACTIVE_DEADLINE_SECONDS = time_limit_sec + 1200` to cover image pull (~10 min cold), first-trajectory pip install (~3 min), and analyzers (~5 min). Tighter buffer killed mvp-002 mid-AIDE.
- AIDE itself runs under `timeout --foreground --signal=TERM --kill-after=10s ${TIME_LIMIT_SECONDS}s` — the soft cap. The k8s activeDeadline is the outer safety net.

## Known issues & deferred work

Logged in the task list (use `TaskList` to inspect). Highlights:

| # | Issue | Why deferred |
|---|---|---|
| #62 | Stage classifier is a flat AST rule table (MVP); real PyCG-Extended integration is task #62 + #70 validation gate | Pilot uses MVP; PyCG upgrade gates the full A/B (#66). 6c/6b overmatch (#104) was fixed in-place by demoting their priority — fallback-only labels now. |
| #71 | Layer-2c LLM judge is unimplemented | Default OFF; only flipped on if pilot's L2a/L2b can't discriminate cells |
| transformers pin | `transformers==4.53.1` hard-pinned in `requirements.in` because `vllm==0.9.2` + `transformers>=5` conflict on the `aimv2` config registration | Bumping `VLLM_TAG` to a release supporting transformers 5.x (e.g., the post-July-2025 nightlies) would unlock newer transformers — but those don't have stable Docker tags, only `nightly-<sha>`. Deferred until methodologically necessary. |
| trl excluded | `trl` is intentionally NOT in `requirements.lock` because trl 1.4+ requires transformers>=4.56, which conflicts with the vllm pin | PEFT tasks needing trl must pin to a pre-1.4 release in per-task requirements.txt or bump VLLM_TAG. |
| hf-warm | Pre-populating `/results/.hf-cache/` is not yet automated (mirrors pip-warm pattern) | Workaround: first GPU trajectory pays a one-time download; subsequent ones hit the PVC. Acceptable for smoke; will hurt parallel A/B if both pods race the download. |
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
- **Bump `MLEVAL_RUN_ID` per sweep** (last shipped: `mvp-003` on house-prices CPU; next: `mvp-004-smoke` for llama-inference GPU smoke, then `mvp-005-ab` for the paired A/B). The orchestrator treats an existing manifest at that run_id as "already done" and won't re-run — fresh sweeps require a fresh ID so PVC paths stay disjoint.
- **k8s Job names are DNS-1123 labels** — no underscores. The orchestrator does `cell.replace("_", "-")` to render `with_skill` → `with-skill` in the Job name. Don't undo this.
- **Limit CLAUDE.md to 250 lines, use references to other docs**
