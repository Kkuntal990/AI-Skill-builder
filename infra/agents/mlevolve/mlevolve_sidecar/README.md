# mlevolve_sidecar

Monkey-patches applied to MLEvolve at import time. Sits BEFORE the
upstream agent in `run_mlevolve.py`'s import order so every patch lands
before any agent module captures the unpatched reference.

## Modules and what they patch

| Module | Patches | Why |
|---|---|---|
| `seed.py` | `random` / `numpy.random` / `torch.manual_seed` | Pin RNG from `$SEED` for paired-seed A/B |
| `openai_apikey_env.py` | `openai.OpenAI(api_key=...)` | Backfill from `$OPENAI_API_KEY` when config has `api_key: ""` |
| `prompt_logger.py` | `llm.openai.{query, generate}` | Capture per-call `{system, user, prompt, output, tokens, t_sec}` to `$MLEVAL_PROMPTS_LOG`. Captures BOTH the kwargs (`query`) and positional/kwarg `prompt` (used by `generate` — stepwise/diff/planner) — see spike-011 root-cause notes in source. |
| `token_budget.py` | `llm.openai.{query, generate}` (default `max_tokens`) | Raise the output-token cap 16384→32768 when a caller passes none. Stops mid-output truncation (the spike-012 `=======` / SyntaxError corruption); wraps outermost (after `prompt_logger`). |
| `skill_retriever.py` | *(loader — no patch)* | Loads a skill **library** from `$MLEVAL_SKILL_LIBRARY` (a dir; scans `*/SKILL.md`, skips `_`-prefixed) or `$MLEVAL_SKILL_PATHS`/`$MLEVAL_SKILL_PATH` (back-compat). Exposes `loaded_skills()` (per-skill `body` + `references` map) and `catalog_text()`. **1.1.0:** also loads a `capabilities.json` sitting beside `SKILL.md` (validated via `capability_schema`) and exposes the manifest-carrying skills through `loaded_capability_skills()` (capability runtime, below). |
| `eval_harness.py` | *(rules only — no patch)* | Task-agnostic benchmark rules (`EVAL_HARNESS_RULES`) + `num_workers=0` rewrite. **Not skill content.** Appended to impl_guideline by `skill_injector`'s wrapper on every codegen node (both cells). Mirror: `infra/tasks/_harness_rules.md`. |
| `capability_linker.py` | *(agent-generic library — no patch; no `mlevolve`/`agents`/`llm` module-scope imports)* | **The capability-linker runtime (sidecar 1.1.0, experimental).** Given a `NodeProfile` + skills carrying a validated `capabilities.json`, runs the 4-stage link (hard compatibility filter → temp-0 selection → dependency closure → budgeted render) and returns a `## Linked ML Capabilities` brief. `link()` never raises. Reached only when `MLEVAL_SKILL_DELIVERY_MODE` ∈ {`capability_task`, `capability_node`}; `legacy` is the default. See the capability-runtime section below. |
| `skill_injector.py` | `agents.{draft,improve,debug,evolution}_agent.{run, get_impl_guideline_from_agent}` (via a `sys.meta_path` import hook) | **The A/B treatment (skills only).** Calls `eval_harness.apply_impl_guideline_harness` first, then dispatches on `$MLEVAL_SKILL_DELIVERY_MODE` (read at call time): `legacy` (default) = progressive disclosure — Tier-0 catalog into EVERY node + per-node temp-0 selector loading relevant skill(s)+references; `capability_task`/`capability_node` route through `capability_linker` instead. |

> **Build-time patch (not a sidecar):** the Kaggle *persona* and *./input*
> framing are neutralized at image-build time by
> [`../patches/de_kaggle.py`](../patches/de_kaggle.py) (a `RUN` step in the
> Dockerfile after the upstream COPY), because the submodule tree is reset by
> `git submodule update` and can't be edited durably. See its docstring.

## Design philosophy

We limit our patches to the minimum needed for (a) the A/B treatment, (b)
keeping the agent on our contract-only tasks, and (c) a trustworthy metric.
Earlier iterations shipped `prompt_overlay.py` (per-task persona /
impl_guideline / review override) and `env_overlay.py` (custom package hint
list); both were removed — the persona problem is now solved more directly at
build time (see below), and we keep the upstream 15-package env hint as-is
(noise for text tasks, but harmless and symmetric across cells).

- **Kaggle persona / `./input` framing → neutralized at build time** by
  `../patches/de_kaggle.py`, not by a sidecar. We found the "Kaggle
  Grandmaster competing on a leaderboard" persona + "read `./input` CSVs"
  framing actively drove off-task drift (the agent solved IMDB sentiment
  classification instead of SAMSum summarization). `de_kaggle.py` replaces
  those strings with neutral "expert ML engineer implementing the task"
  framing and an HF-load `./input`-may-be-empty note.
- **Output contract / metric.** We run `no_submission_mode: False` (NOT True)
  so MLEvolve natively preserves the best node's per-example predictions at
  `best_submission/submission.csv`. Our independent held-out grader
  (`mleval.grader`, run post-exit by `entrypoint.sh`) recomputes the metric
  from that file against held-out references — the **trustworthy A/B number**.
  MLEvolve's own self-reported `Final Validation Score` stays only the
  tree-search signal + a drift diagnostic. (Background:
  `memory/project_held_out_grader_decision`; the field standard is
  artifact + independent grader, not a self-reported scalar.)

## Skill injection — progressive disclosure (the three tiers)

MLEvolve is single-shot codegen (no tool-use file reads), so it can't read a
skill into context on its own. `skill_injector` stands in for that, mapping
Anthropic's Discovery→Activation→Execution onto the universal
`get_impl_guideline_from_agent` seam (called by all four codegen agents):

- **Tier 0 — Discovery (always, every node):** the patched guideline builder
  appends `skill_retriever.catalog_text()` — each skill's name + 1-line
  description + its `references/*.md` filenames — to the
  `"Implementation guideline"` list. The agent is always aware of the whole
  library (~150–200 tokens for 3 skills).
- **Tier 1 — Activation (per node):** `_wrap_run` stashes the current
  `stage` (draft/improve/debug/evolution) + `parent_node` on the agent; the
  guideline builder runs a **temp-0 model selector** once per node
  (`llm.query(..., func_spec=select_skills, model=agent.acfg.feedback.model)`,
  logged by `prompt_logger` as `func_spec_name="select_skills"`). It returns
  which skill(s) to load.
- **Tier 2 — Execution:** the same selector picks which `references/*.md` to
  load (`[]` = SKILL.md only, `["__all__"]` = all, else specific filenames) —
  so we never dump every body into every node.

**Fallback:** empty library → no catalog, no selector (baseline, identical to
without_skill). Selector raises → all skill bodies (SKILL.md only), logged.
Selector returns `[]` → catalog only (the model declined).

## How a task uses it

1. Stage the skill **library** on the PVC, e.g. `/results/skills/{peft-tuning,
   vllm-inference,tabular-baseline}/` (each a `SKILL.md` + `references/`).
2. The orchestrator's `--skill-library /results/skills` populates
   `MLEVAL_SKILL_LIBRARY` (preferred). `--skill-path` (singular) still works
   for back-compat (sets `MLEVAL_SKILL_PATH`).
3. `without_skill` cells get an empty library → `loaded_skills()==[]` → the
   guideline passes through unchanged (no catalog, no selector call).

## Capability-linker runtime (experimental, sidecar 1.1.0)

A second, **experimental** skill-delivery path that sits alongside the legacy
progressive-disclosure injector above. It is **purely additive**: legacy mode is
byte-identical to sidecar 1.0.0, stays the default, and remains the A/B control
(`version.py`) — so a legacy 1.1.0 run and a 1.0.0 run are directly comparable,
and the capability modes are a new regime distinguished by `delivery_mode`.

**Delivery-mode switch — `MLEVAL_SKILL_DELIVERY_MODE`** (read at call time, never
cached at import, so a cell flips it via env without a rebuild):

- `legacy` *(default)* — the unchanged catalog + per-node `SKILL.md` selector
  described above. Any unrecognised value (a typo) also runs legacy, with a loud
  one-time warning, so a mis-set cell never silently ships an untested treatment.
- `capability_task` — link **once** at the first (draft/generate) node, cache the
  `LinkResult` on the shared search agent, and reuse the rendered brief verbatim
  at every later node.
- `capability_node` — **re-link fresh** at every codegen node from that node's
  current search state.

**What it delivers.** Instead of splicing whole `SKILL.md` bodies, the linker
renders a compact `## Linked ML Capabilities` brief from *capability units* — the
structured procedures a skill declares in a `capabilities.json` manifest beside
its `SKILL.md`. The manifest is produced at **build time** by the skill builder
(`--emit-capabilities`); the sidecar only **consumes** it. `skill_retriever.py`
now loads and validates that manifest (via `capability_schema.py` — schema 0.2, a
deterministic validator) and exposes the manifest-carrying skills through
`loaded_capability_skills()`. A skill without a valid, non-empty manifest is
excluded from the capability path entirely — a capability arm **never** mixes in
that skill's legacy body.

**The linker — `capability_linker.py`** is **agent-generic by construction**: it
imports nothing from `mlevolve` / `agents` / `llm` at module scope
(`capability_schema` and `llm.FunctionSpec` are imported lazily inside
functions), so any host that can build a `NodeProfile` and supply an `llm_query`
callable can link. The MLEvolve-specific glue — the `NodeProfile` adapter and the
`llm.query` closure — lives in `skill_injector.py`, never in the linker. Public
API: `NodeProfile`, `LinkBudget`, `LinkResult`, `STAGE_MAP`
(draft→generate · improve→refine · debug→debug · evolution→explore),
`parse_hardware(free-text → {gpu_count, vram_gb, gpu_name})`, and
`link(profile, cap_skills, llm_query, budget) → LinkResult`.

**`link()` runs four stages (HLD §10) and MUST NOT raise** — any internal error
returns an empty `LinkResult` with an `error` telemetry field, so a broken
manifest or a selector transport failure can never break codegen:

- **(A) Deterministic hard compatibility filter — three-valued.** Each unit is
  *known-incompatible* (reject, reason-coded), *known-compatible* (keep), or
  *unknown* (pass through to the selector — never auto-pruned). VRAM only
  hard-filters when the unit's requirement `basis == "explicit-section"`.
  Security reject-on-unknown is scoped to `credentials` **only** (per HLD §10.3);
  `network` / `persistent-service` / `multi-node` stay ordinary three-valued
  unknowns, because rejecting them would wrongly prune legit units (e.g. an HF
  hub fetch, or a `vllm serve` root that legitimately needs network).
- **(B) Temp-0 semantic selection** over the survivors — retry-once-then-decline;
  it never injects-all as a fallback.
- **(C) Dependency closure** — topological order, drop-on-missing.
- **(D) Budgeted rendering** — `LinkBudget` caps `max_units` **roots before
  closure** (default 3) and full reference bodies (default 1).

**Clean-experiment invariant.** The whole capability path in `skill_injector.py`
is wrapped so it can never break codegen and **never silently falls back to
legacy content** (`legacy_fallback` stays False). An empty capability library is
the *no-skill baseline* for these modes — the linker injects nothing rather than
borrowing the legacy selector — and that empty treatment is still logged, so it
reads as evidence rather than an absence.

**Telemetry.** `selection_logger.py` gained `log_capability_node()`, emitting one
`capability_node` record per node (delivery mode, stage, loaded/candidate/
selected ids, hard-filter reasons, dependency add/drop, rendered order, injected
chars, `legacy_fallback`) into the same append-only `selection_events.jsonl`,
stamped with `sidecar_version` for cross-run aggregation (HLD §11).

Design reference: `docs/skill-builder/capability-linker-mvp-hld.md` (§9
NodeProfile, §10 linker, §11 telemetry) and
`docs/skill-builder/capability-linker-mvp-plan.md`.

## Persona sites — now patched by `de_kaggle.py`

The recurring "Kaggle grandmaster attending a competition" phrase (11 sites
across draft / improve / evolution / fusion / aggregation / planner /
result_parse / stepwise) plus the draft-agent competition block, the
`stepwise_coder` "competition-winning code", and the `improve_agent`
"Grandmaster" / "kaggle award-winning" phrasings are all replaced at build
time by `../patches/de_kaggle.py` (each REQUIRED rule asserts it applied, so
an upstream refactor fails the build). These were previously left unpatched
and "accepted as cosmetic" — that turned out to be wrong: combined with the
skill's own classification example they drove measurable off-task drift, so
we now neutralize them.

## Known upstream behaviors we accept (not patched)

- The upstream 15-package env hint (xgboost, lightGBM, timm, etc.) is
  irrelevant noise for our text tasks but harmless and symmetric across
  cells — left as published.
- `no_submission_mode: False` makes result_parse run a content-quality check
  (`engine/validation/quality_check.py`) on the submission; this is a useful
  local anti-laziness guard (rejects empty/constant predictions) and we keep
  it. The mle-bench *format* grader on that path is auto-skipped by
  `use_grading_server: False`.
