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
| `skill_retriever.py` | *(loader — no patch)* | Loads a skill **library** from `$MLEVAL_SKILL_LIBRARY` (a dir; scans `*/SKILL.md`, skips `_`-prefixed) or `$MLEVAL_SKILL_PATHS`/`$MLEVAL_SKILL_PATH` (back-compat). Exposes `loaded_skills()` (per-skill `body` + `references` map) and `catalog_text()`. |
| `skill_injector.py` | `agents.{draft,improve,debug,evolution}_agent.{run, get_impl_guideline_from_agent}` (via a `sys.meta_path` import hook) | **The A/B treatment.** Anthropic progressive disclosure: Tier-0 catalog (name+desc+ref filenames) into EVERY codegen node; a per-node temp-0 model selector (`select_skills` func-call) loads only the relevant skill(s)+references (Tier-1/2). Patches each of the 4 agent modules individually because `from agents.prompts import get_impl_guideline_from_agent` copies the name per-module. |

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
