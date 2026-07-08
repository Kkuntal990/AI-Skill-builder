# Eval-gated build — P0 + P1 implementation plan

Detailed plan to converge `build-skill-from-docs` on Anthropic's **evaluate-to-ship**
method: evaluation becomes part of the build loop, iteration is driven by observed
with/without **behavior** (run through our `skill-tester` agent), and triggering becomes
sibling-aware and situated. Implements P0 + P1 from [anthropic-parity.md](anthropic-parity.md) §8.

Companion to [hld.md](hld.md), [stage1.md](../eval/stage1.md). Cites `file:line` at the seams.

> **Implementation status (branch `skill-phase3`).** M0–M9 are implemented and committed.
> All new behavior sits behind flags that default to today's behavior — `--ship-gate off`,
> `--baseline-probe` off, `--llm-grader` off — so existing builds are unchanged. Every
> milestone was verified **offline** (`py_compile` + stubbed unit tests); a **live**
> end-to-end gated build (calling `claude -p` / `openclaw agent`, costs credit) has not been
> run and needs the user's environment. **M10 (focused-skill splitter, P1.10) is not yet
> built.** New surface: `eval_core.py` (shared primitives); `skill_builder.py` gains
> `run_ship_gate` / `repair_result_for_gate` / `baseline_gap_probe` / `check_doc_faithfulness`
> / `check_artifact_imports` / `cmd_freshness` and the `--ship-gate|--ship-anyway|--eval-agent
> |--max-repair-eval-rounds|--baseline-probe` flags + `freshness` subcommand; `eval_skill.py`
> gains `--siblings|--llm-grader` and the `optimize-description` subcommand.

---

## Goal

Turn the current **generate → validate → write** pipeline into a **generate → evaluate →
repair → gate → ship** loop, where:
- the skill is built into a **staging dir**, evaluated there, and only **promoted** to
  `~/.openclaw/workspace/skills/<name>/` when it clears the gate (or `--ship-anyway`);
- the with/without **functional content test** runs real prompts through **`skill-tester`**
  (the neutral baseline agent) — one cell with the skill loaded, one without — and grades the
  agent's **text answer** on `must_contain`/citation/LLM-judge assertions;
- the **organic-activation** signal keeps running through `claude -p` (available-not-forced,
  the existing `_run_claude_executor`);
- **failing behavior feeds a repair round** (rewrite from transcripts), not just rule hits.

> **SCOPE GUARDRAIL — the build gate is prompt-level only.** It **never** executes the skill's
> end-to-end task, trains a model, or runs code, and needs **no Docker and no GPU**. Both signals
> are cheap LLM calls: (a) the agent *answers* eval prompts and we grade the **text** it returns,
> and (b) `claude -p` measures whether it *reads* the skill unprompted. The heavyweight
> **task-solving A/B** (run a full ML pipeline, held-out grader, Docker + GPU) is **Stage-2
> MLEvolve** — deliberately **out of scope here** (see [Non-nudging note](#non-nudging-note-stage-1-vs-stage-2)).
> A per-build budget is ~30 LLM calls / cents (see [budget](#executors--transport-costtime)),
> not a GPU job.

Two prompt-level signals feed the gate, matching skill-creator's two eval tracks (neither
solves the skill's actual task):

| Signal | Question | Executor | Exists? |
|---|---|---|---|
| **Functional content A/B** | Does the skill's content make the **text answer** better? (no task execution) | `skill-tester` (`openclaw agent`), with/without cells, graded on assertions | ⚠️ in `eval_skill.py`, not in build |
| **Organic activation** | Does the skill fire on the right prompts unprompted? | `claude -p` + `.claude/skills/` catalog (`_run_claude_executor`) | ✅ 3.0-3 |
| **Triggering (judge)** | Would the description be picked over siblings? | LLM judge over description + siblings | ⚠️ vs canned decoys today |

---

## Current-state anchors (what we modify)

`skill_builder.py`: `_pipeline` (2114), `cmd_build` (2623), `cmd_preview` (2606),
`write_evals` (1159), `evaluate_triggering` (1304), `improve_description` (1371),
`write_skill` (1954), `openclaw_skills_check` (1931), `update_lockfile` (1994),
`_add_build_flags` (~2699).
`eval_skill.py`: `cmd_functional` (575), `_run_agent` (447; default `agent="ai-skill-builder"`),
`_run_claude_executor` (352), `_score_assertions` (510), `cmd_triggering` (266),
`cmd_activation` (717), `_build_report` (843), `DEFAULT_PASS_BAR` (816), `_load_pass_bar` (825).
`sb.DECOY_SKILLS` (~1260). `eval_skill.py:43` does `import skill_builder as sb`.

**Constraint:** `eval_skill` → `skill_builder` today. The build must call eval primitives
*without* a circular import → **M0 extracts them into `eval_core.py`** imported by both.

---

## Target build loop (with P0+P1 wired)

```
RESOLVE → FETCH → INTENT
   → GENERATE EVAL PROMPTS (from intent; prompts-first, assertions deferred)   [P0.3]
   → BASELINE: run skill-tester WITHOUT skill on the prompts → gap notes       [P0.3]
   → PLAN (intent + gap-notes) → WRITE BODY → CRITIC+REPAIR (rules, ≤3)
   → BUILD CONTRACT → SYNTHESIZE (staging dir) → REF-SCAN CRITIC
   → DRAFT ASSERTIONS (from body + baseline gaps)                              [P0.3]
   ┌─ EVALUATE (staging):                                                       [P0.1]
   │    • functional A/B: skill-tester with vs without × runs → grade          [P0.2 exec]
   │    • activation: claude -p available-not-forced × runs                     [3.0-3]
   │    • triggering: judge vs real siblings                                    [P1.1]
   │    • analyzer: flag non-discriminating / flaky assertions                  [P1.4]
   ├─ GATE: pass? ── no ──► BEHAVIOR REPAIR (rewrite from failing transcripts)  [P0.2]
   │                          → new iteration-N/ → re-EVALUATE  (≤ N rounds)
   └─ yes ▼
   → DESCRIPTION OPTIMIZE (held-out 60/40, pick-by-test)                        [P1.3]
   → PROMOTE staging → workspace  → openclaw skills check → LOG
```

---

## M0 (prerequisite) — extract `eval_core.py`

Move the transport-agnostic eval primitives out of `eval_skill.py` into a new
`scripts/eval_core.py`, leaving `eval_skill.py` as the CLI wrapper and letting
`skill_builder.py` import the same functions. No behavior change; pure refactor.

- **Move:** `_run_agent`, `_run_claude_executor`, `_score_assertions`, the functional /
  triggering / activation *core* loops (extracted from `cmd_*`), `DEFAULT_PASS_BAR`,
  `_load_pass_bar`, `_build_report`, MCP-signal helpers, `_SKILLTESTER_SYSTEM`.
- **Public API** `eval_core` exposes:
  ```python
  run_functional(skill_dir, tests, *, agent="skill-tester", runs=3, timeout=240,
                 baseline_skill_dir=None, grader="deterministic") -> dict
  run_activation(skill_dir, trigger_set, *, model="opus", runs=3, max_concurrency=3) -> dict
  run_triggering(skill_dir, trigger_set, *, siblings_dir=None, runs=3) -> dict
  analyze(functional_result) -> dict          # non-discriminating / flaky flags   [P1.4]
  build_report(skill_dir, triggering, functional, activation, pass_bar) -> (md, passed)
  ```
- `eval_skill.py` `cmd_*` become thin shims over these. `skill_builder.py` imports
  `eval_core` directly. Circular import resolved.
- **Tests:** existing `eval_skill.py all <skill>` must produce byte-identical reports pre/post
  refactor on a fixture skill.

---

## P0.1 — Eval-gated ship (staging + gate + promote)

**Build into staging, evaluate, promote on pass.**

- **Staging:** `_pipeline` writes the assembled skill to a temp `work_dir` (e.g.
  `$TMPDIR/skillbuild-<name>/iteration-1/`) instead of straight to `SKILLS_DIR`. `write_skill`
  gains an `out_dir` it already has; point it at staging.
- **Gate call:** new `run_ship_gate(work_dir, args) -> {passed, report, metrics}` in
  `cmd_build`, after synthesis + P0 `validate_skill`. It calls
  `eval_core.run_functional/run_activation/run_triggering` + `build_report`.
- **Gate policy** (extends `DEFAULT_PASS_BAR`, all overridable via `evals/pass_bar.json`).
  **Severity principle:** default new checks to **WARN**; reserve **BLOCK** for high-severity
  correctness failures. Too many hard gates makes builds brittle and over-blocks.
  - **Hard (BLOCK):** `triggering_f1_min` ≥ 0.85 · `activation_recall_min` ≥ 0.50 ·
    **doc-faithfulness** high-severity (P1.6) · **executable** compile/import failures (P1.8).
  - **Advisory (WARN, drives repair but doesn't block):** `functional_pass_min` ≥ 0.60 — for
    **ML skills the content A/B saturates** (the base model already answers canonical questions;
    the `peft-tuning` Stage-1 run was 100%/100%, +0 lift), so it mostly catches *regressions*,
    not gains. The *discriminating* cheap signals for a doc→skill tool are **faithfulness +
    executable-sanity + activation + triggering**, not content pass-rate — the P1.6/P1.8/P1.9
    additions below are what make the gate meaningful without task execution.
  - **lift is signal, not gate** (Goodhart-safe, [stage1.md](../eval/stage1.md)); a *negative*
    lift is a **warning that triggers repair**, not a hard block.
- **Promote:** on pass, move staging → `SKILLS_DIR/<name>/` (or `--out`), then
  `openclaw_skills_check` + `update_lockfile`. Persist the eval artifacts into
  `<name>/evals/grading_results/`.
- **Flags** (`_add_build_flags`):
  - `--ship-gate {off|smoke|full}` (default `smoke`): `off` = today's behavior (write always);
    `smoke` = prompts capped, `runs=1`, cheap; `full` = all prompts, `runs=3`.
  - `--ship-anyway`: on gate fail, write with a loud `quality_gate: failed` warning instead
    of refusing (never silently overwrite; still honors `--force`).
  - `--eval-agent` (default `skill-tester`), `--eval-model` (activation `claude -p` model),
    `--max-repair-eval-rounds` (default 2).
- **`quality_gate`** becomes the real ship decision (today it's soft, folded into warnings at
  `_pipeline` 2402-2409). This is the **Phase 3.0-6 hard ship-gate** the code already anticipates
  (`skill_builder.py:2293`).

## P0.2 — Behavior-driven repair (skill-tester with/without)

**Rewrite from what the agent actually did, not from rule violations.**

- **Executor** = `skill-tester` via `eval_core.run_functional(..., agent="skill-tester")`. The
  neutral agent (no bundled skills, [`agents/skill-tester/`](../../agents/skill-tester/)) is the
  clean baseline:
  - **without cell:** bare prompt → `openclaw agent --agent skill-tester -m <prompt>`.
  - **with cell:** prompt + the existing skill marker (`_run_agent` skill_marker,
    `eval_skill.py:588`) → the agent reads the staged `SKILL.md`.
  - `runs` each (≥3 in `full`), captured with full `reply_text` for transcript-grounded repair.
- **Grade** each assertion → `{text, passed, evidence}` (deterministic first, LLM grader tier
  P1.5 for the rest). Collect **failures** = `(prompt, assertion, with_reply, without_reply,
  evidence)` for every with-cell assertion that failed **or** where without ≥ with (no lift).
- **Repair:** new prompt `prompts/repair_from_eval.txt` + `repair_from_eval(body, refs,
  failures, contract) -> (body, refs)`. Unlike `repair_skill_body` (rule-driven full regen),
  this is **gap-targeted**: "these prompts failed these assertions; here is what the agent
  produced with vs without the skill — revise the body/reference so the with-skill run would
  pass, without over-fitting to these exact prompts."
- **Loop:** write revised skill to `iteration-N+1/`, re-evaluate, up to
  `--max-repair-eval-rounds`. Stop early when the gate passes or no assertion improved
  (convergence, mirroring skill-creator's "no meaningful progress").
- **Guardrail (anti-overfit):** the repair prompt is told to generalize; a later
  held-out check (P1.3 for triggering; a functional holdout is P2) guards the description.

## P0.3 — Baseline-first gap capture (evals before docs)

**Target the skill at real gaps, not imagined ones.** **Sequenced early (M3, before the
ship-gate):** the without-skill gaps shape the body, references, assertions, and the repair
target, so this must precede — not follow — hard gating.

- **Reorder:** split `write_evals` (1159) into `write_eval_prompts` (runs right after INTENT,
  from the intent brief + doc TOC — *prompts only*) and `draft_assertions` (runs post-synthesis,
  from the body + baseline transcripts). Mirrors skill-creator "prompts first, assertions later."
- **Baseline probe:** run `skill-tester` **without** the skill on the prompts once → capture
  where the base model already succeeds vs fails. Summarize into `gap_notes`.
- **Thread `gap_notes`** into `plan_structure` (728) + `write_body` (899): "the base model
  already handles X; it fails at Y, Z — prioritize Y, Z; don't spend the body budget on X."
  This is the doc-driven analogue of Anthropic's "run without-skill first, document failures."
- **Scope note:** the baseline probe is a **prompt** run (ask the question, read the text
  answer) — not a task execution. For some builds the prompts don't discriminate (e.g. a pure
  API-reference skill the base model already answers well); then `gap_notes` is empty and the
  pipeline proceeds as today. Non-fatal, best-effort — like `infer_intent`.

---

## P1.1 — Sibling near-miss triggering by default

- `run_triggering` defaults `siblings_dir` to the **install root of co-resident skills**
  (`SKILLS_DIR`, or `--siblings <dir>`), judging the description against **real siblings**
  instead of `sb.DECOY_SKILLS`. Canned decoys become the fallback only when <2 siblings exist.
- The builder's build-time `evaluate_triggering` (1304) already accepts siblings — make
  `eval_core.run_triggering` share that path so build-time and harness agree.
- Emits per-query near-miss confusability so the analyzer (P1.4) can flag a description that
  loses to a specific sibling.

## P1.2 — Situated, realistic trigger prompts (grounded in the fetched substrate)

- Upgrade `prompts/write_evals.txt` (and the new `write_eval_prompts`) to emit **long,
  situated** queries per skill-creator: concrete file paths, column names/values, company names,
  URLs, casual speech, occasional typos — and **genuinely tricky near-misses** that share
  keywords with the skill but need something else. Add 2-3 few-shot good/bad examples in the
  prompt. Keep the 8-10 / 8-10 balance ([stage1.md](../eval/stage1.md) triggering set).
- **Draw prompts from the whole fetched substrate, not just intent/body** (all already pulled
  by `_gather_sources`): **examples/** → tutorial-style tasks; **issues** (pitfalls/
  troubleshooting) → failure-mode prompts; **changelog** → API-migration prompts ("I'm on the
  old `X`, how do I…"); **sibling libraries** → near-miss negatives naming a real competitor.
  Grounding prompts in real usage/failures is what makes the baseline probe (P0.3) and the gate
  *discriminate* instead of saturate.

## P1.3 — Held-out description optimization (port `run_loop.py`)

- Replace the single-shot `improve_description` (1371) accept-if-better logic with a held-out
  loop in `eval_core.optimize_description(skill_dir, trigger_set, *, holdout=0.4, runs=3,
  max_iters=5)`:
  1. 60/40 train/test split of the 20-query trigger set.
  2. Evaluate current description on **train** (3 runs/query), propose a rewrite via `claude -p`.
  3. Iterate ≤5×; **blind the optimizer to test scores** (strip them from history).
  4. Return `best_description` chosen by **test** score (anti-overfit).
- Runs after the gate passes (description tuning shouldn't fight body repair). Report initial
  vs best train/test F1.

## P1.4 — Assertion analyzer (meta-evaluation)

- `eval_core.analyze(functional_result)` flags, per assertion across the with/without cells:
  - **non-discriminating** — passes (or fails) in *both* cells → doesn't measure skill value;
  - **flaky** — high variance across the `runs` (e.g. stddev over a threshold);
  - **time/token** tradeoffs (with-cell much slower/pricier for no pass gain).
- Generalizes today's single saturation flag (both arms ≥ 0.9). Output feeds the report and
  **down-weights non-discriminating assertions in the gate** (a skill shouldn't pass on
  assertions that pass without it). Explicitly *does not* propose skill fixes (that's the
  repair step) — mirrors skill-creator's `analyzer.md` separation.

## P1.5 — LLM grader tier

- Extend `_score_assertions` (510): assertions marked `type: "judge"` (or that can't be checked
  by substring) are graded by a `claude -p` **grader** using `prompts/grade_assertion.txt`,
  returning `{text, passed, evidence}`. Deterministic assertions stay first (preferred).
- Add a **human-calibration** note + optional `--grader-calibration <file>` to spot-check judge
  verdicts against human labels (Anthropic: model graders "require calibration with human
  graders"). Blinding (P2.2) applies here.

## P1.6 — Doc-faithfulness gate (source-grounding) — *near-P0 priority*

Every claim in the skill must trace to fetched source. Generalizes today's shell-command
fabrication check + the reference-scan precondition critic into one gate.
- **Claim grounding:** extract code identifiers (imports, class/function names, CLI flags,
  fenced commands) from the body + references; check each against the fetched source corpus
  (doc + README + examples + changelog). An identifier/command with **no source support** is a
  finding — **BLOCK** for a fabricated import/class/flag (hallucination), **WARN** for a
  plausible-but-unattested detail. Builds on `validate_skill`'s command-fabrication check.
- **Precondition survival:** a precondition stated in the source (VRAM floor, version gate,
  "requires a GPU", "call X before Y") must appear in the body **and** the relevant reference —
  extends the P3 contract critic + `critique_references` to *check presence*, not just ungated
  reproduction.
- **Artifact backing:** every `scripts/`/`templates/` file must be backed by a source example
  or pass a cheap executable smoke test (→ P1.8); else WARN.
- **Why near-P0:** for a doc-distillation tool a hallucinated API is worse than imperfect
  triggering, and this is a *discriminating* cheap signal (unlike the saturating content A/B).

## P1.7 — Source-drift / freshness gate (= deferred R4/R7)

Uses provenance we already stamp (`source.url` / `content_sha256` / the doc's package version).
- **Build-time:** compare the documented version to the latest release (PyPI/releases); **WARN**
  when building against visibly stale docs, or when `doc-cache.json` served a hit older than a
  threshold. Optional **BLOCK** for libraries flagged fast-moving (R7).
- **Re-validation:** a `check`/`freshness` subcommand re-fetches the source, diffs core API
  symbols vs the skill's references, and marks the lockfile entry stale — a concrete home for the
  deferred **R4** ([plan.md](plan.md) Open Items). Cheap: the symbol diff needs no model call.

## P1.8 — Executable artifact sanity (no GPU, no training)

Extends today's `py_compile`/`bash -n` (`validate_templates`/`validate_scripts`) — reaffirming
the [scope guardrail](#goal): **compile / import / signature / example only, never task
execution or GPU.**
- **Always:** `py_compile` (Python) / `bash -n` (shell) — already present.
- **When the package is importable in the build env:** resolve `import`s; check that API symbols
  the artifact calls exist with compatible signatures (`inspect.signature`); optionally run a
  bundled ≤ few-second example on a tiny/mocked input. **Skip (record "unverified: package
  absent"), don't fail,** when the package isn't installed — the builder reads docs and may not
  have `vllm`/`peft` present.
- Severity: compile → BLOCK; import/signature mismatch → BLOCK (likely hallucinated API);
  example runtime error → WARN. Feeds P1.6 artifact-backing. This is recommendation #7's
  "optional cheap executable checks" — kept strictly separate from Stage-2's task-solving A/B.

## P1.9 — Reconstruction / round-trip check (completeness)

*Does the skill actually transmit the knowledge?* Give a **neutral agent only the generated
skill** (no source) and ask it to reconstruct the key workflows, preconditions, and API usage;
compare against **source-derived gold notes** (extracted from doc/examples at build time). Gaps
= knowledge the skill failed to carry → WARN + a repair target.
- Executor: `claude -p` (skill available) → reconstruction; a judge compares to gold notes.
- **Inspired by / adapted from MIND-Skill** — verified: Li et al., 2026,
  [arXiv:2605.08670](https://arxiv.org/abs/2605.08670), *Quality-Guaranteed Skill Generation via
  Multi-Agent Induction and Deduction*. The load-bearing idea is theirs: a **frozen deduction
  agent reconstructs from the induced skill alone, with no access to the source**, scored by a
  **reconstruction loss**. Cite it as *inspired by*, **not** *as described in* — four differences:
  - the paper reconstructs the **execution trajectory** (tool/action trace), not prose
    workflows/preconditions/API;
  - it compares against the **raw source trajectory**, not "gold notes" (doc quality is a
    separate *rubric loss*);
  - it's a **TextGrad gradient signal inside an iterative optimization loop**, not a one-off gate;
  - it **induces** skills from *successful trajectories*; we **distill** from *documentation* —
    there is no source trajectory, so the "source-derived gold notes" substitution is our
    adaptation to the doc-grounded, one-shot setting.
- Heavier than the other checks (gold-note extraction + judge). Reasonable as **P2** unless we
  want it in the first gate.

## P1.10 — Focused-skill splitter (SkillsBench)

If the doc implies **more than 2–3 major workflow families**, don't ship one sprawling skill —
SkillsBench: 2–3 modules **+18.6pp**, comprehensive bundles **−2.9pp**
([retrieval-design](../eval/skill-retrieval-design.md)).
- **At PLAN time:** count workflow families; if >3, either (a) make the SKILL.md body a
  **router** with narrow per-family references, or (b) emit **multiple focused skills**.
- ⚠️ **Constraint conflict:** option (b) breaks the standing **"One URL in, one skill out"** rule
  ([hld.md](hld.md) Constraints). Recommend **(a) router-body as the default now**; treat
  multi-skill emission as an opt-in future (`--split`) that would relax that constraint.
- Upgrades today's `MAX_FOCUSED_MODULES=3` / P2-bloat *warning* into a plan-time routing decision.

---

## Data model & artifacts

Per build, under the staging dir then promoted into `<name>/evals/grading_results/`:
```
iteration-<N>/
  eval_prompts.json         # prompts-first (P0.3)
  baseline.json             # without-skill probe + gap_notes (P0.3)
  functional-<ts>.json      # with/without cells, per-run reply_text, {text,passed,evidence}
  activation-<ts>.json      # organic-activation (3.0-3)
  triggering-<ts>.json      # judge vs siblings (P1.1)
  analysis-<ts>.json        # non-discriminating / flaky flags (P1.4)
  report-<ts>.md            # build_report + gate verdict
  feedback.json             # optional human notes (P2.1 viewer writes this)
```
`manifest`/lockfile records `quality_gate`, `gate_profile`, `repair_eval_rounds`,
and the winning `iteration-N`.

## Executors & transport, cost/time

| Path | Transport | Cost | Notes |
|---|---|---|---|
| Functional A/B (skill-tester) | `openclaw agent` → OpenRouter | paid credit | user-requested; true neutral baseline |
| Activation | `claude -p` (subscription) | no per-token | needs `.claude/skills/` catalog |
| Triggering judge / desc-opt / LLM grader | `_llm_call` → `claude -p` default | no per-token | reuses build transport |

**Budget (per build):** `full` functional ≈ prompts(5) × cells(2) × runs(3) = 30 agent calls,
× (1 + repair rounds). At ~10-30s and ~$0.01-0.04/call → single-digit minutes, cents-to-low-$.
`--ship-gate smoke` (prompts≤3, runs=1) keeps interactive builds fast; `full` for pre-ship.
Alternative cheaper functional path: `claude -p` + `_SKILLTESTER_SYSTEM` persona (subscription)
— offered via `--eval-agent claude` if OpenRouter cost matters.

## Non-nudging note (Stage-1 vs Stage-2)

This with/without functional test **intentionally injects skill availability** (the with-cell
marker tells the agent to read the skill) — that is skill-creator's method and is correct for
**Stage-1** (measuring the skill's *content* value). It is a **different** philosophy from the
**Stage-2 MLEvolve A/B**, which must measure *organic* pickup and forbids any nudging
([see the no-nudging rule](../eval/stage2.md)). Keep them separate: the ship-gate here is
Stage-1; it does not change Stage-2's untouched-instruction discipline.

---

## Sequencing & milestones

Ordered so we **get the eval signal trustworthy before gating on it** (gating on a weak eval
is the top risk):

| # | Milestone | Items | Depends on | Risk |
|---|---|---|---|---|
| **M0** | Extract `eval_core.py` | refactor | — | low (pure move) |
| **M1** | Trustworthy + grounded triggering | P1.1 + P1.2 | M0 | low |
| **M2** | Doc-faithfulness + executable sanity | P1.6 + P1.8 | M0 | med (highest-value correctness gate) |
| **M3** | **Baseline-first gap capture** | P0.3 | M0, M1 | med (pipeline reorder) |
| **M4** | Analyzer + LLM grader | P1.4 + P1.5 | M0 | med (judge calibration) |
| **M5** | Eval-gated ship | P0.1 | M1, M2, M3, M4 | med (cost/time, staging) |
| **M6** | Behavior-driven repair | P0.2 | M5 | med (overfit, loop cost) |
| **M7** | Held-out desc-opt | P1.3 | M1 | low |
| **M8** | Freshness gate | P1.7 | M0 | low |
| **M9** | Reconstruction check (P2-ish) | P1.9 | M2 | med (extra judge cost) |
| **M10** | Focused-skill splitter (router-body) | P1.10 | plan stage | med |

Rationale (updated per review): **(1)** baseline-first (P0.3, now **M3**) moves *before* the
ship-gate — the without-skill gaps must shape the body, references, assertions, and repair
target, which is the whole point of "evals before docs." **(2)** doc-faithfulness + executable
sanity (**M2**) land before the gate because for a doc→skill tool they are the most
*discriminating* cheap correctness signals (the content A/B saturates). **(3)** the gate (M5)
still runs after the signals it depends on are trustworthy. Each milestone ships behind a flag
and defaults to today's behavior until proven; new checks default to **WARN**, promoting to
**BLOCK** only for high-severity correctness failures.

## Testing plan

- **M0:** report-diff on a fixture skill (byte-identical pre/post).
- **M1:** on `peft-tuning` / `vllm-inference`, confirm sibling judging changes the confusion set
  vs canned decoys; assert situated prompts contain paths/identifiers.
- **M2:** seed a deliberately non-discriminating assertion → analyzer flags it; a judge assertion
  with a known answer → grader returns correct `{passed, evidence}`.
- **M3:** a skill that fails its own functional bar must **not** promote under `--ship-gate full`;
  must promote with `--ship-anyway` + `quality_gate: failed`.
- **M4:** an intentionally thin skill improves its functional pass across ≥1 repair round; loop
  stops on convergence.
- **M5:** `gap_notes` non-empty for a task-shaped skill, empty for a pure-reference skill; build
  proceeds either way.
- **M6:** desc-opt improves held-out F1 and selects by test (not train) score.
- **Cost guardrail test:** `smoke` profile stays under a wall-clock/cost budget on CI.

## Risks & mitigations

- **Gating on weak evals** → M1+M2 first; analyzer down-weights non-discriminating assertions.
- **Build cost/latency** → `--ship-gate off|smoke|full`; `smoke` default; subscription path option.
- **Overfitting to eval prompts** → repair prompt told to generalize; held-out desc-opt; keep a
  functional holdout as P2.
- **Model variance** ([deepseek caveat](../eval/stage1.md)) → ≥3 runs, report mean±stddev, gate on mean.
- **skill-tester ambient MCP confound** → irrelevant for pure functional content grading; only
  matters for MCP-triggering prompts (use `main`/no-mcporter for those, per stage1.md).
- **Non-determinism breaking reproducibility** → persist all `reply_text` for offline re-grading;
  record `gate_profile` + model in the manifest.
