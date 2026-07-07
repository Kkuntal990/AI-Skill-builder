# Anthropic skill-creator parity — comparison & replication plan

How our `build-skill-from-docs` (builder 2.1.0) + `skill-tester` setup compares to
Anthropic's official skill-creation and evaluation workflow, and a concrete plan to
replicate the parts we're missing. Companion to [README.md](README.md), [hld.md](hld.md),
[skill-reliability-checklist.md](skill-reliability-checklist.md), and the eval docs under
[../eval/](../eval/overview.md).

Anthropic claims below are cited to primary sources (verified 2026-07, see [Sources](#sources)):
**[OV]** overview · **[BP]** best-practices · **[ENG]** equipping-agents · **[SC]** the
`anthropics/skills` skill-creator repo · **[E]** demystifying-evals. Our claims cite
`file:line`.

---

## TL;DR

**The one fundamental difference:** Anthropic's skill-creator is an **agentic, interactive,
evaluation-gated iterative loop** — a Claude instance interviews the user, writes a
*minimal* draft, then measures it by running a *fresh* Claude with-skill vs a without-skill
baseline on realistic tasks, and rewrites the skill from **observed behavior** until it
converges. Evals come *before* the writing and *gate* the loop.

Ours is an **autonomous, document-grounded, generate-then-validate pipeline** — one doc URL
in, one skill out. Iteration is a *rule-based* body critic+repair loop; the *behavioral*
eval (`eval_skill.py`) is a **separate CLI that the build never runs**, and our AGENTS.md
[explicitly rejects](../../agents/ai-skill-builder/AGENTS.md) the interactive mode, pointing
users to skill-creator for it.

**We match or exceed Anthropic on artifact quality** (progressive disclosure, description
rules, frontmatter gates, one-default-with-escape-hatch, ToC, degrees-of-freedom) **and add
things skill-creator lacks** (provenance/hashing, security + IPI scanning, doc/community
fetch substrate, MCP runtime wiring, hardware/conditional-gating). **We diverge on the
*method*:** we don't interview, we don't build evals first, and evaluation doesn't gate the
build.

**Top 3 gaps to close (the "replicate it" core):**
1. **Make evaluation gate the build** — wire the functional/triggering/activation eval as a
   closing gate (the `quality_gate` is soft today; the "hard ship-gate" is anticipated in
   code as Phase 3.0-6 but unbuilt).
2. **Behavior-driven iteration** — rewrite the skill from with/without transcripts, not just
   from rule violations.
3. **Sibling-aware, situated, held-out triggering** — judge against real co-resident
   siblings (not 5 canned decoys), with situated prompts and a 60/40 held-out description
   optimization.

---

## 1. Anthropic's CREATION checklist (primary-source)

1. **Capture intent before drafting** — mine the conversation, then ask 4 scoping questions
   (what it enables / when it triggers / output format / build test cases?). [SC]
2. **Interview for edge cases, I/O formats, example files, success criteria, deps; research
   via subagents/MCP** before writing test prompts. [SC]
3. **Build evaluations FIRST** — run Claude on representative tasks *without* the skill,
   document specific gaps/failures. [BP][ENG]
4. **Write a *minimal* draft** — `name` + `description`, then just-enough body to close the
   observed gaps. [BP][SC]
5. **Description does the triggering** — third person, "what + when", specific terms,
   slightly "pushy" against undertriggering; all when-to-use lives in the description. [BP][SC]
6. **Frontmatter limits** — `name` ≤64 lowercase/hyphens, no reserved words (anthropic/claude);
   `description` ≤1024, no XML. [OV][BP]
7. **Progressive disclosure** — L1 metadata (~100 tok) / L2 body (<5k tok, <500 lines) /
   L3 bundled resources (unlimited). SKILL.md is a table of contents onward. [OV][BP][SC]
8. **Set degrees of freedom per section** — prose (high) for open tasks, exact scripts (low)
   for fragile ones. [BP]
9. **scripts / references / assets**; references one level deep; ToC on long references
   (>100 lines [BP] / >300 [SC]); split by domain. [BP][SC]
10. **One default + escape hatch**; don't enumerate many options. [BP]
11. **Explain the *why*; prefer imperative prose over ALL-CAPS MUSTs.** [SC]
12. **Write 2–3 realistic test prompts; user signs off; save prompts first, assertions
    later.** ≥3 evals. [SC][BP]
13. **Run with-skill AND baseline as paired subagents in the same turn.** [SC]
14. **Objectively-verifiable assertions; grade genuine (not surface) success; critique the
    assertions too.** [SC]
15. **Aggregate quantitatively** (pass_rate/time/tokens, mean±stddev, delta); flag
    non-discriminating & flaky assertions. [SC]
16. **Human review gate** — surface results in the eval viewer *before* self-correcting;
    read `feedback.json`. [SC]
17. **Improve by generalizing, reading transcripts, and promoting repeated ad-hoc work into
    bundled scripts.** [SC]
18. **Iterate the whole loop** into fresh `iteration-N/` dirs until happy / feedback empty /
    no progress; then expand the test set and re-test at scale. [SC]
19. **Optimize the description on a held-out split** — 20 queries (8–10 trigger / 8–10
    near-miss), 3 runs each, 60/40 train/test, ≤5 iters, pick by *test* score. [SC]
20. **Test across models; validate & package** (`.skill`), preserving the name on update. [BP][SC]

## 2. Anthropic's EVALUATION checklist (primary-source)

1. **Evals before docs** — they are "your source of truth." [BP]
2. **Start from real failures**; baseline = Claude without the skill. [BP][E]
3. **Every functional eval runs with-skill and without-skill, paired, same turn** (baseline =
   no skill for new; pre-edit snapshot for improvements). [SC][BP]
4. **≥3 realistic prompts; prompts before assertions.** [SC][BP]
5. **Deterministic/code graders preferred → LLM graders where necessary → humans to
   calibrate.** Script any checkable assertion. [E][SC]
6. **Assertions must be *discriminating* and evidence-backed** — exact `{text, passed,
   evidence}` schema; passes only on genuine success. [SC]
7. **Meta-evaluate the evals** — grader critiques weak assertions; an analyzer flags
   non-discriminating (pass-in-both) and flaky (high-variance) ones. [SC]
8. **Replicate (≥3 runs/config); report mean±stddev + with/without delta.** [SC][E]
9. **Unambiguous, solvable tasks** (two experts agree on pass/fail; reference solution). [E]
10. **Isolated trials; grade outcomes not paths.** [E]
11. **Always read transcripts — never trust the score alone.** [E][SC]
12. **Human review gate before rewriting.** [SC]
13. **Separate 20-query triggering eval** (8–10 / 8–10); negatives are genuinely tricky
    **near-misses** sharing keywords; prompts long & situated. [SC]
14. **Optimize triggering on a 60/40 held-out split; select by test score; blind the
    optimizer to test scores.** [SC]
15. **Guard judge bias with blinding** (blind A/B comparator) + human calibration. [SC][E]
16. **Iterate to convergence, then scale; watch saturation** (all-pass = refresh; 0%
    pass@100 = broken task). [SC][E]

## 3. Our method — how `build-skill-from-docs` (2.1.0) creates & evals

**Creation pipeline** (`skill_builder.py`, one autonomous run, no user in the loop):
RESOLVE → FETCH(doc/README/examples/[issues/changelog/SE]) → **INTENT** (inferred from doc
or `--intent`, 3.0-2) → EXTRACT hw hints → IPI scan → **PLAN** (1 LLM call) → **WRITE BODY**
(1 LLM call) → **CRITIC+REPAIR** (≤3 rounds, rule-based P1–P4) → **BUILD CONTRACT** (3.0-1)
→ **SYNTHESIZE** refs/templates/scripts/evals in parallel (contract-threaded) → **REF-SCAN
CRITIC** (3.0-1) → **TRIGGERING eval** (description judge; `improve_description` on a miss)
→ ASSEMBLE frontmatter (deterministic) → **VALIDATE** (P0 hard gates) → WRITE → LOG.

- **Gates:** P0 deterministic hard-fail (`validate_skill`: name/desc/dead-pointer/security/
  line-cap/fabrication); P1–P4 *soft* critic+repair (`critique_skill`, ships-with-warning via
  `quality_gate`); reference-scan `P3-ungated-reference` (`critique_references`).
- **Transport:** `_llm_call` → `claude -p` (subscription) default, OpenRouter fallback.
- **Input / source of truth:** a library **documentation URL** (not a user task).
- **Interactivity:** none by design — AGENTS.md refers interactive requests to skill-creator.

**Evaluation** (`eval_skill.py`, a **separate CLI the build never invokes**; run by hand):
- `triggering` — description LLM judge over positives + near-miss negatives, vs **5 canned
  `DECOY_SKILLS`** (real siblings only via the builder's own `--siblings`, not here).
- `functional` — with/without A/B via `openclaw agent` (with-cell = prompt marker
  "read the installed skill"), **deterministic** `must_contain` / `must_contain_any` /
  `must_not_contain` / `expected_citations`; no LLM grader.
- `activation` — organic-activation via `claude -p` (skill available-not-forced in a temp
  `.claude/skills/` catalog) → activation precision/recall/F1 + false-activation rate. **This
  is a near-direct port of skill-creator's `run_eval.py`.**
- Bars (advisory unless run via `pass-bar`): F1 ≥ 0.85, functional ≥ 0.60, citation ≥ 0.50,
  activation_recall ≥ 0.50 (when run). Saturation flag when both arms ≥ 0.9.
- `skill-tester` = a passive baseline eval-*target* (no skills, no memory; loads a skill via
  a prompt marker); its persona is reused inline by the `claude -p` activation executor.

## 4. Fundamental method contrast

| Axis | Anthropic skill-creator | Ours (build-skill-from-docs) |
|---|---|---|
| **Shape** | Agentic, interactive, human-in-the-loop | Autonomous, one-shot script pipeline |
| **Source of truth** | The user's *task* + observed agent behavior | A library *documentation URL* |
| **Intent** | Interview (4 scoping Qs) + research | Inferred from the doc, or `--intent` flag |
| **Order** | Evals **first**, then minimal draft | Draft **first**, evals synthesized last |
| **Iteration driver** | Observed with/without **behavior** in transcripts | **Rule** violations (P1–P4 regex + 1 LLM critic) |
| **Eval ↔ build** | Evaluation **gates** the loop | Evaluation **decoupled** (separate CLI, soft `quality_gate`) |
| **Convergence** | Loop until user-happy / feedback-empty / no-progress | Fixed ≤3 repair rounds; single description-optimize pass |
| **Human role** | Reviews eval viewer before each rewrite | None in the build; manual spot-check after |

## 5. Feature comparison — CREATION

| Feature | Anthropic | Ours | Status |
|---|---|---|---|
| Interactive intent capture (4 Qs) | ✅ [SC] | ⚠️ doc-inferred / `--intent`, non-interactive (3.0-2) | **Partial (by design)** |
| Evals-first / gap-driven | ✅ [BP] | ❌ docs first, evals last | **Gap** |
| Minimal draft | ✅ [SC] | ⚠️ full synthesis from doc plan | Divergent |
| Progressive disclosure (L1/L2/L3, <500 ln) | ✅ [OV] | ✅ target 60–150, cap 500 | **Match** |
| Description: 3rd-person, what+when, pushy | ✅ [BP][SC] | ✅ `critique_skill` P1 + `improve_description` | **Match** |
| Frontmatter limits (name/desc/reserved) | ✅ [OV] | ✅ P0 `validate_skill` (exact same limits) | **Match** |
| Degrees of freedom | ✅ [BP] | ✅ shape-principles + P3 conditional-gating | **Match** |
| One default + escape hatch | ✅ [BP] | ✅ P3 conditional-gating (arguably exceeds) | **Match/exceed** |
| Explain-why / no ALL-CAPS | ✅ [SC] | ✅ `critique_skill` P3-allcaps | **Match** |
| scripts / references / assets; ToC>100; refs 1-deep | ✅ [BP][SC] | ✅ scripts+refs+templates, ToC>100; ⚠️ no `assets/` | **Match** (minor: no assets) |
| Behavior-driven iterate-to-convergence | ✅ [SC] | ❌ ≤3 rule-based repair rounds | **Gap** |
| Provenance (url/hash/version/coverage) | ❌ | ✅ `assemble_frontmatter` | **We exceed** |
| Security + IPI scanning | ❌ | ✅ 60-pattern scan + IPI on community | **We exceed** |
| Doc/community fetch substrate | ❌ (user provides) | ✅ HTTP + `gh` + Stack Exchange | **We exceed** |
| MCP runtime fallback wiring | ⚠️ mentions MCP | ✅ declares mcps + inline triggers + `serve` | **We exceed** |
| Hardware/resource gating | ❌ | ✅ hardware hints + P3 conditional-gating | **We exceed** |

## 6. Feature comparison — EVALUATION

| Feature | Anthropic | Ours | Status |
|---|---|---|---|
| Evaluation gates the loop | ✅ [BP][SC] | ❌ decoupled CLI; `quality_gate` soft; 3.0-6 unbuilt | **Gap (core)** |
| Without-skill baseline | ✅ [SC] | ✅ `functional` A/B (but not in build) | Partial |
| Functional with/without A/B | ✅ [SC] | ✅ `eval_skill.py functional` (prompt-marker cells) | **Match** (decoupled) |
| Deterministic > LLM > human graders | ✅ [E][SC] | ⚠️ deterministic only; LLM grader deferred; manual human | Partial |
| `{text,passed,evidence}` schema | ✅ [SC] | ✅ modeled on it (per stage1.md) | **Match** |
| Discriminating assertions + meta-eval analyzer | ✅ [SC] | ❌ only a saturation flag | **Gap** |
| ≥3 runs, mean±stddev, delta | ✅ [SC] | ⚠️ `--runs 3`, delta yes; stddev not reported | Partial |
| Isolated trials, grade outcomes | ✅ [E] | ✅ fresh `claude -p` / per-trial agent | **Match** |
| Read transcripts, not just score | ✅ [E] | ⚠️ `reply_text` preserved; no enforced read | Partial |
| Human review gate (viewer/feedback) | ✅ [SC] | ❌ none | **Gap** |
| Triggering: 20 queries, near-miss **siblings**, situated | ✅ [SC] | ⚠️ 20 (10+10) but vs **canned decoys**; situated quality unverified | **Gap** |
| Triggering optimize: 60/40 held-out, pick-by-test, blind | ✅ [SC] | ❌ single rewrite, keep-if-better, no holdout/blinding | **Gap** |
| Blind A/B comparator (anti-self-preference) | ✅ [SC] | ❌ | **Gap** |
| Organic-activation measurement | ✅ `run_eval.py` [SC] | ✅ `_run_claude_executor` (3.0-3) — a port | **Match (recent)** |
| Cross-model testing | ✅ [BP] | ❌ | Gap (minor) |

## 7. Where we already match or exceed Anthropic

Keep these — they exist because our use case (autonomously distilling third-party library
docs at scale, then A/B-testing them inside an MLE agent) is different from skill-creator's
(a human co-authoring one bespoke skill). Don't regress them while replicating the loop:

- **Provenance & reproducibility** — content hash, source URL/repo, `builder_version`,
  coverage list in frontmatter.
- **Security posture** — 60-pattern scanner (BLOCK/CAUTION) + ML-safe filter + IPI scan of
  community sources + dead-pointer + shell-fabrication check. skill-creator has none of this.
- **Ingestion substrate** — doc HTML, README, examples, issues, changelog, Stack Exchange —
  skill-creator assumes the human brings the content.
- **MCP runtime tail-coverage** — declared `mcps`, per-workflow inline triggers, skill-as-MCP
  `serve`.
- **Hardware/conditional-gating** — the P3 critic + contract-threaded reference scan (the
  mvp-029 QLoRA fix) generalize Anthropic's "one default + escape hatch" to *resource*
  preconditions.
- **Downstream Stage-2 MLEvolve A/B** — a *real* behavioral eval (held-out grader over a full
  MLE-agent run), which is stronger than skill-creator's single-turn functional A/B — it's
  just post-hoc and external rather than part of the build.
- **The organic-activation executor (3.0-3)** already replicates skill-creator's `run_eval.py`
  triggering method.

## 8. Replication roadmap (prioritized)

Ordered to converge our *method* on Anthropic's while keeping §7. Each item names the file(s)
to change. This is the concrete form of the already-anticipated **Phase 3.0-6 hard ship-gate**
and the "Phase E behavioral eval" direction.

### P0 — make evaluation part of the build loop (the fundamental shift)
- **P0.1 Eval-gated build.** After WRITE, have `cmd_build` optionally run the Stage-1 harness
  (triggering + functional + activation) and turn `quality_gate` into a *hard* gate behind a
  flag (`--eval-gate` / `--ship-gate`). Files: `skill_builder.py cmd_build`, import/shell
  `eval_skill.py`. This is Phase 3.0-6.
- **P0.2 Behavior-driven repair.** Feed functional-A/B failures (with/without transcripts)
  back into a `repair_skill_body` round — iterate on *behavior*, not just P1–P4 rule hits.
  Files: `skill_builder.py` repair loop + a new "repair-from-eval" prompt.
- **P0.3 Baseline-first gap capture.** Before/while synthesizing, run a without-skill baseline
  on the intent's tasks to target *real* gaps (Anthropic's "evals before docs"). Partial for
  doc-driven builds, but even a single baseline probe would align the artifact to gaps.

### P1 — high-value harness upgrades (mostly `eval_skill.py` + prompts)
- **P1.1 Sibling near-miss triggering by default.** Make `cmd_triggering` judge against the
  real co-resident library (`--siblings`) instead of the 5 canned `DECOY_SKILLS`. Files:
  `eval_skill.py cmd_triggering` (the builder already supports `--siblings`).
- **P1.2 Situated trigger prompts.** Upgrade `write_evals.txt` to emit long, detailed,
  realistic queries (file paths, column names, casual speech, typos) per skill-creator, and
  genuinely tricky near-misses. Files: `prompts/write_evals.txt`.
- **P1.3 Held-out description optimization.** Replace the single `improve_description` rewrite
  with a 60/40 train/test loop: 3 runs/query, ≤5 iters, select `best_description` by *test*
  score, blind the optimizer to test scores. Files: `skill_builder.py evaluate_triggering` /
  `improve_description` (port `run_loop.py`).
- **P1.4 Assertion meta-eval / analyzer.** Add an analyzer that flags non-discriminating
  (pass-in-both) and flaky (high-variance) assertions — generalize the saturation flag. Files:
  `eval_skill.py` (new `analyze` step) + report.
- **P1.5 LLM grader tier.** Add a model-based grader for assertions that can't be checked with
  substrings (Anthropic's 2nd tier), with a human-calibration note. Files: `eval_skill.py
  _score_assertions` + a grader prompt.

### P2 — rigor & polish
- **P2.1 Human review gate.** Emit an HTML with/without diff viewer (we have Artifact support)
  before accepting — the `eval_review.html` analog.
- **P2.2 Blind A/B comparator.** For any LLM-judged functional comparison, blind the judge to
  which cell is with-skill (anti-self-preference). Files: `eval_skill.py`.
- **P2.3 Variance reporting.** Report mean±stddev across the 3 runs, not just the mean/delta.
- **P2.4 Cross-model check.** Run triggering/functional on Haiku + Sonnet + Opus.

### Explicitly out of scope (divergent by design, not a gap)
- Full interactive interview mode — AGENTS.md deliberately delegates this to skill-creator;
  our `--intent` flag is the autonomous substitute. Adopt scoping *questions* only if/when we
  add a conversational front-end.

---

## Sources

Anthropic (primary, verified 2026-07):
- **[OV]** Agent Skills overview — https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- **[BP]** Skill authoring best practices — https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- **[ENG]** Equipping agents for the real world with Agent Skills — https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- **[SC]** `anthropics/skills` — `skills/skill-creator/` (SKILL.md, `agents/{grader,comparator,analyzer}.md`, `references/schemas.md`, `scripts/{run_eval,run_loop,improve_description,aggregate_benchmark,package_skill,quick_validate}.py`) — https://github.com/anthropics/skills/tree/main/skills/skill-creator
- **[E]** Demystifying evals for AI agents — https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Skills launch — https://claude.com/blog/skills (redirect from anthropic.com/news/skills)

Ours: `agents/ai-skill-builder/skills/build-skill-from-docs/scripts/{skill_builder.py, eval_skill.py}`,
`agents/ai-skill-builder/AGENTS.md`, `agents/skill-tester/`, and [hld.md](hld.md) /
[stage1.md](../eval/stage1.md).

**Unreconciled in Anthropic's own sources** (carry as ranges, don't hard-code): ToC threshold
>100 [BP] vs >300 [SC]; ALL-CAPS "MUST" is a yellow flag [SC] but offered as a prominence fix
[BP]; L1 metadata "~100 tokens" [OV] vs "~100 words" [SC].
