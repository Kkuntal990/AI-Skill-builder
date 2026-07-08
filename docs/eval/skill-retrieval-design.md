# MLEvolve skill-retrieval design

How the Stage-2 eval framework selects and injects skill content into MLEvolve
codegen trajectories, and the roadmap for hardening that selection.

> **Status (2026-07-07):** the **per-node LLM-selector** architecture in this doc
> is **as-built and live** (`mlevolve_sidecar/skill_{retriever,injector}.py`),
> validated in mvp-032 (first genuine with-skill paired wins). This revision
> supersedes the earlier BM25/FAISS retrieval-index design (see
> [Superseded designs](#superseded-designs)), which was specced but never built.
> **M0–M2 of the roadmap are now BUILT** (branch `skill-retriever` @ `740eb0c`,
> **sidecar 1.0.0**, image `ghcr.io/kkuntal990/mleval-agent:dev` @ `sha256:eee5693…`,
> build-time smoke green). **M3 (ablation arms) and M4 remain unbuilt** and M3 is
> the first step that spends a live GPU Job — gated on explicit approval.

## Core thesis

For a non-tool-using codegen agent, "skill retrieval" means exactly one thing:

> Given a codegen node, select the **smallest amount of skill text** to splice into
> that node's prompt so the next generated code is better.

MLEvolve is a single-shot codegen tree-search agent with **no tool-use file reads**.
It cannot navigate a filesystem, open a `SKILL.md`, or `cat` a reference mid-run. So
the activation decision must be made **at prompt-construction time, before codegen** —
we stand in for the Read action Claude Code would take. This rules out (and we do
**not** build): handing MLEvolve a skill directory, a skill-librarian CLI, runtime
skill search commands, MCP fallback during solving, or agent-side file navigation.
Those become relevant only if MLEvolve ever gains tool-use reads.

Everything reaches the agent through one seam: `get_impl_guideline_from_agent(...)`,
patched by the injector.

## As-built architecture

```text
              ┌─────────────────────────────┐
              │ infra/skills/<name>/         │   3 skills today:
              │   SKILL.md + references/*.md │   peft-tuning, vllm-inference,
              └──────────────┬──────────────┘   tabular-baseline
                             │ import time (once)
                             ▼
              ┌─────────────────────────────┐
              │ skill_retriever.py  (LOADER) │  parse frontmatter → name, description,
              │  _load_all_skills()          │  body (fm stripped), references{file:text}
              │  catalog_text() / loaded_…() │  no patching
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ skill_injector.py (PATCHER)  │  sys.meta_path hook rebinds run +
              │  _SkillPatchFinder           │  get_impl_guideline_from_agent on the
              │  → draft/improve/debug/evol  │  4 codegen agents post-load
              └──────────────┬──────────────┘
                             │ per node
                             ▼
     ┌───────────────────────────────────────────────┐
     │ Per-node selector  (_run_selector, temp-0)     │  select_skills func-call over
     │  ctx = stage + task[:6000] + err_tail/code_head │  the L1 catalog; cached once/node
     └──────────────┬────────────────────────────────┘
                    │
                    ▼
     ┌───────────────────────────────────────────────┐
     │ Injection bundle appended to impl_guideline    │
     │  Tier-0  catalog (ALWAYS, every node)           │  Anthropic progressive disclosure,
     │  Tier-1  selected SKILL.md bodies               │  simulated at prompt-build time —
     │  Tier-2  selected references/*.md               │  NOT interactive agent reads
     └──────────────┬────────────────────────────────┘
                    ▼
             MLEvolve codegen node
```

Three tiers, matching Anthropic Agent Skills (progressive disclosure is *simulated*
at prompt construction, not performed interactively by the agent):

| Tier | What | Where in code | When |
|---|---|---|---|
| **Discovery** | L1 catalog: name + description + reference filenames | `skill_retriever.catalog_text()` → `_wrap_impl_guideline` (`skill_injector.py:281`) | every node, always |
| **Activation** | full `SKILL.md` body of selected skill(s) | `_render_selected_bodies` (`skill_injector.py:214`) | only if selector picks it |
| **Execution** | selected `references/*.md` | same | only if selector picks it |

**A/B switch** (`run_ab.py:158-161`): `with_skill` gets `MLEVAL_SKILL_LIBRARY`;
`without_skill` gets `""` → zero skills load → injector no-ops → pure baseline.

**Fallback semantics** (`skill_injector.py:28-30`, `:217`): library empty → baseline;
selector errors → `_FALLBACK_ALL` (all bodies, no refs); selector returns `[]` →
catalog only.

**Fixed bug (spike-023):** the harness prepends ~3k chars of constant rules to every
task, which pushed the real task past the selector's old 1500-char window → the
selector declined every skill (`selections=[]×6`, treatment silently near-baseline).
Fixed by `_task_for_routing` stripping the header (`skill_injector.py:131`) + a
6000-char routing window (`:128`). **This is exactly the silent-weak-treatment failure
the roadmap's telemetry (M0) exists to make impossible to miss again.**

## Two selection layers (one live, one deferred)

The design has **two conceptually distinct selection layers**. They separate by
**input scope** and **frequency**, and each gates a *different* thing. Today only
Layer 2 is built; Layer 1 is deferred (see the [N=3](#the-fact-that-reorders-priorities-n--3)
rationale and M4 trigger).

| | **Layer 1 — Availability** | **Layer 2 — Activation** |
|---|---|---|
| Question | which of N skills are *available* this run? | which available skill(s) help *this node*? |
| Also called | task-level skill pool | per-node selector (`_run_selector`) |
| Sees | the task instruction only | task + stage + parent error/code |
| Runs | once per **trajectory** (at start) | once per **codegen node** (cached) |
| Gates | the **catalog** (Discovery/Tier-0) | injected **bodies + references** (Tier-1/2) |
| Output | ~8–12 skill pool | selected bodies/refs from the pool |
| Status | ⏸️ **designed, not built** (M4, trigger **N ≥ 25**) | ✅ **live** (`skill_injector.py`) |

**The invariant that keeps them separated properly:** Layer 1 shrinks the *catalog* that
the run sees; Layer 2 picks *content to inject* strictly from within Layer 1's output.
They must not collapse into one call — different context (task-only vs per-node),
different frequency (once vs per-node). A third, orthogonal control sits on top: the
per-node **caps** (M1, optional; **default uncapped**) can bound how much of Layer 2's
selection lands in the prompt.

**Layer 2 is a router, not the codegen agent.** MLEvolve cannot read files mid-run, so it
never pulls a skill organically. Layer 2 is a separate temp-0 `select_skills` LLM call
that decides *on the agent's behalf, before codegen*, and splices the result into the
prompt — "simulated progressive disclosure at prompt-build time," not interactive reads.

**Why Layer 1 is deferred, and what breaks without it at scale.** At N=3 there is nothing
to pool, so Layer 2 routes over the full library and the catalog lists all 3 skills — this
is correct and lean. Grow to N=50 *without* Layer 1 and every node's catalog would list all
50 descriptions and the selector would route over all 50 — the documented distractor
regime: organic activation drops **49% → 31% with distractors** (arXiv 2604.04323). That is
the failure Layer 1 exists to prevent, hence the **N ≥ 25** trigger. The leanness this design
promises at scale is a *property of Layer 1* — it does not hold from Layer 2 alone.

**The only availability lever that exists today is static and manual:** `MLEVAL_SKILL_PATHS`
hand-picks which skills load into the library, and the `MLEVAL_SKILL_LIBRARY`-empty A/B
switch gates the whole cell. Neither is task-driven — neither is Layer 1.

## The fact that reorders priorities: N = 3

The library is **3 skills** (peft-tuning, vllm-inference, tabular-baseline). Every
degradation threshold in the tool/skill-selection literature sits far above N=3:

- RAG-MCP (arXiv 2505.03275): selection holds **>90% below ~30 candidates**, degrades
  in the 31–70 band, collapses beyond ~100.
- SkillComposer (arXiv 2606.32025): top-3 embedding shortlist ≈ oracle retrieval only
  starts to matter at **196 skills**.
- "More Skills, Worse Agents?" / skill shadowing (arXiv 2605.24050): up to 21% drop at
  **202 skills**; selection failure = 68% of the loss, context overhead negligible.

Our two observed failures were **not** catalog-size problems: spike-023 was an
**observability failure** (routing-signal truncation, now fixed) and mvp-032's QLoRA
tilt was a **content/description failure** (peft-tuning's method-menu description). So
the plan's own closing thesis is right — *optimize for selection observability,
per-node relevance, and controlled injection, not a new agentic retrieval layer* — but
its structural centerpieces (task-level pooling, deterministic pre-ranking) are no-ops
at N=3 and are deferred behind explicit scale triggers.

One finding validates the current design outright: "Skills in the Wild"
(arXiv 2604.04323) shows organic self-triggering activation drops **49% → 31% with
distractors**. Our per-node temp-0 selector is effectively a *forced-consideration*
arm — it structurally sidesteps that failure mode.

## Feature verdicts

Grounded in source-verified literature (fetched 2026-07-06; cross-checked against
`reference_skill_failure_mode_lit` + `reference_skill_creation_optimization_lit`) and
against the current code. Legend: ✅ implement · ⚠️ implement simplified · ⏸️ defer
behind a trigger · ❌ reject.

| Feature (from refined plan) | Verdict | Grounding |
|---|---|---|
| **Selection telemetry** (structured per-node event log) | ✅ **first** | spike-023 was silent for a full run. Selector calls already land in `prompts.jsonl` (`func_spec_name:"select_skills"`, `prompt_logger.py:91`) → past runs replayable. Missing: fallback_mode, injected-token counts, node join, without_skill confirmation, analyzer awareness (0 hits in `src/mleval/analyzer/`). |
| **Reasons + `decline_reason` in schema** | ✅ | Cheap attribution; distinguishes "declined" from "silently empty". Bumps behavior (CoT effect) → version the sidecar, don't compare across the boundary. |
| **Per-node caps** (skills + refs) | ✅ built as a lever; **default UNCAPPED** | SkillsBench (2602.12670) motivates *having* the lever — 2–3 skills **+18.6pp**, 4+ **+5.9pp**, comprehensive bundles **−2.9pp**. But the mvp-032 replay found 30/32 selections were DELIBERATE explicit ref-lists (only 2 `["__all__"]`), so a hard cap trims genuine selection, not a dump-everything pathology. Shipped as a symmetric, env-tunable lever (`_render_selected_bodies`); **default uncapped reproduces mvp-032** and whether a finite cap helps is a hypothesis to A/B (Stage 3), not a baked-in baseline. |
| **Ablation arms** (oracle / random / catalog_only / none) | ✅ | SkillComposer's 5-arm design (No/All/Retrieved/Oracle/Gold = 22.2/29.3/44.0/44.0/51.1%) is the template; SkillsBench *lacks* this arm ⇒ publishable wedge. Answers the mvp-032 open question: selection-failure vs content-failure. |
| **Offline replay harness** | ✅ | Near-free: mvp-032 selector contexts already on the PVC. Test selector variants before spending GPU-h. |
| **Hardware fact → selector context** | ✅ (added) | Codegen prompt gets `**Compute**: …A6000 48GB` (`eval_harness.py:171`) but `_selector_user` (`:148`) doesn't — the selector that pulled `quantization.md` in mvp-032 never saw GPU size. Symmetric env fact. |
| **fallback_all → top-k ladder** | ⚠️ | Plan's own threshold: fallback_all fine at ≤5 skills — we have 3. Do only the cheap guard (retry once → fallback_all if N≤5 else catalog_only, always logged). Deterministic ranker it needs shouldn't exist yet. |
| **Rich catalog** (`use_for`/`not_for`/stages) | ⏸️ fix descriptions instead | Anthropic spec is two prose slots — what + when, ≤1024 chars, 3rd person; **no official `use_for`/`not_for` schema**, and no study isolates `not_for` fields (unverified). Measured lever is description quality itself (>10× usage, 2505.18135; 20%→81%, 2510.02554). Our culprit (peft method-menu) is a **builder-side** description fix. Enrichment also fights heterogeneous formats (tabular-baseline is old 0.1.0). |
| **Task-level skill pool** (8–12) | ⏸️ trigger: **N ≥ 25** | No-op at N=3; adds a 2nd LLM call + 2nd silent-empty surface. Agent verified: no clean single- vs two-stage comparison exists at 25–50 scale; hierarchical wins are measured at 196–200K (AgentSkillOS 2603.02176, AnyTool 2402.04253). |
| **Deterministic pre-ranking** (top-20) | ⏸️ trigger: **N ≥ 30** or replay shows wrong-family picks | RAG-MCP's 43.13% vs 13.62% tripling is measured against pools of hundreds+. Hand-authored keyword table = our priors on the scale (nudging risk); prefer neutral embedding shortlist when triggered. |
| **Reference summaries / section injection** | ⏸️ trigger: token telemetry shows ref bloat | Load-time LLM summarization adds import-time LLM dep + nondeterminism across paired cells. Caps capture most of the benefit for free. |
| **Retry-on-empty guardrail** ("are any general MLE skills useful…?") | ❌ | Direct **no-nudging** violation — coaxes toward skills, and names skills not in our library. Empty is a legitimate organic outcome; the wrongly-empty case (spike-023) was a routing bug (fixed) and telemetry catches recurrence in one grep. |
| **Stage-specific prefer/avoid lists** | ❌ | Subtler nudge: telling draft to prefer "fast baseline / validation" injects recipe priors — the mvp-029 QLoRA-tilt mechanism relocated into the selector. Neutral stage awareness already exists (`skill_injector.py:148-164`). |
| **Structured task profile** (`workflow_needs:[...]`) | ❌ as designed | `workflow_needs` vocabulary mirrors skill names ⇒ nudging via feature-engineering. The routing failure it targets was the truncation bug (fixed). Keep only the hardware line. |
| **Per-task unique-skill cap (10)** | ❌ | Solves a problem impossible at N=3; adds cross-node state. Revisit with the task pool. |
| **Librarian CLI / symlinks / MCP fallback / file reads** | ✅ agree — nothing to remove | Confirmed none exist in the codebase. |

## Implementation roadmap

Ordered so everything before the first GPU spend is local or replay-based. Every
sidecar change needs a rebuild on `amusing`; **hard rule stands — no live Jobs without
explicit approval.** **M0–M2 built on `skill-retriever` @ `740eb0c` (sidecar 1.0.0);
M3–M4 unbuilt.**

### M0 — Selection telemetry ✅ BUILT (sidecar 1.0.0)

- New `mlevolve_sidecar/selection_logger.py` + `version.py`: append-only JSONL at
  `$MLEVAL_SELECTION_LOG` (default alongside `prompts.jsonl` on the PVC — **not** the
  plan's off-convention `.openclaw_mle/` path). Events from `_run_selector` +
  `_wrap_impl_guideline`: cell, stage, selected skills/refs,
  `fallback_mode ∈ {none, catalog_only, fallback_all}`, declined flag, injected
  body/ref char counts, selector model, task_chars_seen.
- `without_skill` emits one "0 skills loaded" confirmation at import.
- `entrypoint.sh` exports the path; manifest records it. Analyzer: `adapter_mlevolve`
  ingests events; `aggregate` adds **treatment-empty rate** (% with_skill nodes with
  no body injected).
- **Behavior-neutral** — the telemetry itself changes no selection logic (M1's caps
  are the behavior change; both landed together on this branch).
- **As built:** `adapter_mlevolve` emits per-node `skill_selection`; `aggregate` emits
  a **Skill selection** report section (treatment-empty rate, injected sizes, fallback
  histogram). Local checks + build-time smoke green.

### M1 — Selection hardening ✅ BUILT (sidecar 1.0.0 = comparability boundary)

- Schema: per-selection `reason`, top-level `decline_reason` (strict-safe: both required).
- Injection-cap lever `MLEVAL_SKILL_MAX_PER_NODE` / `MLEVAL_SKILL_MAX_REFS_PER_NODE`
  (counting `__all__` expansion), enforced in `_render_selected_bodies` with logged
  truncation. **Default uncapped** (unset/negative = uncapped, `0` = inject none,
  positive N = cap N); a finite value is an explicit ablation arm, not the baseline.
- Fallback ladder: retry once → `fallback_all` iff N≤5 else catalog_only, always logged.
- One line in `_selector_user`: the `MLEVAL_HARDWARE` compute fact.
- Threaded new env vars: orchestrator → `job.yaml.tmpl` → `entrypoint.sh` (+ `run_ab.py`
  CLI flags `--skill-max-per-node` / `--skill-max-refs-per-node`). Version recorded in
  `manifest.agent.sidecar_version`.
- **As built:** default **uncapped** (reproduces mvp-032 organic behavior). The
  mvp-032 replay found 30/32 selections were deliberate explicit ref-lists (only 2
  `["__all__"]`), so capping to 3 would trim *genuine* selection — hence uncapped
  default, with 3/3 demoted to a Stage-3 ablation arm. `0` = catalog-only ablation.

### M2 — Replay + gold labels ✅ BUILT (local; no rebuild)

- `scripts/replay_skill_selector.py`: parses `select_skills` records from PVC
  `prompts.jsonl` (works on mvp-032/029/028), scores against gold, **simulates the
  caps** (what a given max_skills/max_refs would inject), flags low-recall /
  high-empty-rate. Falls back to `selection_events.jsonl` for new runs.
- Per-task gold labels `infra/tasks/<task>/skill_gold.json` for all 6 tasks —
  `relevant` / `acceptable` / `irrelevant` sets + confidence (jigsaw marked low).
- **Deliverable (pending data):** run it on the pulled mvp-032 PVC dirs for the first
  selector-accuracy table — zero GPU. This is the recommended next action (see
  [Testing at N=3](#testing-at-n3)).

### M3 — Ablation arms ⏳ NOT BUILT (orchestrator + sidecar; rebuild; needs live-Job approval)

- `MLEVAL_SKILL_SELECTOR_MODE = llm | oracle | random | catalog_only | none`, threaded
  per cell. Organic (`llm`) stays the headline treatment; the rest are labeled
  diagnostic arms ⇒ no nudging conflict.
- First use: re-run boolq as organic-vs-oracle. Oracle also tilts to QLoRA ⇒ content
  problem (builder). Oracle wins ⇒ selection problem (harness).

### M4 — Deferred, with explicit re-entry triggers (documented, not built)

- Task-level pool → library **≥ 25**.
- Embedding pre-shortlist → library **≥ 30** or replay shows wrong-family picks.
- Rich catalog → replay shows description-level confusion *after* builder-side repair.
- Reference summaries → token telemetry shows ref bloat.

### Parallel track — not this harness's code

The single highest-leverage item per the literature is **not** in the injection path:
rewrite peft-tuning's method-menu description (QLoRA/DoRA/AdaLoRA named up front) to
Anthropic's what+when shape. That is the skill-builder pipeline's job — see
`docs/skill-builder/skill-reliability-checklist.md` (its #1 lever) and
`project_skillbuilder_vs_anthropic_process`.

## Testing at N=3

With only 3 skills, **be explicit about what a test can and cannot show.** The
scale questions (distractor collapse, task-level pooling, pre-ranking) are
un-testable at N=3 and are correctly deferred (M4). What 3 skills *can* validate:
the **mechanism** (telemetry/caps/fallback land correctly in-cluster), the
**routing accuracy** (does the selector pick the right skill), the **attribution**
(is a with-skill loss a selection failure or a content failure), and the **caps
regression** (did bounding `["__all__"]` change the mvp-032 outcome). Test those;
don't over-claim scale.

Four stages, cheapest first:

1. **Replay on existing PVC data — DONE (2026-07-07), zero-GPU.** Ran
   `scripts/replay_skill_selector.py` over mvp-032/029/028. Result: the spike-023
   header-strip fix is validated (post-fix recall **0.59–0.75** / empty **0.00–0.21**
   vs pre-fix spike-018 recall **0.30** / empty **0.49**); **wrong-family = 0** in
   every run (the selector under-triggers, never mis-picks). A 3-ref cap *would* have
   truncated 12/25 mvp-032 nodes — but the categorization showed those were
   DELIBERATE explicit 4–5-ref picks (only 2 `["__all__"]`), which is why the ref cap
   is **not** the default. **No approval needed — reads persisted artifacts.**

2. **Live plumbing smoke — 1 trajectory, needs approval.** One task × `with_skill`
   × 1 seed, small step/time budget, on `:dev`. Purpose is *plumbing, not science*:
   confirm `selection_events.jsonl` lands, `cell_init` fires, `node_selection`
   records carry the fields, caps truncate, `aggregate` computes treatment-empty
   rate, and `manifest.agent.sidecar_version == 1.0.0`. ~1 short pod.

3. **Regression A/B + cap ablation — needs approval.** Re-run boolq (the mvp-032
   QLoRA-tilt case) on `:dev`. Two questions, one sweep: (a) *regression* —
   `without_skill` vs `with_skill` **uncapped** (the default, = mvp-032 recipe):
   confirm telemetry lands and the with-skill result reproduces, nothing broken by
   the sidecar changes; (b) *cap ablation* — add a `with_skill` arm at
   `--skill-max-refs-per-node 3`: does bounding the peft `quantization.md` pull move
   the QLoRA-tilt loss? If yes, the cap is *earned*; if not, the tilt is a content
   problem (builder fix). Needs **no new code** — the current image supports it via
   the CLI flag.

4. **Attribution arms — needs M3 (unbuilt) + approval.** `oracle` / `random` /
   `catalog_only` / `none` separate selection-failure from content-failure. Build
   M3 first; then organic-vs-oracle on boolq settles whether the mvp-032 QLoRA tilt
   is the *selector's* fault (fixable here) or the *skill's* (a builder fix).

**Growing the library for a scale test without 25 curated skills:** add a handful
of realistic **off-domain distractor** skills and use them **offline only** — feed
them to the replay harness (stage 1) to watch selector precision as N climbs toward
10–15. This de-risks the M4 triggers (task pool at N≥25, pre-ranking at N≥30)
cheaply. **Do not** route synthetic skills through a live A/B — they would confound
outcomes; they are a selector-accuracy probe, not a treatment. Real library growth
is the skill-builder's job, tracked separately.

## Evaluation layers

Because injection is prompt-construction-based, evaluate three layers (per
SkillComposer / Skills-in-the-Wild methodology):

1. **Selector** — Recall@k, Precision@k, wrong-family rate, empty-selection rate,
   fallback rate. New headline metric: **treatment-empty rate** (% with_skill nodes
   with no body injected).
2. **Injection** — tokens injected per node, bodies/refs count, catalog cost, and
   **effective injection rate** (% nodes where selected content survived prompt
   construction/truncation — the metric that would have caught spike-023).
3. **Downstream (MLEvolve)** — the M3 arms (No / catalog_only / random / oracle /
   organic) against: valid-submission rate, held-out score, compile/runtime failure
   rate, debug-recovery rate, token cost. Attribution stays on persisted
   `held_out_score.json` only (per `feedback_result_numbers_from_held_out_score`).

## Superseded designs

- **BM25/FAISS retrieval index** (this doc, pre-2026-07 revision): chunk SKILL.md at
  H2/H3 boundaries, index with `HybridRetriever`, inject top-k chunks via a
  `build_chat_prompt_for_model` wrapper. **Never built** — MLEvolve's stepwise codegen
  path doesn't route through `build_chat_prompt_for_model` (same reason `prompt_overlay`
  was removed, see `mlevolve_sidecar/__init__.py` history note). Replaced by the
  universal `get_impl_guideline_from_agent` seam + LLM selector, which reaches all four
  codegen stages. The retrieval-index approach is the M4 fallback *if* N grows past the
  point where an LLM selector over the full catalog degrades.
- **spike-004 concat-into-description.md**: dumped 75,845 chars of fenced markdown into
  the system message → DeepSeek-Flash mimicked the fences → every `runfile_0.py` opened
  with ` ```python ` → SyntaxError cascade (18 SEARCH-marker leaks). Verified the skill
  *content* worked (58 LoRA mentions vs 16 baseline, correct Qwen MLP target_modules) —
  the *delivery* was the failure. This is why injected bodies are now spliced into the
  impl_guideline (structured), never concatenated raw into the system prompt.

## Prior art (verified)

- **AutoMLGen** (arXiv 2510.08511, the canonical MLEvolve citation): §3.2 Knowledge
  Base injects model-recipes **at draft time only**; Table 3 ablation worth +9 medal
  points on MLE-Bench-Lite. Materialized at `upstream/engine/coldstart/*.json` as a
  `{category:{model:code_template}}` taxonomy; we keep it **disabled**
  (`use_coldstart:false`) — our tasks aren't in its {Image, Detection, Segmentation,
  NLP, Audio, Others} taxonomy, and prose+decision-tree+multi-file skills don't fit its
  JSON shape. We keep AutoMLGen's *architectural position* (knowledge prior at codegen
  time) but extend from draft-only to draft/improve/debug/evolution and swap the
  storage/retrieval layer.
- **Selection at scale** — Anthropic best-practices ("description critical… potentially
  100+ skills", what+when, 3rd person, ≤1024 chars, body ≤500 lines); RAG-MCP
  (2505.03275); SkillComposer (2606.32025); AgentSkillOS (2603.02176); AnyTool
  (2402.04253); "Looking Is Not Picking" (2606.16364); "How Many Tools…" (2605.24660);
  Less-is-More (2411.15399); EASYTOOL (2401.06201).
- **Skill benefit/harm** — SkillsBench (2602.12670): +16.2pp avg, 16/84 tasks negative
  (worst −39.3pp), comprehensive −2.9pp, 2–3 skills optimal, self-gen −1.3pp;
  Skills-in-the-Wild (2604.04323): distractors halve activation, loading ≠ utility;
  skill shadowing (2605.24050): selection failure = 68% of loss at 202 skills.
- **Description as the lever** — 2505.18135 (>10× usage from edited descriptions),
  2510.02554 (name+desc tuning 20%→81%).

## Scope notes

- **Cross-skill conflict**: if two skills match, both bodies get injected; the LLM
  resolves contradictions — out of harness scope (but caps bound the blast radius).
- **Versioning / hot-reload**: trajectories pin the skill snapshot at startup;
  mid-trajectory updates unsupported.
- **True L3 (`bash: cat ref.md`)**: needs a ReAct loop MLEvolve lacks; we simulate it
  by having the selector pre-choose references. Revisit only if MLEvolve gains tool use.
- **Skill authoring** is out of scope — see `agents/ai-skill-builder/` and the
  description-repair parallel track above.

## Related docs

- `docs/eval/overview.md` — two-stage pipeline
- `docs/eval/stage2.md` — A/B methodology
- `infra/agents/mlevolve/mlevolve_sidecar/README.md` — sidecar patching surface
- `docs/skill-builder/skill-reliability-checklist.md` — description-as-lever (parallel track)
