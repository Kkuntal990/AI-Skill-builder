# Capability-Linker MVP — Verified Positioning, Corpus Feasibility, and Refined Plan

**Status:** planning; companion to the [capability-linker MVP HLD](capability-linker-mvp-hld.md)

**Date:** 2026-07-15

**Inputs synthesized here:**

1. A source-verified literature sweep (~40 works, every arXiv page fetched 2026-07-15; all
   2026 results are author claims unless noted).
2. A structural feasibility scan of
   [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs)
   — a **reference library** used to pick target libraries/domains and to stress-test the
   schema/compiler on diverse real doc-skills. It is NOT the source of production skills:
   every shipped skill is created by our own builder. Corpus skills serve E0/E1 only and
   never enter the MLEvolve eval library (§4.1).
3. A ground-truth audit of every code seam the HLD §15 names (all exist; details in §4).

---

## 1. Executive summary

**Literature:** no single verified work covers the HLD's four-part combination, and no work
anywhere implements two of its clauses — *hard hardware/artifact compatibility gating before LLM
selection* and *contraindication-compliance / selection-correctness evaluation of doc-derived
units*. But every individual clause now has a published neighbor, several of them absent from the
HLD's related-work table. Four narrowings are mandatory (§2.2), the most important being that
**MLEvolve itself natively ships state-conditioned retrieval** ("Retrospective Memory"), so
per-node retrieval as such cannot be claimed even within the host agent. The claim survives as a
**systems composition + domain instantiation**, not a representational or mechanistic first —
which is consistent with the HLD's own §1 framing, now with the receipts to defend it.

**Feasibility:** compiling AI-Research-SKILLs-style doc skills into `capabilities.json` is
**feasible with compiler inference**. 94 of 98 skills follow one strict template in which the
hardest schema fields (`applicability.when`, procedure steps, recovery, hardware constraints)
are pre-authored and near-mechanically extractable. Three honest gaps: upstream provenance
bottoms out at the skill file; ~⅓ of units need synthesized `validation`/`avoid_when`; and the
corpus's scale (98 skills → ~350–650 units) turns selection-at-scale from a deferred question
into a real MVP condition. The 4 raw-scraped skills (axolotl, llama-factory, unsloth, deepspeed)
must be excluded or re-authored.

**Plan:** keep the HLD's gate structure but re-target it: the compiler is a **builder stage**
with two entries (normal build `--emit-capabilities` + `compile-existing` package ingest), every
gate is **fully automated** (no human in the loop — the builder's evaluate-to-ship machinery
extends to the manifest), the schema gets a two-tier provenance model, synthesized-field flags,
and agent-generic stage classes, and E2's replay library grows to ~10–12 skills using corpus
categories as a natural distractor set. The linker core is written against an abstract
NodeProfile so only a thin adapter is MLEvolve-specific (§4.6). Everything through E2 is
zero-GPU. Total small-MVP surface: ~3–4 weeks of offline work before any cluster decision.

---

## 2. Literature verdict

### 2.1 Verification status

All ten works cited in HLD §3/§18 exist with accurate titles and (with the corrections in §2.5)
accurate one-line characterizations. The sweep additionally examined ~30 works the HLD does not
cite; the ones that matter are in §2.4. Full axis-by-axis notes live in the session record; this
section keeps only what changes decisions.

**The key question — answered.** *No existing MLE/data-science agent re-selects external
knowledge or skills per search node conditioned on node state (stage, parent error, hardware).*
The decomposition:

| Axis of the claim | Closest published neighbor | What the neighbor lacks |
|---|---|---|
| Retrieval inside an MLE tree search, more than once per task | AutoMind (arXiv:2506.10974): papers at draft actions, Kaggle tricks at improve actions | Query is task-label similarity; blind to parent code/error/hardware; silent at debug |
| Per-node, state-scoped context injection in an MLE search agent | ML-Master (arXiv:2506.16499): memory assembled at every node expansion | Injects only its *own* exploration history, scoped topologically (parent+siblings); zero external knowledge |
| Error-conditioned retrieval in an MLE agent | MLEvolve itself (arXiv:2606.06473): error-message-as-query at debug; plan-as-query at planning | Self-experience + a static curated model list, not doc-derived units; no contracts; no gating; per-node frequency unspecified in the paper |
| Feedback-conditioned per-iteration re-selection of external material | DS-Agent (arXiv:2402.17453, linear loop over Kaggle cases); DeepEvolve (arXiv:2510.06056, open-web research per evolutionary iteration) | Not tree search / not MLE-pipeline domain; free-form text; no gating |
| State-conditioned per-step skill re-selection | SGDR (arXiv:2606.04391, web agents); SkillReranker (arXiv:2607.06283, embodied/text games) | Not MLE; no hardware/artifact gating; no dependency closure |
| Structured skill contracts (applicability/validation/recovery fields) | OpenClaw-Skill (arXiv:2606.16774); Anything2Skill (arXiv:2606.09316, incl. contraindications) | Trajectory-distilled or generic-domain; no machine-checkable `requires`; contraindications never *enforced* or compliance-measured |
| Selection + budget + rendering as one controlled object | SkillsInjector (arXiv:2605.29794, incl. anti-shadowing "Not for:" clauses) | Once per task, offline, task-only conditioning; no closure |
| Docs → precondition/effect contracts gating what the agent sees | Contract2Tool (arXiv:2606.07904) | Gates *executable tools* with symbolic conditions; no guidance units, no MLE search, no provenance |

**Uncovered by any surveyed work:** (a) hard hardware/artifact compatibility filtering of
knowledge before LLM selection; (b) capability units compiled from library documentation with
source provenance; (c) mechanism-level evaluation of selection correctness / adoption /
contraindication compliance prior to task-lift claims.

### 2.2 Four mandatory narrowings

1. **MLEvolve's Retrospective Memory (arXiv:2606.06473) — mandatory repositioning.** The host
   agent natively does task-typed cold-start KB lookup at init (repo:
   `engine/coldstart/models_guidance_classified.json`) plus plan- and error-conditioned retrieval
   over its own experience (BM25 + FAISS, reciprocal rank fusion). Consequences: (i) the HLD's
   R18 row must say so; (ii) the claim narrows to *doc-derived, source-grounded units re-linked
   uniformly at every code-generation node* vs self-experience memory at planning/debug triggers;
   (iii) **the E3 protocol must control for native memory** — determine whether the vendored
   snapshot (@26bde89) ships it and hold its configuration identical across all arms (§4.5).
2. **AutoMind (arXiv:2506.10974)** kills any "first knowledge retrieval inside an MLE tree
   search" phrasing. Narrow to node-*state* conditioning (parent code, error, hardware),
   retrieval at debug nodes, and contracted units vs plain-text tricks.
3. **SkillReranker + SkillsInjector + SGDR** jointly kill generic "adaptive/dynamic skill
   selection" novelty. Narrow to the MLE graph-search setting + compatibility gating + dependency
   closure. Cite SkillsInjector's explicit once-per-task limitation as the direct foil.
4. **OpenClaw-Skill + Anything2Skill + the agent-skills survey (arXiv:2605.07358)** kill
   schema-field novelty — applicability/requires/validation/recovery as fields, and even the
   abstraction "skill = (instructions, resources, applicability conditions)" are published.
   Narrow the representation claim to: compiled from library documentation, with provenance and
   *machine-checkable* requires (runtime/hardware/artifacts).

### 2.3 The residual defensible gap, restated

> Doc-compiled capability units with provenance and machine-checkable requires-contracts,
> hard hardware/artifact compatibility filtering **before** LLM selection, independent re-linking
> at **every node** of an MLE graph search conditioned on (stage, task, parent code/error,
> hardware), with dependency closure and budgeted rendering, evaluated **first at mechanism
> level** (selection correctness, adoption, contraindication compliance).

Every individual clause has a neighbor; the conjunction — in particular the compatibility-gating
and contraindication-compliance clauses — has none as of 2026-07-15. This matches the HLD's
"hypothesis to test, not a first claim" stance; the sweep confirms the stance is necessary, not
merely cautious.

Conceptual honesty requirements (from the operator-learning and skills-formalization lines):

- The capability unit **is** a STRIPS/HTN-style operator adapted to non-executable guidance. Cite
  the lineage — Guan et al. (arXiv:2305.14909), NL precondition/effect world models
  (arXiv:2409.12278), Text2World (arXiv:2502.13092), the planning-modelers survey
  (arXiv:2503.18971), SayCan (arXiv:2204.01691) — and say "operator-style guidance units," never
  "a novel schema."
- **Contrast Contract2Tool explicitly**: contracts gate executable tools with machine-evaluable
  symbolic conditions; capability units gate prose guidance whose validation/recovery are
  advisory text the agent self-applies during code generation. Without the contrast the papers
  look identical.
- **Adopt the survey vocabulary** (arXiv:2605.07358): corpus-derived acquisition, applicability
  conditions, dependency-aware retrieval, context-aware dynamic selection. Position inside the
  taxonomy as (arguably) the first *system* instantiating that full column for ML-engineering
  guidance.
- **Position relative to Agent Skills as a machine-checkable formalization layer**, ideally
  compiling to/from spec-compliant SKILL.md. Anthropic's spec already carries informal analogues
  of most fields (description when-to-use; free-text ≤500-char `compatibility`; in-body
  validation loops; depth-wise progressive disclosure). Claiming skills "lack applicability or
  validation guidance" would be factually wrong; they lack *enforceable, typed, state-conditioned*
  versions, and the spec has no inter-skill dependency edges and no negative applicability.
- Safe novelty claims: doc-version provenance enabling staleness detection; typed artifact
  ontology + hardware minimums as a hard pre-filter; the reference/procedure/diagnostic/workflow
  kind taxonomy; the end-to-end compile→link system for MLE agents with mechanism-first eval.
  Claims to avoid: "novel schema," "first to structure documentation for agents," "first
  precondition-aware selection," "first state-conditioned context injection."

### 2.4 Must-add citations (not in the HLD today)

| Work | Why it must be cited |
|---|---|
| MLEvolve Retrospective Memory (in R18, arXiv:2606.06473) | Host agent's native per-state retrieval; the first reviewer objection |
| Contract2Tool (arXiv:2606.07904) | Docs → symbolic precondition/effect contracts → selection gating; conceptual twin |
| Agent-skills survey (arXiv:2605.07358) | Formalizes skills with applicability conditions + state-conditioned selection; supplies the vocabulary |
| SoK: Agentic Skills (arXiv:2602.20867) | Defines skills with applicability conditions + progressive disclosure patterns |
| SGDR (arXiv:2606.04391) | Per-step state-grounded skill retrieval replacing task-level retrieval (web agents) |
| SkillReranker (arXiv:2607.06283) | State-conditioned per-interval skill re-ranking (embodied/text) |
| SkillsInjector (arXiv:2605.29794) | Selection+budget+rendering as one object; anti-shadowing rendering; once-per-task foil |
| OpenClaw-Skill (arXiv:2606.16774) | Skill records with applicability/requires/validation/recovery fields (trajectory-distilled) |
| ML-Master (arXiv:2506.16499) | Per-node state-scoped injection precedent (internal memory only) |
| MLE-STAR (arXiv:2506.15692) | One-shot task-level web retrieval in MLE; debug retrieves nothing |
| SELA (arXiv:2410.17238) | Static self-generated insight pool attached per MCTS node |
| CBR R&D-Agent (arXiv:2606.05250) | Structured cases + injection provenance + reuse-detection (adoption-adjacent) |
| DeepEvolve (arXiv:2510.06056) | Per-iteration external research inside evolutionary code search |
| Operator-learning lineage (arXiv:2305.14909, 2409.12278, 2502.13092, 2503.18971, 2204.01691) | The schema's intellectual ancestry; omission reads as ignorance |
| EasyTool (arXiv:2401.06201) / DRAFT (arXiv:2410.08197) / Gorilla (arXiv:2305.15334) | Docs → structured unit compilation precedents |
| RCR-Router (arXiv:2508.04903) | Stage-aware budgeted context routing |
| Skill eval/evolution survey (arXiv:2606.11435) | Confirms the mechanism-metric slot (selection correctness / contraindication compliance) is open |

### 2.5 Corrections the HLD needs (factual, not stylistic)

1. **R18/MLEvolve row**: add Retrospective Memory (cold-start model-guidance KB + per-step
   BM25/FAISS experience retrieval at planning and debug). Also reconcile node naming: the paper
   describes Progressive-MCGS operations (primary expansion / intra-branch evolution /
   cross-branch reference / multi-branch aggregation); draft/improve/debug/evolution is the
   vendored snapshot's agent-module vocabulary — state which snapshot (@26bde89) is patched.
2. **R12/HASP row**: HASP *does* include mechanism-level evaluation (trigger/intervention
   analysis). The HLD may only claim its specific axes (build fidelity, selection correctness,
   contraindication compliance) as unoccupied, not mechanism-level eval per se.
3. **R9/SkillFoundry row**: note it ships per-skill *tests* (build-fidelity-adjacent) and is
   self-evolving.
4. **R17/SkillsBench**: inventory is 87 tasks in v4 (2026-06-14); headline +16.6pp; 16/84 tasks
   negative in the earlier version our memos cite.
5. **R7/Skill Seekers**: it has a static quality gate (threshold-scored, non-zero exit) — "no
   evaluation" would overstate.

---

## 3. Corpus feasibility

### 3.1 The corpus as measured (fetched 2026-07-15)

- **98 skills, 23 numbered categories**; MIT license; created 2025-11; last push 2026-06-16
  (v1.7.2); actively maintained; built by a "Skill Seeker MCP" scraper pipeline then curated
  against Anthropic's vendored best-practices docs.
- **94/98 follow one strict template**: frontmatter (name, description, version, author, license,
  tags, usually version-pinned `dependencies`) → When-to-use → Quick start →
  Workflows-with-checklists → When-vs-alternatives → Common issues → Hardware requirements →
  `references/*.md` → Resources footer. Median curated bundle ≈ 3,700 words (~5k tokens);
  SKILL.md mean ≈ 1,400 words. No `scripts/` anywhere — pure doc skills.
- **4/98 are raw scraper dumps** (single lines up to 61k chars; unsloth 195k words): `axolotl`,
  `llama-factory`, `unsloth` (all of `03-fine-tuning` except `peft`), and `deepspeed`.
  **Exclude from the MVP** (re-author later if needed). PEFT is the only curated fine-tuning
  skill.
- **Relevant categories for us:** 03-fine-tuning (peft), 12-inference-serving (vllm, sglang,
  tensorrt-llm, llama-cpp), 04-mech-interp (transformer-lens, nnsight, saelens, +1),
  05-data-processing (ray-data, nemo-curator), 06-post-training (8 skills),
  08-distributed-training (6), 10-optimization (7: bitsandbytes, awq, gptq, gguf,
  flash-attention, …), 11-evaluation (lm-evaluation-harness, bigcode-eval, nemo-evaluator),
  13-mlops (4).
- Skills load in our retriever **as-is** (frontmatter carries `name` + `description`; body
  regex-parseable) — the legacy arm can consume corpus skills without modification.

### 3.2 Field-by-field extraction matrix (from an 8-skill sample)

| Schema field | Verdict | Notes |
|---|---|---|
| `id`, `title`, `summary` | DIRECT | Section headers are crisp |
| `kind` | INFERABLE (high conf.) | Checklist workflows → procedure/workflow; tables → reference; verify-steps → diagnostic |
| `applicability.when` | DIRECT | Pre-authored twice: description + "Use X when:" bullets |
| `applicability.avoid_when` | DIRECT at skill level; INFERABLE→partially ABSENT at unit level | Skill-level contraindications authored (peft: "use full FT when <1B"; flash-attn: "Not supported: V100, CPU"); per-unit avoid_when often needs inverting conditional advice |
| `requires.runtime` | INFERABLE (reliable) | `tensor_parallel_size`→multi-gpu; `vllm serve`→persistent-service; `HF_TOKEN`→credentials |
| `requires.hardware` | DIRECT ~half, INFERABLE rest | Explicit "Hardware requirements" sections + VRAM tables; numbers are **unattributed** (risk §3.3) |
| `requires.artifacts` / `produces` | ABSENT as typing; INFERABLE from code | `save_pretrained(...)` → ml:adapter; namespace is compiler-invented |
| `procedure.steps` | DIRECT (curated) | Numbered, bolded, copy-paste checklists |
| `reference_files` | DIRECT | Explicit topic-labeled pointers |
| `validation` | DIRECT ~⅓, INFERABLE ~⅓, **ABSENT ~⅓** | Gold standard exists (flash-attn "diff <1e-3", vllm "TTFT < 500ms"); mandatory-validation forces synthesis for the rest |
| `recovery` | DIRECT | Universal "Common issues" keyed by verbatim error strings |
| `sources` | DIRECT for skill-internal; **ABSENT for upstream** | Resources-footer root URLs only; e.g. vllm's 4 reference files have 0 links in 1,212 lines |
| `depends_on` | ABSENT explicit; INFERABLE | Step ordering + integration sections (peft→vLLM/TRL) give edges |
| 3–7 units/skill | Fits naturally | Curated skills decompose to ≈4–7 units |

### 3.3 Top risks and their mitigations

| Risk | Mitigation (now part of the plan) |
|---|---|
| Provenance bottoms out at the skill file; upstream anchors don't exist; the skills' own numbers are uncited | Two-tier provenance (§4.2): `sources[].kind = skill-package \| upstream-doc`, `upstream_verified: false` by default. Never fabricate upstream anchors. E1 adds a cheap upstream spot-check (~20 numeric claims) to quantify corpus trustworthiness as a limitation |
| ~⅓ of units get compiler-**synthesized** validation/avoid_when — plausible-looking, unexecuted | Compiler self-reports `synthesized_fields` per unit (§4.2); E1 audits extracted vs synthesized separately; synthesized-claim support counts against H2 |
| Unattributed VRAM/perf numbers graduating into hard planner constraints | Linker policy: hard-filter only on gpu-required / min_gpu_count; `vram_gb` is advisory unless extracted from an explicit Hardware-requirements section (`hardware.basis` flag, §4.2) |
| 4 scraped outliers (incl. 3 of 4 fine-tuning skills) | Excluded from MVP corpus; peft is the curated fine-tuning representative |
| Scale: 98 skills → ~350–650 units; heavy applicability overlap (peft/unsloth/axolotl/llama-factory all claim LoRA; bitsandbytes-QLoRA duplicates peft-QLoRA) — the over-selection/shadowing failure we've already documented | MVP library stays curated-and-audited (~10–12 skills); overlap becomes an E2 *feature*: category siblings are the distractor set the 3-skill library never gave us. Full-scale retrieval stays deferred per HLD §16 |
| Corpus staleness (authored ~2025-11, `version: 1.0.0` everywhere, no update tracking) | Pin a snapshot (commit SHA) of the skills used; record in provenance; staleness detection is a *payoff* of doc-version provenance, not a blocker |

One pointed alignment: the corpus's PEFT skill carries the **same ungated QLoRA-forward framing
that caused the mvp-029/mvp-032 with-skill failures**. Compiling `avoid_when` by inverting its
conditionals ("QLoRA when <24GB" → "avoid QLoRA when VRAM ≥ 40GB") is exactly the fix that RCA
prescribed — this is the MVP's best motivating example and should appear in E0.

### 3.4 Verdict

**Feasible with compiler inference; no corpus re-authoring required.** The 94 curated skills map
onto the schema with DIRECT extraction for when/steps/recovery/reference-pointers/hardware and
reliable inference for kind/runtime/artifacts/depends_on, provided (a) sources downgrade to
skill-internal locators, (b) synthesized validation/avoid_when are flagged and audited, (c) the
4 scraped outliers are excluded.

---

## 4. Design deltas vs the HLD

Ground truth from the seam audit that shapes these deltas: all §15 paths exist.
`skill_retriever.py` parses only `name`/`description` (no structured-metadata handling — the
manifest loader is a clean addition); the selector already receives stage + hardware + capped
task/parent context and returns strict function-call JSON; telemetry
(`selection_events.jsonl`) has an additive schema; injection is **uncapped by default**
(`MLEVAL_SKILL_MAX_PER_NODE` = ∞), so the HLD's ≤3-unit budget is a real treatment change;
`scripts/replay_skill_selector.py` exists and spike-018 samsum provides 80 recorded
`select_skills` contexts with gold labels locally.

### 4.1 The compiler is a builder stage (no standalone tooling, no human artifacts)

The capability manifest is a **second compiled form of the same skill** — same bundle, same
ship-gate, one more build output — produced by `skill_builder.py` itself. Every production skill
is created by our builder; the capability manifest rides on that build. One compiler module, one
gate, two entry points:

- **`--emit-capabilities`** on a normal build (THE canonical path): docs URL → builder →
  `SKILL.md` + references + `capabilities.json` in one coherent authored pass. Runs at the
  existing post-reference slot (body + references + Skill Contract materialized, frontmatter not
  yet assembled). These skills carry true upstream provenance (`docs_sha256`, `library_version`).
  Every skill that enters the MLEvolve eval library (E2/E3) is built this way.
- **`compile-existing <skill-dir>`** subcommand (secondary utility, not a skill source): compiles
  a manifest from an on-disk bundle without rebuilding. Two uses only — (a) retrofit a manifest
  onto skills our builder already produced before the flag existed (`infra/skills/*`), and (b)
  compile the **reference-corpus** skills as an E0/E1 stress corpus. Provenance grounds in the
  pinned package snapshot (`sources.kind = skill-package`), never fabricated upstream anchors.
  Corpus outputs are test artifacts; they never ship in the eval library.

The compiler uses a schema-constrained function-calling helper (new to the builder — it
currently regex-extracts JSON; the sidecar's `FunctionSpec` pattern is the model) over the
existing `MLEVAL_LLM_TRANSPORT` (claude-CLI subscription default). Outputs are machine artifacts
only: `capabilities.json` plus fidelity metrics appended to the machine-readable eval report
under `evals/` that `run_gate` already consumes. No review tables, no human-facing documents.

### 4.2 Schema 0.2 deltas (all additive)

```json
"sources": [{
  "kind": "skill-package | upstream-doc",
  "locator": "SKILL.md#lora-workflow | references/quantization.md#awq",
  "upstream_url": "optional",
  "upstream_verified": false
}],
"synthesized_fields": ["validation", "applicability.avoid_when"],
"requires": { "hardware": { "vram_gb": 24, "basis": "explicit-section | inferred" } }
```

- `sources[].kind` makes two-tier provenance honest: units compiled from corpus skills ground in
  the pinned skill-package snapshot; upstream anchors only when they actually exist.
- `synthesized_fields` is the compiler's self-report of fields it authored without direct textual
  support; E1 audits these at a stricter bar.
- `hardware.basis` drives the linker's hard-vs-advisory policy (§3.3).

### 4.3 NodeProfile hardware parsing

`MLEVAL_HARDWARE` reaches the selector today as one free-text string ("1 NVIDIA RTX A6000 GPU
(48 GB VRAM), 8 CPUs, 32 GB RAM"). The hard filter needs structured facts: add a deterministic
regex parser → `{gpu_count, gpu_name, vram_gb}`; parse failure → `unknown` → no hard filter
(three-valued logic per HLD §10.3). No new state sources.

### 4.4 Library scale for the MVP

The HLD's §4.3 "three skills" constraint is superseded: the E2 replay library grows to ~10–12
skills — **all built by our own builder** (`--emit-capabilities`), covering the target
libraries plus category-sibling distractors (an optimization, an evaluation, an mlops skill).
The builder is fed the docs for those libraries; corpus skills are NOT reused as ship/distractor
artifacts (they stay E0/E1-only). This makes H3's "reduce irrelevant or contraindicated
exposure" measurable against realistic near-miss distractors and directly tests the documented
over-selection failure mode. Task-level pooling / embedding pre-ranking stay deferred behind
their scale triggers.

### 4.5 Native-memory control (new methodological requirement)

Before E3: determine whether vendored MLEvolve @26bde89 contains Retrospective Memory
(`engine/coldstart/`, memory-enhanced planning modes) and whether our spike config enables it.
Whatever the answer, its configuration must be **identical across all four arms** and recorded in
the manifest. If enabled, the paper must discuss capability-linking *on top of* native memory; if
absent from the snapshot, say so and cite the paper's mechanism anyway (§2.2.1).

### 4.6 Generic core vs MLEvolve adapter

The method is piloted on MLEvolve but designed to port. The split mirrors the repo's existing
plugin axis (harness code vs `infra/agents/<name>/`):

**Agent-agnostic (the method):**

- `capabilities.json` schema + deterministic validator — pure data, no agent coupling.
- The compiler — a builder stage consuming a skill bundle; knows nothing about the consuming
  agent.
- The linker core — hard filter → temp-0 selection → dependency closure → budgeted rendering,
  written against an abstract `NodeProfile {stage, task_text, parent_code, parent_error,
  hardware, runtime_capabilities}`. Any agent that can produce those fields can link.
- **Canonical stage classes** in the schema: `generate | refine | debug | explore` replace the
  HLD's `delivery_hints.mlevolve_stages`; adapters map native vocabulary onto them.
- The telemetry event schema and the perturbation-fixture format (§E1′/E2′).

**MLEvolve adapter (the pilot's plugin):**

- `skill_injector.py` dispatch — the `sys.meta_path` rebind of `get_impl_guideline_from_agent`
  on the 4 codegen agents, and the `impl_guideline` landing site.
- `NodeProfile` construction from MLEvolve node objects: stage from the agent module
  (draft/improve/debug/evolution → generate/refine/debug/explore), `parent.term_out /
  analysis / code` with the existing caps, task text via `_task_for_routing`, hardware parsed
  from `MLEVAL_HARDWARE`.
- `selection_events.jsonl` placement beside `prompts.jsonl`; the replay adapter over our
  `prompt_logger` format; env plumbing (`MLEVAL_SKILL_DELIVERY_MODE`, library path) through
  entrypoint/job template; the image rebuild.
- Vendored MLEvolve source stays untouched (sidecar-only invariant holds).

For the MVP the generic core ships as dependency-free modules inside `mlevolve_sidecar/`
(generic by construction, not yet by packaging); extracting them to `src/mleval/` or a
standalone package post-MVP is mechanical. Porting to a second agent = one new adapter
(NodeProfile constructor + injection seam + stage map) — the same cost shape as adding an agent
plugin today.

---

## 5. Refined experiments

Gate order and kill criteria are unchanged from HLD §12–13 except where noted. Everything
through E2′ is zero-GPU and requires no cluster access. Corpus skills are used from a pinned
commit SHA.

### E0′ — schema capacity probe (agent-authored, no pipeline code, ~1 day)

> **Status: DONE 2026-07-17 — PASS.** 24 units across peft/vllm/transformer-lens/ray-data +
> held-out verl, all VALID; held-out needed 0 new top-level fields. Schema 0.2 frozen
> (`capability_schema.py`, 17 validator tests). Artifacts + findings (F1–F6):
> `docs/skill-builder/capability-mvp/e0/e0_report.json`. Held-out pick was **verl**
> (06-post-training). Next gate: W3 compiler + E1′.

**What it tests:** whether the schema *can* represent heterogeneous ML skills at all, before
any compiler exists. Cheap falsification: if a best-case author with the source text in front
of it cannot encode a library without breaking the schema, no compiler prompt will. "Author"
is the orchestrating research agent, not a human — the artifacts are probe `capabilities.json`
files checked by the deterministic validator; no human labor, no human-facing documents. Second
function: the probe manifests become the reference standard E1′ compares compiler output
against, which is what lets a later failure be localized as "schema can't say it" vs "compiler
didn't say it."

- **Dev libraries (curated corpus skills as source material):** `peft` (03), `vllm` (12),
  `transformer-lens` (04), `ray-data` (05) — chosen for maximal representation pressure
  (conditional procedures + hardware gates; persistent services; read-only diagnostics;
  pipelines).
- **Held-out:** one curated skill from 06-post-training or 08-distributed-training (picked from
  the actual curated list at execution time; deepspeed is a scraped outlier). Schema freezes
  *before* it is encoded; the gate is that it fits with no new top-level field.
- 3–5 units per library; record the fraction of `validation`/`avoid_when` that had to be
  synthesized (calibrates E1′ — the feasibility scan predicts ~⅓).
- **Must-include unit:** peft QLoRA with an inverted-conditional `avoid_when` ("VRAM ≥ 40GB and
  no memory pressure") — the mvp-029 RCA case.
- Pass gate: ≥80% of sampled workflows encode without loss; held-out needs no schema change.

### E1′ — compiler fidelity gate (fully automated, ~3–4 days to build, minutes to run)

> **Status: grounding gates PASS 2026-07-17** (W3 shipped as builder 2.3.0:
> `capability_compiler.py` + `compile-existing` + `--emit-capabilities`; W4 grounding critic:
> `capability_grounding.py`, sonnet critic vs default-model compiler). 7/7 skills compiled
> VALID (45 units; 5 first-attempt, 2 needed one repair round); 657 claims audited: strict
> support **0.9894**, fabrication **1.06%** (6 caught — mostly cross-context borrowing),
> applicability **0.9556**, validation-usable **0.9688**, correction proxy **0.0444**. All five
> gates pass. Pending inside E1: upstream spot-check + behavioral analogs (fold into W5/W6) +
> run_gate wiring. Full numbers: `docs/skill-builder/capability-mvp/e1/e1_report.json`.
> Ops note: claude-CLI transport must run sequentially in a foreground shell — rapid
> succession/background shells fail transiently and silently fall back to OpenRouter (stale
> key → 401).

**What it tests:** whether the *pipeline* produces faithful manifests — is the compiler making
things up? E0′ established the schema has the capacity; E1′ establishes the builder fills it
without unsupported claims. No human review anywhere; the gate reuses and extends the builder's
existing evaluate-to-ship machinery:

1. **Deterministic validator** (schema, dead refs, DAG acyclicity, stage classes, security
   scan) — hard fail.
2. **Grounding critic:** every unit claim checked against the source bundle by a *different
   model* than the compiler (no self-grading), following the existing `check_doc_faithfulness`
   pattern, with a bounded critique→repair loop like the body's. Claims listed in
   `synthesized_fields` face a stricter bar: entailed-by-source or explicitly flagged, never
   silently asserted.
3. **Behavioral analog of the Stage-1 eval (`eval_core` extension):** (a) a **linking eval** —
   per-skill fixtures of synthetic NodeProfiles (right-stage / wrong-stage / wrong-hardware /
   sibling-distractor) on which the linker must select or abstain correctly — the triggering-F1
   analog; (b) a **functional A/B** — the existing executor pattern answering
   scripted-assertion probes with the rendered capability brief vs without — the told-to-read
   analog.
4. **Automated upstream spot-check:** ~20 numeric claims (VRAM figures, thresholds) fetched and
   verified against upstream docs — not a gate; a quantified limitation, and the trigger for
   demoting corpus numbers to advisory.

- Runs on the same 5 corpus skills (`compile-existing`) + our own built `peft-tuning` and
  `vllm-inference` (`--emit-capabilities`; provenance-rich contrast).
- All metrics land in the machine-readable eval report under `evals/`; `run_gate` consumes them
  like every other bar.
- Pass gates: support ≥0.90 (w.r.t. the skill-package source); applicability ≥0.85; usable
  validation ≥0.80; material-correction proxy ≤0.20 (measured as repair-loop convergence — the
  fraction of fields the critic forced to change); extracted-vs-synthesized reported separately.

### E2′ — offline replay + perturbation (~1 week)

> **Status: W5 DONE + deterministic E2 PASS 2026-07-18.** W5 shipped the runtime (sidecar
> 1.1.0): `capability_linker.py` (agent-generic 4-stage linker), `capabilities.json` loading in
> `skill_retriever`, the `MLEVAL_SKILL_DELIVERY_MODE` dispatch in `skill_injector`, capability
> telemetry — **88 tests green**, legacy path byte-unchanged (default), no silent legacy
> fallback in capability arms. W6 shipped `scripts/replay_capability_linker.py` +
> `tests/test_capability_perturbation.py` and ran the **deterministic** E2 on the local
> spike-018 replay (80 reconstructed nodes) + the perturbation suite. Deterministic gates PASS:
> expected-response 3/3 (100%) on relevant perturbations, invariance 6/6 (100%) on irrelevant,
> **0/80 hard-constraint violations** on real nodes + 0 across 18 perturbation admitted-unit
> checks. Numbers: `docs/skill-builder/capability-mvp/e2/e2_report.json`.
>
> **LLM-selection pass DONE 2026-07-18 (W6+).** Ran the real temperature-0 `--with-selector`
> replay over all 80 nodes (sequential foreground claude-CLI, resumable per-node cache): **ACTUAL
> exposure reduction 82.37%** vs legacy (295 → 52 units; ≥30% gate **PASS**), **0/80 hard-constraint
> violations**, and a clean stage gradient — the selector injects on generate (6.5→2.0 units/node)
> and refine (4.33→1.67) but **abstains on debug (3.11→0) and explore (2.0→0.17)**, the per-node
> contraindication-compliance signal `capability_task` can't produce. A validity guard
> (`selector_health`, `masked_failure==0`) distinguishes real abstention from silent transport
> failures — it caught one masked node (spurious 82.71%), which was evicted+re-run to the corrected
> 82.37%. Machine record: `e2_selector_report.json` + per-node `e2_selector_nodes.jsonl`. **Still
> pending inside E2:** selected-unit precision/recall/abstention **vs node-level gold** — the
> selections are now persisted for grading, but scoring correctness needs an *independent* gold
> label set (no human in loop → stronger-model / multi-vote adjudication, not the same selector).

- Implement loader + NodeProfile + linker + renderer behind
  `MLEVAL_SKILL_DELIVERY_MODE = legacy | capability_task | capability_node` (runtime code,
  exercised entirely offline). **[DONE — W5]**
- **Replay corpus:** spike-018 samsum (80 recorded `select_skills` contexts, local; deterministic
  replay + `--with-selector` LLM pass both **done**). Per-node selections persisted at
  `e2_selector_nodes.jsonl`; node-level *gold* labels are **not yet built** (the remaining follow-on).
  Optional widening: pull mvp-032 gsm8k/boolq selection contexts off the PVC via a
  1-CPU/2-Gi reader pod (no Job approval needed; tear down after).
- **Library:** audited peft + vllm capability manifests + 6–10 distractor manifests compiled from
  category siblings (bitsandbytes, flash-attention, lm-evaluation-harness, one mlops skill, …).
- Gold capability labels for 30–50 stratified nodes; perturbation pairs per HLD §12-E2
  (hardware perturbations now exercise the structured parser). The perturbation-fixture format
  is the same one E1′'s linking eval uses per skill — built once, used at build time (per-skill
  gate) and at campaign level (cross-library replay).
- Set the capability-mode token budgets from these replay distributions (HLD §10.2-D's
  requirement), and record legacy-uncapped vs capability-budgeted context sizes explicitly —
  H3's "within legacy prompt budget" needs the legacy baseline measured, since legacy is
  currently uncapped.
- Pass gates: unchanged.

### E3′ — gated live pilot (unchanged shape; extra controls)

- Prerequisites: E1′+E2′ pass, native-memory audit done (§4.5), `make ab-plan` workflow,
  **explicit user approval** (hard rule).
- Arms, tasks, seeds: as HLD (no_skill / legacy_node / capability_task / capability_node ×
  {gsm8k or boolq, llama-inference, house-prices} × 1 seed = 12 trajectories; extend to 3 seeds
  only if mechanism metrics warrant).
- **First-run simplification (user decision 2026-07-18):** do NOT run all four arms up front.
  Run **`capability_node` only**, and reuse the existing mvp-032 `legacy`/`no_skill` baselines —
  valid only after a **one-trajectory `no_skill` reproduction check** on the rebuilt image (the
  linker is a flag-gated addition, so the baseline should reproduce; if it doesn't, the
  environment drifted and legacy/no_skill run fresh). Mechanism metrics (adoption, compliance,
  per-node selection) are self-contained on the `capability_node` trajectories. This drops
  `capability_task` from the first pass (and with it the H4 decomposition) — add it back only if
  `capability_node` shows signal.
- Additional controls: identical native-memory config across arms; library = audited peft + vllm
  manifests only (house-prices remains the abstention control); selection telemetry gains the
  capability-event fields (HLD §11) — additive, analyzer-compatible.

---

## 6. Small-MVP work plan

Status as of 2026-07-18: **W1–W6+ DONE and passing** (E0/E1/E2-deterministic + the ≥30%
exposure-reduction gate all pass); **W6++ (gold precision/recall) and W7 (live pilot) pending.**

| # | Work item | Where | Status |
|---|---|---|---|
| W1 | Schema 0.2 + deterministic validator + tests | `mlevolve_sidecar/capability_schema.py` (17 tests) | **DONE** — schema 0.2 frozen |
| W2 | E0′ probe manifests (5 libraries) + gate verdict | `docs/skill-builder/capability-mvp/e0/` | **DONE — E0 PASS** (24 units, held-out verl 0 new fields) |
| W3 | Builder compiler stage (`--emit-capabilities` + `compile-existing`) | `skill_builder.py` 2.3.0 + `capability_compiler.py` + `prompts/compile_capabilities.txt` | **DONE** |
| W4 | E1′ automated fidelity gate (grounding critic + repair loop) | `capability_grounding.py` + `prompts/critique_capabilities.txt` + `docs/.../e1/` | **DONE — E1 grounding PASS** (0.989 support, 1.06% fabrication) |
| W5 | Sidecar runtime: loader + NodeProfile + linker + injector dispatch + telemetry + tests | `mlevolve_sidecar/capability_linker.py` (new) + `skill_retriever`/`skill_injector`/`selection_logger` (extended); sidecar 1.1.0 | **DONE** (88 tests) |
| W6 | Offline replay + perturbation fixtures + E2 report (deterministic) | `scripts/replay_capability_linker.py` + `tests/test_capability_perturbation.py` + `docs/.../e2/e2_report.json` | **DONE — deterministic E2 PASS** (80 nodes, 0 violations; perturbation 100%) |
| W6+ | `--with-selector` LLM replay pass for the ≥30% exposure-reduction gate (+ `selector_health` validity guard) | `scripts/replay_capability_linker.py --with-selector` + `docs/.../e2/e2_selector_report.json` + `e2_selector_nodes.jsonl` | **DONE — exposure gate PASS** (82.37% reduction, 80/80 nodes, masked_failure 0; abstains on debug/explore) |
| W6++ | Node-level gold precision/recall/abstention scoring of the persisted selections | (grade `e2_selector_nodes.jsonl` vs an independent gold set) | **PENDING** — remaining E2 follow-on; needs independent gold (not the same selector) |
| W7 | *(gated)* image rebuild + live MLEvolve pilot + adoption audit | `make docker-mlevolve` / orchestrator | **PENDING** — needs native-memory audit (§4.5) + explicit user approval |

Sequencing (all satisfied through W6+): W1→W2 gate (E0); W3→W4 gate (E1); W5→W6 gate (E2);
W6→W6+ (exposure gate); W7 only after user approval. Kill criteria: HLD §13 verbatim, plus: if the E1′ upstream
spot-check finds a high error rate in corpus numeric claims, demote all corpus-derived hardware
numbers to advisory permanently (never hard-gate on them).

## 7. Open items

1. **`--with-selector` LLM replay pass — DONE 2026-07-18.** Exposure-reduction gate **PASS**
   (82.37% reduction over 80 nodes, `masked_failure==0`; abstains on debug/explore). The remaining
   H3 piece is **selected-unit precision/recall/abstention vs node-level gold** (W6++): the
   selections are persisted (`e2_selector_nodes.jsonl`); grading needs an *independent* gold set
   (no human in loop ⇒ stronger-model or multi-vote adjudication with full context, never the same
   selector — else it's circular). Next runnable step.
2. **Compiler coverage-recall (NEW, 2026-07-18)** — the coverage-recall audit
   (`capability_coverage_audit.py`, the recall axis missing from the precision-only E1 grounding
   critic) shows the compiler UNDER-produces units: peft captured 30/51 (58.8%) and vllm only
   13/61 (21.3%) of the solvable issues their own source teaches, gaps concentrated in
   explore+debug. This substantially explains the E2 selector's debug/explore abstention as a
   *compilation* artifact, not pure pruning — so exposure-reduction and coverage-recall are in
   tension and must be reported together (`e1/coverage_audit_summary.json`). Fix is a BUILDER
   change (compiler chunking/prompt to lift cross-cutting operational knowledge — error traps,
   decision points, memory hygiene — into units) + rebuild; NOT hand-authoring units. Feeds the
   W6++ gold step (grade recall against full source, not just compiled units).
   **→ Design directions + verified literature: [capability-packaging-directions.md](capability-packaging-directions.md)**
   (2026-07-19): *guarantee recall structurally, buy precision semantically* — two-layer index
   (deterministic AST enumeration + LLM annotation, totality as an O(|T|) lint), aspect sweeps
   for cross-cutting concerns, typed-graph lints (the stage-consistency check explains all 90
   E2 `dep_incompatible` rejections), two-tier runtime fallback (ToC ≈ 900 tok/skill), needle
   regression suite; migration M1–M4. Verified novelty: no published system audits a compiled
   skill library against its source corpus.
3. **Vendored-MLEvolve memory audit** (§4.5) — grep `infra/agents/mlevolve/upstream/` for
   `coldstart` / memory-enhanced planning; check our `config.yaml`. Blocks E3/W7 only.
3. **Corpus snapshot pinning** — vendor the eval-library-source skills at a recorded SHA (only if
   the reference corpus is used; the eval library itself is builder-made per §4.1/§4.4).
4. **E3 first-run simplification (user decision, 2026-07-18)** — run `capability_node` only and
   reuse mvp-032 legacy/no_skill baselines after a **one-trajectory `no_skill` reproduction check**
   on the rebuilt image (validates the image is a pure flag-gated addition before trusting the
   reused numbers). Drops `capability_task` from the first pass (drops the H4 decomposition; add it
   back only if `capability_node` shows signal). See §5/E3′.
5. **Optional mvp-032 PVC pull** for replay diversity (reader pod, 1 CPU / 2 Gi, tear down after).
6. **HLD amendments** — apply §2.5 corrections + §2.4 citations to
   `capability-linker-mvp-hld.md` §3/§18 (the R18 Retrospective-Memory correction is now applied;
   remaining citations pending paper submission).

**Settled:** held-out library = verl (W2); grounding-critic model = sonnet, ≠ compiler (W4).

## 8. New references (beyond HLD §18)

MLE-agent landscape: AIDE arXiv:2502.13138 · AutoMind arXiv:2506.10974 · DS-Agent
arXiv:2402.17453 · SELA arXiv:2410.17238 · MLE-STAR arXiv:2506.15692 · R&D-Agent
arXiv:2505.14738 · CBR R&D-Agent arXiv:2606.05250 · ML-Master arXiv:2506.16499 · Agent K
arXiv:2411.03562 · AutoKaggle arXiv:2410.20424 · FM Agent arXiv:2510.26144 · DeepEvolve
arXiv:2510.06056.

Skills / selection: SoK arXiv:2602.20867 · agent-skills survey arXiv:2605.07358 · SkillsInjector
arXiv:2605.29794 · OpenClaw-Skill arXiv:2606.16774 · SkillReranker arXiv:2607.06283 · SGDR
arXiv:2606.04391 · SkillTester arXiv:2603.28815 · skill eval/evolution survey arXiv:2606.11435 ·
Skilldex arXiv:2604.16911.

Operator/contract lineage: SayCan arXiv:2204.01691 · Guan et al. arXiv:2305.14909 · NL
precondition/effect world models arXiv:2409.12278 · Text2World arXiv:2502.13092 ·
planning-modelers survey arXiv:2503.18971 · ConceptAgent arXiv:2410.06108 · Contract2Tool
arXiv:2606.07904.

Docs→structured units / context: EasyTool arXiv:2401.06201 · DRAFT arXiv:2410.08197 · Gorilla
arXiv:2305.15334 · API-Bank arXiv:2304.08244 · EVOR arXiv:2402.12317 · GraphSkill
arXiv:2603.06620 · RCR-Router arXiv:2508.04903 · StateAct arXiv:2410.02810 · context-engineering
survey arXiv:2507.13334 · Agent Skills spec (agentskills.io/specification) + Anthropic
engineering post + best-practices docs.

Corpus: Orchestra-Research/AI-Research-SKILLs (GitHub, MIT, v1.7.2, 2026-06-16).

All 2026 arXiv results are author claims verified to exist (pages fetched 2026-07-15) but not
independently reproduced. Re-run the sweep immediately before paper submission.
