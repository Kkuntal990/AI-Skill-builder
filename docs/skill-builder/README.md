# AI-Skill Builder — System Index

Entry point for the **skill-builder** system: the OpenClaw agent that turns a Python
package's documentation URL into an installable, progressive-disclosure `SKILL.md`.
Start here, then follow the map into the detailed design and evaluation docs.

**Current version:** builder `2.1.0` (Skills-3.0 — intent capture + contract-threaded
references + P3 conditional-gating critic). Provenance is stamped into every generated
skill's frontmatter (`metadata.openclaw.source.builder_version`).

---

## What it is (in one paragraph)

`build-skill-from-docs` is the **author** in an author↔tester split. Its `build <url>`
command runs a deterministic Python pipeline (`skill_builder.py`) that generates → critiques
→ repairs the skill artifact, making many single-shot LLM calls (plan → write → critique →
repair → synthesize) through one `_llm_call` dispatcher (`claude -p` on the Claude
subscription by default, OpenRouter fallback). **Behavioral evaluation is not done here** —
when the ship-gate is on (`--ship-gate`), the builder *delegates* it to the **`skill-tester`
agent** (a sub-agent call), which runs the shared `eval_core` harness and returns a JSON
verdict; the builder then repairs and re-gates. The builder holds **zero behavioral-eval
logic** — it authors and repairs; the tester scores. The durable output is the skill
artifact; MCP is a build-time fetcher and a runtime tail-coverage fallback, never the product.

## Design docs (read in this order)

| Doc | What it covers |
|---|---|
| [hld.md](hld.md) | **High-level design.** Architecture, the full pipeline (RESOLVE → INTENT → PLAN → WRITE → CRITIC/REPAIR → BUILD CONTRACT → SYNTHESIZE → REFERENCE-SCAN → TRIGGERING → VALIDATE → WRITE), MCP seams, sources ingested, URL handling, generated-skill structure, subcommands, and the inlined eval-methodology spec. |
| [plan.md](plan.md) | **Phase history + open items.** Phase 1.0 → 2.1 → 3.0 progression, verified-end-to-end table, deferred work (R1/R3/R4/R6/R7/R9), and the invocation cheat sheet. |
| [skill-shape-principles.md](skill-shape-principles.md) | **What content a skill should carry** — required sections, inline per-workflow MCP triggers, "a precondition travels with its action", scripts-vs-templates, auto-ToC, anti-patterns. |
| [skill-reliability-checklist.md](skill-reliability-checklist.md) | **The reliability bar the creator enforces** — P0 hard gates + P1–P4 quality checks, each tagged machine-enforceable (`det`) or LLM-checkable (`llm`), sourced to Anthropic-primary / peer-reviewed / derived. |
| [anthropic-parity.md](anthropic-parity.md) | **Comparison vs Anthropic's skill-creator** — primary-source creation + evaluation checklists, feature-by-feature comparison, the fundamental-method contrast (autonomous doc→skill vs interactive eval-gated loop), and a prioritized replication roadmap. |
| [eval-gate-plan.md](eval-gate-plan.md) | **P0+P1 implementation plan** — the eval-gated build loop: staging→evaluate→repair→gate→promote, `skill-tester` as the with/without behavior executor, `eval_core.py` refactor, sequenced milestones M0–M6. |

## Evaluation (does the skill actually help?)

The builder is one half of a two-stage evaluation pipeline; the eval docs live under
`docs/eval/` (per repo convention, all skill-evaluation work lives there).

| Doc | Scope |
|---|---|
| [../eval/overview.md](../eval/overview.md) | Two-stage pipeline summary + current status. |
| [../eval/stage1.md](../eval/stage1.md) | **Stage 1 — local skill eval** (in `eval_skill.py`): triggering F1, functional A/B (told-to-read), **organic-activation** (`activation` subcommand, `claude -p`), MCP signal capture, pass bars. |
| [../eval/stage2.md](../eval/stage2.md) | **Stage 2 — MLEvolve A/B**: paired with/without-skill full MLE-agent runs, held-out grader (L1) + 6×16 sub-stage attribution (L2) + cost (L3). |
| [../eval/skill-retrieval-design.md](../eval/skill-retrieval-design.md) | **Runtime skill retrieval** roadmap — the hybrid retrieve → LLM-rerank router for scaling from ~3 skills to a 50-skill library. The acknowledged critical path before the library can grow. |

> **Naming caveat:** `hld.md`'s "Stage 2 = `mle-skill-bench`" is a *speculative, unimplemented*
> spec. The real, running Stage 2 is the MLEvolve A/B in [../eval/stage2.md](../eval/stage2.md).
> They are different things; don't conflate them.

## Sibling agents & bundled skills ("sub-" units)

The builder is one of three agents in the skills program. It also bundles operational
skills beyond the core builder.

| Unit | Where | Role |
|---|---|---|
| `ai-skill-builder` | [`agents/ai-skill-builder/`](../../agents/ai-skill-builder/) | This agent. Core skill: `build-skill-from-docs`. |
| ↳ `build-skill-from-docs` | [`.../skills/build-skill-from-docs/`](../../agents/ai-skill-builder/skills/build-skill-from-docs/) | The author pipeline (`skill_builder.py`) + `prompts/` + authoring `references/`. Delegates behavioral eval to `skill-tester`. |
| ↳ ops skills | `.../skills/{build-mleval-image,monitor-mleval-job,refresh-mleval-pvc}/` | Eval-runtime helpers (image build, Job monitoring, PVC staging) — unrelated to skill creation. |
| `ai-skill-scout` | [`agents/ai-skill-scout/`](../../agents/ai-skill-scout/) · [docs](../skill-scout/hld.md) | Finds & safely installs *existing* skills from GitHub. The builder imports Scout's security scanner. |
| `skill-tester` | [`agents/skill-tester/`](../../agents/skill-tester/) | **The tester.** Owns behavioral skill eval via its `evaluate-skill` skill (`eval_core.py` + `eval_skill.py`): triggering, activation, functional A/B, description optimization, gate verdict. The builder delegates gating here; can also serve as a clean A/B baseline. |

## Code layout

```
agents/ai-skill-builder/skills/build-skill-from-docs/   # AUTHOR
├── SKILL.md                  the agent's tool definition (pipeline + flags)
├── scripts/
│   ├── skill_builder.py      build pipeline (BUILDER_VERSION = 2.1.0); delegates eval to skill-tester
│   └── prompts/*.txt         per-phase LLM prompts (intent_capture, plan_structure, write_skill_body,
│                             critique_skill, repair_skill_body, write_reference, improve_description, …)
└── references/               authoring rules the validator enforces
    ├── skill-anatomy.md · frontmatter-spec.md · anti-patterns.md

agents/skill-tester/skills/evaluate-skill/            # TESTER (owns behavioral eval)
├── SKILL.md                  contract: dir + profile → run harness → return verdict JSON
└── scripts/
    ├── eval_core.py          shared eval primitives (triggering judge, activation, functional,
    │                         optimize_description, run_gate, baseline_probe) — self-contained
    └── eval_skill.py         CLI over eval_core (triggering/functional/activation/all/gate/
                              baseline-probe/optimize-description/report/pass-bar)
```

The repo-root [`CLAUDE.md`](../../CLAUDE.md) is the whole-repo index (builder + scout + tester +
eval framework + infra); this file is the builder-centered slice of it.
