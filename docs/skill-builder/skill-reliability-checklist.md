# Skill-reliability checklist (creator-enforceable)

A prioritized checklist of what makes a **highly reliable** Agent Skill, sourced from Anthropic's
official docs + the `anthropics/skills` repo + peer-reviewed tool-doc research (deep-research,
2026-06-19). Every item is satisfiable by **one skill in isolation** (no cross-skill bridges) and is
either **machine-enforceable** or **LLM-checkable at build time**, so `build-skill-from-docs` can gate on it.

Companion to [skill-shape-principles.md](skill-shape-principles.md) (which covers *what content* a skill
should carry); this doc covers *the reliability bar* the creator must enforce.

**Legend** — Source: `[A]` Anthropic-primary · `[P]` peer-reviewed paper · `[C]` community · `[D]` our derivation.
Enforce: `det` deterministic string check (hard-fail) · `llm` LLM-check at build (warn/gate).
Our status: ✅ done · ⚠️ partial · ❌ missing/wrong. **As of builder 2.0.0 (2026-06-24) the P0 gates, the P4 scope-honesty critic, and the P1/P3 mechanical checks are enforced in `validate_skill` + `critique_skill`; validated live (vllm-inference genre wall caught; peft-tuning rebuilt clean).** **Builder 2.1.0 (2026-06-28) adds the P3 conditional-gating critic (block→repair): a resource-heavy/optional technique presented as a default, or a workflow whose precondition is stranded in the Decision Tree, is flagged — the fix for the mvp-029 QLoRA RCA where an ungated "Memory-efficient QLoRA" checklist trained 4-bit on a 48GB GPU.**

---

## P0 — Hard gates (frontmatter). Deterministic; hard-fail the build. `[A]`

These are the **only Anthropic-validated hard limits** — safe to reject a skill on.

| Check | Rule | Enforce | Ours |
|---|---|---|---|
| `name` present, charset | lowercase letters/numbers/hyphens only; ≤ 64 chars; no XML tags; not `anthropic`/`claude` | det | ✅ P0 gate (2.0.0) |
| `description` present | non-empty; ≤ 1024 chars; no XML tags | det | ✅ P0 gate (2.0.0) |
| Exactly the two required fields exist | `name` + `description` required; optional `license`/`metadata`/`allowed-tools` permitted (do **not** over-restrict — issue #249 myth) | det | ✅ |
| No dead pointers | every `references/`, `scripts/`, `templates/` file named in the body must exist on disk | det | ✅ P0 gate (2.0.0); rebuilt peft-tuning ships no phantom `templates/*.py` |

## P1 — Discovery / description. **The single highest-leverage lever.** `[A][P]`

The `description` is injected into the system prompt and is *the* mechanism Claude uses to pick among 100+
skills. Failed discovery zeroes out every other quality. Wording alone swings invocation **>10×** with
capability held fixed (arXiv:2505.18135); description-only rewrites lift tool-use accuracy by tens of
percent with no fine-tuning (arXiv:2602.20426).

| Check | Rule | Enforce | Ours |
|---|---|---|---|
| Third person | no "I/you/we"; inconsistent POV breaks discovery | llm/det | ✅ prompt rule + critic P1-person |
| **Both clauses** | description states BOTH *what it does* AND *specific when-to-use triggers/keywords* | llm | ✅ prompt requires what+when + 2-3 fire-situations; critic P1-when-clause |
| Specific terms | name concrete capabilities (class/algorithm names), not generic prose | llm | ✅ |
| **"Pushy" trigger** | include an explicit "use whenever the user … even if they don't ask for …" clause to counter Claude's documented **under**triggering | llm | ✅ required in write_skill_body + improve_description |
| Bidirectional tuning | tune to cut BOTH false-negatives (never fires) AND false-positives (over-fires) | llm + eval | ✅ evaluate_triggering bidirectional (negative_prompts) + sibling-aware (--siblings) |
| Skill-level when-to-use lives **here**, not the body | (but keep *intra-skill* conditional routing in the body — don't strip decision logic) | llm | ⚠️ |

## P2 — Structure / progressive disclosure. Mostly deterministic; gate as **warning**. `[A]`

Three levels: (1) metadata name+description, ~100 words, always loaded; (2) SKILL.md body **< 500 lines**,
loaded on trigger; (3) `references/`/`scripts/` unlimited, loaded only as needed — scripts **executed via
bash without loading their contents** into context.

| Check | Rule | Enforce | Ours |
|---|---|---|---|
| Body line cap | `wc -l` < 500 (soft "optimal" heuristic — **warn, don't hard-fail**) | det (warn) | ✅ we cap tighter (target 80–180, hard 300) |
| Anti-monolith | a near-cap body with **no** Level-3 references is a smell → split into `references/*.md` with pointers | llm | ⚠️ |
| Scripts marked execute-not-read | `## Scripts` says "execute, don't read"; scripts `chmod 0755` | det | ✅ (1.4.0) |
| Reference ToC | references > 100 lines get a `## Contents` ToC (partial-read visibility) | det | ✅ (1.4.0) |

## P3 — Content quality. `llm`-checkable; gate as warning. `[A]`

| Check | Rule | Enforce | Ours |
|---|---|---|---|
| **No rigid ALL-CAPS** | ALWAYS/NEVER/MUST in caps + "super rigid structures" are a yellow flag → **explain the *why*** instead (theory of mind) | det+llm | ✅ critic P3-allcaps + prompt rule |
| One default; heavy path = escape-hatch | prefer a single **cheapest-that-works** default + escape-hatches; a resource-heavy/optional technique (4-bit/QLoRA, 8-bit, distributed, separate inference engine) is gated on its precondition, never the headline default | llm | ✅ prompt rule + critic P3-conditional-gating (block) |
| **Precondition travels with action** | a workflow that is the conditional branch of a Decision Tree row (or a heavy/optional technique) must restate its gate in the intro + first step — not leave the condition stranded in the Decision Tree (the mvp-029 QLoRA failure) | llm | ✅ critic P3-conditional-gating BLOCK → repair loop |
| No time-sensitive info | nothing that goes stale ("after July 2026 …"); use an `## Old Patterns` `<details>` for legacy | det+llm | ✅ critic P3-time-sensitive |
| Consistent terminology | pick one term per concept, use it throughout | llm | ❌ |
| No Windows paths | forward slashes only | det | ✅ critic P3-windows-path |
| No voodoo constants | magic numbers in scripts need a rationale comment (Ousterhout's law) | det+llm | ❌ |
| Declare deps | don't assume packages installed; list/verify them | llm | ⚠️ |
| Self-contained constraints | make implicit prerequisites + input/output contracts explicit (agents can't resolve human-tolerable ambiguity) | llm | ⚠️ |

## P4 — Scope honesty. **The fix for the vLLM failure.** `[D]` (weakest-sourced — our derivation)

There is **no Anthropic-endorsed "NOT for X" pattern**, and negative scope walls collide with the documented
undertriggering tendency. In our gsm8k run, vLLM's *"NOT for: fine-tuning"* line made the agent scope the
skill out of a fine-tune-**then-evaluate** task where it was a valid eval sub-step.

| Check | Rule | Enforce | Ours |
|---|---|---|---|
| Prefer positive triggers | lead with "use whenever / also useful as a sub-step for …"; don't gate behind exclusions | llm | ✅ prompts emit positive escape-hatches |
| **No task-genre exclusions** | flag any "NOT for / don't use" clause that names an **adjacent task** rather than a **method limitation** | llm | ✅ critic P4 BLOCK → repair loop (validated live: caught vllm-inference genre wall) |
| Method-scoped exclusions only | exclusions may say "this skill doesn't *do* X" (capability), never "don't use this when the task involves X" (genre) | llm | ✅ prompts + critic P4 |
| No use-when ↔ NOT-for contradiction | a skill claiming "running evals" in-scope must not exclude "fine-tuning" if eval is a sub-step of it | llm | ✅ critic P4 (use-when ↔ NOT-for) |

---

## Caveats (carry these into any gate design)

- **Hard vs soft.** Only **P0 frontmatter** rules are Anthropic-validated → hard-fail. The ~100-word metadata
  and < 500-line body budgets are explicitly soft ("for optimal performance") → **warn**, never reject.
- **Scope-honesty (P4) is our own derivation**, built on (a) the documented undertriggering tendency,
  (b) our single vLLM incident, (c) general tool-doc ambiguity research. No published positive/negative-scope
  taxonomy exists (a five-dimension description taxonomy was *refuted* in this run). Treat as a hypothesis to
  validate with our own A/B, not settled practice.
- **Magnitudes don't transfer.** ">10×" and "+60%" come from competitive/auto-rewriter benchmark settings.
  The *lever* (description quality) transfers; the percentages do not — don't promise them.
- **Pushy-vs-overtrigger is an open tradeoff** with no closed-form single-skill rule — especially since we
  can't coordinate descriptions across the 10+ sibling skills competing for the same triggers. Resolve it
  empirically via the triggering eval (now competing against *real* siblings).
- **Refuted, do not adopt:** "no other YAML fields allowed"; "selection is based *entirely* on text / is
  fragile"; the five-dimension description taxonomy.
- **Fast-moving standard** (Agent Skills GA ~2025-10). Re-verify `anthropics/skills` paths before hard-coding.

## Sources
- Anthropic — Agent Skills **best practices** · platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices `[A]`
- Anthropic — Agent Skills **overview** · platform.claude.com/docs/en/agents-and-tools/agent-skills/overview `[A]`
- Anthropic — **Equipping agents** with Agent Skills (engineering) `[A]`
- Anthropic — **Writing tools for agents** (engineering) `[A]`
- `anthropics/skills` — **skill-creator/SKILL.md** (pushy descriptions, 3-level loading, ALL-CAPS yellow flag) `[A]`
- arXiv:2505.18135 (EMNLP 2025) — description wording → >10× invocation `[P]`
- arXiv:2602.20426 (Feb 2026) — description-layer rewrite → +tens-of-% tool-use accuracy, no fine-tuning `[P]`
- Community curated lists: `travisvn/awesome-claude-skills`, `ComposioHQ/awesome-claude-skills`, `obra/superpowers` `[C]`
