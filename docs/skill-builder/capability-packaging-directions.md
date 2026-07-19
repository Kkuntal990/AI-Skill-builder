# Packaging a skill for precise per-node insertion — problem formalization & design directions

*2026-07-19. Research memo. Motivated by the coverage-recall audit
(`capability-mvp/e1/coverage_audit_summary.json`): the compiler captures only 58.8% (peft)
/ 21.3% (vllm) of the solvable issues its own source teaches, so the runtime linker can
never inject the missing fixes — the per-node gate is only as good as the index beneath it.
This memo formalizes the problem, diagnoses why the current pipeline is brittle, and
proposes directions grounded in verified literature (§4) and in deterministic measurements
of our own artifacts (§2).*

---

## 1. The problem, formalized

**Given.** A skill document **D** — a markdown tree **T** (SKILL.md + `references/*.md`;
files → headings → leaf spans: paragraphs, code blocks, admonitions). A stream of nodes
from a tree-search MLE agent; each node **n** has a machine-readable profile
**p(n)** = (stage ∈ {generate, refine, debug, explore}, task text, parent code/error/
analysis, hardware). A context budget **k** (currently ≤3 units/node).

**Wanted.** An injection policy **π : n ↦ content ⊆ D, |π(n)| ≤ k** such that for every
node n and every doc-taught fix **f** relevant at n, **f is *reachable* under π at n** —
the policy *can* deliver it (recall), and mostly *does not* deliver irrelevant or
contraindicated content (precision / compliance).

**Our π factorizes into three layers** (this is the useful part):

```
π  =  select  ∘  gate  ∘  index
        LLM       det.      LLM (compile time)
```

- **index** : D → U — compile-time; the unit set U with an anchor map
  **a : U → 2^Leaves(T)** (`sources[].locator`) and guards
  **g : U → predicates** (`applicability`, `requires`, `delivery.stages`, `depends_on`).
- **gate** — deterministic three-valued filter over the machine-checkable guard parts
  (`capability_linker._hard_filter` + dependency closure).
- **select** — temp-0 LLM choice among gate survivors.

**A recall miss factorizes accordingly.** Fix f, relevant at node n, can be lost at:

| Layer | Miss mode | Evidence in our artifacts |
|---|---|---|
| **index** | no u ∈ U with f ∈ a(u) | 41–79% of source-taught issues have no unit (coverage audit); the spike-018 gradient-checkpointing fix is the type specimen |
| **gate** | guard wrongly excludes a relevant u at n | 2 stage-inconsistent `depends_on` edges prune `qlora-small-gpu` and `quantize-to-fit-vram` at exactly `debug` — 100% of E2's 90 `dep_incompatible` rejections (§2.3) |
| **select** | LLM declines a relevant survivor | unmeasured (needs node-level gold — W6++) |
| **render** | budget drops the span (`max_reference_files=1`, whole-file locators) | §2.2: only 8–12% of sections are anchor-addressable; the rest ride whole-file pointers |

**Design principle (the memo's thesis).** *Guarantee recall structurally; buy precision
semantically.* Recall must be an invariant of the **symbolic** layers (index totality +
gate soundness), machine-checkable without an LLM. The **neural** layer (selector) should
only ever *narrow* — never be the thing that makes content unreachable. Today both recall
and precision are entrusted to one open-vocabulary LLM compile call, which is why the
system is brittle.

## 2. Deterministic measurements of our current artifacts (2026-07-19)

All computed by script from the shipped bundles + manifests — no LLM in any number.

### 2.1 The structural denominator exists

The doc AST gives a countable, stable denominator — no LLM enumeration needed:

| | peft-tuning | vllm-inference |
|---|---|---|
| files | 5 | 6 |
| heading-anchored sections | **66** | **74** |

(The coverage audit's LLM-extracted "51 / 61 solvable issues" is a *soft* denominator —
useful for discovery, wrong as a metric. §5 replaces it.)

### 2.2 Per-section routability is 8–12%

Units already carry heading-anchored locators (`SKILL.md#fine-tune-a-model-with-lora`) —
the schema *requires* ≥1 per unit and 0 are dangling. But:

| | peft | vllm |
|---|---|---|
| locators total | 15 | 14 |
| … anchored to a section | 10 | 8 |
| … whole-file only | 3 (files) | 5 (files) |
| sections reachable via anchored locator | **8/66 = 12.1%** | **6/74 = 8.1%** |

A whole-file pointer defeats per-node routing: the renderer either injects the entire
file or nothing, and the selector cannot address the one span that matters.

### 2.3 The gate has a lintable defect class

For every dependency edge u → d, the closure prunes u at any stage where d is not
deliverable. The invariant **stages(u) ⊆ stages(d)** is checkable in O(|E|):

```
peft: qlora-small-gpu [debug,generate] → loraconfig-core-parameters [generate,refine]  ⇒ PRUNED at debug
vllm: quantize-to-fit-vram [debug,generate] → quantization-method-selection [explore,generate] ⇒ PRUNED at debug
```

These two edges account for **all 90** `dep_incompatible` rejections in the E2 replay
(2 units × 45 debug nodes). A 15-line validator lint catches at build time what the LLM
audit found only anecdotally. Note the semantics: `depends_on` conflates *"prerequisite
knowledge"* (LoraConfig params are context for QLoRA) with *"must be co-deliverable"*
(what the closure enforces). The lint forces the compiler to pick one meaning per edge.

### 2.4 A ToC-as-catalog is cheap

Rendering every heading as one line costs ~889 tokens (peft) / ~968 (vllm) — small enough
to sit inside the selector's context as a guaranteed-total fallback catalog (§5-D4).

## 3. Diagnosis: why the current packaging is brittle

1. **Enumeration and annotation are fused.** One LLM call decides *what units exist* and
   *how they are guarded*. Open-vocabulary enumeration has no totality constraint, so
   under-production is silent (21–59% issue coverage; 8–12% routability).
2. **The audit's denominator is itself LLM-generated.** "Solvable issues" is one model's
   enumeration at one granularity — soft in magnitude, unfalsifiable as a gate metric.
3. **Cross-cutting concerns fall through section-shaped chunking.** The
   gradient-checkpointing fix is *scattered* (AOP's term) as side-notes inside QLoRA and
   integration sections; section-based extraction has no pass that sweeps a concern
   across the whole doc. This is a systematic class: error→fix pairs, memory/resource
   knobs, version constraints, decision tables.
4. **Dependency edges carry no consistency invariant** (§2.3) — silent stage pruning.
5. **Whole-file locators defeat the budget renderer** (§2.2).
6. **What is *not* broken:** the hard gate itself (0/80 violations twice), unit precision
   (98.9% grounded, 1.06% fabrication), and the minimal machine-checkable guard
   vocabulary (runtime/gpu/vram/stage — deliberately small; see the expert-systems
   lesson, §4). Keep all of it.

## 4. Verified literature anchors

*All items verified from publisher/arXiv/OASIS pages, 2026-07-19 (four independent
verification sweeps). One-line relevance each; URLs are canonical.*

### 4.1 Retrieval & document indexing — the *enrich, don't select* pattern

Every successful doc-indexing system below guarantees coverage **by construction** and
spends LLM effort on *enrichment* (context, summaries, structure) — never on deciding
which content survives. Our compiler does selective survival; that is the deviation.

- **Dense X Retrieval** (Chen et al., EMNLP 2024, [2312.06648](https://arxiv.org/abs/2312.06648)) —
  retrieval granularity is a first-class design choice; their "propositions" come from an
  **exhaustive** decomposition pass over every passage — every source fact yields a unit.
- **RAPTOR** (Sarthi et al., ICLR 2024, [2401.18059](https://arxiv.org/abs/2401.18059)) —
  tree index by recursive clustering+summarization; **abstraction is additive** — leaves
  retain 100% of source text, retrieval can always fall through to any leaf.
- **GraphRAG** (Edge et al., 2024, [2404.16130](https://arxiv.org/abs/2404.16130)) — the
  closest architectural sibling (LLM builds a structured index at compile time), inheriting
  the same extraction-recall risk; mitigates with repeated **"gleaning"** extraction rounds
  per chunk — directly transferable to our compiler.
- **Anthropic Contextual Retrieval** (2024, [blog](https://www.anthropic.com/engineering/contextual-retrieval)) —
  augment-don't-distill: every chunk kept (coverage = 100% by construction), LLM adds
  situating context; −49% retrieval-failure rate.
- **DocPrompting** (Zhou et al., ICLR 2023, [2207.05987](https://arxiv.org/abs/2207.05987)) —
  retrieves doc sections into code-gen prompts; the pool is **every** doc section, so any
  doc-taught API is reachable. The no-compile baseline pole (successor: **EVOR/ARKS**,
  [2402.12317](https://arxiv.org/abs/2402.12317)).
- **ColBERT** (Khattab & Zaharia, SIGIR 2020, [2004.12832](https://arxiv.org/abs/2004.12832)) —
  the canonical index-vs-scan trade-off: fine-granularity late interaction buys scan-like
  recall at index-like cost.
- **Chunking evaluation** (Smith & Troynikov, Chroma 2024,
  [report](https://www.trychroma.com/research/evaluating-chunking)) — chunking strategy
  alone moves **token-level recall by ~9 points**; introduces token-level recall/IoU
  metrics for chunking. **Element/structure-aware chunking** (Jimeno Yepes et al., 2024,
  [2402.05131](https://arxiv.org/abs/2402.05131)) — chunking along the doc's own structural
  elements beats structure-blind splitting.

### 4.2 Classic computer science

- **Guarded commands** (Dijkstra, CACM 1975, [DOI](https://dl.acm.org/doi/10.1145/360933.360975)) —
  guard→statement pairs with nondeterministic choice among *enabled* commands: **safety
  comes from the guards, so any choice among enabled units is admissible — selection only
  affects utility.** The correctness frame for our gate/selector split.
- **Design by Contract** (Meyer, IEEE Computer 1992, [DOI](https://dl.acm.org/doi/10.1109/2.161279)) —
  `applicability.when/avoid_when` + `requires` are machine-checkable preconditions;
  contraindication compliance is precondition checking.
- **Rete** (Forgy, Artificial Intelligence 1982, [DOI](https://www.sciencedirect.com/science/article/abs/pii/0004370282900200)) —
  compile condition-action rules into a network for cheap repeated matching against
  changing state; our per-node link cycle is a production-system match.
- **Knowledge-acquisition bottleneck** (Feigenbaum, IJCAI-77,
  [PDF](https://www.ijcai.org/Proceedings/77-2/Papers/092.pdf): "the acquisition of domain
  knowledge was the bottleneck problem…" — the compressed phrase crystallized later;
  Hayes-Roth, Waterman & Lenat, *Building Expert Systems*, 1983) — extracting correct
  condition-action knowledge into the rule base is the bottleneck, not runtime matching.
  Our 21–59% under-coverage is a modern instance. Lesson: keep the hard-guard vocabulary
  **small and machine-checkable** (we do); don't grow it into an ontology.
- **A Knowledge Compilation Map** (Darwiche & Marquis, JAIR 2002,
  [DOI](https://doi.org/10.1613/jair.989)) — compilation trades offline effort for
  tractable online queries; a compiled form is only useful if it is *complete for the
  query class*. Our query class is "what does the doc say about situation s?" — the audit
  shows our compilation is not query-complete.
- **Submodular maximization** (Nemhauser, Wolsey & Fisher, Math. Prog. 1978,
  [DOI](https://link.springer.com/article/10.1007/BF01588971)) — greedy 1−1/e for
  cardinality-constrained monotone submodular objectives (the k≤3 selection);
  **set-cover greedy** (Chvátal, Math. OR 1979,
  [DOI](https://pubsonline.informs.org/doi/abs/10.1287/moor.4.3.233)) — ln(d) guarantee;
  coverage-deficit vocabulary.
- **Typestate** (Strom & Yemini, IEEE TSE 1986, [DOI](https://ieeexplore.ieee.org/document/6312929)) —
  legal operations depend on *state*, checkable statically; `delivery.stages` violations
  are typestate violations (see the lint, §2.3).
- **Formal Concept Analysis** (Ganter & Wille, Springer 1999,
  [DOI](https://link.springer.com/book/10.1007/978-3-642-59830-2)) — optional: a
  (fixes × attributes) concept lattice exposes systematic enumeration holes.

### 4.3 Software engineering & technical writing

- **DITA 1.3** (OASIS 2015, [spec](https://docs.oasis-open.org/dita/dita/v1.3/dita-v1.3-part3-all-inclusive.html)) —
  the tech-writing industry's mature precedent for exactly our design: **typed topics**
  (concept/task/reference + a dedicated **troubleshooting** type added in 1.3 —
  condition→cause→remedy ≈ our `diagnostic` kind), **conditional processing** via
  profiling attributes (audience/platform/product ≈ our `applicability`/`requires`), and
  **maps** for assembly ("topics themselves can be relatively context-free").
- **Information Mapping** (Horn, *Mapping Hypertext*, 1989; Horn et al. 1969) — typed
  information blocks, "one idea per block, one block per idea" — the norm violated when a
  memory-tuning tip is buried inside a QLoRA how-to.
- **Aspect-Oriented Programming** (Kiczales et al., ECOOP 1997,
  [DOI](https://link.springer.com/chapter/10.1007/BFb0053381)) — **scattering & tangling**:
  cross-cutting concerns end up scattered across the primary decomposition; the remedy is
  a separate modularization + weaving. Names our gradient-checkpointing failure class.
- **Traceability** — Gotel & Finkelstein (ICRE 1994,
  [DOI](https://ieeexplore.ieee.org/document/292398/)): the canonical traceability-problem
  analysis; Antoniol et al. (IEEE TSE 2002,
  [DOI](https://ieeexplore.ieee.org/document/1041053)): IR-based doc↔code link recovery.
  Our locator-carrying manifest *is* a traceability matrix; under-coverage is a
  traceability-completeness failure.
- **Documentation incompleteness & coverage linting** — Uddin & Robillard (IEEE Software
  2015, [PDF](https://www.cs.mcgill.ca/~martin/papers/ieeesw2015.pdf)): incompleteness is
  empirically the top API-doc failure; industrial inventory-diff linters
  ([interrogate](https://interrogate.readthedocs.io/), docstr-coverage) mechanically diff
  docs against an enumerable artifact inventory — the exact shape of our totality check,
  but over code symbols; ours is over doc spans.

### 4.4 The 2025–26 agent-skill line (novelty position)

**Verified emptiness (Q1):** no published system measures whether a compiled skill/tool
library **covers its source corpus**. Anything2Skill
([2606.09316](https://arxiv.org/abs/2606.09316), taxonomy-guided skill compiler with
evidence windows), SkillWiki ([2606.16523](https://arxiv.org/abs/2606.16523),
evidence-linked skill assets), and Corpus2Skill
([2604.14572](https://arxiv.org/abs/2604.14572), corpus → navigable skill directory) all
compile without measuring what they missed; "Skill Coverage"
([2606.20659](https://arxiv.org/abs/2606.20659)) measures the *inverse* (trajectory
coverage of skill prose). **Our compilation-recall metric is novel.**

**New mandatory citations (gap narrows but survives):**
- **Graph-of-Skills** ([2604.05333](https://arxiv.org/abs/2604.05333)) — closest published
  pipeline to our linker: offline skill graph, PageRank-style dependency retrieval,
  **context-budgeted hydration**. No hard machine-checkable applicability/hardware gate;
  not per-node inside a tree search.
- **SkillOps** ([2605.13716](https://arxiv.org/abs/2605.13716)) — skill-library
  maintenance with runtime precondition checks (composition safety, not context injection).
- **R3-Skill** ([2606.03565](https://arxiv.org/abs/2606.03565)) — skill routing needs
  query-conditioned *compatibility*, trained on LLM-rejection annotations — the learned
  analogue of our `avoid_when`.
- **ToC/tree navigation retrieval:** LATTICE ([2510.13217](https://arxiv.org/abs/2510.13217)),
  BookRAG ([2512.03413](https://arxiv.org/abs/2512.03413) — ToC-tree index, SOTA *recall*
  on manuals), HORMA ([2606.11680](https://arxiv.org/abs/2606.11680) — the
  **missing-vs-misleading context** failure split ≈ our under-coverage vs over-selection),
  GraphSkill ([2603.06620](https://arxiv.org/abs/2603.06620)), Search-o1
  ([2501.05366](https://arxiv.org/abs/2501.05366) — model-triggered when-to-retrieve, the
  soft pole opposite our hard gate), ExploraCoder
  ([2412.05366](https://arxiv.org/abs/2412.05366) — execution probing as a complementary
  channel for facts the compiler missed). API-doc RAG evidence: Chen et al.
  ([2503.15231](https://arxiv.org/abs/2503.15231)) — **code examples are the most valuable
  doc component** for code-gen.
- **Anthropic Agent Skills** (Zhang, Lazuka & Murag, Oct 2025,
  [blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)) —
  three-level progressive disclosure; SKILL.md as "a well-organized manual that starts
  with a table of contents." Their coverage story is **purely iterative/empirical**
  ("observe where agents struggle") — no compile-time completeness check. Design fork to
  name in the paper: **agent-navigates-index** (Anthropic; interactive) vs
  **linker-pushes-units** (ours; forced by non-interactive MLEvolve).

## 5. Design directions

**D0 — what we do NOT change.** The hard gate (0/80 violations twice), the *minimal*
machine-checkable guard vocabulary (runtime/gpu/vram/stage — the expert-systems lesson,
§4.2: don't grow the guards into an ontology; the semantic residual belongs to the
selector), the precision pipeline (98.9% grounded), and provenance locators. In Dijkstra's
frame (§4.2): guards give safety, selection gives utility — so recall repairs go in the
index and gate, never by weakening the gate.

### D1 — Structural totality: a two-layer index with recall by construction

Split the fused compile call into **deterministic enumeration + LLM annotation**:

- **Base layer (total):** enumerate entries from the markdown AST — every heading section
  (66 peft / 74 vllm, §2.1) becomes an addressable entry with a stable anchor and an
  LLM-written one-liner + guards (kind/stages/`when`/`requires`), DITA-style typed topics
  with profiling attributes (§4.3). The LLM no longer decides *what exists* — only *how
  each thing is labeled*. Totality is a machine-checked invariant:
  **∪ₐ a(U) ⊇ Leaves(T)**, checkable in O(|T|) with no LLM — an inventory-diff lint in the
  `interrogate` mold (§4.3), and the exhaustive-decomposition pattern of Dense X /
  RAPTOR-leaves / Contextual Retrieval (§4.1).
- **Overlay layer (curated):** keep today's semantic units as an overlay (they are good —
  98.9% grounded), deduplicated against base entries by anchor. Overlay units carry richer
  procedures; base entries guarantee nothing is unreachable.

The linker analogy is exact: sections *export symbols*; the index is the symbol table; an
unexported section is a **link error at build time**, not a silent runtime gap.

### D2 — Aspect sweeps for cross-cutting concerns

Section-shaped enumeration still misses *scattered* concerns (§4.3, Kiczales): the
gradient-checkpointing fix lives as side-notes inside QLoRA/integration sections. Add
dedicated compiler passes — one per concern class — that sweep the **whole** doc and emit
overlay units pointing into spans across sections:

- error→fix pairs (DITA troubleshooting shape: condition→cause→remedy),
- memory/resource knobs (the OOM class),
- version/compatibility constraints,
- decision tables ("use X when …").

This is AOP for documentation: modularize the scattered concern, weave at link time.
GraphRAG's *gleaning* (§4.1) says iterate each sweep until a pass returns nothing new —
recall via repeated extraction, the loop-until-dry pattern we already use elsewhere.

### D3 — Typed-graph invariants (build-time lints)

Deterministic validator additions (`capability_schema.validate_manifest`):

1. **Stage consistency:** for every `depends_on` edge u→d, `stages(u) ⊆ stages(d)` —
   already catches both real defects and explains all 90 E2 `dep_incompatible` rejections
   (§2.3). Typestate checking (§4.2) over the unit graph. Forces the compiler to split
   `depends_on` into *prerequisite-knowledge* vs *co-delivery* edges.
2. **Anchor resolution:** every `#anchor` locator must resolve to a real heading
   (currently 0 dangling — keep it enforced).
3. **Anchored-locator requirement:** `procedure`/`diagnostic` units must carry at least
   one *section-anchored* locator (whole-file pointers defeat routing, §2.2).
4. **Totality** (from D1): every leaf span reachable from ≥1 entry.

### D4 — Two-tier runtime delivery with a guaranteed fallback channel

Give the selector a two-tier catalog: **tier 1** = gated curated units (as today);
**tier 2** = the base-layer ToC (~900 tokens/skill, §2.4) from which it may request a
section when no unit fits. Then *any* doc span remains reachable at every node — a
structural recall floor at runtime, not only at compile time. Precedents: BookRAG's
ToC-tree recall, LATTICE navigation, DocPrompting's index-everything pool (§4.1, §4.4);
this is Anthropic's progressive disclosure adapted to a non-interactive agent by moving
navigation *into the selector's action space* (the push-vs-navigate fork, §4.4).

**Known risk:** catalog dilution / skill shadowing (selection failure dominates skill
loss in the literature we hold). Mitigations: hierarchical selection (pick file → pick
section, LATTICE-style) and one-line ToC entries; measure drift with the E2 replay before
trusting it (§6-M4). Corpus2Skill's navigate-vs-retrieve failure taxonomy (§4.4) tells us
where this works (single-domain structured corpora — us) and where it fails (open-domain
factoid pools — not us).

### D5 — Deterministic + behavioral evaluation (retire the LLM denominator as a gate)

- **Structural metrics (no LLM):** section routability (today 8–12% anchored, §2.2) and
  totality — token-level recall framing per Chroma (§4.1).
- **Behavioral needles:** a regression suite from real trajectories — needle #1 is the
  spike-018 training-OOM node: *given this recorded profile, is the span containing
  `gradient_checkpointing_enable` reachable/injected?* Every RCA adds a needle. Axes per
  HORMA's missing-vs-misleading split (§4.4): under-coverage vs over-selection.
- **The LLM issue-audit** (`capability_coverage_audit.py`) stays as a *discovery* tool —
  it found real gaps — but its LLM-enumerated denominator is not a gate metric.

### D6 — Principled selection objective (framing, lower priority)

Model per-node injected value as monotone coverage of node needs → greedy selection
carries the 1−1/e guarantee (NWF 1978, cardinality-constrained; §4.2); the coverage
audit's deficit is a set-cover deficit (Chvátal). Explains why the budget should cap the
*rendered closure*, not roots, and gives the paper a clean objective statement. FCA (§4.2)
optional for systematic gap analysis over (fixes × guard attributes).

## 6. Migration plan (incremental; each step independently valuable)

| Step | What | Where | Cost | Depends on |
|---|---|---|---|---|
| **M1** | Lints: stage-consistency + anchor-resolution + anchored-locator-required; structural routability + totality metrics; fix the 2 bad edges (compiler re-emit) | `capability_schema.py` validate_manifest; `capability_coverage_audit.py` structural section | ~1 day, **zero LLM** | — |
| **M2** | Compiler 0.2 — two-pass: (A) deterministic markdown-AST enumeration → base entries with anchors; (B) LLM annotation per entry + aspect sweeps (D2, gleaning-style loop-until-dry); current units become the overlay, deduped by anchor | `capability_compiler.py` + prompts | mid | M1 |
| **M3** | Linker: candidate space = overlay units ∪ base entries; two-tier selector catalog (units first, ToC lines second); telemetry records which tier served each selection | `capability_linker.py`, `skill_injector.py` | mid | M2 |
| **M4** | Eval: needle suite (spike-018 OOM + the 2 peft high-severity gaps + 3–5 vllm high gaps); re-run the E2 replay — coverage becomes ~structural-100% by construction, so measure the *precision* side: selected-unit drift, abstention pattern, dilution | `scripts/replay_capability_linker.py`, new `tests/test_capability_needles.py` | mid | M3 |

**Paper positioning (from §4.4):** the compilation-recall metric is novel (verified
emptiness); Graph-of-Skills and SkillOps join the must-cite list; the contribution
statement gains a second leg — *hard compatibility gating* (existing) + *coverage-audited
compilation with structural totality* (new) — and the push-vs-navigate fork names our
non-interactive constraint honestly.

---

*Deterministic measurements in §2 are reproducible from the shipped bundles/manifests;
scripts inline in the session log. Literature verified 2026-07-19; verification notes
(incl. the Feigenbaum phrasing caveat and NWF-vs-KMN "budgeted" terminology) preserved in
§4. Related: `capability-linker-mvp-plan.md` §7 (open items), `capability-mvp/e1/coverage_audit_summary.json`.*
