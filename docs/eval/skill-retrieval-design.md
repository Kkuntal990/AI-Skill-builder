# MLEvolve skill-retrieval design (Path A)

Design for how the eval framework injects skill content into MLEvolve trajectories. Supersedes the spike-004 "concat-into-description.md" approach that broke on prompt-format mimicry (38% fence-at-start rate, SyntaxError cascade).

## Background

### What spike-004 verified

The skill **content** worked: spike-004 (with peft-tuning skill) produced 58 LoRA-related code mentions vs spike-003's 16 (no skill), and added Qwen-correct MLP target_modules (`gate_proj, up_proj, down_proj`) plus QLoRA 4-bit quantization that spike-003 missed. The skill **delivery** broke: concatenating 75 845 chars of fenced markdown into `description.md` (system message) made DeepSeek-Flash mimic the format. Result: every runfile_0.py started with ` ```python ` → SyntaxError on line 1 → cascade into 18 SEARCH-marker leaks.

### Prior art (verified, not assumed)

- **AutoMLGen paper** (arXiv 2510.08511, the canonical MLEvolve citation): §3.2 describes an explicit Knowledge Base mechanism — model-recipes injected only at draft time. Table 3 ablation: KB worth +9 medal points on MLE-Bench-Lite (50% → 41% without).
- **Materialized in code** at `upstream/engine/coldstart/{competition_tag_classified,models_guidance_classified}.json` — a `{category: {model: code_template}}` taxonomy, classifier-driven.
- We **disabled** it (`use_coldstart: false`) because our tasks aren't in {Image, Detection, Segmentation, NLP, Audio, Others}.
- No public precedent for extending the KB beyond its model-recipe schema (verified across 41 forks, BioXArena, MLE-Bench). Our use case (Anthropic-Skills bundles with prose, decision trees, multi-file references) doesn't fit the JSON shape.

### Why we substitute instead of fork

The AutoMLGen KB shape is `Code_template + Description` strings. Our skills carry sections like `## Choosing LoRA rank`, `## Common QLoRA bugs`, decision trees, references/*.md. We can't shoehorn them into the model-recipe schema without losing structure. We keep AutoMLGen's **architectural position** (knowledge prior at draft/improve time) but swap the **storage/retrieval layer**.

## Operating principle

The user-stated requirement:

> Model should know about existence of skill(s); when a relevant problem arises it should recall and inject the skill context; when not actively used it should not bloat our context.

Made operational as three prompt states:

| State | What model sees | Size |
|---|---|---|
| **Idle** — task touches no skill | L1 catalog only | ~400 tok per skill in system prompt |
| **Recalled** — task matches a skill | L1 catalog + top-k chunks from index | + ~3-5 KB in user message |
| **Multi-skill** — task touches several | L1 catalog + top-k chunks blended across skills | + ~3-5 KB (capped by k, not by N) |

The critical invariant: **bloat scales with N skills only in L1, not in retrieved content**.

## Architecture

```
                                    ┌─────────────────────────┐
                                    │  infra/skills/<name>/   │
                                    │    SKILL.md             │
                                    │    references/*.md      │
                                    └────────────┬────────────┘
                                                 │ (trajectory start)
                                                 ▼
                              ┌────────────────────────────────────┐
                              │  skill_retriever sidecar           │
                              │                                    │
                              │  1. Chunker: H2/H3 boundaries +    │
                              │     whole-file fallback            │
                              │  2. BM25Okapi + FAISS index        │
                              │     (reuses MLEvolve HybridRetriever│
                              │     if available; else local copy) │
                              │  3. L1 catalog: skill name+desc    │
                              │     in Anthropic YAML              │
                              └──────────┬─────────────────────────┘
                                         │ (per LLM call via wrapper)
                                         ▼
                ┌──────────────────────────────────────────────────────────┐
                │  build_chat_prompt_for_model wrapper (sidecar monkey patch)│
                │                                                          │
                │  call:                                                   │
                │    (model, introduction, user_prompt, assistant_prefix)  │
                │                                                          │
                │  introduction (system):                                  │
                │    → persona_overlay's identity                          │
                │    → append "## Available skills" YAML catalog (L1)      │
                │                                                          │
                │  user_prompt:                                            │
                │    → if stage in {draft, improve, debug}:                │
                │        query = user_prompt content                       │
                │        chunks = retriever.search(query, k=3)             │
                │        if max(chunks.score) >= THRESHOLD:                │
                │          insert # Skill Reference section before # Instructions
                │                                                          │
                │  → call original build_chat_prompt_for_model             │
                └──────────────────────────────────────────────────────────┘
```

### Hook site rationale

`build_chat_prompt_for_model` (in `upstream/agents/planner/base_planner.py`) is the single chokepoint for every LLM call that goes through MLEvolve's planner path. We already monkey-patch it for persona overlay (with dual-bind to `agents.planner` re-export site — see `prompt_overlay.py`). Wrapping it once handles both surfaces (system + user) without modifying upstream code.

Stage detection is content-based (heuristic match on user_prompt markers like `"# Task description"`, `"# Memory"`, `"Solution sketch guideline"`). If we ever need precise stage signals, we add thread-local context set by caller — but the heuristic is sufficient for MVP.

## Bloat budget

Worked numbers assuming `400 tok` per L1 catalog entry, `1.5 KB` per retrieved chunk, k=3.

### Per-state cost

| Library size | Idle system bloat | Recall user bloat | Total per relevant call |
|---|---|---|---|
| 1 skill | +400 tok (~1.6 KB) | +4.5 KB | ~6 KB |
| 5 skills | +2 KB | +4.5 KB | ~6.5 KB |
| 10 skills | +4 KB | +4.5 KB | ~8.5 KB |
| 20 skills | +8 KB | +4.5 KB | ~12.5 KB |

### Compared to spike-004 baseline

- spike-004 system_message: **80 223 chars** (~20 KB) — always-on bloat regardless of relevance
- This design system_message (5 skills, idle): **~7 KB** — 3× smaller
- This design system_message + recall (5 skills): **~12 KB** — still 2× smaller than spike-004
- 20-skill library at recall is still smaller than spike-004's single-skill always-on

## Skill format (Anthropic-style)

`infra/skills/<name>/` layout, **unchanged**:

```
infra/skills/peft-tuning/
  SKILL.md
  references/
    qwen_targets.md
    llama_targets.md
    common_bugs.md
    ...
  requirements.txt
```

`SKILL.md` front matter (added if not present) used for L1 catalog generation:

```yaml
---
name: peft-tuning
description: |
  LoRA/QLoRA recipes for fine-tuning HuggingFace transformers.
  Covers target_modules per architecture (Qwen, Llama, Mistral),
  rank/alpha selection, QLoRA quantization config, common bugs,
  SFTTrainer integration, evaluation patterns.
triggers: [lora, qlora, peft, fine-tune, adapter, target_modules, sft]
---
```

Body content below the frontmatter is **not modified** — chunked at H2/H3 boundaries, indexed verbatim.

## Chunking strategy

1. **SKILL.md**: split on H2 (`## ...`) and H3 (`### ...`) headings. Each section becomes one chunk. Title prefix retained for the indexer's BM25 vocabulary.
2. **references/*.md**: each file becomes one chunk by default. If a single reference exceeds 4 KB, also split it on H2/H3 boundaries.
3. **Metadata per chunk**: `{skill_name, source_file, heading, body}`. Used for retrieval result rendering.

### Why H2/H3 boundaries

Anthropic-Skills convention organizes SKILL.md around `## <topic>` sections. Splitting at heading boundaries preserves semantic coherence (each chunk is a self-contained topic) and matches how a human reader would navigate the doc. Token-window chunking (e.g., 512 tokens overlap) would split decision trees and code blocks awkwardly.

## Retrieval

### Implementation choice

**Reuse MLEvolve's `HybridRetriever`** (`upstream/agents/memory/retriever.py`) if it exposes a stable `add_document` / `search` interface. Otherwise stand up a local copy (~150 LoC: BM25Okapi + FAISS IndexFlatL2 + RRF fusion).

Reuse benefit: zero new deps in the image, consistency with MLEvolve's `# Memory` retrieval semantics. Local-copy fallback exists because MLEvolve's retriever is built for `(plan, code, metric)` tuples, not arbitrary chunks — may not have a clean public API.

### Parameters

| Param | Value | Rationale |
|---|---|---|
| **k** | 3 | One SKILL.md section + 1-2 reference details — usually enough for draft turn |
| **score threshold** | 0.30 RRF | Strict gate (matches "no bloat when idle" principle) — tune after smoke |
| **stages enabled** | draft, improve, debug | AutoMLGen paper limits to draft; we extend per our use case (PEFT debugging benefits from skill mid-trajectory) |
| **stages disabled** | code-review, parse-result, planner-JSON | Structured outputs — skill irrelevant |

### Threshold logic

```
chunks = retriever.search(query, k=3)
if not chunks or max(c.score for c in chunks) < THRESHOLD:
    # No injection — only L1 catalog remains visible
    return user_prompt_unchanged
else:
    rendered = render_chunks(chunks)
    return insert_before_instructions(user_prompt, rendered)
```

## Implementation surface

### New files

| File | Purpose | LoC est. |
|---|---|---|
| `infra/agents/mlevolve/mlevolve_sidecar/skill_retriever.py` | Chunker + indexer + retrieval API | ~150 |
| `infra/agents/mlevolve/mlevolve_sidecar/skill_catalog.py` | L1 catalog YAML builder | ~40 |

### Edits to existing files

| File | Change | LoC est. |
|---|---|---|
| `infra/agents/mlevolve/mlevolve_sidecar/skill_inject.py` | Repurpose: no longer concatenates into description.md. Instead initializes skill_retriever and produces L1 catalog text. | -40 / +30 |
| `infra/agents/mlevolve/mlevolve_sidecar/prompt_overlay.py` | Extend `_patched_build_chat_prompt_for_model` wrapper to (a) append L1 catalog after persona-overlay intro, (b) call retriever and inject `# Skill Reference` into user_prompt for draft/improve/debug stages | ~50 |
| `infra/agents/mlevolve/mlevolve_sidecar/__init__.py` | Register `skill_retriever` import after `prompt_overlay` | +1 |
| `infra/agents/mlevolve/_smoke_imports.py` | Add 3 regression assertions: retriever importable, chunker handles peft-tuning skill, end-to-end roundtrip with sample query returns ≥1 chunk | ~20 |
| `infra/agents/mlevolve/entrypoint.sh` | Set `MLEVAL_SKILL_DIR` env var (path to staged skill on PVC) so retriever can load chunks at runtime | ~5 |

### Upstream MLEvolve files NOT modified

- `upstream/agents/planner/base_planner.py:104` (`build_chat_prompt_for_model`) — wrapped via existing monkey-patch surface
- `upstream/agents/draft_agent.py:_draft` — untouched
- `upstream/engine/coldstart/*` — left as-is (still disabled via `use_coldstart: false`)

## Smoke verification (spike-005)

### Pre-flight (build-time, in `_smoke_imports.py`)

1. `skill_retriever.SkillIndex` constructs from `infra/skills/peft-tuning/`
2. Chunker produces ≥5 chunks from peft-tuning skill
3. Retriever returns ≥1 chunk for query `"Fine-tune Qwen on dialogue summarization with LoRA"`
4. Monkey-patch dual-bind verified (`agents.planner.build_chat_prompt_for_model is wrapper`)

### Runtime (spike-005, SAMSum × 1 seed × 1 trajectory)

| Metric | Target | Source |
|---|---|---|
| `system_message` size (median over calls) | ≤ 7 KB | `prompts.jsonl` row sizes |
| `user_message` size (draft/improve/debug calls, median) | 5-10 KB | same |
| Fence-at-start rate of LLM responses | ≤ 16% (matches spike-003 baseline) | python regex over `output` field |
| LoRA-related code mentions in generated content | ≥ 50 (target ≥ 58 = spike-004) | grep over verbose.log |
| target_modules includes MLP layers (`gate_proj` etc.) | Yes | grep |
| QLoRA 4-bit quantization attempted | Yes | grep `BitsAndBytesConfig` |
| SyntaxError count in execution | 0 (vs spike-004's 7) | verbose.log |
| Best-solution.py non-empty | Yes | `stat` on PVC |
| Final Validation Score parsed | Any non-null float | `journal.json` |

### Failure paths

| Outcome | Interpretation | Next action |
|---|---|---|
| Fence rate spikes again | Catalog YAML itself has fences (frontmatter) — strip on render | Patch catalog formatter |
| Retrieval returns wrong chunks | Threshold or k mis-tuned | Adjust empirically on next smoke |
| Best-solution empty, SyntaxError 0 | Different failure mode (e.g., training crash) — skill not at fault | Separate issue, fence regression fixed |
| LoRA mentions back to ~16 | Threshold too strict — chunks not making it into prompt | Drop threshold or expand stage coverage |

## Scope notes

### Out of scope for this design

- **Skill-builder integration**: how skills get authored (handled by `agents/ai-skill-builder/`). This design only specifies the consumer side.
- **Skill versioning / hot-reload**: trajectories pick up the skill snapshot at startup. Updating a skill mid-trajectory is not supported.
- **Cross-skill conflict resolution**: if two skills' chunks both match a query, retriever blends them. If they give contradictory advice, the LLM resolves it — out of harness scope.
- **L3 progressive disclosure**: Anthropic-Skills' "model emits `bash: cat ref.md`" pattern needs ReAct loop, which MLEvolve doesn't have. References are indexed alongside SKILL.md content; retrieval surfaces them when relevant. We do not extend MLEvolve with tool use in this design.

### Future extensions worth flagging

- **Per-stage chunk pools**: train-time chunks vs eval-time chunks vs debug-time chunks, retrieved against stage-specific weights
- **Skill citation logging**: every retrieval logged with chunk_id + score → enables L2c attribution ("did the skill chunk that surfaced actually inform the code?")
- **Model-driven loading**: if we ever build a ReAct loop into MLEvolve, swap the retriever's auto-injection for model-emitted `load_skill(name)` calls

## Open question (defer to implementation)

The AutoMLGen paper restricts KB to `s_init = Init(T, R_KB(T))` — initial draft only. We are extending to draft + improve + debug. This is a defensible methodological extension because PEFT debugging (OOM diagnosis, target_modules tweaking) is exactly when the skill's value is highest. If we want a clean comparison to the paper's claim, we can gate the extension behind a config flag and report results both ways.

## Related docs

- `docs/eval/overview.md` — eval framework two-stage pipeline
- `docs/eval/stage2.md` — A/B framework methodology
- `infra/agents/mlevolve/mlevolve_sidecar/README.md` — sidecar patching surface
- `docs/skill-builder/skill-shape-principles.md` — Anthropic-Skills authoring conventions
