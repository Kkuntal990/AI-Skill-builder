# PLAN: AI-Skill Builder — OpenClaw Agent

> For the design overview, see **[hld.md](hld.md)**.

**Status:** 2.0.0 — closed critic→repair loop + Claude-subscription transport (see Phase 2.0)
**Date:** 2026-06-24
**Scope:** An OpenClaw agent that turns a Python package's docs URL into a progressive-disclosure SKILL.md + references + templates + evals, optionally augmented with curated community gotchas and runtime MCP fallback.

---

## 1. Goal

A single command — `openclaw agent --agent ai-skill-builder -m "Build a skill from <url>"` — produces:

1. A reproducible, version-pinned, opinionated SKILL.md the agent can load via progressive disclosure
2. Topical `references/*.md` deep-dives
3. `py_compile`-validated runnable templates
4. Stack-Exchange-sourced community gotchas with CC BY-SA attribution
5. Runtime MCP routing instructions (Context7 fallback) baked into the body
6. Provenance metadata (URL, repo, fetched_at, content_sha256, builder_version, coverage)

The skill is the durable artifact. MCPs are optional fetchers (build time) and runtime tail-coverage.

---

## 2. Phase History

### Phase 2.0 — Closed critic→repair loop + subscription transport (Jun 24, complete) — LATEST

Evolved the one-shot template into a closed generate→gate→critic→repair loop (Anthropic skill-creator pattern); see [hld.md](hld.md) Pipeline.
- **P0 hard gates** in `validate_skill`: `name` charset/≤64/reserved-word/XML, `description` ≤1024/XML, and **dead-pointer** (every cited `references|scripts|templates/<f>` must be bundled) — reject the build.
- **Quality critic + bounded repair** (`critique_skill` + `prompts/critique_skill.txt`, `repair_skill_body`): deterministic P3/P1/P2 regex checks + one abstain-when-unsure LLM call for **P4 scope-honesty** (task-genre "NOT for" walls); ≤3 repair rounds, ship-with-warning + a `quality_gate` field.
- **Sibling-aware + bidirectional triggering** (`--siblings`, `negative_prompts`).
- **Claude-subscription transport**: `_llm_call` dispatcher → `claude -p` by default (no API credit), OpenRouter fallback (`MLEVAL_LLM_TRANSPORT`). All OpenClaw agents also moved onto the Claude subscription.
- **Re-issued skills**: peft-tuning rebuilt clean (no genre wall, no phantom templates, `builder_version: 2.0.0`, triggering win_rate 1.0 vs real siblings); vllm-inference rebuilt. The old 1.x builds carried genre walls + (peft) phantom `templates/*.py` dead pointers.

### Phase 1.0 — Bootstrap (Apr 2026, complete)
- Agent workspace files (`AGENTS.md`, `SOUL.md`, `IDENTITY.md`, `TOOLS.md`)
- `find-ai-skill` skill scaffolding
- `skill_builder.py` skeleton: RESOLVE → FETCH → PLAN → SYNTHESIZE → VALIDATE → WRITE → LOG
- HF docs map + RTD subdomain inference
- Frontmatter assembled deterministically (LLM never writes it)
- 60-pattern security scanner inherited from Skill Scout

### Phase 1.1 — Open issues + troubleshooting (Apr 24, complete)
- `fetch_open_issues` + reaction-weighted ranking
- `extract_stack_traces` regex side-channel
- `write_troubleshooting` LLM stage with Symptom/Fix bullets
- ML-safe filter (skip `model.eval()` / `model.exec()` false positives in injection scanner)
- GitHub repo-root URL optimization (skip HTML chrome, use README directly)

### Phase 1.2 — Templates, workflows, hardware, when-to-use (Apr 24, complete)
- Plan-stage flags: `include_hardware_section`, `include_when_to_use_section`
- Hardware-hint regex extraction (VRAM / A100 / FSDP / DeepSpeed / quantization markers)
- Workflows with copy-paste checklists
- Templates synthesis with `python -m py_compile` validation
- "When to Use / NOT for" block naming concrete competitor tools
- Triggering eval loop: LLM judge vs 5 canned decoys + description optimizer

### Phase 1.3 — Community + provenance + MCP declarations (May 7, complete)
- `--with-community` flag wired through pipeline
- `fetch_stackexchange_qas_for_repo` (top-voted SO Q&As, IPI-scanned at fetch time)
- `fetch_question_issues` for closed `question`-labeled GitHub issues
- `distill_community_gotchas.txt` prompt + `write_community_gotchas` LLM stage
- `references/community-gotchas.md` with CC BY-SA attribution preserved per item
- `assemble_frontmatter` extended with `mcps`, `provenance`, `coverage`
- `_mcp_defaults_for(repo, url)` — HF packages get `hf-mcp/doc_search` preferred + Context7 fallback; non-HF gets Context7-only
- Provenance: `url`, `repo`, `fetched_at` (ISO 8601), `content_sha256`, `builder_version`
- Coverage list of substrates that contributed to the build

### Phase 1.4 — Runtime MCP integration (May 7, complete)
- **L1 declaration** already shipped in 1.3.
- **L2 server registration:** Context7 wired in both `mcporter` (the CLI agents call through `bash`) and `openclaw mcp` registries. Smoke-tested: returns `/huggingface/peft` with 1183 snippets.
- **L3 Path A — body routing:** `prompts/write_skill_body.txt` extended to require an `## Looking things up live (MCP fallback)` section in every generated SKILL.md, with verbatim `mcporter call context7.resolve-library-id ... → context7.query-docs ...` instructions and a discovery hint (`mcporter list-tools context7`).
- **L3 Path C — skill-as-MCP:** new `skill_builder.py serve <skill-dir>` subcommand running an MCP stdio server (stdlib only, no `mcp` SDK dep) that exposes `search_skill_refs(query)` over the skill's `references/`. Token-overlap ranking, no embeddings.
- **Custom-domain map:** `_DOCS_DOMAIN_MAP` static dict for non-HF / non-RTD package subdomains (vllm, unsloth, dspy, langchain, ray, lightning, langgraph, crewai, smolagents, haystack, guardrails, lmdeploy, bentoml, litellm).
- **Default model switched to `anthropic/claude-sonnet-4.6`** (faster, cheaper than Opus for synthesis; Opus available as override).

### Phase 1.5 — Skill-shape SOTA pattern + eval-harness fixes (May 24, complete)

Triggered by literature review (Anthropic Skills best-practices, LlamaIndex Skills-vs-MCP empirical work) showing the May 7 "MCP tail section" pattern was bolted-on and not driving agent behavior. Validated by an A/B against the rebuilt vLLM skill. See companion doc [skill-shape-principles.md](skill-shape-principles.md).

**Builder upgrades (`scripts/skill_builder.py`, BUILDER_VERSION 1.3.0 → 1.4.0):**
- **Decision tree section** — new `## Decision Tree` slot before `## Common Workflows`. Planner emits 2-6 routing rows for libraries with meaningful choices to make. New prompt: `prompts/write_decision_tree.txt` (refiner, optional second-pass).
- **Inline per-workflow MCP triggers** — each workflow checklist now ends with a `**MCP fallback**: if X is not in references/Y.md, call ...` step. Replaces the bolted-on tail section that agents were ignoring.
- **OpenClaw native MCP naming** — prompts now teach `context7__resolve-library-id` and `context7__query-docs` (double underscore), not Anthropic's `Context7:get-library-docs` colon syntax. This was the single biggest defect in the May 7 1.4 release: agents read the instructions but couldn't call the named tools.
- **`scripts/` tier** — new `--with-scripts` flag generates 1-3 SHORT utility scripts (<60 lines, bash or python) that the agent **executes** (not reads as reference). Validated via `py_compile` / `bash -n`. New prompt: `prompts/write_scripts.txt`.
- **`old_patterns` section** — under `--with-version-notes`, planner emits deprecated APIs as a collapsed `<details>` block.
- **Auto-ToC** — `_inject_toc_if_long` post-processor prepends `## Contents` to any reference >100 lines that lacks one (per Anthropic best-practices: agents previewing with `head -N` miss content past the cutoff).

**Security-scanner false-positive fixes (`scripts/skill_builder.py`):**
- `_is_ml_safe` now whitelists `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env-var-name references unless paired with an actual `sk-...` value. Previously blocked legitimate OpenAI-compatible API docs (e.g. vLLM).
- New `_DEMOTE_TO_CAUTION` set demotes HTTP/networking patterns (`curl POST`, `curl pipe`, `curl data send`, `requests.post`, `netcat`, `socket.connect`) from BLOCK to warning. These are inherent to any REST API + distributed-system docs; skill-scout's scanner was calibrated for low-trust skill discovery, not high-trust first-party generation. Real exfiltration patterns (SSH key cat, reverse shells, sudo, setuid, `.env` reads) remain BLOCK.

**Eval-harness fixes (`scripts/eval_skill.py`):**
- `_extract_tool_signals` extended with a fourth signal: **outcome narration**. Detects "Fetched via Context7", "LibraryId: /...", and similar patterns that prove MCP fired even when OpenClaw's `agentMeta.toolSummary` returns `null` (a runtime bug observed on multi-step / subagent-spawning responses). Without this signal, ~50% of real MCP calls were misclassified as `clean_miss`.
- Trial output now preserves full `reply_text` (previously only `reply_chars` length). Enables offline re-grading when assertions or detection patterns change — necessary because rebuilding to re-run costs $2-4 per pass.

**vLLM skill rebuild (artifact at `infra/skills/vllm-inference/`):**
- 6 references (vs 4 previously), 3 executable scripts, 6-row decision tree, 4 inline MCP triggers (per workflow), old-patterns section, ToCs on all 5 references >100 lines. `builder_version: 1.4.0` in frontmatter.

**Empirical findings worth carrying forward:**
- **Ambient-MCP confound** — when the test agent has `mcporter` bundled (as `skill-tester` does), Context7 is reachable regardless of which skill is loaded. The "with_skill vs without_skill" A/B for MCP-must-use prompts shows MCP firing in **both** cells. Use `main` agent for clean MCP-triggering A/B.
- **DeepSeek V4 Pro day-to-day variance is huge** — same prompts, same skill, different days produced median reply lengths of 430 chars vs 28 chars. Single-run absolute-pass-rate comparisons across days are unreliable.
- **`toolSummary` is unreliable on OpenClaw** — null for multi-step responses; the outcome-narration signal is required to recover the false negatives. Worth filing upstream.

---

## 3. Verified end-to-end

| Skill built | Files | Triggering | MCP fallback verified? |
|---|---|---|---|
| `trl-training` (HF) | 9 — SKILL.md + 4 refs + 3 templates + evals.json | 3/3 vs decoys | Refs covered all test prompts (no MCP needed) |
| `peft-tuning` v2.1 (HF) | 12 — SKILL.md + 6 refs incl. community-gotchas.md + 3 templates + evals.json | 3/3 build-time vs decoys; **F1=1.000** on 20-prompt Stage-1 eval | ✅ BOFT trajectory: agent ran `mcporter call context7.query-docs libraryId=/huggingface/peft query="BOFT BOFTConfig..."` after judging refs thin. Self-corrected our prompt's wrong tool name (`get-library-docs` → `query-docs`) by running `mcporter list-tools context7`. |
| `vllm-inference` (non-HF) | 6 — SKILL.md + 4 refs + evals.json | 3/3 vs decoys | Domain map missing at build time → no repo resolved → community/templates skipped. Fixed in 1.4 via `_DOCS_DOMAIN_MAP`. |

Trajectory artifacts: `/tmp/agent-trajectories/01-09-*.json` capture the full reasoning chain including tool-call evidence.

### Stage-1 evaluation results (peft-tuning, 2026-05-08)

Anthropic-pattern eval via `eval_skill.py`:

| Metric | Result | Target |
|---|---|---|
| Triggering F1 | **1.000** (10 TP / 10 TN, 60 judge calls) | ≥ 0.85 ✅ |
| Functional pass (with-skill) | 100% (5/5) | n/a |
| Functional pass (without-skill) | 100% (5/5) | < 100% expected |
| Lift | +0.0pp | +10–20pp |
| Citation rate (with-skill) | 60% (3/5) | ≥ 80% |
| Token cost ratio | input 2.4×, output 1.8× | ≤ 2× |

Saturation diagnosis: the 5 functional prompts are too in-distribution — Sonnet's training covers canonical PEFT questions, so deterministic `must_contain` checks pass with or without the skill. The skill's actual lift surfaces on **harder, niche, version-specific** questions (the BOFT trajectory verified this — agent fell through to Context7 for content not in our refs). Stage-2 (`mle-skill-bench`) is designed to surface this lift; see [HLD Evaluation Methodology](hld.md#evaluation-methodology) for the spec grounded in MLAlgo-Bench (EMNLP 2025) + MLE-Bench / RE-Bench / τ-bench.

Persisted at `~/.openclaw/workspace/skills/peft-tuning/evals/`:
- `triggering.json` (20 prompts: 10+10)
- `functional.json` (5 prompts with assertions)
- `grading_results/triggering-*.json` and `functional-*.json` per-run outputs

---

## 4. Open Items

### Deferred — possible future work

| ID | Item | Trigger to revisit |
|---|---|---|
| R1 | Build-time MCP doc substrate (replace HTML scrape with HF Docs MCP / RTD `/api/v3/search/` / Context7) | When a package's HTML strip yields too much chrome for the LLM to digest cleanly |
| R3 | Skill-as-MCP `serve` is stdlib-only token-overlap. Could upgrade to embedding-based ranking via a local SBERT model | When tail-coverage queries return wrong sections often enough to matter |
| R4 | `freshness check` subcommand — re-fetch source, diff core API symbols vs references, mark stale in lockfile | When a built skill rots (caught a wrong API in production) |
| R6 | MCP-baseline arm in triggering eval — judge against "use MCP instead of any skill" as a 7th option | When we want signal on whether a skill is worth distilling at all |
| R7 | Refuse builds for fast-moving libraries (use R4 drift signal as a gate) | When we waste distill cost on libraries where the artifact decays in weeks |
| R9 | `--llms-txt-only` fast path for libraries that publish `llms-full.txt` | When a library has a high-quality `llms-full.txt` we'd prefer over HTML scraping |

### Mitigations identified, not yet implemented

| Limitation | Fix | Cost |
|---|---|---|
| Case C: no skill match → MCP silent | Add a global system-prompt MCP routing block to the agent runtime config | tiny — append a 10-line block to `~/.openclaw/agents/<id>/agent/system-extra.md` |
| Static `_DOCS_DOMAIN_MAP` requires manual updates | Add page-link inspection (regex over fetched HTML for first `github.com/<owner>/<repo>`) as Tier 2 fallback | ~30 LOC, no new LLM cost |
| `community-gotchas.md` only covers Stack Overflow | Add curated Discourse forum scraping for HF packages (their forums have signal) | medium — IPI risk profile is higher than SO; needs careful ingestion |
| MCP routing in body is advisory, not mandatory | Optional validation hook that scans replies for `mcporter call` invocations on questions matching version-specific patterns | ~60 LOC; non-blocking warning |
| Stage-1 functional prompts are saturated (peft-tuning case) | Implement Stage-2 `mle-skill-bench` per [HLD §Evaluation Methodology](hld.md#evaluation-methodology): runnable container tasks, EScore = ∆Score × pass-rate, two-judge calibration, hardware-sizing assertions | medium — ~300 LOC harness + Docker per task. Authority: MLAlgo-Bench (EMNLP 2025), MLE-Bench, RE-Bench |

---

## 5. Invocation Cheat Sheet

```bash
# Conversational build via OpenClaw
openclaw agent --agent ai-skill-builder --json --timeout 600 \
  -m "Build a skill from <url> with --with-templates --with-troubleshooting --with-community"

# Direct script (LLM flows through the Claude subscription via `claude -p` by default)
python3 agents/ai-skill-builder/skills/build-skill-from-docs/scripts/skill_builder.py \
  build <url> --with-templates --with-troubleshooting --with-community --name <name> --force

# Dry-runs
python3 .../skill_builder.py sources <owner/repo>     # what would be fetched
python3 .../skill_builder.py plan <url>               # one LLM call → structure plan only
python3 .../skill_builder.py preview <url>            # full synth, print to stdout, no write

# Skill-as-MCP (Path C)
python3 .../skill_builder.py serve ~/.openclaw/workspace/skills/<name>

# MCP registration (one-time)
mcporter config add context7 --command npx --arg "-y" --arg "@upstash/context7-mcp" \
  --transport stdio --scope home
openclaw mcp set context7 '{"command":"npx","args":["-y","@upstash/context7-mcp"]}'

# Verify MCP wiring
mcporter list                                              # should show context7
mcporter call context7.resolve-library-id query="test" libraryName=peft   # smoke test
```

---

## 6. File Layout

```text
agents/ai-skill-builder/
├── AGENTS.md                                 Operational playbook
├── SOUL.md                                   Personality
├── IDENTITY.md                               Builder 🛠️
├── TOOLS.md                                  Binaries + MCP server registration steps
├── .gitignore                                Ignores data/
├── data/                                     (gitignored)
│   ├── built-skills.json                    Lockfile
│   ├── doc-cache.json                       1-hour TTL HTTP cache
│   └── skill-builder-audit.log              Append-only build log
└── skills/build-skill-from-docs/
    ├── SKILL.md                             Tool definition
    ├── references/                          Anatomy / frontmatter / anti-patterns
    └── scripts/
        ├── skill_builder.py                 ~1700 LOC pipeline
        └── prompts/
            ├── plan_structure.txt
            ├── write_skill_body.txt          Includes `## Looking things up live` mandate (Phase 1.4)
            ├── write_reference.txt
            ├── write_template.txt
            ├── write_evals.txt
            ├── distill_pitfalls.txt
            ├── write_troubleshooting.txt
            ├── distill_community_gotchas.txt Phase 1.3
            ├── judge_triggering.txt
            └── improve_description.txt
```
