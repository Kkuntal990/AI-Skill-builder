# HLD: AI-Skill Builder — OpenClaw Agent

**Status:** 2.0.0 — closed critic→repair loop + Claude-subscription transport (2026-06-24)
**Companion to:** [../skill-scout/hld.md](../skill-scout/hld.md) (Skill Scout — finds existing skills) · [plan.md](plan.md) (phase progression and open items) · [skill-shape-principles.md](skill-shape-principles.md) (authoring guidance)

## Goal

An OpenClaw agent that generates an OpenClaw-formatted SKILL.md for a Python package from a single URL. Fetches documentation, repository, and curated community sources; synthesizes a progressive-disclosure skill augmented with cited gotchas and runtime-MCP declarations; validates; installs to the workspace.

## MCP's Role in This Architecture

MCP appears in three distinct seams. The skill is still the durable artifact; MCPs are fetchers (build time) and fallbacks (runtime).

| Seam | Status | Where | Purpose |
|---|---|---|---|
| ① Build-time doc substrate | **deferred** | would replace `urllib` HTML scrape in `_gather_sources` | HF Docs MCP / RTD `/api/v3/search/` / Context7 for cleaner snippets fed to the distiller. Today: HTTP + `gh api` + Stack Exchange REST. |
| ② Build-time community substrate | **partial — direct REST** | `_gather_sources` + `--with-community` | Stack Exchange API (CC BY-SA, attributable) + GitHub closed `question` issues → IPI-scanned → `references/community-gotchas.md`. Direct REST today; would be MCP-wrapped in seam ① work. |
| ③ Runtime tail-coverage | **wired (Phase 1.4)** | frontmatter declaration + body routing block + skill-as-MCP `serve` | Each skill declares preferred + fallback MCPs in `metadata.openclaw.mcps`. `## Looking things up live (MCP fallback)` body section gives the agent verbatim `mcporter` invocations. `skill_builder.py serve <skill-dir>` exposes the skill's `references/` as its own MCP server. |

**Reddit is intentionally not a build-time source.** Google's Apr 2026 indirect-prompt-injection research reports a 32% rise in IPI content in static blogs/forums Nov 2025→Feb 2026; five poisoned docs flip a RAG response 90% of the time. Stack Exchange (licensed, structured, vote-deduplicated, attribution-preserving) is the safer community substrate.

**No MCP calls happen at build time today.** The script uses `urllib` (HTTP), `gh api` (GitHub CLI), Stack Exchange REST (`api.stackexchange.com`), and the Claude subscription via the local `claude -p` CLI (OpenRouter is a fallback transport). MCP is a runtime concern only.

## How It Works

1. User gives the agent a URL (hosted docs page, or GitHub repo root)
2. Agent invokes `skill_builder.py build <url>`
3. Script resolves `owner/repo` from the URL and fetches sources in parallel (doc, README, examples; optionally issues, changelog, Stack Exchange Q&As)
4. Regex side-channel extracts hardware-relevant sentences (VRAM/GPU/memory) from doc + README for the body prompt
5. One LLM call plans structure: `references/*.md` files + workflows + templates + section flags (hardware, when-to-use)
6. LLM drafts the SKILL.md body; a critic + repair loop (≤3 rounds) scores P1-P4 and auto-fixes any task-genre scope wall, then parallel LLM calls synthesize each reference file, runnable template/script, evals, and (when `--with-community`) `community-gotchas.md`
7. Each template is validated with `python -m py_compile`; each script via `py_compile`/`bash -n`
8. Triggering eval loop judges the description against the real co-resident siblings (`--siblings`) or canned decoys per eval prompt; rewrites the description once if any prompt fails (bidirectional — also catches over-triggering)
9. Script assembles YAML frontmatter deterministically: `name`, `requires.bins`, `install`, runtime MCP declarations (`mcps.preferred`, `mcps.fallback`), and provenance (`source.{url, repo, fetched_at, content_sha256, builder_version}`, `coverage`)
10. Script validates with **P0 hard gates** (name/description rules + dead-pointer) plus the security scan + line-count cap + shell-command fabrication check + template/script compile check + IPI scan + `openclaw skills check`
11. Script writes `SKILL.md + references/ + templates/ + evals/` to `~/.openclaw/workspace/skills/<name>/`
12. Script updates `built-skills.json` lockfile and appends to `skill-builder-audit.log`

## Pipeline

Since 2.0.0 the pipeline is a **closed generate → gate → critic → repair loop**, not a one-shot template:

```text
RESOLVE -> FETCH (doc, README, examples, [issues], [changelog], [SE Q&As])
       -> EXTRACT hardware hints (regex side-channel on doc + README)
       -> IPI SCAN on community sources (drop poisoned docs before LLM ingest)
       -> PLAN STRUCTURE (LLM — references, workflows, decision tree, MCP triggers, scripts, flags)
       -> WRITE BODY (LLM)
       -> CRITIC + REPAIR loop (<=3 rounds):
             critique P1-P4 (regex P3 + abstain-when-unsure LLM for P4 scope-honesty / P1)
          |- if any BLOCK finding (task-genre wall): repair body (full regen) -> re-critique
       -> SYNTHESIZE in parallel:
             references/*.md + [templates/*.py] + [scripts/*] + evals/evals.json
           + [pitfalls.md] + [troubleshooting.md] + [community-gotchas.md]
       -> EVAL TRIGGERING (LLM judge vs real siblings [--siblings] or canned decoys; bidirectional)
          |- if win_rate < 1.0: IMPROVE DESCRIPTION (LLM) -> RE-JUDGE -> accept if better
       -> ASSEMBLE frontmatter (deterministic): name + install + mcps + provenance (builder_version)
       -> VALIDATE (P0 hard gates + dead-pointer + scan + line cap + fabrication + openclaw check)
       -> WRITE to workspace
       -> LOG
```

| Stage | What happens |
|-------|-------------|
| Resolve | Parse URL → derive `owner/repo` when possible |
| Fetch | `urllib` for doc HTML (regex-stripped to markdown); `gh api` for README, examples, issues, changelog |
| Extract hardware hints | Regex-grep doc + README for VRAM / GPU / A100 / H100 / DeepSpeed / FSDP / quantization mentions; up to 12 sentences |
| Plan structure | One LLM call returns `references`, `workflows`, `decision_tree`, `mcp_workflow_triggers`, `templates`/`scripts` (0-3 each), and section flags |
| Write body | One LLM call drafts the SKILL.md body from the plan |
| Critic + repair | `critique_skill` scores P1-P4 (deterministic regex for P3 anti-patterns + mechanical P1/P2-bloat; one abstain-when-unsure LLM call for P4 scope-honesty + semantic P1). Any BLOCK (task-genre "NOT for" wall) triggers `repair_skill_body` (full regen, not diff-patch); loop ≤3 rounds. Residual ships as warnings + a `quality_gate` field |
| Synthesize | Parallel `claude -p` (Claude subscription) calls — one per reference/template/script/evals file. Transport switchable via `MLEVAL_LLM_TRANSPORT` (claude default, openrouter fallback) |
| Eval triggering | Per eval prompt the LLM judges our description against the real co-resident siblings (`--siblings`) or 5 canned decoys; bidirectional (also detects false positives on should-NOT-trigger prompts) |
| Improve description | If `win_rate < 1.0`, one LLM call rewrites the description (third-person, pushy, what+when) from failing prompts |
| Assemble | Python writes YAML frontmatter (LLM never writes frontmatter); stamps `builder_version` provenance |
| Validate | **P0 hard gates** (`name` charset/≤64/reserved-word/XML, `description` ≤1024/XML, **dead-pointer**) reject the build; plus Scout's scanner + ML-safe filter, line cap 500, shell-fabrication check |
| Write | `~/.openclaw/workspace/skills/<name>/{SKILL.md, references/*.md, [templates|scripts]/*, evals/evals.json}` |
| Log | Append to `skill-builder-audit.log`; update `built-skills.json` lockfile |

## Sources Ingested

| Source | Default? | Output section |
|---|---|---|
| Doc page (user-supplied URL) | Always | Quickstart, core APIs, references/*.md decomposition |
| Repo `README.md` | Always | Install section, canonical commands |
| Repo `examples/` listing | Always | Code references in `references/*.md` |
| Closed `bug`-labeled issues, sorted by reactions (top 20) | `--with-pitfalls` | `references/pitfalls.md` |
| Open issues (≥3 reactions) + closed bugs + regex-extracted tracebacks | `--with-troubleshooting` | `references/troubleshooting.md` with `**Symptom:** ... **Fix:** ...` bullets grouped by error theme; unresolved items marked `**Known issue:**` |
| Stack Exchange top-voted Qs tagged `<package>` (top 15) + closed `question`-labeled issues | `--with-community` | `references/community-gotchas.md` with `**Symptom:** ... **Fix:** ... **Source:**` bullets, CC BY-SA attribution preserved per item |
| `CHANGELOG.md` / latest release notes | `--with-version-notes` | Version notes in SKILL.md |

**Excluded by design:** Reddit, Discord, HF Discourse forums. Indirect-prompt-injection risk in unmoderated forum text dominates marginal signal; if community knowledge gaps remain after Stack Exchange + closed issues, add a runtime MCP fallback (Seam ③) rather than baking forum text into the skill.

## URL Handling

| URL type | Doc source | Notes |
|---|---|---|
| `huggingface.co/docs/<pkg>/...` | HTML strip | Repo auto-derived via `_HF_DOCS_MAP` |
| `<project>.readthedocs.io/...` | HTML strip | Repo inferred from subdomain |
| `github.com/<owner>/<repo>` | README via `gh api` (single fetch, no HTML chrome) | Repo-root detection skips HTML path |
| `docs.<package>.<tld>` (vllm, unsloth, dspy, langchain, ray, lightning, …) | HTML strip | Repo via `_DOCS_DOMAIN_MAP` static lookup (Phase 1.4) |
| Any other `http(s)` URL | HTML strip | Repo only derived if URL matches a known pattern; without it, gh-issues / community-gotchas / install-detection are skipped |

URL validation: `http(s)` only, ≤2048 chars, no `file://` or local paths.

`_DOCS_DOMAIN_MAP` is a static dict in `skill_builder.py`. Adding a new ML library is a one-line entry: `"docs.foo.io": "fooorg/foo"`. The page-link inspection / PyPI-metadata / `gh search` resolver tiers proposed in earlier discussions are intentionally deferred — empirical data so far shows the static map handles the common case cheaply.

## Generated Skill Structure

Every generated skill includes:

- `SKILL.md` (80-180 line target, hard cap 300) with sections:
  - Description + Installation + Quick Start
  - **Decision Tree** — 2-6 routing rows (Phase 1.5+) for libraries with meaningful choices between modes/strategies/algorithms
  - **Common Workflows** — 2-4 named workflows, each with a copy-paste checklist that ends with a **per-workflow MCP-fallback step** (Phase 1.5+) using OpenClaw-native naming `context7__resolve-library-id` / `context7__query-docs`
  - **When to Use** — "Use when..." cases + "NOT for (use alternatives instead)" cases naming concrete competitor tools
  - **Hardware Requirements** — included when regex-extracted hardware hints exist; lists GPU/VRAM/multi-GPU/mixed-precision guidance
  - **Templates** — bullet list of runnable Python scripts in `templates/` (read-as-reference)
  - **Scripts** — bullet list of executable utilities in `scripts/` (Phase 1.5+, executed via bash — *not* read into context)
  - **Old Patterns** — collapsed `<details>` listing deprecated APIs (Phase 1.5+, when `--with-version-notes` and the docs mention deprecations)
  - **References** — bullet list of `references/*.md` files
  - **Looking things up live** — short tail-coverage MCP fallback section (Phase 1.5: tightened from 18 to ~14 lines; native OpenClaw naming)
- `references/*.md` — 2-5 topical deep-dives (domain-variant decomposition). Auto-prepended `## Contents` ToC when ≥100 lines (Phase 1.5+) so agents previewing with `head -N` see the full scope.
- `templates/*.py` — 0-3 runnable, `py_compile`-validated Python scripts with `# TODO:` customization points (when `--with-templates` is set)
- `scripts/*.{sh,py}` — 0-3 SHORT (<60 line) utility scripts (Phase 1.5+, when `--with-scripts` is set). Bash via `bash -n`, Python via `py_compile` validated. `chmod 0o755`. Authoring guidance in [skill-shape-principles.md](skill-shape-principles.md).
- `evals/evals.json` — 2-3 realistic user prompts (unless `--no-evals`)
- `references/pitfalls.md` — closed bug fixes (when `--with-pitfalls`)
- `references/troubleshooting.md` — open issues + stack traces (when `--with-troubleshooting`)
- `references/community-gotchas.md` — Stack Exchange + closed-issue gotchas with CC BY-SA attribution (when `--with-community`)

Frontmatter (Phase 1.3+):

```yaml
metadata:
  openclaw:
    emoji: 🤖
    requires: { bins: [python3] }
    install: [...]
    mcps:
      preferred: [hf-mcp/doc_search, hf-mcp/doc_fetch]   # for HF packages
      fallback: [context7/get-library-docs]              # universal
    source:
      url: https://huggingface.co/docs/peft/index
      repo: huggingface/peft
      fetched_at: 2026-05-07T18:30:00Z
      content_sha256: <hex>
      builder_version: 1.4.0
    coverage: [html, gh-readme, gh-issues-open, stackexchange, gh-issues-question-closed]
```

For non-HF packages, `mcps.preferred` is `[]` and `mcps.fallback` carries Context7 only. The package family is detected by `_mcp_defaults_for(repo, url)`.

## Runtime MCP Integration (Phase 1.4)

Three layers must be wired for an agent to actually invoke MCP when using a skill:

| Layer | Responsibility | Status today |
|---|---|---|
| **L1 — Declaration** | `metadata.openclaw.mcps` in SKILL.md frontmatter | ✅ Auto-emitted by `assemble_frontmatter`. |
| **L2 — Server registration** | MCP server reachable from agent's shell environment | ✅ `mcporter config add context7 --command npx --arg "-y" --arg "@upstash/context7-mcp"` + duplicate via `openclaw mcp set context7 ...`. |
| **L3 — Agent runtime glue** | Routing logic the agent reads and acts on | ✅ Phase 1.5 rewrite: **per-workflow inline MCP-fallback steps** (each `## Common Workflows` checklist ends with a `**MCP fallback**: ...` step) plus a short tail `## Looking things up live` section. Both use OpenClaw-native tool naming (`context7__resolve-library-id` / `context7__query-docs`, double underscore — NOT Anthropic's `Context7:get-library-docs` colon syntax). The May 7 tail-section-only pattern was bolted on and didn't drive behavior; inline triggers + correct naming were the fix. See [skill-shape-principles.md](skill-shape-principles.md). |

Path C (skill-as-MCP) is also wired: `python3 skill_builder.py serve <skill-dir>` runs an MCP stdio server (stdlib only, no `mcp` SDK dependency) exposing `search_skill_refs(query)` over the skill's `references/`. Register with `mcporter config add <name> --command python3 --arg <abs-path-to-skill_builder.py> --arg serve --arg <abs-path-to-skill-dir> --transport stdio`.

### When MCPs Actually Fire — five conditions

An MCP call happens only when **all five** hold:

1. The skill description matches the user message → skill body loads.
2. The agent reads the `## Looking things up live` section → routing instructions become visible.
3. The bundled `references/` don't fully answer the question → agent judges them insufficient.
4. The agent has a shell tool (`bash` / `exec`) → can run `mcporter call ...`.
5. The MCP server is registered and reachable (`mcporter list` shows it).

If any one fails, MCP doesn't fire.

### Three Cases

| Case | Skill triggers? | References cover? | MCP fires? | Notes |
|---|---|---|---|---|
| A — reference miss | Yes | Partial / no | ✅ | Most common. BOFT trajectory verified this. |
| B — out-of-scope but skill loaded | Yes | n/a (different topic) | ⚠️ depends on agent judgment | The body's routing block doesn't restrict queries to the skill's package; the agent *may* pivot Context7 to another lib, but it's heuristic. |
| C — no skill matches | No | n/a | ❌ | Routing block lives inside skill body; if no skill loads, the agent never sees the instructions. Limitation today; mitigated by adding a global system-prompt MCP routing block (deferred fix). |

### Token Economics

MCP servers in our setup are **opt-in via shell**, not native auto-mounted tools. Cost ladder:

| Component | Tokens per turn (idle) | Tokens per turn (in use) |
|---|---|---|
| Skill metadata (description) | ~100 | ~100 |
| SKILL.md body when triggered | 0 | ~5,000 |
| `bash` tool definition | ~50 | ~50 |
| Context7 + skill-mcp tool definitions | **0** | only the response chunks (hundreds of tokens) |
| References (only the .md files bash-read) | 0 | ~1,500 each |

Native auto-mount (Claude Desktop / Cursor) would inject Context7's full tool catalog into every turn — the "tens of thousands of tokens" critique. Our `mcporter`-via-bash routing dodges that cost: idle MCP servers cost nothing.

## Triggering Eval Loop

Generated skills include 2–3 realistic test prompts in `evals/evals.json`. After synthesis, each prompt is judged by an LLM against the generated description plus 5 canned decoy skills (`data-preprocessing`, `model-evaluation`, `experiment-tracking`, `vector-retrieval`, `deployment-serving`). For each prompt the judge picks one skill and records a reason.

If `win_rate < 1.0`, one LLM call rewrites the description using the failing prompts and the judge's reasons. The loop re-judges and accepts the new description only if `win_rate` strictly improved. The full report (initial + revised scores, per-prompt choices and reasons) is returned in the build result.

Toggle off with `--no-eval-triggering`.

## Validation & Security

Reuses Skill Scout's 60-pattern regex scanner across 7 categories (exfiltration, injection, destructive, obfuscation, credential access, privilege escalation, reverse shell). Applied post-synthesis, pre-write.

Builder-specific layers:

- **ML-safe filter** — `model.eval()` and other method-call `eval()`/`exec()` invocations are excluded from the injection pattern (only bare builtin calls block).
- **Frontmatter parse** — YAML must include `name` and `description`.
- **Line cap** — SKILL.md ≤ 500 lines (target 60–150).
- **Shell-command fabrication check** — commands in fenced `bash/sh` blocks must appear in the source docs/README; unknown verbs surface as warnings.
- **Template compile check** — each `templates/*.py` runs `python -m py_compile`; failures become warnings.
- **`openclaw skills check`** — run post-write when writing to the default workspace.

BLOCK hits reject the build; CAUTION hits surface as warnings.

## File Layout

### Agent workspace (in this repo)

```text
agents/ai-skill-builder/
├── AGENTS.md                                  Operational playbook
├── SOUL.md                                    Personality
├── IDENTITY.md                                Name/emoji (Builder 🛠️)
├── TOOLS.md                                   Binaries + paths
├── .gitignore                                 Ignores data/
└── skills/
    └── build-skill-from-docs/
        ├── SKILL.md                           Agent's tool definition
        ├── references/
        │   ├── skill-anatomy.md               Progressive-disclosure rules
        │   ├── frontmatter-spec.md            OpenClaw metadata schema
        │   └── anti-patterns.md               Traps rejected by the validator
        └── scripts/
            ├── skill_builder.py               Pipeline
            └── prompts/
                ├── plan_structure.txt         LLM: doc TOC → references + workflows + templates + section flags
                ├── write_skill_body.txt       LLM: doc + workflows + hardware hints → SKILL.md body
                ├── write_reference.txt        LLM: doc + topic → references/<topic>.md
                ├── write_template.txt         LLM: workflow + doc → templates/*.py (py_compile validated)
                ├── write_evals.txt            LLM: body → evals/evals.json
                ├── distill_pitfalls.txt       LLM: closed bugs → pitfalls.md
                ├── write_troubleshooting.txt  LLM: open+closed issues+traces → troubleshooting.md
                ├── judge_triggering.txt       LLM: user msg + skills → which fires
                └── improve_description.txt    LLM: failing prompts → better description
```

### Runtime state (gitignored, in `data/`)

- `built-skills.json` — lockfile of generated skills (source URL, content hash, build timestamp)
- `doc-cache.json` — 1-hour TTL cache of fetched doc pages
- `skill-builder-audit.log` — append-only log of all `build` invocations

## Invocation

```bash
# Via gateway (conversational)
openclaw agent --agent ai-skill-builder -m "Build a skill from <url>"

# Via embedded agent (no gateway)
openclaw agent --agent ai-skill-builder --local -m "..."

# Via dashboard
openclaw dashboard

# Direct script (LLM calls go through the Claude subscription via `claude -p` by default)
python3 agents/ai-skill-builder/skills/build-skill-from-docs/scripts/skill_builder.py build <url>
```

## Subcommands

| Subcommand | What it does |
|---|---|
| `build <url>` | Full pipeline. Flags: `--name X`, `--with-pitfalls`, `--with-troubleshooting`, `--with-templates`, `--with-community`, `--with-version-notes`, `--no-evals`, `--no-eval-triggering`, `--force`, `--out <dir>` |
| `preview <url>` | Run full synthesis (including eval loop) but print to stdout instead of writing |
| `plan <url>` | Only run PLAN STRUCTURE; show the file decomposition and why |
| `sources <owner/repo>` | Dry-run: show what would be fetched (URLs, sizes, counts) |
| `built` | Dump `built-skills.json` |
| `serve <skill-dir>` | Run a stdio MCP server exposing `search_skill_refs(query)` over the skill's `references/` (Phase 1.4, Path C of L3 integration) |

## Constraints

- **One URL in, one skill out.** No multi-page crawling.
- **LLM never writes frontmatter.** Python assembles it deterministically.
- **Never auto-overwrite.** Name collisions require `--force`.
- **No installation to workspace from non-default `--out`.** `openclaw skills check` is skipped for out-of-workspace writes.
- **LLM transport (no API credit by default).** Default: the local Claude Code CLI (`claude -p`) → Claude subscription. `MLEVAL_LLM_TRANSPORT=openrouter` switches to the paid OpenRouter path (`OPENROUTER_API_KEY` or the `openrouter:default` auth profile; model `OPENROUTER_MODEL`). Build aborts clearly if no transport is available.

## Honest Limitations

- **No build-time MCP usage today.** Seam ① is deferred. Build still works through HTTP + `gh` + SE REST.
- **Case C — agent is blind to Context7 when no skill matches.** Routing instructions live inside SKILL bodies; if no skill triggers, the agent never sees them. Mitigation deferred (a global system-prompt MCP block would fix it).
- **Static `_DOCS_DOMAIN_MAP` requires manual updates** for new ML library subdomains. The page-link-inspection / PyPI / `gh search` resolver tiers are designed but not implemented.
- **Skill-as-MCP search is keyword token overlap, not semantic.** Adequate for tail coverage, not for precise lookup; a future improvement is to add embedding-based ranking if a local embedding model is available.
- **Context7 self-hosting via `npx` requires Node.** Documented dependency; covered in `TOOLS.md`.

## Evaluation Methodology

Two-stage evaluation pipeline. **Stage 1** is implemented today and ships as `agents/ai-skill-builder/skills/build-skill-from-docs/scripts/eval_skill.py`. **Stage 2** (`mle-skill-bench`) is specified here for ML-engineering-grade evaluation; not yet implemented.

The methodology is grounded in [MLAlgo-Bench (Wang et al., EMNLP Findings 2025)](https://aclanthology.org/2025.findings-emnlp.772/) — the closest peer-reviewed analogue to "did the agent follow the prescribed recipe or take a shortcut?" — combined with the Anthropic `skill-creator` 20-prompt 60/40 protocol and the MLE-Bench / RE-Bench containerised-runnable pattern.

### Stage 1 — Anthropic-pattern eval (implemented)

Per the [`anthropics/skills/skills/skill-creator/SKILL.md`](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) protocol. Four metrics, one harness:

| Metric | Method | Field target | peft-tuning result |
|---|---|---|---|
| **Triggering F1** | 10 should-trigger + 10 near-miss decoy prompts × 3 runs each. LLM judge picks among target skill + 5 canned decoys + `none`. | ≥ 0.85 (Anthropic) | **1.000** (10/10 TP, 10/10 TN, 60 calls) |
| **Functional pass rate** | 5 prompts with deterministic `must_contain` / `must_not_contain` / `expected_citations` assertions, run with-skill and without-skill. Schema follows skill-creator's `{text, passed, evidence}`. | n/a (relative) | **100% with-skill / 100% without-skill** — see saturation note |
| **Lift** | with-skill pass-rate minus without-skill pass-rate, in percentage points. | +10–20pp ([Sogl 2025](https://dev.to/danielsogl/skills-without-evals-are-just-markdown-and-hope-3a71) reported +16pp on a real skill) | **+0.0pp** — saturation; prompts too in-distribution |
| **Citation rate** | % of with-skill replies that name a `references/*.md` or `templates/*.py` file. | ≥ 80% | **60%** (3/5) |
| **Token cost ratio** | with-skill (input + output) / without-skill | ≤ 2× | input 2.4×, output 1.8× |

Runtime cost ≈ ~2 minutes for triggering (60 parallel judge calls), ~5–10 minutes for functional A/B (5 sequential prompts × 2 conditions through OpenClaw gateway).

The peft-tuning saturation result is informative: on canonical PEFT questions Sonnet's training data already covers the answer, so deterministic `must_contain` checks pass with or without the skill. The skill's actual lift shows up on **harder, niche, version-specific** questions — verified separately by the BOFT trajectory (`/tmp/agent-trajectories/06-mcp-fallback-boft.json`) where the agent fell through to Context7 to pull a complete `BOFTConfig` parameter table the without-skill arm wouldn't have had. Stage 2 is designed to surface this lift.

### Stage 2 — MLE-skill-bench (specified, not implemented)

A containerised, runnable benchmark for ML-engineering skills. Combines:
- MLAlgo-Bench's **EScore = ∆Score × pass-rate** (its eq. 4) — rewards a skill only when it *both* lifts pass-rate *and* keeps recipe fidelity high. Detects the recipe-ignoring / pass-rate-gaming shortcut (MLAlgo-Bench Table 6 — agent ignores prescribed recipe to game pass-rate).
- MLE-Bench / RE-Bench's **container isolation + held-out test data + fixed compute budget** so OOM and wall-clock failures count as evaluation failures.
- Anthropic skill-creator's **same-turn paired runs** for variance control + `text/passed/evidence` schema.
- Eugene Yan's **two-judge averaging** (one Claude + one non-Claude) to neutralise the +25pp self-preference bias measured for Claude-v1.

#### The 8 evaluation dimensions for MLE skills

Each dimension has at least one primary-source justification:

| # | Dimension | What we measure | Source |
|---|---|---|---|
| 1 | **Triggering precision/recall** | F1 over 20-prompt 60/40 split, 3 runs/query | Anthropic `skill-creator` lines 339–394 |
| 2 | **API correctness / hallucination on symbol names** | Rate of fabricated function/argument names; calls to non-existent classes | MLAlgo-Bench error taxonomy: 13% non-existent variable/function, 16% incorrect library; [BigCodeBench](https://arxiv.org/abs/2406.15877) top model 60% vs human 97% |
| 3 | **Workflow ordering correctness** | Multi-step pipeline tasks (SFT→RM→PPO; SFT-checkpoint before DPO) | [τ-bench (Yao et al. 2024)](https://arxiv.org/abs/2406.12045): GPT-4o <50% with `pass^8 < 25%` consistency. [MCP-Bench](https://arxiv.org/abs/2508.20453) trajectory-planning rubric tier. |
| 4 | **Hardware sizing accuracy** | Does "QLoRA-7B fits in 24 GB" hold? Container caps memory; OOM = fail | [MLE-Bench (Chan et al. 2024)](https://arxiv.org/abs/2410.07095): 75 Kaggle competitions, isolated containers. [RE-Bench (METR 2024)](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/): 7 environments under fixed compute, 71-expert human baseline. |
| 5 | **Convergence-aware recipes** | Run reduced training (e.g., 100 SFT / 50 DPO steps); assert loss decreases; final loss < calibrated threshold | [MLAgentBench (Huang et al. ICML 2024)](https://arxiv.org/abs/2310.03302): 13 ML tasks, Claude-3-Opus 37.5%. MLAlgo-Bench's ∆Score under Kaggle min-max bounds. |
| 6 | **Failure-mode coverage** | Does the skill anticipate the OOM, target_modules error, SFTConfig API break? Test prompts include known broken inputs. | [Sogl 2025](https://dev.to/danielsogl/skills-without-evals-are-just-markdown-and-hope-3a71): six idiom-correction prompts produced +16pp lift on @ngrx/signals (uncalibrated, single-author but instrumented). Anthropic skill-creator `agents/analyzer.md` "non-discriminating assertion" warning. |
| 7 | **Version-pinned behavior** | Pin `requirements.txt` lockfile per task; container runs against that lockfile only | MLE-Bench contamination warning. [Inspect_AI](https://inspect.aisi.org.uk/) Docker/Kubernetes sandboxing. |
| 8 | **LLM-judge calibration for subjective assertions** | κ ≥ 0.6 against ≥3 ML-engineer human raters; two-judge averaging | [Eugene Yan, *Evaluating LLM-Evaluators*](https://eugeneyan.com/writing/llm-evaluators/): position bias 70%, verbosity bias >90%, self-preference +25pp Claude-v1, +10pp GPT-4. [ToolEmu (Ruan et al. 2024)](https://arxiv.org/abs/2309.15817): LM safety judge agrees with humans 68.8%. |

#### Test-set composition

For each skill (`peft-tuning`, `trl-training`, `vllm-inference`, …), build:

```
~/.openclaw/workspace/skills/<skill>/evals/mle-skill-bench/
├── triggering.json                       20 prompts (10+10), 3 runs each
├── deterministic.json                    15-20 prompts with must_contain assertions
├── llm_judge.json                        10-15 prompts with Likert-1-5 instruction-following rubric
├── runnable/                             10-15 containerised end-to-end tasks
│   └── <task-id>/
│       ├── Dockerfile                    pinned requirements.txt (version-locked)
│       ├── task.md                       prompt + budget (e.g. "DPO 50 steps, lr=1e-6")
│       ├── assertions.json               (pass/fail, loss-decreasing, peak-mem<X)
│       ├── golden_reference.py           reference implementation
│       └── run_eval.sh                   spins container, records exit + loss curve
└── grading_results/                      timestamped per-run outputs
```

Total ~105 observations per skill — within the field-standard band (MLAlgo-Bench 121, MLAgentBench 13, RE-Bench 7, GAIA 466, τ-bench ~140 per domain).

#### Headline metric

Adapted from MLAlgo-Bench eq. 4:

```
SkillScore     = ∆Score(with_skill) − ∆Score(without_skill)
EffectiveLift  = SkillScore × pass_rate(with_skill)
```

A skill that lifts pass-rate but lowers fidelity (the recipe-ignoring / pass-rate-gaming shortcut, MLAlgo-Bench Table 6) gets penalised; only skills that lift *both* pass and recipe fidelity score well.

#### Acceptance thresholds

| Threshold | Source |
|---|---|
| Triggering F1 ≥ 0.85 | Anthropic skill-creator |
| EffectiveLift ≥ +10pp | Sogl +16pp benchmark |
| Citation rate ≥ 80% on with-skill capability runs | Field convention |
| Token cost ratio ≤ 2× without-skill baseline | Sogl measurements |
| Median wall ≤ 60s, median tokens ≤ 30k, median cost ≤ $0.10 per task | Sogl ($0.04 cold / $0.004 cached at Sonnet pricing) |
| LLM-judge κ ≥ 0.6 vs ≥3 ML-engineer human raters on 30 sampled tasks | Eugene Yan; MLAlgo-Bench Tables 4 / 12 (achieved 0.67–0.72) |

#### Anti-saturation discipline

When any skill clears 95% on the capability set, expand the eval set with harder cases (FSDP+LoRA, gradient-checkpointing tradeoffs, multi-node accelerate). Anthropic skill-creator's analyzer pass flags non-discriminating assertions for the same reason. Sogl: "100% capability saturation means evals only catch regressions, not gains" — when this happens, harden the test prompts.

### Sources cited (curated)

**Seed paper**
- [MLAlgo-Bench (Wang et al., EMNLP Findings 2025) — *Can Machines Implement Machine Learning Algorithms?*](https://aclanthology.org/2025.findings-emnlp.772/) — instruction-fidelity benchmark, EScore metric, recipe-ignoring / pass-rate-gaming shortcut warning (Table 6).

**MLE benchmarks (Dimensions 4, 5, 7)**
- [MLE-Bench (Chan et al., OpenAI 2024)](https://arxiv.org/abs/2410.07095)
- [RE-Bench (METR 2024)](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/)
- [MLAgentBench (Huang et al., ICML 2024)](https://arxiv.org/abs/2310.03302)
- [MLGym (Nathani et al. 2025)](https://arxiv.org/abs/2502.14499)

**Multi-step / tool-use trajectory (Dimension 3)**
- [τ-bench (Yao et al. 2024)](https://arxiv.org/abs/2406.12045)
- [MCP-Bench (Wang et al. 2025)](https://arxiv.org/abs/2508.20453)
- [GAIA (Mialon et al. 2023)](https://arxiv.org/abs/2311.12983)

**Code correctness baselines (Dimension 2)**
- [BigCodeBench (Zhuo et al. 2024)](https://arxiv.org/abs/2406.15877)
- [SWE-Bench / SWE-Bench Verified](https://www.swebench.com/)

**Skill evaluation methodology (Dimensions 1, 6, 8)**
- [Anthropic skill-creator SKILL.md](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) — canonical 20-prompt 60/40 + `text/passed/evidence` schema
- [Hamel Husain — *Evals Skills for Coding Agents*](https://hamel.dev/blog/posts/evals-skills/)
- [Daniel Sogl — *Skills Without Evals Are Just Markdown and Hope*](https://dev.to/danielsogl/skills-without-evals-are-just-markdown-and-hope-3a71) (uncalibrated single-author, instrumented)
- [Eugene Yan — *Evaluating LLM-Evaluators*](https://eugeneyan.com/writing/llm-evaluators/)

**Eval frameworks**
- [Inspect_AI (UK AISI)](https://inspect.aisi.org.uk/)
- [DSPy (Khattab et al. 2023)](https://arxiv.org/abs/2310.03714)
- [ToolEmu (Ruan et al., ICLR 2024)](https://arxiv.org/abs/2309.15817)

## Integration with Skill Scout

- Builder's validator imports Scout's `BLOCK_PATTERNS` and `CAUTION_PATTERNS` — one source of truth.
- Builder's output conforms to Scout's expected format, so Scout can find and install Builder-generated skills.
