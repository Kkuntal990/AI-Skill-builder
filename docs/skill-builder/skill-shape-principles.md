# Skill-shape principles

Operational guidance for authoring OpenClaw skills the agent will actually use well. Distilled from Anthropic's official Skills documentation (May 2026), LlamaIndex's empirical work on Skills vs MCP tools (May 2026), and our own evals on `peft-tuning` and `vllm-inference`.

## Core principle: shape what MCP can't

A skill earns its place in the agent's context window only if it provides something the MCP-returned API docs can't:

- **Workflow choreography** — what order to do things, what stage to run a validator, which decision precedes which.
- **Project defaults** — the canonical quantization, the right tensor-parallel size for your topology, the standard checkpoint format.
- **Decision trees** — when X vs when Y, with concrete criteria the agent can apply.
- **Anti-patterns** — the things the docs don't tell you NOT to do.

When LlamaIndex tested giving an agent both a skill and an MCP-doc tool, the skill was "rarely invoked, and often did not yield substantially better results" — because the MCP docs answered everything the skill duplicated. The corollary: a skill that just paraphrases the docs is dead weight; a skill that *teaches* the workflow is high leverage.

Source: [LlamaIndex — Skills vs MCP tools for agents](https://www.llamaindex.ai/blog/skills-vs-mcp-tools-for-agents-when-to-use-what).

## Required sections (when applicable)

The builder emits these in this order. Each is optional based on whether the source library warrants it; planner decides.

1. **`# <Skill Name>`** — title + opening-paragraph description starting with an action verb, including at least one "Use when..." clause.
2. **`## Installation`** — canonical install command from the README.
3. **`## Quick Start`** — fastest zero-to-working example from the docs.
4. **`## Decision Tree`** — routing table for choices between modes/strategies. Only emitted when the library has 2-6 meaningful choices to make (e.g. online vs offline serving, FP8 vs AWQ vs GPTQ, tensor vs pipeline parallelism). Single-option scenarios don't belong here.
5. **`## Common Workflows`** — 2-4 multi-step procedures as copy-paste checklists. **Each checklist ends with a per-workflow MCP-fallback step** (see "inline MCP triggers" below).
6. **`## When to Use`** — concrete positive triggers + method-level escape-hatches ("for &lt;a capability this skill lacks&gt;, use &lt;tool&gt; instead"). Scope exclusions to capabilities the skill does not implement, **never** to adjacent task genres the skill could be a sub-step of (a genre-exclusion makes the agent scope the skill out of whole tasks — see Anti-patterns).
7. **`## Hardware Requirements`** — concrete VRAM/GPU numbers from docs, only when surfaced.
8. **`## Templates`** — runnable Python scripts in `templates/` (read-as-reference).
9. **`## Scripts`** — executable utilities in `scripts/` (executed via bash, not read into context).
10. **`## Old Patterns`** — collapsed `<details>` listing deprecated APIs. Only when `--with-version-notes` and the docs mention deprecations.
11. **`## References`** — short pointers to each `references/*.md`.
12. **`## Looking things up live (MCP fallback)`** — short (~14 line) tail-coverage section.

## Pattern: inline MCP triggers per workflow

Anthropic's official guidance says skills should "complement Model Context Protocol (MCP) servers by teaching agents more complex workflows that involve external tools." The empirical refinement: a single "MCP fallback" section at the bottom of SKILL.md is **bolted on and gets ignored** — agents read the workflow checklists and act on them.

So every workflow checklist ends with one MCP step:

```
### Serve a quantized model with LoRA

Copy this checklist:

- [ ] Step 1: Choose quantization format (FP8 / AWQ / GPTQ) and download checkpoint
- [ ] Step 2: Launch server with `--quantization X --enable-lora`
- [ ] Step 3: Verify reduced VRAM via `/metrics`
- [ ] **MCP fallback**: if your quant format isn't in `references/quantization-and-memory.md`, call `context7__resolve-library-id` with `libraryName="vllm"`, then `context7__query-docs` with the returned libraryId and `query="<format>"` — skip if references covered your case.
```

The MCP step explicitly says *when* to fall through (refs don't cover it) and *what to call* (specific tools with parameters). The agent treats it as the natural next step in the workflow, not an optional appendix.

## Pattern: MCP tool naming is runtime-specific

**OpenClaw uses double-underscore prefixing for native MCP tools.** Anthropic's official docs use single-colon naming (`Context7:get-library-docs`). They are NOT interchangeable.

| Runtime | Naming | Example |
|---|---|---|
| Anthropic first-party (Claude API / Claude Code) | `ServerName:tool_name` | `Context7:get-library-docs` |
| OpenClaw | `server__tool-name` | `context7__query-docs` |

Additionally: the actual Context7 tool is named `query-docs`, **not** `get-library-docs` (which is a name Anthropic's docs use in examples but isn't the live tool name). Always verify the actual registered tool names against `agentMeta.tools` in a smoke test before baking them into a skill.

Library IDs are also runtime-specific. Context7 returns IDs like `/websites/vllm_ai_en` or `/<org>/<project>`; the format depends on what `resolve-library-id` returns at runtime. Never hardcode — always instruct the agent to call `resolve-library-id` first, then pass the result to `query-docs`.

A skill that uses the wrong tool naming will produce 0 MCP calls even though the agent reads the instructions. This was observed empirically with vLLM in 2026-05-24: replacing `Context7:get-library-docs libraryId="/vllm-project/vllm"` with `context7__resolve-library-id` + `context7__query-docs` was the difference between "agent ignores SKILL.md MCP step" and "agent calls Context7 and returns live data."

## Pattern: scripts vs templates

Two distinct tiers with different intent:

| Dir | Intent | Token cost | Use when |
|---|---|---|---|
| `templates/` | **Read as reference** — complete end-to-end Python example to copy and adapt | Full file content loaded if agent reads it | The user needs a worked example for a workflow |
| `scripts/` | **Execute via bash** — short utility (<60 lines) for a specific check or operation | Only output consumed; file body never loaded | Health checks, validators, probes, single-purpose helpers |

Anthropic's spec: *"For most utility scripts, execution is preferred because it's more reliable and efficient."* The SKILL.md `## Scripts` section explicitly tells the agent to **execute, not read** — and the builder marks scripts as `chmod 0o755` so they run without intermediate `python` / `bash` invocations.

## Pattern: auto-ToC on long references

Anthropic's spec: *"For reference files longer than 100 lines, include a table of contents at the top. This ensures Claude can see the full scope of available information even when previewing with partial reads."*

The builder auto-prepends `## Contents` to any reference >100 lines that doesn't already have one. The LLM is also asked to write one itself (which it usually does, better than the auto-injected fallback).

This matters because the agent often peeks at a reference with `head -N` to decide whether to read the whole file. Without a ToC at the top, content past the cutoff is invisible to the decision.

## Pattern: old patterns in collapsed details

For libraries with breaking changes (vLLM v0/v1 split, HF transformers API churn, etc.), emit a collapsed `<details>` block:

```
## Old Patterns

<details>
<summary>Deprecated APIs (kept for historical context)</summary>

- **`AsyncLLMEngine` direct instantiation** (deprecated in 0.4.x) — use `AsyncLLM` instead.
- **vLLM V0 scheduler** (deprecated in 0.6.0) — V1 engine is now default.

</details>
```

The `<details>` keeps the deprecated info out of the agent's primary attention but available if it asks about a legacy pattern.

## Anti-patterns

- **Don't hardcode MCP libraryIds.** They change. Always teach `resolve-library-id` first.
- **Don't write workflow checklists without per-step MCP routing.** The bottom-of-page MCP section gets ignored.
- **Don't list every option.** Anthropic's spec: *"Don't present multiple approaches unless necessary."* Pick a default, then escape-hatch.
- **Don't exclude task genres in "When to Use".** A negative "NOT for: &lt;task genre&gt;" wall makes the agent scope the skill out of any task that *contains* that genre as a step — e.g. an inference skill that says "NOT for: fine-tuning" gets dropped from a fine-tune-then-evaluate task where it was the right tool for the eval step (observed on `vllm-inference`, 2026-06). Frame limits as positive, capability-scoped escape-hatches ("for &lt;a method this skill lacks&gt;, use Y instead") instead. No Anthropic source endorses "NOT for X" exclusion blocks, and they collide with Claude's documented tendency to under-trigger.
- **Don't use time-sensitive language** ("after July 2026 use X"). Use the `## Old Patterns` collapsed section for legacy info instead.
- **Don't include API key references with literal values** (e.g. `OPENAI_API_KEY=sk-abc123`) — the security scanner correctly blocks these. Env var name references (`os.environ["OPENAI_API_KEY"]`) are fine and recognized as benign.

## Sources

- [Anthropic — Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) — official spec.
- [Anthropic — Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — design framing, Skills ↔ MCP relationship.
- [LlamaIndex — Skills vs MCP tools for agents: when to use what](https://www.llamaindex.ai/blog/skills-vs-mcp-tools-for-agents-when-to-use-what) — empirical caveat: skills rarely invoked when MCP duplicates content.
- [Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — eval vocabulary used in `docs/eval/stage1.md`.
