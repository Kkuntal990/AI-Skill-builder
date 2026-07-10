# Sub-agent eval orchestration — parallel, background, script-orchestrated

**Status:** implemented (2026-07-10) — P1–P3 shipped; see the Phasing section. Prompted by the
live full-gate failure on `peft-tuning`.
Governs how the Stage-1 behavioral eval fans out sub-agents. Companion to
[stage1.md](stage1.md) and the builder's [eval-gate-plan.md](../skill-builder/eval-gate-plan.md).

## The problem

The `full` ship-gate, delegated to a single `skill-tester` agent turn, **orphaned**: the
turn returned in ~6s having emitted only a Bash tool call (launching a ~23-min eval), so it
detached and left `eval_skill.py` running with no way to return the verdict. Root cause is a
nested, synchronous, sequential fan-out:

```
builder (python) → openclaw agent --agent skill-tester   ← ONE conversation turn (Layer 1)
                     └─ bash: python eval_skill.py gate --profile full
                          └─ run_functional: 24× openclaw agent --agent main  ← SEQUENTIAL (Layer 2)
```

Layer 1 has to stay attached for the whole ~23 min while Layer 2 runs 24 sequential agent
turns. It doesn't. This is the anti-pattern; the SDK even documents the exact symptom: *"a
subagent whose only output was tool calls with no text fails … terminated early."*

Two related gaps surfaced alongside it:
- **MCP never fires** in the functional A/B — the executor is `main`, which has no MCP client,
  so every trial is `clean_miss` (0.0), even tests designed to need live docs.
- **The gate verdict drops the MCP signals** — `run_functional` captures them but the verdict
  only surfaces pass-rate/lift.

## What Anthropic does (the solved pattern)

Sub-agents in the Claude Agent SDK are a first-class primitive with exactly the properties
that avoid our failure ([SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents),
[building agents](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)):

- **Parallel** — "multiple subagents run concurrently, so independent subtasks finish in the
  time of the slowest one rather than the sum of all of them."
- **Background by default** (v2.1.198) — an Agent call runs in the background unless the result
  is needed inline; a `background: true` field forces it. Async, built in.
- **Context-isolated**, with **per-agent `tools` / `model` / `mcpServers` / `skills`** and
  **bounded nesting** (5 levels).
- **Grader loop** ("Performance Outcomes") — a separate grader sends each sub-agent back to
  revise until it meets a rubric.
- **Scale → outside the conversation** — "for runs that coordinate dozens to hundreds of
  agents, use the `Workflow` tool, which moves orchestration into a **script the runtime
  executes outside the conversation context**." Turn-by-turn delegation is only for a *few*
  sub-agents per turn.

Anthropic's own `skill-creator` tests this way: parallel with-skill/without-skill **executor**
sub-agents per case + a **grader** sub-agent + a blind **comparator** + an **analyzer**, with
async completion notifications; it degrades to sequential inline only when sub-agents are
unavailable.

Prior art to borrow from: [skillgrade](https://github.com/mgechev/skillgrade) ("unit tests for
agent skills"; graders score workspace state; local+Docker), the
[Agent-as-a-Judge survey](https://github.com/ModalityDance/Awesome-Agent-as-a-Judge),
[AgentBench](https://github.com/THUDM/AgentBench),
[SkillVetBench](https://arxiv.org/html/2606.15899v1) (LLM-judge over the skill artifact), and
Langfuse's [MCP-agent evaluation cookbook](https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation).

**We already run this shape in-house:** the MLEvolve `entrypoint.sh` (`setsid` a long job →
write a result file → the orchestrator polls) is exactly the background/script pattern. Reuse
it; don't reinvent.

## skill-tester removed

We briefly had the builder **delegate the gate to a `skill-tester` agent** (the author↔tester
split). Live testing showed that's redundant and harmful: the eval is a **deterministic
pipeline** (triggering → activation → functional, with only *scripted* `claude -p` judge/grader
calls) — there is no orchestration-level decision for an LLM to make. Wrapping it in an agent
turn added the orphan failure, nondeterministic JSON, and cost, for **zero reasoning value**.
Anthropic's own split confirms this: `skill-creator` is an agent because a *human* talks to it,
but its deterministic parts are **scripts**. With no human in our loop, the orchestrator agent
has no role, so **`skill-tester` was deleted** (2026-07-10). What survives is the good part —
`eval_core` is a self-contained module (it just lives in the builder's `scripts/` again, imported
directly). The one necessary agent is the **executor** (§3), which was never `skill-tester`.

## Design

Map the pattern onto our OpenClaw stack via four components.

### 1. Script-orchestrated, no orchestrator agent (fix the orphan)

The orphan came from wrapping a deterministic, long, fan-out eval inside **one `skill-tester`
agent turn**. There is no reasoning for an LLM to do at the orchestration level, so **the
orchestrator agent was removed entirely** (see [the skill-tester removal note](#skill-tester-removed)).
The builder now calls the eval **directly as a module** — `eval_core.run_gate(...)` — which
is the Anthropic "orchestrate many sub-agents from a script outside the conversation" pattern.

| Profile | Orchestration | Rationale |
|---|---|---|
| **smoke** (default) | Builder calls `eval_core.run_gate(profile="smoke")` in-process (~1 min, triggering only). | No agent turn to keep alive; deterministic. |
| **full** (opt-in) | Builder calls `eval_core.run_gate(profile="full")`; `run_functional` **fans out parallel executor sub-agents** (§2). For very long runs, launch the call **detached** (`setsid` → write `verdict.json` → poll), mirroring the MLEvolve `entrypoint.sh` pattern. | full = "dozens of executor turns" → orchestrate from a script; parallelism keeps it short enough that detachment is usually unnecessary. |

The only agent the eval spawns is the **executor** inside `run_functional` (§2/§3) —
measuring agent behavior requires an agent. Orchestration, grading, and analysis are plain
code. This is exactly the direct run that completed in ~23 min (now ~5 min with §2's
parallelism); the failed path was the redundant `skill-tester` turn wrapped around it.

### 2. Parallel, isolated executors (fix speed — likely the biggest single win)

`run_functional` runs its trials **sequentially** today (24 turns → ~20 min). Switch to a
`ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY)`; each `_run_agent` turn is already
independent and context-isolated.

- `EVAL_CONCURRENCY` default **4** (env `MLEVAL_EVAL_CONCURRENCY`), deliberately under the
  claude-code plugin's `maxConcurrentSessions: 5`, with **retry + backoff on
  `FailoverError`/session-limit**.
- 24 trials ÷ 4 ≈ ~5 min vs ~20. This alone may make `full` short enough to sync-delegate —
  but the design does **not** depend on that; tier 1 is the robust fix.

### 3. One MCP-capable executor for both content and MCP (fix MCP-fallback)

A **single** executor identity — **`skill-eval-target`** = minimal identity **+ scoped
`mcpServers: [context7]`**, nothing else — used for **both** the content A/B and the
MCP-fallback measurement (not `skill-tester`, so no self-nesting). This replaces the earlier
two-executor split (`main` for content, a separate agent for MCP); one agent is simpler, more
realistic (real agents have tools), and fires MCP in the *normal* functional run so the
4-signal capture stops reading all-zero.

The "ambient-MCP confound" (an MCP-capable executor can fire MCP in the *without*-skill cell
too) is resolved by measuring the **delta**, not absolutes:
- **content pass-rate lift** = with-skill − without-skill (as today).
- **MCP-firing lift** = with-skill firing rate − without-skill firing rate. Both cells *can*
  reach context7; only the with-skill cell is *told* to (`context7__query-docs`), so the delta
  attributes firing to the skill's instruction.

Trade-off to record: we give up the "pure content lift with **zero** tools" number. If the
without-skill cell answers by hitting context7 itself, content lift reads *smaller* — a **true
finding** (the skill's prose is redundant with live docs), not noise.

To actually *exercise* the fallback, tag a few **beyond-`references/`** prompts in the same
functional set (questions the skill's static content can't answer, so MCP is the only path).
These are tagged cases in the *same* eval on the *same* executor — **not** a separate agent or
eval mode. Grade them on the `best_case`/`stealth_use` firing delta.

### 4. Surface signals + grader (fix the reporting gap)

- Add `mcp_metrics` to the gate verdict: `actual_mcp_call_rate` + `mcp_classification_counts`
  from the functional run, plus the mcp-fallback firing rate. (Today the verdict drops these.)
- Grader unchanged: `_score_assertions` (deterministic) + optional `claude -p` grader +
  `analyze()` (non-discriminating/flaky) — already the skill-creator grader shape.

## Phasing (each independently shippable, low→high risk)

1. **P1 — Parallelize `run_functional`** ✅ **done** (2026-07-10) — `ThreadPoolExecutor`
   at `EVAL_CONCURRENCY` (env `MLEVAL_EVAL_CONCURRENCY`, default 4), keyed by
   `(test_idx, side, run)` + bounded retry-on-empty. Biggest speed win.
2. **P2 — `full` runs in-process from the builder** ✅ **done** — `run_ship_gate` calls
   `eval_core.run_gate` directly (no orchestrator agent). Detachment (`setsid`→verdict-file→poll)
   proved unnecessary once P1 cut wall-clock. *(Delegation to a skill-tester turn is removed.)*
3. **P3 — single `skill-eval-target` executor** ✅ **done** (2026-07-10) — minimal MCP-capable
   agent (`agents/skill-eval-target/`, bare persona + `mcporter`/context7, no content skills),
   registered in `openclaw.json`, is now the functional-A/B executor (`FUNCTIONAL_EXECUTOR`,
   was `main`). `write_evals` emits a tagged beyond-`references/` `mcp_fallback` case whenever the
   skill declares an MCP; `run_functional` splits content-lift (content tests only) from the
   MCP-firing **delta** (with−without, both cells MCP-capable) and the fallback-subset firing;
   `run_gate` surfaces all of it as `mcp_metrics`. Advisory (never blocks the ship).
4. **P4 — (optional)** background-subagent framing + grader-revision loop ("Performance
   Outcomes").

## Caveats

- **Session limits are the real ceiling** — the concurrency cap must stay conservative and
  retry on `FailoverError` (we hit rolling-window session limits during validation).
- **Parallelizing changes trial order/RNG** — fine for independent trials; note it in results.
- **`skill-eval-target` is registered** (openclaw.json `agents.list` + repo agent dir
  `agents/skill-eval-target/`). It reaches context7 via the bundled `mcporter` skill (the
  OpenClaw MCP config is global — there is no per-agent `mcpServers` field — so scoping is via
  the agent's whitelisted skills/binaries, as the old `skill-tester` did). Override the executor
  with `MLEVAL_FUNCTIONAL_EXECUTOR` (e.g. back to `main`) if it isn't registered.
- **MCP-firing capture across the gateway.** `_run_agent` injects a sidecar `mcporter` PATH
  wrapper, but agent turns run in the gateway daemon (not the client subprocess), so the
  sidecar log stays empty; likewise the skill's SKILL.md tells the agent to call *native*
  `context7__…` tools, but `skill-eval-target` reaches context7 via the `mcporter` **bash**
  route — so `tool_summary.tools` is empty too. Net: the **reply-text / outcome** signal is
  the only reliable ground truth for gateway-routed runs. Validated live on peft (2026-07-10):
  the executor *did* query context7 ("Queried live PEFT docs (context7, `/huggingface/peft`)…
  use_rslora default is False") and answered correctly, but the first run scored it
  `clean_miss` because the narration patterns were too narrow. Fixed in `_extract_tool_signals`:
  a resolved **library-id path** (`/org/project`, a resolve-library-id *return value*, gated on
  a declared-server mention so local file paths can't spoof it) now counts as actual-call
  ground truth, plus the realistic narration phrases. The with−without delta holds either way.
- **`run_gate` doesn't persist per-test reply text** — only aggregate `mcp_metrics`. Debugging
  a zero delta means re-running the functional A/B standalone (`eval_skill.py functional`) to
  see replies. A cheap future win: dump the full functional result (with `reply_text`) into the
  bundle's `evals/`.
- **`full` stays opt-in.** Default is `smoke`; only an explicit `full` pays the fan-out cost.

## Sources

- Claude Agent SDK — [Subagents](https://code.claude.com/docs/en/agent-sdk/subagents) ·
  [Building agents](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) ·
  [Dynamic workflows](https://code.claude.com/docs/en/workflows)
- Anthropic `skill-creator` — https://github.com/anthropics/skills/tree/main/skills/skill-creator
- [skillgrade](https://github.com/mgechev/skillgrade) ·
  [Agent-as-a-Judge survey](https://github.com/ModalityDance/Awesome-Agent-as-a-Judge) ·
  [AgentBench](https://github.com/THUDM/AgentBench) ·
  [SkillVetBench](https://arxiv.org/html/2606.15899v1) ·
  [Langfuse MCP-agent eval](https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation)
