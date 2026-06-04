# Stage 1 — Local skill eval

Fast, CI-style evaluation of a skill in isolation. Runs on every `build-skill-from-docs` invocation. ~5 min, ~$1 per skill.

Follows Anthropic's vocabulary verbatim ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), 2026-01):

- **Task** — a single test with defined inputs and success criteria.
- **Trial** — one attempt at a task. We run `n=3` trials per task to average over model stochasticity.
- **Grader** — logic that scores one aspect of the response. Three tiers in order of preference: **code-based → model-based → human**.
- **Eval saturation** — when the skill passes ≥90% of solvable tasks. Trigger to add harder tasks.

## Test agent

`eval_skill.py` defaults to `--agent ai-skill-builder` (a CLI default, not the doc's recommendation). Choose based on what you're testing:

| Agent | Bootstrap | mcporter ambient? | Best for |
|---|---|---|---|
| `main` | ~0K (no SOUL/IDENTITY/AGENTS files; no bundled skills) | No | Measuring the marginal lift of a skill in isolation. The cleanest A/B baseline. |
| `skill-tester` | ~50K (OpenClaw system prompt + tools, plus the `mcporter` baseline skill) | **Yes** | Testing MCP-dependent skill features end-to-end — but see ambient-MCP confound below. |
| `ai-skill-builder` | heavy — 18+ baseline skills | Yes | Avoid for skill A/B: baseline skills cover too much adjacent knowledge. |

**The ambient-MCP confound** (discovered 2026-05-24): when the test agent has `mcporter` bundled (as `skill-tester` does), Context7 is reachable regardless of whether the skill is loaded. A test that asks the agent to fetch live docs will fire MCP in **both** the with-skill and without-skill cells. So an MCP-must-use prompt cannot directly measure whether *the skill* triggers MCP — it can only measure whether the skill teaches better workflow choreography around MCP calls. To A/B-test the *triggering* of MCP from skill instructions, use `main` (no bundled mcporter).

Invocation:
```bash
python3 .../eval_skill.py all ~/.openclaw/workspace/skills/<skill> \
  --agent main --runs 3                # clean lift measurement
# or
python3 .../eval_skill.py all ~/.openclaw/workspace/skills/<skill> \
  --agent skill-tester --runs 3        # MCP outcome plumbing
```

## MCP signal capture (four signals)

Every functional trial captures four orthogonal MCP signals. Three were original; the fourth (outcome narration) was added 2026-05-24 to recover false negatives from a runtime bug — see below.

| Signal | Source | Catches |
|---|---|---|
| 1. Native tool calls | OpenClaw `agentMeta.toolSummary.tools[]` filtered by declared MCP servers | `context7__query-docs`, `context7__resolve-library-id` etc. — the canonical OpenClaw MCP invocation path |
| 2. Bash sidecar log | Temporary `mcporter` PATH shim wrapping the real binary | Bash CLI invocations like `mcporter call context7.query-docs ...` |
| 3. Text narration (call syntax) | Regex on reply text for `mcporter call` and `<server>__` patterns | Agent narrating the call syntax verbatim |
| 4. **Text narration (outcome)** | Regex on reply for "Fetched via Context7", "LibraryId: /...", `/websites/...`, etc. | Agent reporting *results* from an MCP call without naming the tool — the realistic case |

Classification combines them into one verdict per trial:

| Class | Means |
|---|---|
| `best_case` | Any of 1/2/4 fired AND signal 3 narrates the call |
| `stealth_use` | Any of 1/2/4 fired without narration |
| `lip_service` | Signal 3 narrated but no actual fire (1/2/4 all empty) |
| `clean_miss` | None — no MCP usage attempted |

**Why signal 4 is necessary** (the `toolSummary`-null bug): OpenClaw's `agentMeta.toolSummary` is sometimes `null` for multi-step responses (likely subagent spawning loses top-level aggregation). Without signal 4, real MCP calls register as `clean_miss` whenever toolSummary is null. The 2026-05-24 diag captured an agent reply with verbatim Context7 output (`LibraryId: /websites/vllm_ai_en` plus the actual flag name `--cudagraph-capture-sizes`) and `toolSummary: null` — by signals 1–3 alone, this would be a false negative. Signal 4 recovers it as `best_case`.

The sidecar log is authoritative for the bash CLI path; the native tools path needs signals 1+4 because OpenClaw `toolSummary` is unreliable.

## Reply text preservation

As of 2026-05-24, `eval_skill.py` writes full `reply_text` into each trial's JSON (previously only `reply_chars`). This enables offline re-grading when assertions or detection patterns change. File sizes grow ~5×; acceptable trade for the ability to apply harness fixes without re-spending on LLM calls.

## Model variance caveat

`openrouter/deepseek/deepseek-v4-pro` shows large day-to-day reply-length variance. Same prompts, same skill, different days produced median reply lengths of **430 chars one day and 28 chars another**. Pass-rate measurements from a single run can swing 30+ pp purely from this variance. Multi-day median-of-medians or model-of-models is needed before declaring one skill version better than another in absolute terms.

## Eval sets and graders

| File | Anthropic role | Size |
|---|---|---|
| `evals/triggering.json` | Tasks — does the skill activate on the right prompt? | 10 should-trigger + 10 near-miss |
| `evals/functional.json` | Tasks — does the response contain required content + citations? | 5 tasks, each with `must_contain` / `must_not_contain` / `expected_citations` |

| Tier | What it scores | Implementation |
|---|---|---|
| Code-based | Triggering F1, regex `must_contain` / `must_not_contain`, citation presence | Already in `scripts/eval_skill.py` |
| Model-based | Faithfulness, citation accuracy for nuanced functional tasks | Deferred to v0.3 |
| Human | Spot-check 5 random transcripts per skill build | Manual review, weekly |

## Metrics reported per skill

- Triggering F1, precision, recall
- Functional pass rate (per task, averaged over `n=3` trials)
- Citation accuracy (fraction of `expected_citations` matched)
- **Actual MCP call rate (sidecar log)** — ground-truth invocation count
- **Narrated MCP rate (text regex)** — what the agent claims to have done
- **Cross-signal classification histogram** — `best_case` / `stealth_use` / `lip_service` / `clean_miss`
- Lift (with-skill pass − without-skill pass), reported as signal not gate
- Eval-saturation flag (`true` if both arms ≥ 0.9)
- Token in/out ratio (with-skill / without-skill)

## Pass bar

Three gates, AND-combined. **All measured on the with-skill cell only.** Without-skill cell feeds the lift number only.

- Triggering F1 ≥ 0.85
- Functional pass rate ≥ 0.6
- Citation accuracy ≥ 0.5

Lift is *signal not gate* in v0.1 — Goodhart-safe choice. A skill that performs well in absolute terms passes even if the base model already covered most prompts. `saturated=true` (both arms ≥ 0.9) flags for human review.

## Sources

- [Anthropic — *Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — 2026-01. Source of the vocabulary (task / trial / grader / saturation) and grader hierarchy.
