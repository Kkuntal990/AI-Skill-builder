# Stage 1 — Local skill eval

Fast, CI-style evaluation of a skill in isolation. Runs on every `build-skill-from-docs` invocation. ~5 min, ~$1 per skill.

Follows Anthropic's vocabulary verbatim ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), 2026-01):

- **Task** — a single test with defined inputs and success criteria.
- **Trial** — one attempt at a task. We run `n=3` trials per task to average over model stochasticity.
- **Grader** — logic that scores one aspect of the response. Three tiers in order of preference: **code-based → model-based → human**.
- **Eval saturation** — when the skill passes ≥90% of solvable tasks. Trigger to add harder tasks.

## Locked test agent: `main`

All Stage 1 A/B runs use the **`main` openclaw agent** as the test target. `main` is:

- No workspace dir, no `SOUL.md` / `AGENTS.md` / personality injection
- No baseline skills attached (zero `skills.entries[]` at boot)
- Default model only (currently `openrouter/deepseek/deepseek-v4-pro`)

Rationale: `main` is the cleanest baseline. Other agents we tested (`ai-skill-builder` with 18 baseline skills, `skill-tester` with auto-regenerated personality prompts) muddied the signal — baseline skills already cover PEFT-adjacent knowledge, suppressing the marginal lift a new skill contributes.

Invocation:
```bash
python3 .../eval_skill.py functional ~/.openclaw/workspace/skills/<skill> \
  --agent main --runs 3
```

## MCP signal capture (Signal 3 — sidecar wrapper)

Every functional trial is wrapped in a temporary `mcporter` PATH shim that JSONL-logs every invocation. Ground-truth answer to "did the agent actually call MCP."

| Class | Means |
|---|---|
| `best_case` | Sidecar log shows call AND reply text narrates it |
| `stealth_use` | Sidecar log shows call but reply doesn't narrate (agent was silently competent) |
| `lip_service` | Reply narrates "I called MCP" but sidecar log is empty (hallucinated call) |
| `clean_miss` | Neither — no MCP usage attempted |

The sidecar log is authoritative because openclaw's `toolSummary.tools[]` does not track Bash subprocesses (how `mcporter` is invoked).

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
