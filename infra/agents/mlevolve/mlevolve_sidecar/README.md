# mlevolve_sidecar

Monkey-patches applied to MLEvolve at import time. Sits BEFORE the
upstream agent in `run_mlevolve.py`'s import order so every patch lands
before any agent module captures the unpatched reference.

## Modules and what they patch

| Module | Patches | Why |
|---|---|---|
| `seed.py` | `random` / `numpy.random` / `torch.manual_seed` | Pin RNG from `$SEED` for paired-seed A/B |
| `openai_apikey_env.py` | `openai.OpenAI(api_key=...)` | Backfill from `$OPENAI_API_KEY` when config has `api_key: ""` |
| `prompt_logger.py` | `llm.openai.{query, generate}` | Capture per-call `{system, user, prompt, output, tokens, t_sec}` to `$MLEVAL_PROMPTS_LOG`. Captures BOTH the kwargs (`query`) and positional/kwarg `prompt` (used by `generate` — stepwise/diff/planner) — see spike-011 root-cause notes in source. |
| `skill_retriever.py` | `agents.prompts.environment.get_prompt_environment` | Loads skill(s) from `$MLEVAL_SKILL_PATHS` (colon-separated; falls back to `$MLEVAL_SKILL_PATH` singular). Splices a two-tier (catalog + full body) skill block into `prompt["Instructions"]`. The dict slot is reached by both stepwise's StepAgent + MetaAgent (via `prompt_base["Instructions"]` copy) and the non-stepwise draft path. |

## Design philosophy

We deliberately limit our patches to the minimum needed for the A/B
treatment. Earlier iterations also shipped `prompt_overlay.py` (per-task
persona / impl_guideline / review override) and `env_overlay.py` (custom
package hint list). Both were removed in favor of trusting MLEvolve's
published configuration:

- The "Kaggle Grandmaster" persona is generic ML expertise framing —
  it does not push toward submission.csv on its own; that is gated by
  the `no_submission_mode: True` flag in `config.yaml`.
- The upstream 15-package env hint (xgboost, lightGBM, timm, etc.) is
  irrelevant noise for our text tasks but harmless and symmetric across
  cells.

If a future task genuinely needs persona override or env list rewrite,
re-introduce a narrowly scoped patch then — but the default position is
to follow MLEvolve as published.

## Skill injection — slot layout

When `MLEVAL_SKILL_PATHS` (or `MLEVAL_SKILL_PATH` singular) is set, the
patched `get_prompt_environment` returns:

```python
{
    "Installed Packages": "...",   # upstream-untouched
    "Available Skills":   "- **peft-tuning**: <description from frontmatter>\n- ...",
    "Skill Reference":    "### Skill: peft-tuning\n\n<full SKILL.md body>\n\n#### references/...\n\n...",
}
```

Both keys appear only when at least one skill loads. Multiple skills
(Anthropic up-to-8 pattern) are supported by passing a colon-separated
list; their bodies are concatenated under the same key.

## How a task uses it

1. Stage the skill on the PVC (e.g. `/results/data/peft-tuning/`).
2. The orchestrator's `--skill-path` (singular) populates
   `MLEVAL_SKILL_PATH`. For multi-skill A/B, set
   `MLEVAL_SKILL_PATHS=/path/one:/path/two` via env or template.
3. Tasks WITHOUT a skill path get no `Available Skills` / `Skill Reference`
   blocks — the upstream env dict passes through unchanged.

## Known upstream behaviors we accept (not patched)

- `agents/result_parse_agent.py:153` hardcodes "Kaggle grandmaster" for
  the result parser. Pollutes `prompts.jsonl` cosmetically; does not
  affect generated code.
- `agents/coder/stepwise_coder.py:312` has its own inline persona for the
  stepwise MetaAgent merge. Fires under `use_stepwise_generation=True`
  (hardcoded in upstream `engine/agent_search.py:57`).
- `agents/improve_agent.py:276` `_IMPROVE_DIFF_INTRODUCTION` is also Kaggle-
  shaped. Fires in `use_diff_mode=True` (our config).

These are cosmetic — they don't change library choice or contract. If
they ever start materially affecting agent behavior, add a narrow patch.
