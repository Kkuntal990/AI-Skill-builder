# mlevolve_sidecar

Monkey-patches applied to MLEvolve at import time. Sits BEFORE the
upstream agent in `run_mlevolve.py`'s import order so every patch lands
before any agent module captures the unpatched reference.

## Modules and what they patch

| Module | Patches | Why |
|---|---|---|
| `seed.py` | `random` / `numpy.random` / `torch.manual_seed` | Pin RNG from `$SEED` for paired-seed A/B |
| `openai_apikey_env.py` | `openai.OpenAI(api_key=...)` | Backfill from `$OPENAI_API_KEY` when config has `api_key: ""` |
| `prompt_logger.py` | `llm.openai.{query, generate}` | Capture per-call `{system, user, output, tokens, t_sec}` to `$MLEVAL_PROMPTS_LOG` |
| `prompt_overlay.py` | 3 prompt surfaces (see below) | Per-task declarative overrides for tasks that don't fit MLEvolve's hardcoded Kaggle-shape contract |
| `skill_inject.py` | description.md content | Splice a skill cookbook into the task brief for `with_skill` cells |

## prompt_overlay

The three monkey-patches, with their dual-bind invariant:

| Patched surface | Where defined | Where re-exported | Why dual-bind |
|---|---|---|---|
| `build_chat_prompt_for_model` | `agents/planner/base_planner.py:104` | `agents/planner/__init__.py:14` | 7 agent files import via the package re-export. Patching only the defining submodule leaves the re-export pointing at the original. |
| `get_impl_guideline_from_agent` | `agents/prompts/impl_guideline.py:8` | `agents/prompts/__init__.py:10` | 6 agent files import via the re-export. |
| `get_code_review_guidelines` | `agents/prompts/validation_template_prompts.py:33` | (same module call site at line 29) | Single in-module call. Dual-bind not needed because the call resolves via the module's globals at call time. |

The build-time smoke (`_smoke_imports.py`) asserts the patch identity
holds after import — this is the regression guard for upstream refactors.

### Activation

Set `MLEVOLVE_PROMPT_OVERLAY=/path/to/overlay.yaml`. If the env var is
unset or the file is missing/malformed, the overlay falls through to
upstream MLEvolve prompts (silent — with a warning log).

### Schema

See `overlays/peft_rouge.yaml` for the canonical example. Four optional
keys:

```yaml
persona:
  identity: str               # FULL replacement of upstream intro
instructions:
  what_to_produce: list[str]  # Becomes the "Implementation guideline" block
  self_check: list[str]       # Becomes the "Self-Check" checklist
review_facts:
  output_location: str        # Splices over the hardcoded "submission.csv" line
```

Missing keys → upstream behavior for that surface. Wrong types → drop with
warning, fall back to upstream.

### How a task uses it

1. Create `infra/tasks/<task>/prompt_overlay.yaml`
2. The entrypoint detects it next to `instruction.md` and exports
   `MLEVOLVE_PROMPT_OVERLAY` automatically.
3. Tasks WITHOUT an overlay get MLEvolve defaults (no behavior change for
   Kaggle-shape tasks like `lmsys-chatbot-arena`).

### Known residuals (acceptable for MVP)

- `agents/result_parse_agent.py:153` hardcodes "Kaggle grandmaster" for
  the result parser. Pollutes `prompts.jsonl` but does not affect generated
  code. Patch in a future iteration if it shows up in agent behavior.
- `agents/coder/stepwise_coder.py:312` has its own inline persona used
  in stepwise mode. Our config already disables stepwise; if we ever
  enable it, this becomes the 4th surface.

### Out of scope for MVP

- `omit_fragments` — skip hardcoded blocks (ROBUSTNESS, LEAKAGE_PREVENTION, etc.)
- `stepwise_mode: false` toggle
- `allowed_packages_extra`
- per-step persona overrides

Add when a task needs them; the YAML schema is loose (extra keys ignored).
