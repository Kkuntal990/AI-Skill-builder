---
name: evaluate-skill
description: "Evaluate an OpenClaw skill bundle and return a machine-readable verdict — triggering F1 (LLM judge vs real sibling skills), organic activation (does claude -p read the skill on its own), functional with/without-skill A/B, held-out description optimization, and a baseline-first gap probe. Use whenever you are asked to run a skill's ship-gate, score/benchmark a skill, check whether a skill triggers, optimize a skill's description, or probe what a base model misses without a skill — especially when the ai-skill-builder delegates gating to you. Prompt-level only: no task execution, no GPU."
metadata:
  {
    "openclaw":
      {
        "emoji": "🧪",
        "requires": { "bins": ["python3", "claude"] }
      }
  }
---

# evaluate-skill

This skill is the **tester** half of the author↔tester split. The `ai-skill-builder`
agent (the author) generates a skill, then **delegates** the behavioral evaluation to
you. You run the shared eval harness (`scripts/eval_core.py`, driven by
`scripts/eval_skill.py`) against a skill bundle on disk and **return the JSON verdict
verbatim** so the caller can parse it.

The harness owns all the eval logic (triggering judge, decoy/sibling competitors,
organic-activation executor, functional grader, description optimizer). You just invoke
it and relay its JSON. Do **not** re-implement scoring or paraphrase results.

## The one rule

**Run the script, return ONLY its JSON stdout.** No preamble, no summary, no markdown
fences around it unless asked. The caller (usually the builder) parses your reply as
JSON. Extra prose breaks that parse.

## What the caller asks for, and what you run

The eval scripts live in this skill's `scripts/` directory. Resolve that directory from
where this SKILL.md is loaded and invoke with `python3`. `<SCRIPTS>` below = that path.

### 1. Ship-gate (the builder's main delegation)

> "run the ship-gate on the skill bundle at `<dir>`, profile `<smoke|full>`, siblings `<sib-dir>`"

```bash
python3 <SCRIPTS>/eval_skill.py gate <dir> --profile <smoke|full> --siblings <sib-dir>
```

- `smoke` — triggering only, 1 run/prompt (cheap). `full` — triggering ×3 + organic activation
  **+ the functional with/without-skill A/B** (runs real agent turns via the `main` baseline;
  minutes, not seconds).
- Reads `<dir>/evals/triggering.json` and (in `full`) `<dir>/evals/functional.json` — both
  written by the author — plus `<dir>/pass_bar.json`.
- Emits the **behavioral verdict** JSON: `{passed, reasons, ran, skipped, triggering_metrics,
  activation_metrics, functional_metrics, functional_analysis, failing_positives, report}`.
  **Triggering + activation gate (hard); the functional A/B is ADVISORY** — its pass-rate/lift
  and the analyzer's non-discriminating/flaky flags are reported but never added to `reasons`
  (the assertions are self-authored). The builder folds in its own artifact `quality_gate`
  afterward — that part is not your job.

Return that JSON object and nothing else.

### 2. Baseline-first gap probe

> "run baseline-probe; payload at `<path>`"  (payload = JSON `{intent, doc_excerpt, n_questions}`)

```bash
python3 <SCRIPTS>/eval_skill.py baseline-probe --payload-file <path>
```

Runs the **base** model (no skill, via the `main` agent — never `skill-tester`, to avoid
recursion) on intent-derived questions and summarizes what it misses. Emits
`{"gap_notes": "..."}`. Return it verbatim (`gap_notes` is `""` if the baseline already
answers well).

### 3. Manual / full evaluation (a human asks directly)

```bash
python3 <SCRIPTS>/eval_skill.py all <dir> [--with-activation] [--siblings <dir>] [--llm-grader]
python3 <SCRIPTS>/eval_skill.py triggering <dir>            # triggering F1 only
python3 <SCRIPTS>/eval_skill.py activation <dir>            # organic activation only
python3 <SCRIPTS>/eval_skill.py optimize-description <dir>  # 60/40 held-out description tuning
python3 <SCRIPTS>/eval_skill.py report <dir>                # markdown summary of latest results
python3 <SCRIPTS>/eval_skill.py pass-bar <dir>              # exit 0 PASS / 1 FAIL vs pass_bar.json
```

For a human, you may add a short plain-language summary *after* printing the JSON. For a
delegated/automated request, JSON only.

## Notes

- **Transport:** the judge, activation executor, and LLM grader all run on `claude -p`
  (the Claude subscription), model `$SKILLBUILD_LLM_MODEL` (default `opus`). No OpenRouter
  credit is spent unless a functional `--agent` run is explicitly requested.
- **Prompt-level only.** This never trains a model or runs a task end-to-end — that is the
  Stage-2 MLEvolve A/B, a separate system. If asked to "run the full task A/B," say that is
  out of scope for this skill.
- **Recursion guard.** The functional executor defaults to `ai-skill-builder`/`main`, and
  baseline-probe defaults to `main`. Do not point any executor at `skill-tester` — that
  would spawn this agent inside itself.
- If a script exits non-zero or prints no JSON, report the stderr tail plainly so the caller
  can see the failure rather than a silent empty verdict.
