"""Eval-harness rules — injected into MLEvolve impl_guideline (NOT skill content).

WHY THIS MODULE EXISTS (separate from skill_injector.py)
--------------------------------------------------------
Task-agnostic benchmark rules (held-out grading, validate tool, resource budget,
num_workers safety) must reach BOTH A/B cells identically via MLEvolve's native
**Implementation guideline** channel — the same per-node prompt block upstream
already uses for exec timeout and submission path.

Those rules are NOT skills and must not live in skill_injector.py, which only
implements Anthropic progressive disclosure (catalog + per-node skill selector).
skill_injector *calls* ``apply_impl_guideline_harness`` from here inside its
``get_impl_guideline_from_agent`` wrapper because that wrapper is already the
universal seam on all four codegen agents.

Operative source for runtime text: ``EVAL_HARNESS_RULES`` below.
Human-readable mirror: ``infra/tasks/_harness_rules.md`` (keep in sync).

Task-specific contract (data files, model pin, metric, columns) stays in each
task's ``instruction.md`` → ``description.md`` only.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Sentinel in infra/tasks/_harness_rules.md — skill_injector strips through this
# when routing (legacy C1 prepend; harmless no-op when task text has no marker).
HARNESS_RULES_MARKER = "<!-- END_HARNESS_RULES -->"

# Appended once per node to result["Implementation guideline"]. Keep in lockstep
# with infra/tasks/_harness_rules.md (minus the doc-only conflict-priority prose).
EVAL_HARNESS_RULES: list[str] = [
    "",
    "🔒 **Held-out evaluation rules** (this benchmark):",
    "• Use exclusively the data loader and files named in the task description. "
    "Do not download datasets from the internet or reconstruct held-out labels "
    "from external sources.",
    "• Predictions must come from a model you train on the provided training "
    "data — no hand-written, copied, or looked-up answers; do not train or "
    "select on the held-out test set.",
    "• The test inputs have their TARGETS WITHHELD — graded against references "
    "you cannot see.",
    "• Estimate your `Final Validation Score` on a validation split you hold out "
    "from the TRAINING data — never on the test set.",
    "• Your `./submission/submission.csv` is the ONLY thing graded (independently, "
    "after the run). Your printed score is just the search signal — it is not the "
    "official result.",
    "• Validate your submission FORMAT before finishing (this does NOT reveal your "
    "score): `python -m mleval.grader.validate submission/submission.csv` — fix any "
    "INVALID before you stop.",
    "• **Resource budget:** each execution of your script has a per-run wall-clock "
    "cap (also shown above in these guidelines). Training AND evaluation must "
    "finish within it; a killed run keeps only whatever submission was already "
    "written. For generation-heavy tasks, batch your decoding.",
]

# MLEvolve impl_guideline hardcodes num_workers>=2; fork-after-CUDA segfaults.
_NUM_WORKERS_BAD = "num_workers>=2"
_NUM_WORKERS_FIX = (
    "• Use DataLoader with num_workers=0 (forking workers AFTER CUDA/model init "
    "can segfault the process, exit 139); only raise it if you create the loader "
    "before any CUDA call"
)


def apply_impl_guideline_harness(result) -> None:
    """Inject eval-harness rules + fix the num_workers line. Cell-agnostic.

    Mutates ``result["Implementation guideline"]`` in place. Safe/no-op if the
    structure is unexpected. Idempotent within a node.
    """
    try:
        gl = result.get("Implementation guideline")
        if not isinstance(gl, list):
            return
        for i, line in enumerate(gl):
            if isinstance(line, str) and _NUM_WORKERS_BAD in line:
                gl[i] = _NUM_WORKERS_FIX
        if not any(isinstance(l, str) and "Held-out evaluation rules" in l for l in gl):
            gl.extend(EVAL_HARNESS_RULES)
    except Exception as e:  # noqa: BLE001 — never break codegen
        logger.warning("[eval_harness] impl_guideline injection failed: %s", e)
