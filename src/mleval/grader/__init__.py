"""Independent, held-out grading of agent prediction artifacts.

WHY THIS EXISTS
---------------
MLEvolve's per-node ``metric`` is the score the *agent itself* prints to
stdout (parsed by an LLM in ``result_parse_agent``). That number is the
tree-search signal — and it is gameable: a trajectory that drifts to a
different task (e.g. IMDB sentiment classification) will happily print a
high "Final Validation Score" that has nothing to do with the real task.

This module computes the **trustworthy** A/B metric the way every credible
agent benchmark does (METR Task Standard, mle-bench, SWE-bench): the agent
emits a per-example *prediction artifact* and an *independent grader*
recomputes the metric against held-out references. We grade the predictions
MLEvolve preserved for the best node (``best_submission/submission.csv``,
available because we run with ``no_submission_mode: False``), NOT a number
the agent reported. A drifted solution either produces no valid artifact or
predictions whose ids don't match the task's test set → it scores 0.

The references are public for SAMSum (the HF test split), so secrecy is not
the protection — *recomputation* is. We never trust the agent's self-report.

The ROUGE-L implementation here is intentionally self-contained (LCS-based
F1, lowercase word tokenisation, no stemming) so the grader is
zero-dependency, fully deterministic, and unit-testable with hand-computed
values. It is the authoritative A/B metric; swap in ``rouge_score`` only if
absolute comparability to published numbers is ever required.
"""
from __future__ import annotations

from .grade import GradeResult, grade_predictions
from .rouge import rouge_l_f, tokenize

__all__ = ["GradeResult", "grade_predictions", "rouge_l_f", "tokenize"]
