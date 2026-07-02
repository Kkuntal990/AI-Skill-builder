"""Task-specific predicates for llama-inference.

Loaded by ``mleval.analyzer.state_predicates`` via file-path import. Each
predicate takes ``$MLEVAL_OUTPUT_DIR`` and returns bool.

This task has no CSV submission — the metric is parsed from stdout of the
agent's final modified inference script. Predicates therefore focus on
"did the agent actually produce a runnable inference script and did it
emit the expected metric line?".
"""

from __future__ import annotations

import re
from pathlib import Path

METRIC_PREFIX = "Average per token generation time"
_METRIC_LINE = re.compile(
    re.escape(METRIC_PREFIX) + r"\s*:\s*\[?\s*([0-9eE+\-.]+)"
)


def _find_per_step_dirs(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("working_dirs/op_*"))


def _has_python_file(step_dir: Path) -> bool:
    for p in step_dir.rglob("*.py"):
        try:
            if p.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def inference_script_present(output_dir: Path) -> bool:
    """At least one per-step snapshot contains a non-empty .py file."""
    return any(_has_python_file(d) for d in _find_per_step_dirs(output_dir))


def _run_log_paths(output_dir: Path) -> list[Path]:
    """Where the agent's captured stdout lands — try a few likely locations."""
    candidates = list(output_dir.glob("agent_logs/*.log"))
    candidates += list(output_dir.glob("mlevolve_runs/**/*.log"))
    candidates += list(output_dir.glob("working_dirs/op_*/run.log"))
    return [p for p in candidates if p.is_file()]


def metric_line_emitted(output_dir: Path) -> bool:
    """Some captured log contains the 'Average per token generation time:' line."""
    for p in _run_log_paths(output_dir):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if _METRIC_LINE.search(text):
            return True
    return False


def metric_value_positive_finite(output_dir: Path) -> bool:
    """Last emitted metric value parses to a positive finite float."""
    last_val: float | None = None
    for p in _run_log_paths(output_dir):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in _METRIC_LINE.finditer(text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            last_val = v
    if last_val is None:
        return False
    import math
    return math.isfinite(last_val) and last_val > 0


def do_not_edit_block_preserved(output_dir: Path) -> bool:
    """No emitted .py file modifies the frozen block (model/tokenizer/dataset
    load + tokenize_and_filter_function + generation/context length).

    Heuristic check: any .py that contains both delimiters must contain the
    original load lines verbatim. If neither delimiter is present we treat
    the file as a fresh helper (not a modified starter) and skip.
    """
    must_contain = (
        "generation_length = 1",
        "context_length = 128",
        # Allow the model/tokenizer paths to be substituted (the starter's
        # decapoda-research repo is dead); only the structural lines below.
        ".from_pretrained(",
        'load_dataset("wikitext"',
        "def tokenize_and_filter_function",
    )
    for step_dir in _find_per_step_dirs(output_dir):
        for p in step_dir.rglob("*.py"):
            try:
                src = p.read_text(errors="ignore")
            except OSError:
                continue
            if "#### DO NOT EDIT" not in src and "######" not in src:
                continue
            if not all(needle in src for needle in must_contain):
                return False
    return True


PREDICATES = {
    "inference_script_present": inference_script_present,
    "metric_line_emitted": metric_line_emitted,
    "metric_value_positive_finite": metric_value_positive_finite,
    "do_not_edit_block_preserved": do_not_edit_block_preserved,
}
