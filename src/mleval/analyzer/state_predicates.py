"""Run agent-agnostic + per-task state predicates over a trajectory's outputs.

Predicates are deterministic boolean functions over the artifacts AIDE
produces (workspace dir, best_solution.py, journal metrics). Two sources:

    Generic predicates (this file): apply to any AIDE run. Cheap checks like
    "did the agent ever produce a non-buggy node" or "is the best metric
    finite".

    Per-task predicates: live in ``infra/tasks/<task>/predicates.py`` and
    expose a ``PREDICATES: dict[str, Callable[[Path], bool]]`` mapping. The
    file is loaded by-path from $MLEVAL_TASK_INSTRUCTION_PATH's parent dir
    so tasks don't need to be importable Python packages.

Output is one ``state.json`` per trajectory in $MLEVAL_OUTPUT_DIR.

CLI:
    python -m mleval.analyzer.state_predicates $MLEVAL_OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

# ---- generic predicates ---------------------------------------------------


def _load_journal(output_dir: Path) -> dict | None:
    matches = list(output_dir.rglob("journal.json"))
    if not matches:
        return None
    return json.loads(matches[0].read_text())


def has_best_solution(output_dir: Path) -> bool:
    # AIDE preserves the best node as best_solution.py; MLEvolve preserves it as
    # best_submission/submission.csv (no_submission_mode:False). Accept either.
    return any(output_dir.rglob("best_solution.py")) or any(
        output_dir.rglob("best_submission/submission.csv")
    )


def at_least_one_non_buggy_node(output_dir: Path) -> bool:
    j = _load_journal(output_dir)
    if not j:
        return False
    return any(not n.get("is_buggy", True) for n in j.get("nodes", []))


def best_metric_finite(output_dir: Path) -> bool:
    j = _load_journal(output_dir)
    if not j:
        return False
    for n in j.get("nodes", []):
        m = n.get("metric")
        if m and isinstance(m.get("value"), (int, float)) and math.isfinite(m["value"]):
            return True
    return False


def prompts_log_present(output_dir: Path) -> bool:
    p = output_dir / "prompts.jsonl"
    return p.is_file() and p.stat().st_size > 0


GENERIC_PREDICATES: dict[str, Callable[[Path], bool]] = {
    "has_best_solution": has_best_solution,
    "at_least_one_non_buggy_node": at_least_one_non_buggy_node,
    "best_metric_finite": best_metric_finite,
    "prompts_log_present": prompts_log_present,
}


# ---- per-task predicate loading -------------------------------------------


def _load_task_predicates(task_predicates_path: Path) -> dict[str, Callable[[Path], bool]]:
    """Import a task's ``predicates.py`` by file path and return its ``PREDICATES`` dict."""
    if not task_predicates_path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location(
        f"_task_predicates_{task_predicates_path.parent.name}",
        task_predicates_path,
    )
    if not spec or not spec.loader:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "PREDICATES", {})


def _resolve_task_predicates() -> dict[str, Callable[[Path], bool]]:
    """Find per-task predicates next to the instruction file (if any)."""
    instr = os.environ.get("MLEVAL_TASK_INSTRUCTION_PATH", "")
    if not instr:
        return {}
    return _load_task_predicates(Path(instr).parent / "predicates.py")


# ---- runner ----------------------------------------------------------------


def evaluate(output_dir: Path) -> dict[str, bool]:
    """Run all predicates and return the result mapping."""
    results: dict[str, bool] = {}
    for name, fn in GENERIC_PREDICATES.items():
        try:
            results[name] = bool(fn(output_dir))
        except Exception as exc:  # noqa: BLE001
            results[name] = False
            results[f"{name}__error"] = str(exc)  # type: ignore[assignment]
    for name, fn in _resolve_task_predicates().items():
        try:
            results[f"task__{name}"] = bool(fn(output_dir))
        except Exception as exc:  # noqa: BLE001
            results[f"task__{name}"] = False
            results[f"task__{name}__error"] = str(exc)  # type: ignore[assignment]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate state predicates over a trajectory")
    parser.add_argument("output_dir", type=Path, help="$MLEVAL_OUTPUT_DIR")
    args = parser.parse_args(argv)
    results = evaluate(args.output_dir)
    out_path = args.output_dir / "state.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[state_predicates] wrote {out_path} ({sum(1 for v in results.values() if v is True)}/{len(results)} passing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
