"""Post-trajectory analyzer.

Pure Python modules that run *after* an agent trajectory completes. Each
module is single-purpose:

    adapter_mlevolve    — MLEvolve runs/<ts>_<exp>/journal.json + our
                          sidecar prompts.jsonl -> trajectory.jsonl
                          (one record per node, universal schema)
    stage_classifier    — AST + import patterns -> 6x16 stage label
    state_predicates    — task-agnostic + per-task assertions over outputs
    aggregate           — cross-trajectory L1 outcome + L3 cost rollup

Designed to run *inside the container* after the agent finishes (the
entrypoint invokes them), so the trajectory output is fully
self-describing on the PVC. Local aggregation reads those outputs.

History: an earlier ``adapter_aide`` lived here when the harness was
AIDE-shaped. Removed when we pivoted to MLEvolve on the mlevolve-smoke
branch. ``state_predicates`` and ``metrics`` still reference AIDE-shaped
artifacts (journal.json fields like ``parent``/``is_buggy``/``metric.value``);
those will be updated once the MLEvolve spike validates the architecture
(see ``docs/eval/stage2.md`` MLEvolve-spike section for status).
"""
