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

History: this package previously shipped a second adapter for a different
agent; it was removed when the harness standardized on MLEvolve.
``state_predicates`` and ``metrics`` still assume the shared journal shape
(journal.json fields like ``parent``/``is_buggy``/``metric.value``);
those will be updated once the MLEvolve spike validates the architecture
(see ``docs/eval/stage2.md`` MLEvolve-spike section for status).
"""
