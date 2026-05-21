"""Post-trajectory analyzer.

Pure Python modules that run *after* an AIDE trajectory completes. Each
module is single-purpose:

    adapter_aide        — journal.json + prompts.jsonl -> trajectory.jsonl
                          (one record per node, AIRA-style schema)
    stage_classifier    — AST + import patterns -> 6x16 stage label
    state_predicates    — task-agnostic + per-task assertions over outputs
    aggregate           — cross-trajectory L1 outcome + L3 cost rollup

Designed to run *inside the container* after AIDE finishes (the entrypoint
invokes them), so the trajectory output is fully self-describing on the PVC.
Local aggregation reads those self-describing outputs.
"""
