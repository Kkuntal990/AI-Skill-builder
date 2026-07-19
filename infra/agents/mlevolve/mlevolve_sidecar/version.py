"""Single source of truth for the MLEvolve sidecar version (side-effect-free).

Bump on ANY change that alters what reaches the agent — the selector schema,
the per-node injection caps, the routing context, or the injected content. It is
the **A/B comparability boundary**: recorded in ``manifest.agent.sidecar_version``
and stamped into every ``selection_events.jsonl`` record so a cross-run
aggregation can tell which trajectories ran under which selection regime.

History:
  1.0.0  first versioned sidecar — per-node selection telemetry
         (selection_logger) + hard injection caps (MLEVAL_SKILL_MAX_PER_NODE /
         MLEVAL_SKILL_MAX_REFS_PER_NODE, default 3/3) + per-selection reason /
         decline_reason + retry-once→small-lib fallback ladder + hardware fact in
         the selector routing context. Pre-1.0.0 runs (mvp-032 and earlier) have
         NO sidecar_version field → treat as the uncapped, un-instrumented
         baseline regime.
  1.1.0  capability-linker runtime (W5) — additive: the MLEVAL_SKILL_DELIVERY_MODE
         switch (legacy | capability_task | capability_node) + capability_linker
         (hard compatibility filter → temp-0 selection → dependency closure →
         budgeted rendering) + capabilities.json loading in skill_retriever +
         capability_node telemetry. LEGACY mode is byte-identical to 1.0.0 (the
         default and A/B control), so 1.0.0 and 1.1.0 legacy runs are comparable;
         the capability modes are a new regime distinguished by delivery_mode.

This module MUST stay import-free (no ``from . import ...``, no stdlib side
effects) so entrypoint.sh can read the constant by exec'ing the file directly,
without importing the package (which would apply every monkey-patch).
"""

SIDECAR_VERSION = "1.1.0"
