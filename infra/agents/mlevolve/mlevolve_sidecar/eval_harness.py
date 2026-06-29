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
import os
import subprocess
from collections import Counter

logger = logging.getLogger(__name__)

# Hardware string shown to the agent, mirroring MLE-bench's additional_notes
# `Compute` line (openai/mle-bench: agents/aide/start.sh derives ${HARDWARE} from
# `nvidia-smi --query-gpu=name` and additional_notes.txt injects
# "You have access to ${HARDWARE} ..."). We extend MLE-bench by also reporting
# VRAM (its query omits it) because that is the fact a memory-gated decision
# (e.g. "quantize only if the model doesn't fit") actually needs. Cell-agnostic
# (both A/B arms get the identical line) and purely factual — it states the
# environment, never which method to use.
_HARDWARE_CACHE: str | None = None


def _detect_hardware() -> str:
    """Return e.g. '1 NVIDIA RTX A6000 GPU (48 GB VRAM)' or 'a CPU'.

    Prefers MLEVAL_HARDWARE (set once by entrypoint.sh, the faithful MLE-bench
    start.sh mirror); falls back to querying nvidia-smi directly so the line is
    still correct if the env var is unset. Queried once and cached.
    """
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None:
        return _HARDWARE_CACHE
    env = os.environ.get("MLEVAL_HARDWARE", "").strip()
    if env:
        _HARDWARE_CACHE = env
        return env
    gpu_part = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if lines:  # group identical GPUs into a count prefix, like MLE-bench's uniq -c
            parts = []
            for spec, n in Counter(lines).items():
                name, _, mem = spec.partition(",")
                name = name.strip()
                try:
                    gb = int(round(int(mem.strip()) / 1024))
                    parts.append(f"{n} {name} GPU ({gb} GB VRAM)")
                except ValueError:
                    parts.append(f"{n} {name} GPU")
            gpu_part = ", ".join(parts)
    except Exception as e:  # noqa: BLE001 — never break codegen
        logger.warning("[eval_harness] GPU detection failed: %s", e)
    # CPU/RAM from the cgroup (pod's allotted limits), not os.cpu_count()/MemTotal
    # which report the NODE's totals on k8s.
    cpus = _cgroup_cpus()
    ram_gb = _cgroup_ram_gb()
    rest = f"{cpus} CPUs, {ram_gb} GB RAM"
    hw = f"{gpu_part}, {rest}" if gpu_part else f"{rest} (no GPU)"
    _HARDWARE_CACHE = hw
    return hw


def _cgroup_cpus() -> str:
    """CPU cores from cgroup v2 cpu.max ('quota period'); 'max' → os.cpu_count()."""
    try:
        quota, _, period = open("/sys/fs/cgroup/cpu.max").read().strip().partition(" ")
        if quota != "max" and period and int(period) > 0:
            return str(max(1, round(int(quota) / int(period))))
    except Exception:  # noqa: BLE001
        pass
    return str(os.cpu_count() or "?")


def _cgroup_ram_gb() -> str:
    """RAM (GB) from cgroup v2 memory.max (bytes); 'max'/unreadable → MemTotal."""
    try:
        raw = open("/sys/fs/cgroup/memory.max").read().strip()
        if raw != "max":
            return str(round(int(raw) / (1024 ** 3)))
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return str(round(int(line.split()[1]) / (1024 ** 2)))
    except Exception:  # noqa: BLE001
        pass
    return "?"

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
        # Compute/hardware line — mirror MLE-bench, co-located with the upstream
        # "Resource Budget" (time/steps/exec) line; append if that line is absent.
        if not any(isinstance(l, str) and "**Compute**" in l for l in gl):
            compute_line = (
                f"**Compute**: You have access to {_detect_hardware()} with the "
                "appropriate drivers installed."
            )
            idx = next((i for i, l in enumerate(gl)
                        if isinstance(l, str) and "Resource Budget" in l), None)
            if idx is not None:
                gl.insert(idx + 1, compute_line)
            else:
                gl.append(compute_line)
        if not any(isinstance(l, str) and "Held-out evaluation rules" in l for l in gl):
            gl.extend(EVAL_HARNESS_RULES)
    except Exception as e:  # noqa: BLE001 — never break codegen
        logger.warning("[eval_harness] impl_guideline injection failed: %s", e)
