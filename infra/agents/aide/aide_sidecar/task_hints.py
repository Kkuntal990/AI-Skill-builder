"""Prepend hardware + pitfall hints to AIDE's task description.

Why: mainline AIDE's prompts never tell the LLM what hardware exists or
what fork-after-CUDA pitfalls to avoid (verified by grep on aide/agent.py).
Without that context, DeepSeek-v4-flash deterministically generates
``DataLoader(num_workers=N>0)`` and ``dataset.map(num_proc=N>0)`` after
``model.to('cuda')`` on the llama-inference task, which crashes its own
subprocess and leaks memory (mvp-016/017 OOMs).

MLE-Bench's reference setup compensates the same way: ``start.sh`` probes
``nvidia-smi`` and prepends ``**Compute**: You have access to ${HARDWARE}
with the appropriate drivers installed.`` to the task description before
passing it as ``desc_file=`` (see openai/mle-bench agents/aide/start.sh).
We do the same plus an explicit pitfall list distilled from observed
crashes.

Placement: wraps ``aide.utils.config.load_task_desc`` like skill_inject,
so the hints are part of the task_desc that every code-gen and judge call
sees. Order in :mod:`__init__` puts this BEFORE skill_inject so the final
prompt reads: hardware hints → task → skill (if any).

Knob: ``MLEVAL_TASK_HINTS_DISABLE=1`` no-ops the patch (for measuring the
A/B impact of the hints themselves on agent behaviour).
"""

from __future__ import annotations

import os

import aide.utils.config as _config

_DISABLED = os.environ.get("MLEVAL_TASK_HINTS_DISABLE", "0") == "1"

# Kept literal and stable (no env interpolation) so the prompt is
# reproducible across runs without depending on runtime probe values.
# If hardware ever differs, override via $MLEVAL_TASK_HINTS_OVERRIDE.
_DEFAULT_HINTS = """\
**Compute environment**
- 1× NVIDIA RTX A6000 GPU (48 GiB VRAM), CUDA drivers installed
- 2 CPU cores, ~96 GiB RAM available to the process
- Disk: PVC-backed scratch at /results (slow); /cache and /dev/shm (fast)

**Pitfalls observed in prior trajectories — avoid**
- Do NOT use ``DataLoader(num_workers=N)`` with N>0 after a model has
  been moved to CUDA. The default fork start-method causes worker
  subprocesses to re-init CUDA and crash with
  ``RuntimeError: Cannot re-initialize CUDA in forked subprocess``.
  The wedged workers ignore SIGKILL (CUDA driver pins them) and leak
  ~15 GiB each. Use ``num_workers=0``, or call
  ``torch.multiprocessing.set_start_method("spawn", force=True)`` at
  the very top of your script before any CUDA op.
- Do NOT use ``dataset.map(num_proc=N)`` with N>0 after ``model.to('cuda')``
  or ``Accelerator()``. Same root cause.
- ``Accelerator(mixed_precision="bf16")`` calls
  ``torch.cuda.is_bf16_supported()`` which initializes CUDA early;
  combine with later forks and you get the same crash. Either keep all
  multiprocessing strictly before any CUDA op, or use ``num_workers=0``.
"""

_HINTS = os.environ.get("MLEVAL_TASK_HINTS_OVERRIDE", _DEFAULT_HINTS)
_original_load_task_desc = _config.load_task_desc


def _load_task_desc_with_hints(cfg):
    desc = _original_load_task_desc(cfg)
    if _DISABLED or not _HINTS.strip():
        return desc
    if isinstance(desc, dict):
        new = dict(desc)
        # Stable key name so the analyzer can detect/strip if needed.
        new["Compute environment and pitfalls"] = _HINTS
        return new
    # String desc: prepend with a horizontal rule so the original is
    # visually distinct.
    return f"{_HINTS}\n\n---\n\n{desc}"


_config.load_task_desc = _load_task_desc_with_hints
