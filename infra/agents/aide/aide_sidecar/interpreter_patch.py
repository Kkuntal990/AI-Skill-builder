"""Snapshot AIDE's per-step working_dir + stage skill scripts into it.

AIDE's ``Interpreter.run()`` writes ``runfile.py`` into ``working_dir``,
executes it, then deletes the file before returning (see
``aide/interpreter.py``). The working_dir itself persists across steps but
its contents (model checkpoints, intermediate CSVs) get overwritten or
removed by AIDE's cleanup logic.

This patch does two things per call:

1. **Before** ``Interpreter.run``: idempotently copy the skill's
   ``scripts/`` directory (if any) into ``working_dir/scripts/`` so prompts
   like ``bash scripts/check_vram.sh`` actually find the file. OpenClaw
   skills bundle executable helpers alongside the markdown; without this
   copy, every reference to ``scripts/*`` in the spliced SKILL.md is a dead
   pointer. Idempotent via ``dst.exists()`` check — copied once per
   trajectory's working_dir, no-op thereafter.

2. **After** ``Interpreter.run``: copy whatever exists in ``working_dir``
   to ``$MLEVAL_OUTPUT_DIR/working_dirs/op_<step>/``. This snapshot is what
   ``state_predicates`` reads to assert task-specific post-conditions
   (``submission_present``, ``checkpoint_saved``, etc.).

Size caveat: PEFT runs produce multi-MB checkpoints; with 20 steps that's
hundreds of MB per trajectory. We skip files matching SKIP_GLOBS (caches,
huge model weights) to keep the PVC under control.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path

import aide.interpreter as _interp

from . import skill_inject as _skill_inject

_OUTPUT_DIR = Path(os.environ.get("MLEVAL_OUTPUT_DIR", "."))
_SNAPSHOT_ROOT = _OUTPUT_DIR / "working_dirs"

# Globs to skip when snapshotting; matched against file basename. shutil.copy2
# is kernel-streamed (not buffered in Python), so the OOM risk is PVC bloat
# not RAM. Extended in code-review C2 to cover formats AIDE writes during
# inference / dataset prep on LLM tasks.
SKIP_GLOBS = (
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.gguf",
    "*.arrow",
    "*.parquet",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.pickle",
    "*.tar.gz",
    "__pycache__",
    ".cache",
)


def _should_skip(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_GLOBS)


def _copy_filtered(src: Path, dst: Path) -> None:
    """Copy src tree to dst, skipping SKIP_GLOBS by name."""
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not _should_skip(d)]
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            if _should_skip(f):
                continue
            try:
                shutil.copy2(Path(root) / f, target_root / f)
            except Exception:  # noqa: BLE001
                # Best-effort; some files (sockets, pipes) can't be copied.
                continue


_step_counter = {"n": 0}
_original_run = _interp.Interpreter.run


def _maybe_stage_skill_scripts(working_dir: Path) -> None:
    """Copy the skill's scripts/ dir into working_dir/scripts/ if not done.

    Idempotent: returns early when the destination already exists. So once
    AIDE's working_dir is populated, subsequent steps are O(1) stat calls.
    """
    skill_dir = _skill_inject.get_skill_dir()
    if skill_dir is None:
        return
    scripts_src = skill_dir / "scripts"
    if not scripts_src.is_dir():
        return
    scripts_dst = working_dir / "scripts"
    if scripts_dst.exists():
        return
    try:
        shutil.copytree(scripts_src, scripts_dst)
    except Exception:  # noqa: BLE001
        # Best-effort; if it fails the agent sees the same dead-pointer
        # state as before the patch, which is recoverable.
        pass


def _wrapped_run(self, code, *args, **kwargs):
    wd = Path(getattr(self, "working_dir", "")) if hasattr(self, "working_dir") else None
    if wd and wd.is_dir():
        try:
            _maybe_stage_skill_scripts(wd)
        except Exception:  # noqa: BLE001
            pass

    result = _original_run(self, code, *args, **kwargs)
    step = _step_counter["n"]
    _step_counter["n"] += 1
    try:
        if wd and wd.is_dir():
            _copy_filtered(wd, _SNAPSHOT_ROOT / f"op_{step:03d}")
    except Exception:  # noqa: BLE001
        # Never crash AIDE because of a snapshot failure.
        pass
    return result


_interp.Interpreter.run = _wrapped_run
