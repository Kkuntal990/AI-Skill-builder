"""Snapshot AIDE's per-step working_dir so state predicates can inspect it.

AIDE's ``Interpreter.run()`` writes ``runfile.py`` into ``working_dir``,
executes it, then deletes the file before returning (see
``aide/interpreter.py``). The working_dir itself persists across steps but
its contents (model checkpoints, intermediate CSVs) get overwritten or
removed by AIDE's cleanup logic.

We wrap ``Interpreter.run`` so that AFTER each execution we copy whatever
exists in ``working_dir`` to ``$MLEVAL_OUTPUT_DIR/working_dirs/op_<step>/``.
This snapshot is what ``state_predicates`` reads to assert task-specific
post-conditions (``submission_present``, ``checkpoint_saved``, etc.).

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

_OUTPUT_DIR = Path(os.environ.get("MLEVAL_OUTPUT_DIR", "."))
_SNAPSHOT_ROOT = _OUTPUT_DIR / "working_dirs"

# Globs to skip when snapshotting; matched against file basename.
SKIP_GLOBS = (
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.ckpt",
    "*.h5",
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


def _wrapped_run(self, code, *args, **kwargs):
    result = _original_run(self, code, *args, **kwargs)
    step = _step_counter["n"]
    _step_counter["n"] += 1
    try:
        wd = Path(getattr(self, "working_dir", "")) if hasattr(self, "working_dir") else None
        if wd and wd.is_dir():
            _copy_filtered(wd, _SNAPSHOT_ROOT / f"op_{step:03d}")
    except Exception:  # noqa: BLE001
        # Never crash AIDE because of a snapshot failure.
        pass
    return result


_interp.Interpreter.run = _wrapped_run
