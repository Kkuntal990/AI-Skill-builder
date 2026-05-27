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
import logging
import os
import signal
import shutil
from pathlib import Path

import aide.interpreter as _interp

from . import skill_inject as _skill_inject

_logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# cleanup_session + _run_session patches
# ---------------------------------------------------------------------------
# Upstream `Interpreter.cleanup_session` (verified at SHA 40dcf28) ends with
# an unguarded `self.process.close()` in its finally clause. Python's
# multiprocessing.Process.close() raises `ValueError: Cannot close a process
# while it is still running` whenever is_alive() is True — and the doc
# behavior is intentional (CPython issue #94661 is open-with-no-fix).
#
# That ValueError bubbles uncaught through `agent.step → run.run`, killing
# the entire trajectory after a single LLM-generated draft. We hit this on
# mvp-015-job: a draft used DataLoader(num_workers=2) AFTER model.to(cuda).
# The forked worker tried to re-init CUDA, hung, and ignored SIGKILL because
# the CUDA driver pins forked-child contexts. AIDE's escalation only signals
# the immediate child pid (os.kill, not killpg), so the wedged DataLoader
# grandchildren survive — keeping the parent alive — and close() raises.
#
# Two-part fix, both monkey-patched here so we don't fork AIDE:
#   1. cleanup_session: swallow ValueError on close() and drop the process
#      reference so the agent loop can spawn a fresh interpreter for the
#      next draft. Adds an os.killpg escalation between AIDE's
#      process.kill() and the close() attempt.
#   2. _run_session: call os.setsid() at child entry so the child becomes
#      its own session leader. Without this, the child inherits AIDE's
#      PGID and killpg(getpgid(child_pid)) would kill AIDE itself. The
#      guard `pgid != os.getpgid(0)` below double-checks before signalling.
#
# References: Bihui-Jin/aideml is the most-defensive AIDE fork on github
# (full try/except cleanup, is_alive re-check). AutoMind adds re-terminate
# but still doesn't wrap close(). Inspect AI / OpenDevin / AIRA-dojo all
# isolate at the container/kernel level instead — a larger refactor we
# defer until this surgical fix proves insufficient.


_original_cleanup_session = _interp.Interpreter.cleanup_session
_original_run_session = _interp.Interpreter._run_session


def _patched_run_session(self, *args, **kwargs):
    try:
        os.setsid()
    except OSError:
        # already a session leader or no permission — best-effort
        pass
    return _original_run_session(self, *args, **kwargs)


def _safe_cleanup_session(self) -> None:
    proc = getattr(self, "process", None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.exitcode is None:
            proc.kill()
            proc.join(timeout=0.5)
        if proc.exitcode is None and proc.pid:
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, PermissionError, OSError):
                pgid = None
            # Don't signal our own group — _patched_run_session above should
            # have given the child its own session, but stay defensive.
            if pgid and pgid != os.getpgid(0):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            proc.join(timeout=1.0)
    except Exception as e:  # noqa: BLE001
        _logger.error("cleanup_session: escalation error: %s", e)
    finally:
        try:
            proc.close()
        except ValueError:
            # Child still alive after killpg — most likely CUDA-pinned. Drop
            # the reference and continue; the next draft gets a fresh
            # interpreter. The leaked subprocess holds some GPU memory until
            # the pod ends, which is the cost of trajectory survival.
            _logger.warning(
                "cleanup_session: process still alive after killpg; "
                "leaking subprocess pid=%s to keep trajectory alive",
                getattr(proc, "pid", "?"),
            )
        except Exception as e:  # noqa: BLE001
            _logger.error("cleanup_session: unexpected close() error: %s", e)
        self.process = None


_interp.Interpreter._run_session = _patched_run_session
_interp.Interpreter.cleanup_session = _safe_cleanup_session
