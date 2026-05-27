"""Launch MLEvolve from our harness.

Why this script exists (instead of just shelling out to MLEvolve's run.py):
- Import the sidecar BEFORE MLEvolve loads, so the monkey-patches that
  log each LLM call into prompts.jsonl take effect on the first call.
- Provide a stable entry symbol for the entrypoint (the upstream's run.py
  has a module-level run() that does sys.exit if config is wrong; we wrap
  it so failures bubble up as Python exceptions and land in stdout).

Config is loaded from MLEvolve's default location (config/config.yaml),
which the entrypoint overwrites with our rendered version before invoking
this script. We don't pass argv overrides — the YAML already has everything.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Path setup must happen BEFORE the sidecar import. The sidecar's
# prompt_logger does `import llm.openai` at module load, which only
# resolves if /workspace/mlevolve is on sys.path (llm is a top-level
# package inside MLEvolve's repo). Set up BOTH:
#   /workspace          → so `import mlevolve_sidecar` resolves
#   /workspace/mlevolve → so `import llm.openai` (which the sidecar does
#                          at its own import time) resolves
# Also chdir here so MLEvolve's later `from config import Config`
# (relative to its repo root) works.
#
# Mistake-not-to-repeat: the build-time smoke in the Dockerfile uses
# `python -c` from inside /workspace/mlevolve, which makes llm
# cwd-resolvable. The actual entrypoint uses
# `python /workspace/run_mlevolve.py`, which sets sys.path[0] to
# /workspace (the script's directory), NOT cwd. Two different import
# environments; the smoke was a false negative until this ordering fix.
MLEVOLVE_ROOT = Path("/workspace/mlevolve")
if not MLEVOLVE_ROOT.is_dir():
    print(f"[run_mlevolve] FATAL: MLEvolve not found at {MLEVOLVE_ROOT}", file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(MLEVOLVE_ROOT))
sys.path.insert(0, "/workspace")
os.chdir(MLEVOLVE_ROOT)

# Sidecar must import BEFORE MLEvolve so monkey-patches take effect.
# Order encoded in mlevolve_sidecar/__init__.py docstring.
import mlevolve_sidecar  # noqa: F401, E402


def main() -> int:
    # MLEvolve's run.py is `def run(): ...` at module level. Import-and-call.
    try:
        from run import run as _run
    except ImportError as e:
        print(f"[run_mlevolve] FATAL: cannot import MLEvolve run.py: {e}", file=sys.stderr)
        return 3

    try:
        _run()
    except SystemExit as e:
        # MLEvolve calls sys.exit on some completion paths; propagate code
        return int(e.code) if e.code is not None else 0
    except Exception:
        print("[run_mlevolve] FATAL: unhandled exception in run():", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
