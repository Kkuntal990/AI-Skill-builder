"""Sidecar patches that monkey-patch AIDE in-place.

Imported by run_aide.py BEFORE aide.agent / aide.run. Each submodule applies
its patch at import time. Order matters — seed first (RNG must be set before
anything random runs), backend wrapper second (must run before any LLM call),
skill_inject third (before Agent is constructed).

    1. seed              random / numpy / torch seed pin from $SEED
    2. backend_wrapper   per-LLM-call (prompt, response, tokens) -> JSONL
    3. skill_inject      splice $MLEVAL_SKILL_PATH into AIDE's task_desc
    4. interpreter_patch (placeholder) working_dir preservation [task #63]
"""

from . import seed             # noqa: F401
from . import backend_wrapper  # noqa: F401
from . import skill_inject     # noqa: F401
from . import interpreter_patch  # noqa: F401
