"""Sidecar patches that monkey-patch AIDE in-place.

Imported by run_aide.py BEFORE aide.agent / aide.run. Each submodule applies
its patch at import time. Order:

    1. backend_wrapper   per-LLM-call (prompt, response, tokens) -> JSONL
    2. seed              random / numpy / torch seed pin from $SEED
    3. interpreter_patch (placeholder) working_dir preservation [task #63]
"""

from . import backend_wrapper  # noqa: F401
from . import seed             # noqa: F401
from . import interpreter_patch  # noqa: F401
