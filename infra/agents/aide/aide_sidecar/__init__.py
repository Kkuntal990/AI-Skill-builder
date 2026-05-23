"""Sidecar patches that monkey-patch AIDE in-place.

Imported by run_aide.py BEFORE aide.agent / aide.run. Each submodule applies
its patch at import time. Order matters — seed and openai_timeout first
(must take effect before any LLM client or RNG-using code runs), then the
backend wrapper, skill injection, and interpreter capture.

    1. seed              random / numpy / torch seed pin from $SEED
    2. openai_timeout    inject finite HTTP timeout into openai client
    3. backend_wrapper   per-LLM-call (prompt, response, tokens) -> JSONL
    4. skill_inject      splice $MLEVAL_SKILL_PATH into AIDE's task_desc
    5. interpreter_patch working_dir preservation
"""

from . import seed             # noqa: F401
from . import openai_timeout   # noqa: F401
from . import backend_wrapper  # noqa: F401
from . import skill_inject     # noqa: F401
from . import interpreter_patch  # noqa: F401
