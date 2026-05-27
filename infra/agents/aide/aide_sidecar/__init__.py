"""Sidecar patches that monkey-patch AIDE in-place.

Imported by run_aide.py BEFORE aide.agent / aide.run. Each submodule applies
its patch at import time. Order matters — seed and openai_timeout first
(must take effect before any LLM client or RNG-using code runs), then the
backend wrapper, skill injection, and interpreter capture.

    1. seed              random / numpy / torch seed pin from $SEED
    2. openai_timeout    inject finite HTTP timeout into openai client
    3. backend_wrapper   per-LLM-call (prompt, response, tokens) -> JSONL
    4. task_hints        prepend hardware + fork-after-CUDA pitfalls
    5. skill_inject      splice $MLEVAL_SKILL_PATH into AIDE's task_desc
    6. interpreter_patch working_dir preservation + cleanup_session fix

Both task_hints and skill_inject wrap aide.utils.config.load_task_desc.
task_hints is imported first so its wrapper is the inner call; the final
prompt reads: hardware/pitfalls (hints) → original task → skill (if any).
"""

from . import seed             # noqa: F401
from . import openai_timeout   # noqa: F401
from . import backend_wrapper  # noqa: F401
from . import task_hints       # noqa: F401
from . import skill_inject     # noqa: F401
from . import interpreter_patch  # noqa: F401
