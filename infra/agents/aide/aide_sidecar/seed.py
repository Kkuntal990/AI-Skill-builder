"""Pin every RNG that AIDE or agent-generated code might touch.

Why so many seeds:
    - Python ``random``: AIDE's ``_search_policy`` calls ``random.random()``
      and ``random.choice()`` (aide/agent.py:71-92).
    - AIDE's ``_prompt_environment`` calls ``random.shuffle(pkgs)``
      (aide/agent.py:109) — list order in the system prompt affects LLM
      output even at temperature 0.
    - ``numpy.random``: legacy RandomState path.
    - ``numpy.random.default_rng()``: new-style Generator. If unseeded, it
      uses ``os.urandom`` and is nondeterministic. We can't seed it
      *globally* but we can ensure it's deterministic by monkey-patching
      ``np.random.default_rng`` to use a seeded factory when no seed is given.
    - ``torch.manual_seed`` + ``torch.cuda.manual_seed_all``.

Reproducibility caveat:
    Network-latency-driven retry counts in AIDE's backoff layer can change
    the *number* of random calls between steps. With the same seed, the
    sequence of resulting random values is still deterministic, but the
    *which* sample each random call returns depends on how many prior calls
    were made. Paired with/without runs may diverge if their retry patterns
    differ. We can't fix this from the outside; it's a known limitation.
"""

from __future__ import annotations

import os
import random

_seed = int(os.environ.get("SEED", "0"))
random.seed(_seed)

try:
    import numpy as np

    np.random.seed(_seed)
    # Force new-style default_rng to be seeded when called without args, so
    # downstream code like `np.random.default_rng()` is deterministic.
    _orig_default_rng = np.random.default_rng

    def _seeded_default_rng(seed=None):
        if seed is None:
            seed = _seed
        return _orig_default_rng(seed)

    np.random.default_rng = _seeded_default_rng
except ImportError:
    pass

try:
    import torch

    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    # Forbid nondeterministic ops where possible (PyTorch may still fall back).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass
except ImportError:
    pass

os.environ.setdefault("PYTHONHASHSEED", str(_seed))
