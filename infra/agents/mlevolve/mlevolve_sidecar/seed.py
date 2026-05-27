"""Pin random / numpy / torch RNGs from $SEED at sidecar import time.

MLEvolve has its own seed in cfg.agent.seed but that fires later in the
loop. We pin here so any code imported between sidecar load and MLEvolve's
own seed call (e.g. tokenizer init, model.from_pretrained) is reproducible
across seed cells.
"""
from __future__ import annotations

import os
import random


def _pin() -> None:
    seed_str = os.environ.get("SEED", "0")
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 0
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        try:
            np.random.default_rng(seed)  # noqa: F841 — factory-side seeding
        except Exception:
            pass
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_pin()
