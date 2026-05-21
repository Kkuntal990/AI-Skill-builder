"""Pin Python / NumPy / PyTorch RNG seeds from $SEED.

AIDE's _search_policy uses random.choice / random.random but never calls
random.seed(). Combined with the default LLM temperature of 0.5, two runs
with the same SEED env var diverge. We seed the three relevant RNGs here;
LLM-side determinism is enforced by passing `agent.code.temp=0` and
`agent.feedback.temp=0` from the entrypoint.
"""

import os
import random

_seed = int(os.environ.get("SEED", "0"))
random.seed(_seed)

try:
    import numpy as np

    np.random.seed(_seed)
except ImportError:
    pass

try:
    import torch

    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
except ImportError:
    pass
