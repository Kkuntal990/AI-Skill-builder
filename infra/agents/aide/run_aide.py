"""Shim that loads our sidecar patches before invoking AIDE.

`aide.agent` and `aide.journal2report` do `from .backend import query` at
module-load time, which captures the original `query` reference. To make
our wrapped version visible everywhere, we must patch `aide.backend.query`
BEFORE either of those modules is imported.

Order matters:
    1. `import aide_sidecar` loads `aide.backend`, then replaces its `query`
       attribute with our wrapper.
    2. `from aide.run import run` then triggers `from .agent import Agent`,
       which triggers `from .backend import query` — picking up our wrapper.

Usage (inside the container):

    python /workspace/run_aide.py \\
        data_dir=/path/to/data \\
        goal="..." eval="..." \\
        agent.code.model="..." agent.code.base_url="..." agent.code.api_key="..." \\
        agent.feedback.model="..." agent.feedback.base_url="..." agent.feedback.api_key="..." \\
        agent.steps=20 \\
        log_dir=/path/to/logs workspace_dir=/path/to/ws exp_name=trajectory_id

AIDE uses OmegaConf.from_cli() so CLI args are `key=value` overrides on
config/config.yaml.
"""

import aide_sidecar  # noqa: F401 — applies monkey-patches at import time
from aide.run import run

if __name__ == "__main__":
    run()
