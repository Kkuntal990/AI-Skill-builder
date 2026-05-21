"""mleval — Stage-2 skill evaluation harness for MLE agents.

This package is the local-side companion to the agent container image
(``infra/agents/<name>/Dockerfile``). It provides orchestration utilities,
trajectory parsing, and metric computation. The heavy ML dependencies
(PyTorch, transformers, the agent under test) live inside the container,
not here.
"""

__version__ = "0.1.0.dev0"
