"""A/B sweep orchestrator.

Spawns one k8s Job per (task × cell × seed), waits for completion, pulls
results off the PVC, runs cross-trajectory aggregation. See ``run_ab.py``.
"""
