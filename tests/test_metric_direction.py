"""Tests for the metric-direction pin (spike-026 inverted-search root cause).

MLEvolve's LLM determine_metric_direction flipped maximize->minimize on the same
clean gsm8k task (1/4 trajectories), inverting the MCGS search. The pin sets the
direction deterministically from MLEVAL_METRIC_MAXIMIZE and makes the per-node
validator trust it instead of marking nodes buggy on a per-node LLM flip.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD = (
    Path(__file__).resolve().parents[1]
    / "infra/agents/mlevolve/mlevolve_sidecar/metric_direction.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_metric_direction_probe", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # pure stdlib; _install() is a harmless no-op here
    return mod


class _Agent:
    metric_maximize = None
    metric_maximize_reasoning = None


class _Node:
    id = "n1"
    metric = None


class _MetricValue:
    def __init__(self, value, maximize):
        self.value = value
        self.maximize = maximize


class _Module:
    MetricValue = _MetricValue


def test_pinned_maximize_parsing(monkeypatch):
    inj = _load()
    monkeypatch.delenv("MLEVAL_METRIC_MAXIMIZE", raising=False)
    assert inj._pinned_maximize() is None  # unset -> not pinned
    for v in ("1", "true", "yes", "max"):
        monkeypatch.setenv("MLEVAL_METRIC_MAXIMIZE", v)
        assert inj._pinned_maximize() is True
    for v in ("0", "false", "no", "minimize"):
        monkeypatch.setenv("MLEVAL_METRIC_MAXIMIZE", v)
        assert inj._pinned_maximize() is False


def test_determine_pins_and_skips_llm(monkeypatch):
    inj = _load()
    called = {"orig": 0}

    def orig(agent):
        called["orig"] += 1

    patched = inj._make_determine(orig)
    a = _Agent()
    # pinned maximize -> sets True, LLM (orig) NOT called
    monkeypatch.setenv("MLEVAL_METRIC_MAXIMIZE", "1")
    patched(a)
    assert a.metric_maximize is True and called["orig"] == 0
    # pinned minimize
    monkeypatch.setenv("MLEVAL_METRIC_MAXIMIZE", "0")
    patched(a)
    assert a.metric_maximize is False and called["orig"] == 0
    # unset -> defers to original LLM behavior
    monkeypatch.delenv("MLEVAL_METRIC_MAXIMIZE", raising=False)
    patched(a)
    assert called["orig"] == 1


def test_validate_trusts_pin_no_buggy(monkeypatch):
    inj = _load()
    called = {"orig": 0}

    def orig(agent, node, response):
        called["orig"] += 1
        node.is_buggy = True  # the behavior we want to AVOID under the pin

    patched = inj._make_validate(orig, _Module())
    a = _Agent(); a.metric_maximize = True
    n = _Node()
    monkeypatch.setenv("MLEVAL_METRIC_MAXIMIZE", "1")
    # per-node LLM says minimize (lower_is_better=True) — must NOT mark buggy
    patched(a, n, {"metric": 0.61, "lower_is_better": True})
    assert called["orig"] == 0
    assert getattr(n, "is_buggy", None) is not True
    assert n.metric.value == 0.61 and n.metric.maximize is True
    # unset -> defers to original validator
    monkeypatch.delenv("MLEVAL_METRIC_MAXIMIZE", raising=False)
    patched(a, _Node(), {"metric": 0.5, "lower_is_better": True})
    assert called["orig"] == 1


def test_idempotent_wrap():
    inj = _load()
    def orig(agent): ...
    p1 = inj._make_determine(orig)
    p2 = inj._make_determine(p1)  # already wrapped -> returned as-is
    assert p2 is p1
