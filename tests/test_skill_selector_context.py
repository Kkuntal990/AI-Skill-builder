"""Selector-context regression test (spike-023 root cause).

C1 prepends a ~3 KB shared _harness_rules.md ahead of every task instruction.
The skill router truncated the task to the first 1500 chars to build its
context, so it saw ONLY the rules boilerplate — never the model/method/data —
and declined every skill (selections=[] x6), silently emptying the with_skill
treatment. The router must route on the task-specific text, not the rules.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SIDECAR = _REPO / "infra/agents/mlevolve/mlevolve_sidecar"
_INJ = _SIDECAR / "skill_injector.py"


def _load_injector():
    # skill_injector.py does `from . import skill_retriever`, so it must load as
    # a package member. Register a stub parent package and load the (stdlib-only)
    # skill_retriever first; the package __init__ (which imports upstream
    # MLEvolve) is deliberately NOT executed, so this stays dependency-free.
    pkg = "mlevolve_sidecar"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(_SIDECAR)]
        sys.modules[pkg] = m
    for sub in ("skill_retriever", "skill_injector"):
        full = f"{pkg}.{sub}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full, _SIDECAR / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:  # pragma: no cover
            import pytest

            pytest.skip(f"{sub} import needs upstream MLEvolve: {e}")
    return sys.modules[f"{pkg}.skill_injector"]


def _real_task_desc() -> str:
    rules = (_REPO / "infra/tasks/_harness_rules.md").read_text()
    inst = (_REPO / "infra/tasks/gsm8k/instruction.md").read_text()
    return rules + "\n" + inst  # exactly what entrypoint.sh concatenates


def test_marker_present_in_harness_rules():
    rules = (_REPO / "infra/tasks/_harness_rules.md").read_text()
    assert "<!-- END_HARNESS_RULES -->" in rules
    # Must be the final non-empty line so the strip leaves pure task text.
    assert rules.rstrip().endswith("<!-- END_HARNESS_RULES -->")


def test_routing_strips_rules_and_exposes_task_signal():
    inj = _load_injector()
    routed = inj._task_for_routing(_real_task_desc())
    # The constant rules boilerplate must be gone.
    assert "END_HARNESS_RULES" not in routed
    assert "Provided data only" not in routed  # a rules-only phrase
    # The task-specific routing signals must now be present within the cap.
    head = routed[: inj._SELECTOR_TASK_CHARS]
    for kw in ("Fine-tune", "causal LM", "Qwen", "LoRA", "batch", "left-padding"):
        assert kw in head, f"routing signal '{kw}' missing from selector context"


def test_routing_noop_without_marker():
    """Pre-C1 tasks (no rules header) are passed through unchanged."""
    inj = _load_injector()
    plain = "## Description\nFine-tune Qwen with LoRA.\n"
    assert inj._task_for_routing(plain) == plain


def test_cap_raised_above_old_1500():
    inj = _load_injector()
    assert inj._SELECTOR_TASK_CHARS >= 5000  # old value 1500 cut all signal
