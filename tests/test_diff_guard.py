"""Regression tests for the SEARCH/REPLACE diff guard (mlevolve_sidecar.diff_guard).

Pins the fix for the `=======` patch-corruption defect proven against the real
spike-018 gsm8k/samsum journals: the coder model emits malformed SEARCH/REPLACE
blocks (a spurious trailing `=======` divider, and/or reasoning prose leaking
into the replace text) that the stock patcher bakes into the runfile, producing
invalid Python and a debug death-spiral.

The guard (a) NORMALIZES spurious extra dividers so the intended edit still
applies, and (b) guarantees the bulletproof invariant: a CHANGED result is
always valid Python, else it reverts to last-good (count=0).

The MLEvolve patcher lives in a git submodule, so both modules are loaded by
path to keep the test self-contained.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PATCHER = _REPO / "infra/agents/mlevolve/upstream/agents/coder/diff_coder/patcher.py"
_GUARD = _REPO / "infra/agents/mlevolve/mlevolve_sidecar/diff_guard.py"

pytestmark = pytest.mark.skipif(
    not _PATCHER.exists(),
    reason="MLEvolve submodule not checked out (patcher.py absent)",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def guarded():
    patcher = _load("mleval_test_patcher", _PATCHER)
    guard = _load("mleval_test_diff_guard", _GUARD)
    cls = patcher.SearchReplacePatcher
    cls.apply_patch = guard._wrap_apply_patch(cls.apply_patch)
    return cls, guard


_ORIG = "x = 1\ny = 2\nz = 3\n"


def _valid(text):
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def test_trailing_divider_is_normalized_and_applies(guarded):
    cls, _ = guarded
    patch = (
        "<<<<<<< SEARCH\n"
        "y = 2\n"
        "=======\n"
        "y = 22\n"
        "=======\n"  # spurious trailing divider (the proven bug)
        ">>>>>>> REPLACE\n"
    )
    res, count = cls().apply_patch(patch, _ORIG, strict=False)
    assert count == 1
    assert "y = 22" in res and "=======" not in res
    assert _valid(res)


def test_many_extra_dividers_recovered(guarded):
    cls, _ = guarded
    patch = (
        "<<<<<<< SEARCH\n"
        "z = 3\n"
        "=======\n"
        "z = 33\n"
        "=======\n=======\n=======\n"  # several spurious dividers
        ">>>>>>> REPLACE\n"
    )
    res, count = cls().apply_patch(patch, _ORIG, strict=False)
    assert count == 1 and "z = 33" in res and "=======" not in res and _valid(res)


def test_clean_patch_unaffected(guarded):
    cls, _ = guarded
    patch = "<<<<<<< SEARCH\nx = 1\n=======\nx = 99\n>>>>>>> REPLACE\n"
    res, count = cls().apply_patch(patch, _ORIG, strict=False)
    assert count == 1 and "x = 99" in res and _valid(res)


def test_prose_leak_is_reverted(guarded):
    """Empty-SEARCH insertion of reasoning prose -> invalid -> revert to last-good."""
    cls, _ = guarded
    patch = (
        "<<<<<<< SEARCH\n"
        "=======\n"
        "Wait, looking at the error more carefully:\n"  # prose, not code
        ">>>>>>> REPLACE\n"
    )
    res, count = cls().apply_patch(patch, _ORIG, strict=False)
    assert count == 0 and res == _ORIG  # reverted, last-good preserved


def test_invalid_edit_reverts_not_executes(guarded):
    cls, _ = guarded
    patch = "<<<<<<< SEARCH\ny = 2\n=======\ny = (1 +\n>>>>>>> REPLACE\n"  # unbalanced paren
    res, count = cls().apply_patch(patch, _ORIG, strict=False)
    assert count == 0 and res == _ORIG


def test_normalize_idempotent_on_clean(guarded):
    _, guard = guarded
    clean = "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n"
    assert guard._normalize_blocks(clean) == clean.rstrip("\n")


def test_marker_and_validity_helpers(guarded):
    _, guard = guarded
    assert guard._has_markers("ok = 1\n>>>>>>> REPLACE\n")
    assert not guard._has_markers("ok = 1\nbanner = '====== not seven'\n")
    assert guard._is_valid_python("a = 1\n")
    assert not guard._is_valid_python("=======\n")
