"""Diff-apply guard — hardens MLEvolve's SEARCH/REPLACE patcher against
malformed model output (the proven root cause of `=======` code corruption).

Root cause (reproduced deterministically against spike-018 gsm8k/samsum
journals): the coder model intermittently emits a SEARCH/REPLACE block with a
REDUNDANT extra `=======` divider before `>>>>>>> REPLACE`, e.g.

    <<<<<<< SEARCH
    <old code>
    =======          <- the real divider
    <new code>
    =======          <- spurious extra divider (the bug)
    >>>>>>> REPLACE

`patcher.SearchReplacePatcher`'s regex captures the replace group as everything
up to `>>>>>>> REPLACE`, so the trailing `=======` is baked into the replacement
text and written verbatim into the runfile -> instant SyntaxError, exec_time~0s.
`apply.py` then accepts any result with count>0 (no syntax check), and once a
node's code carries `=======` the next debug node quotes it back in its SEARCH
half -> a corruption death-spiral (markers grew 4 -> 31 -> 99 across a run).

The task1 anti-truncation fix (max_tokens 16384->32768) does NOT touch this: the
responses are COMPLETE, not truncated. This is the never-built "layer 3" guard.

Two layers, both applied by wrapping `SearchReplacePatcher.apply_patch`:

  1. NORMALIZE (recover the node): before applying, drop spurious extra
     `=======` divider lines inside each SEARCH/REPLACE block, keeping only the
     first (the legitimate divider). The well-intended edit then applies cleanly
     and the node is recovered instead of merely rejected.

  2. VALIDATE + REVERT (safety net, never execute corruption): after applying,
     if the result does not `ast.parse`, return (original_text, 0) — i.e.
     "nothing applied" — so the caller retries / keeps the last-good code. The
     ast check is the bulletproof invariant: it subsumes BOTH observed
     corruption modes — stray `=======` markers AND model prose leaking into the
     file (e.g. a trailing "Wait, looking at the error..." line, seen in the
     real journals) — and anything else that yields unrunnable code. Because an
     invalid runfile scores zero no matter what, rejecting+retrying is never
     worse than executing it, and crucially it stops a broken result from
     becoming the parent of the next debug node — which is what turns one bad
     patch into the observed death-spiral (3 valid nodes, then 3 invalid in a
     row, markers growing 4 -> 7 -> 38).

Installed via a sys.meta_path hook (mirrors skill_injector) because
`agents.coder.diff_coder.patcher` loads long after this sidecar imports.
Fair for the A/B: it's a harness-level fix applied identically to both cells.
"""
from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import logging
import re
import sys

logger = logging.getLogger("MLEvolve")

_TARGET = "agents.coder.diff_coder.patcher"

# Stripped-line matchers for the three conflict markers. The patcher itself
# requires exactly 7 of each char (`<{7}`/`={7}`/`>{7}`); we mirror that.
_FENCE_SEARCH = re.compile(r"<{7}\s*SEARCH")
_FENCE_REPLACE = re.compile(r">{7}\s*REPLACE")
_DIVIDER = re.compile(r"={7,}\Z")


def _normalize_blocks(patch_text: str) -> str:
    """Drop spurious *extra* `=======` dividers inside each SEARCH/REPLACE block.

    Keeps the first divider per block (the legitimate one, including the empty
    SEARCH/empty REPLACE deletion case) and removes any subsequent divider lines
    before the closing `>>>>>>> REPLACE` fence. Idempotent on well-formed input.
    """
    out: list[str] = []
    in_block = False
    seen_divider = False
    for line in patch_text.splitlines():
        s = line.strip()
        if _FENCE_SEARCH.match(s):
            in_block = True
            seen_divider = False
            out.append(line)
        elif _FENCE_REPLACE.match(s):
            in_block = False
            seen_divider = False
            out.append(line)
        elif in_block and _DIVIDER.fullmatch(s):
            if seen_divider:
                continue  # spurious extra divider -> drop
            seen_divider = True
            out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def _is_valid_python(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


def _has_markers(text: str) -> bool:
    """True iff `text` carries any residual conflict marker line (diagnostic)."""
    for line in text.splitlines():
        s = line.strip()
        if _FENCE_SEARCH.match(s) or _FENCE_REPLACE.match(s) or _DIVIDER.fullmatch(s):
            return True
    return False


def _wrap_apply_patch(orig):
    if getattr(orig, "_mleval_diff_guarded", False):
        return orig

    def apply_patch(self, patch_text, original_text, strict=True):
        normalized = _normalize_blocks(patch_text)
        if normalized != patch_text:
            logger.info(
                "[diff_guard] normalized spurious SEARCH/REPLACE divider(s) "
                "before apply"
            )
        new_text, count = orig(self, normalized, original_text, strict=strict)
        # Bulletproof invariant: never hand back a CHANGED result that is not
        # valid Python. Catches stray ======= markers, prose leak, and any other
        # malformation. Reverting (count=0) stops the corrupt result from
        # executing AND from becoming the next debug node's parent (the spiral).
        if count > 0 and new_text != original_text and not _is_valid_python(new_text):
            logger.warning(
                "[diff_guard] reverted invalid patch result (ast.parse failed "
                "after applying %d block(s); markers=%s); keeping last-good code",
                count, _has_markers(new_text),
            )
            return original_text, 0
        return new_text, count

    apply_patch._mleval_diff_guarded = True
    return apply_patch


def _patch_module(module) -> None:
    cls = getattr(module, "SearchReplacePatcher", None)
    if cls is None:
        logger.warning("[diff_guard] %s has no SearchReplacePatcher; not patched", _TARGET)
        return
    if getattr(cls.apply_patch, "_mleval_diff_guarded", False):
        return
    cls.apply_patch = _wrap_apply_patch(cls.apply_patch)
    logger.info("[diff_guard] hardened SearchReplacePatcher.apply_patch")


class _DiffGuardFinder(importlib.abc.MetaPathFinder):
    """Wrap exec_module for the patcher module so we patch it post-load."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        if getattr(loader, "_mleval_diff_wrapped_exec", False):
            return spec
        orig_exec = loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            try:
                _patch_module(module)
            except Exception as e:  # noqa: BLE001 — never break codegen import
                logger.warning("[diff_guard] post-load patch failed: %s", e)

        loader.exec_module = exec_module
        loader._mleval_diff_wrapped_exec = True
        return spec


# Install at the front; cover the (unlikely) already-loaded case too.
sys.meta_path.insert(0, _DiffGuardFinder())
_already = sys.modules.get(_TARGET)
if _already is not None:
    _patch_module(_already)

logger.info("[diff_guard] registered import hook for %s", _TARGET)
