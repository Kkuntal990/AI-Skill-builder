"""Deterministic, zero-dependency ROUGE-L (LCS-based F1).

Definition (sentence-level ROUGE-L, the standard summarisation metric):

    LCS   = length of the longest common subsequence of the token lists
    P     = LCS / len(candidate_tokens)
    R     = LCS / len(reference_tokens)
    F1    = 2 P R / (P + R)            (beta = 1)

Tokenisation is lowercase + ``[a-z0-9]+`` word matching. No stemming — this
keeps the metric reproducible across environments and trivially testable.
Empty candidate or reference → 0.0 (never a divide-by-zero).

This matches ``rouge_score``'s rougeL F-measure shape (beta=1, F1) up to the
stemmer/tokeniser choice; for an A/B comparison (with-skill vs without) only
internal consistency matters, and this is consistent and deterministic.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation. Non-strings (NaN floats etc.) → empty."""
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (two-row DP, O(len(a)*len(b))).

    Keeps two rows of the DP table — summaries are short (tens of tokens),
    so this is comfortably fast over the full test split and stays obviously
    correct.
    """
    if not a or not b:
        return 0
    m = len(b)
    prev = [0] * (m + 1)
    for token_a in a:
        curr = [0] * (m + 1)
        for j in range(m):
            if token_a == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = curr[j] if curr[j] >= prev[j + 1] else prev[j + 1]
        prev = curr
    return prev[m]


def rouge_l_f(candidate: str, reference: str) -> float:
    """Sentence-level ROUGE-L F1 between two strings, in ``[0.0, 1.0]``."""
    cand = tokenize(candidate)
    ref = tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    prec = lcs / len(cand)
    rec = lcs / len(ref)
    return 2.0 * prec * rec / (prec + rec)


def mean_rouge_l_f(pairs: list[tuple[str, str]]) -> float:
    """Mean ROUGE-L F1 over (candidate, reference) pairs. Empty → 0.0."""
    if not pairs:
        return 0.0
    return sum(rouge_l_f(c, r) for c, r in pairs) / len(pairs)
