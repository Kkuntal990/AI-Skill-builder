"""Skill retrieval — chunks SKILL.md + references/*.md, indexes BM25, retrieves top-k per turn.

Replaces the spike-004 ``skill_inject`` approach (concatenate everything into
description.md → 80 KB system_message → LLM mimics fenced format → SyntaxError
cascade). Design in ``docs/eval/skill-retrieval-design.md``.

Path A: keep skill content out of the system message. L1 catalog goes into the
persona/intro; L2/L3 chunks are retrieved per LLM call and injected into the
user_prompt only when relevance exceeds a threshold (matches the user's "no
bloat when idle" principle).

Reads ``MLEVAL_SKILL_PATH`` env var at import time. Accepts either:

  * A single skill directory (with SKILL.md and optional references/*.md)
  * A parent directory containing multiple skill subdirs (skill-library mode)
  * An empty / unset value (no-op; ``current_index()`` returns None)

BM25 implementation is inline (~50 LoC) because rank_bm25 is not in the runtime
image. Standard Okapi formula (k1=1.5, b=0.75). Quality is good enough for
~50-200 chunks; swap for rank_bm25 + Hybrid-vector retrieval in a follow-up if
needed.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_VAR = "MLEVAL_SKILL_PATH"

# BM25 hyperparams (standard Okapi values).
_BM25_K1 = 1.5
_BM25_B = 0.75

# Retrieval gating defaults. We use TWO gates because pure score-based
# thresholds either let every query "look relevant" (normalized) or vary
# wildly across queries (raw BM25). Both gates must pass:
#
#   1. MIN_RAW_SCORE: absolute BM25 floor — top chunk must score at least
#      this much. Catches "query has nothing in common with the index."
#   2. MIN_QUERY_TERMS: distinct query tokens (after stop-word filtering)
#      that must appear in the top chunk. Catches "query matches one
#      noisy term that happens to be in lots of chunks."
#
# Tuned empirically against the peft-tuning skill: irrelevant queries like
# "confusion matrix on tabular features" should return [].
DEFAULT_K = 3
DEFAULT_MIN_RAW_SCORE = 3.0
DEFAULT_MIN_QUERY_TERMS = 2

# Tokens that are too common to count toward MIN_QUERY_TERMS. Conservative —
# we keep most words because chunk content includes lots of technical terms
# we DO want to count (LoRA, QLoRA, batch, model, etc.). Just strip the
# 50-or-so most generic words that show up in every English prompt.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i if in is it its of on or"
    " our s such t that the their them then there these they this to was we were"
    " what when which who will with you your".split()
)

# Stages where skill chunks are injected. AutoMLGen paper (arXiv 2510.08511 §3.2)
# limits KB to "initial solution" (draft only); we extend to improve/debug
# because PEFT debugging is exactly when the skill's value shows up. The
# stage signals are read from user_prompt content, since MLEvolve doesn't
# pass stage through to build_chat_prompt_for_model. See ``detect_stage``.
INJECTION_STAGES = {"draft", "improve", "debug"}


@dataclass(frozen=True)
class Chunk:
    """One indexed unit. Heading is empty for whole-file (references) chunks."""

    skill_name: str
    source_file: str  # relative to skill dir, e.g. "SKILL.md" or "references/qwen.md"
    heading: str  # the H2/H3 text, or "" for whole-file
    body: str
    # Set at retrieval time, not index time.
    score: float = 0.0


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

# Match H2/H3 headings at line start. We don't split on H1 because SKILL.md
# typically has a single H1 title that wraps the whole doc.
_H2_H3_RE = re.compile(r"^(##{1,2})\s+(.+?)\s*$", re.MULTILINE)


def _strip_frontmatter(text: str) -> str:
    """Drop YAML frontmatter ``---\n...\n---`` if present at start of file.

    Used by the catalog builder for skill-level metadata. Bodies indexed for
    retrieval should NOT contain it — frontmatter would otherwise leak into
    BM25 term frequencies (e.g. the word ``description`` always appearing).
    """
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + 5 :]


def _chunk_markdown_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs at H2/H3 boundaries.

    Content before the first heading becomes ``("", <preamble>)``. Returns
    [("", body)] if no headings found (single-chunk fallback).
    """
    matches = list(_H2_H3_RE.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    # Preamble before first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body or heading:
            sections.append((heading, body))
    return sections


def _load_skill_dir(skill_dir: Path) -> tuple[str, dict | None, list[Chunk]]:
    """Load a single skill directory.

    Returns (skill_name, frontmatter_dict_or_None, chunks). Frontmatter is
    parsed from SKILL.md's leading YAML block; on failure we return None and
    fall back to deriving the catalog entry from filename + first paragraph.
    """
    skill_name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    chunks: list[Chunk] = []
    frontmatter: dict | None = None

    if skill_md.is_file():
        raw = skill_md.read_text()
        frontmatter = _parse_frontmatter(raw)
        body = _strip_frontmatter(raw)
        for heading, content in _chunk_markdown_by_headings(body):
            if not content:
                continue
            chunks.append(
                Chunk(
                    skill_name=skill_name,
                    source_file="SKILL.md",
                    heading=heading,
                    body=content,
                )
            )

    refs_dir = skill_dir / "references"
    if refs_dir.is_dir():
        for ref in sorted(refs_dir.glob("*.md")):
            content = ref.read_text().strip()
            if not content:
                continue
            # Each reference is one chunk by default. Split further only if
            # it's large enough to risk OOMing the per-turn context.
            if len(content) > 4096:
                for heading, body in _chunk_markdown_by_headings(content):
                    if body:
                        chunks.append(
                            Chunk(
                                skill_name=skill_name,
                                source_file=f"references/{ref.name}",
                                heading=heading,
                                body=body,
                            )
                        )
            else:
                chunks.append(
                    Chunk(
                        skill_name=skill_name,
                        source_file=f"references/{ref.name}",
                        heading="",
                        body=content,
                    )
                )

    return skill_name, frontmatter, chunks


def _parse_frontmatter(text: str) -> dict | None:
    """Minimal YAML frontmatter parser. Just enough for ``name``, ``description``,
    ``triggers``. Avoids adding PyYAML to the runtime image.

    Supports:
      ``name: foo``
      ``description: |\n  multi-line\n  block``
      ``triggers: [a, b, c]``
    Returns None if no frontmatter or parse fails — caller falls back.
    """
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    block = text[4:end]

    out: dict = {}
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "|":
            # Multi-line block — gather indented continuation lines.
            block_lines: list[str] = []
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                block_lines.append(lines[i][2:] if lines[i].startswith("  ") else "")
                i += 1
            out[key] = "\n".join(block_lines).strip()
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            out[key] = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
        else:
            out[key] = value.strip().strip('"').strip("'")
    return out or None


# ---------------------------------------------------------------------------
# BM25 (inline)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + alnum word split. Code-aware (keeps underscores so
    ``target_modules`` stays one token, ``q_proj`` matches ``q_proj``).
    """
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Minimal Okapi BM25 over a fixed corpus."""

    def __init__(self, tokenized_docs: list[list[str]]):
        self.docs = tokenized_docs
        self.n = len(tokenized_docs)
        self.doc_len = [len(d) for d in tokenized_docs]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — the +1 keeps IDF
        # non-negative for terms in all docs (BM25-plus variant).
        df: Counter[str] = Counter()
        for d in tokenized_docs:
            df.update(set(d))
        self.idf: dict[str, float] = {
            term: math.log((self.n - cnt + 0.5) / (cnt + 0.5) + 1.0)
            for term, cnt in df.items()
        }
        self.tf: list[Counter[str]] = [Counter(d) for d in tokenized_docs]

    def score(self, query_tokens: list[str]) -> list[float]:
        if not self.n:
            return []
        scores = [0.0] * self.n
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                ft = tf.get(term)
                if not ft:
                    continue
                dl = self.doc_len[i]
                denom = ft + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * (dl / self.avgdl if self.avgdl else 1.0)
                )
                scores[i] += idf * (ft * (_BM25_K1 + 1.0)) / denom
        return scores


# ---------------------------------------------------------------------------
# SkillIndex — public API
# ---------------------------------------------------------------------------


class SkillIndex:
    """A retriever over a (small) set of skill chunks.

    Construct from a list of Chunks; query via ``search``. Catalog text for
    the L1 system-prompt slot is exposed via ``catalog_text``.
    """

    def __init__(self, chunks: list[Chunk], frontmatter: dict[str, dict | None]):
        self.chunks = chunks
        self.frontmatter = frontmatter  # skill_name → frontmatter dict | None
        # Index over chunk_body + heading (heading boosts term importance).
        texts = [f"{c.heading} {c.heading} {c.body}" for c in chunks]
        self._bm25 = _BM25([_tokenize(t) for t in texts])

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def skill_names(self) -> list[str]:
        return sorted({c.skill_name for c in self.chunks})

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        min_raw_score: float = DEFAULT_MIN_RAW_SCORE,
        min_query_terms: int = DEFAULT_MIN_QUERY_TERMS,
    ) -> list[Chunk]:
        """Return up to k chunks judged genuinely relevant to the query.

        Two-gate filter (see module docstring): the top-1 chunk must
        (a) score above ``min_raw_score`` on absolute BM25, AND
        (b) contain at least ``min_query_terms`` distinct non-stopword
        tokens from the query in its body. If the top chunk fails either
        gate we return []. Sub-top chunks must score within 50% of the
        top — otherwise they're noise.

        Score attached to each returned chunk is the raw BM25 score for
        traceability (logged in render_chunks).
        """
        if not self.chunks:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Distinct non-stopword query terms — used by gate (b).
        q_significant = frozenset(t for t in q_tokens if t not in _STOPWORDS)
        if not q_significant:
            return []

        raw_scores = self._bm25.score(q_tokens)
        if not raw_scores or max(raw_scores) <= 0:
            return []

        ranked = sorted(range(len(self.chunks)), key=lambda i: raw_scores[i], reverse=True)
        top_idx = ranked[0]
        top_score = raw_scores[top_idx]

        # Gate (a): absolute BM25 floor on top chunk.
        if top_score < min_raw_score:
            return []

        # Gate (b): distinct query terms present in top chunk's body.
        top_body_tokens = frozenset(_tokenize(self.chunks[top_idx].body))
        overlap = q_significant & top_body_tokens
        if len(overlap) < min_query_terms:
            return []

        # Sub-top: keep chunks within 50% of top score (relative cutoff).
        out: list[Chunk] = []
        floor = top_score * 0.5
        for i in ranked[:k]:
            if raw_scores[i] < floor:
                break
            c = self.chunks[i]
            out.append(
                Chunk(
                    skill_name=c.skill_name,
                    source_file=c.source_file,
                    heading=c.heading,
                    body=c.body,
                    score=raw_scores[i],
                )
            )
        return out

    def catalog_text(self) -> str:
        """Render the L1 catalog as an Anthropic-style YAML block.

        Always emitted (idle state) so the model knows what skills exist
        even when retrieval is empty.
        """
        lines = ["## Available skills", "", "```yaml"]
        for name in self.skill_names:
            fm = self.frontmatter.get(name) or {}
            description = (fm.get("description") or _derive_description(name, self.chunks)).strip()
            triggers = fm.get("triggers") or []
            lines.append(f"- name: {name}")
            # Indent description block under |
            lines.append(f"  description: |")
            for d in description.splitlines() or [""]:
                lines.append(f"    {d}")
            if triggers:
                trig_str = ", ".join(triggers)
                lines.append(f"  triggers: [{trig_str}]")
        lines.append("```")
        return "\n".join(lines)


def _derive_description(skill_name: str, chunks: list[Chunk]) -> str:
    """Fallback when SKILL.md lacks frontmatter: use the first SKILL.md chunk
    body (truncated). Keeps the catalog non-empty for legacy skills."""
    for c in chunks:
        if c.skill_name == skill_name and c.source_file == "SKILL.md":
            snippet = c.body.split("\n\n", 1)[0]
            return snippet[:400] + ("..." if len(snippet) > 400 else "")
    return f"(no description for {skill_name})"


# ---------------------------------------------------------------------------
# Loader — env-var driven
# ---------------------------------------------------------------------------


def load_skill_index(skill_path: str | None = None) -> SkillIndex | None:
    """Build a SkillIndex from a path. Path may be:

      * A single skill directory containing SKILL.md
      * A parent containing multiple skill subdirs (library mode)
      * Unset / nonexistent → returns None (caller treats as "no skill")
    """
    path_str = skill_path or os.environ.get(ENV_VAR, "").strip()
    if not path_str:
        return None
    root = Path(path_str)
    if not root.exists():
        logger.warning("[skill_retriever] %s does not exist; no index built", root)
        return None

    skill_dirs: list[Path] = []
    if (root / "SKILL.md").is_file():
        skill_dirs = [root]
    elif root.is_dir():
        # Library mode: any direct subdir with SKILL.md is a skill.
        skill_dirs = sorted(
            sub for sub in root.iterdir() if sub.is_dir() and (sub / "SKILL.md").is_file()
        )

    if not skill_dirs:
        logger.warning(
            "[skill_retriever] no SKILL.md found under %s; no index built", root
        )
        return None

    all_chunks: list[Chunk] = []
    frontmatter_by_skill: dict[str, dict | None] = {}
    for sd in skill_dirs:
        name, fm, chunks = _load_skill_dir(sd)
        frontmatter_by_skill[name] = fm
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning(
            "[skill_retriever] %d skill dir(s) found but no chunks produced", len(skill_dirs)
        )
        return None

    idx = SkillIndex(all_chunks, frontmatter_by_skill)
    logger.info(
        "[skill_retriever] indexed %d chunks across %d skill(s): %s",
        len(idx),
        len(idx.skill_names),
        idx.skill_names,
    )
    return idx


# ---------------------------------------------------------------------------
# Module-level singleton + accessor
# ---------------------------------------------------------------------------

_INDEX: SkillIndex | None = load_skill_index()


def current_index() -> SkillIndex | None:
    """Return the active SkillIndex (built at import time, may be None)."""
    return _INDEX


def reload(skill_path: str | None = None) -> SkillIndex | None:
    """Rebuild the index. Used by the build-time smoke test."""
    global _INDEX
    _INDEX = load_skill_index(skill_path)
    return _INDEX


# ---------------------------------------------------------------------------
# Stage detection + chunk rendering helpers (used by prompt_overlay wrapper)
# ---------------------------------------------------------------------------

# Heuristic markers per stage. MLEvolve doesn't pass stage info through to
# build_chat_prompt_for_model, so we read it off the user_prompt content.
# Drafts contain the "Solution sketch guideline" block (only in
# draft_agent.py:101); improves/debugs contain "previous solution" markers.
_STAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "draft": ("Solution sketch guideline", "Solution Plan and Code"),
    "improve": ("Previous solution", "Previously executed", "improvement"),
    "debug": ("Bug analysis", "Previous Bug", "buggy", "Error message"),
}


def detect_stage(user_prompt: str) -> str | None:
    """Best-effort stage detection from user_prompt content. None means
    "structured/non-codegen" — we skip injection for those (parse_result,
    code_review, etc.) to avoid polluting them with skill prose.
    """
    if not user_prompt:
        return None
    for stage, markers in _STAGE_MARKERS.items():
        if any(m in user_prompt for m in markers):
            return stage
    return None


def render_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a # Skill Reference section.

    Body fences are preserved (the skill is authoritative reference). The
    wrapping framing tells the LLM this is REFERENCE, not output template
    — reduces the format-mimicry pressure that broke spike-004.
    """
    if not chunks:
        return ""
    lines = [
        "",
        "# Skill Reference",
        "",
        "*The following reference material is retrieved from the skill library. "
        "Use it to inform your solution. DO NOT mimic its markdown formatting in "
        "your output — your response should be Python code, not documentation.*",
        "",
    ]
    for c in chunks:
        header = f"## From {c.skill_name}"
        if c.source_file != "SKILL.md":
            header += f" > {c.source_file}"
        if c.heading:
            header += f" > {c.heading}"
        header += f" (score={c.score:.2f})"
        lines.append(header)
        lines.append("")
        lines.append(c.body)
        lines.append("")
    return "\n".join(lines)


def inject_into_user_prompt(user_prompt: str, chunks_block: str) -> str:
    """Insert the Skill Reference block before the # Instructions heading.

    Falls back to prepending the block at the top if the heading isn't
    present (defensive — keeps the chunks reachable in unusual layouts).
    """
    if not chunks_block:
        return user_prompt
    marker = "\n# Instructions"
    idx = user_prompt.find(marker)
    if idx < 0:
        return chunks_block + "\n" + user_prompt
    return user_prompt[:idx] + chunks_block + user_prompt[idx:]
