"""Schema + loader for per-task prompt overlays.

A bare dataclass (no pydantic — avoids a Rust-wheel dep in the image; the
4-field schema is too small to justify schema-validation infrastructure).
Missing fields fall through to upstream MLEvolve prompts; this fallback
is the universal "missing key" behavior and is safe by construction.

YAML schema (all keys optional, top-level lookups via .get):

    persona:
      identity: str               # FULL system message — replaces "Kaggle
                                  #   Grandmaster..." intro in its entirety.

    instructions:
      what_to_produce: list[str]  # Becomes the "Implementation guideline"
      self_check: list[str]       #   block injected into every codegen prompt.

    review_facts:
      output_location: str        # Spliced over the hardcoded
                                  #   "Submission File Location: Must save
                                  #   the submission to ./submission/...".

Anything else in the YAML is silently ignored (forward-compat for future
overlay surfaces — omit_fragments, stepwise_mode, etc.).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_VAR = "MLEVOLVE_PROMPT_OVERLAY"


@dataclass(frozen=True)
class Overlay:
    """Loaded overlay (immutable). All fields optional; None = "use upstream"."""

    persona_identity: str | None = None
    what_to_produce: list[str] | None = None
    self_check: list[str] | None = None
    output_location: str | None = None

    @property
    def is_empty(self) -> bool:
        """True iff no override is set — all patches will pass through."""
        return (
            self.persona_identity is None
            and self.what_to_produce is None
            and self.self_check is None
            and self.output_location is None
        )


def load_overlay(path: str | os.PathLike | None = None) -> Overlay:
    """Load overlay from ``path`` (or ``$MLEVOLVE_PROMPT_OVERLAY``).

    Returns an empty ``Overlay()`` on any of:
      - path argument and env var both unset
      - file missing
      - YAML parse error
      - schema mismatch (e.g. non-list ``what_to_produce``)

    All failure modes log a warning and proceed — task continues with
    upstream MLEvolve prompts.
    """
    p = path if path is not None else os.environ.get(ENV_VAR)
    if not p:
        return Overlay()

    fp = Path(p)
    if not fp.is_file():
        logger.warning("[overlay] %s not found; using upstream defaults", fp)
        return Overlay()

    try:
        import yaml  # local import — yaml is already a transitive dep
        data = yaml.safe_load(fp.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("[overlay] failed to parse %s (%s); using upstream defaults", fp, e)
        return Overlay()

    persona = (data.get("persona") or {}).get("identity")
    instr = data.get("instructions") or {}
    what = instr.get("what_to_produce")
    chk = instr.get("self_check")
    review = (data.get("review_facts") or {}).get("output_location")

    # Light shape check — wrong type → drop silently rather than crash.
    if persona is not None and not isinstance(persona, str):
        logger.warning("[overlay] persona.identity must be str; dropping")
        persona = None
    if what is not None and not (isinstance(what, list) and all(isinstance(x, str) for x in what)):
        logger.warning("[overlay] instructions.what_to_produce must be list[str]; dropping")
        what = None
    if chk is not None and not (isinstance(chk, list) and all(isinstance(x, str) for x in chk)):
        logger.warning("[overlay] instructions.self_check must be list[str]; dropping")
        chk = None
    if review is not None and not isinstance(review, str):
        logger.warning("[overlay] review_facts.output_location must be str; dropping")
        review = None

    ov = Overlay(
        persona_identity=persona,
        what_to_produce=what,
        self_check=chk,
        output_location=review,
    )
    if not ov.is_empty:
        logger.info(
            "[overlay] loaded from %s: persona=%s instructions=%s review=%s",
            fp,
            persona is not None,
            (what is not None) or (chk is not None),
            review is not None,
        )
    return ov
