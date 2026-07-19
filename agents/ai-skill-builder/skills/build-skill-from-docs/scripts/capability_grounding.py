"""Grounding audit for compiled capability manifests (E1 fidelity gate, W4).

A second model (never the compiler's) audits every claim in a compiled
capabilities.json against the source skill package, labeling each item
supported / entailed / unsupported. Aggregates map onto the E1 pass bars
(docs/skill-builder/capability-linker-mvp-plan.md §5/E1′):

- support rate over claims in fields NOT flagged synthesized  (gate >= 0.90)
- fabrication rate: unsupported claims in unflagged fields    (the killer metric)
- applicability correctness                                    (gate >= 0.85)
- validation usability for executable units                    (gate >= 0.80)

Like the compiler, the LLM transport is injected. The audit itself uses a
bounded parse-retry (no repair semantics — an auditor that can't produce JSON
is re-asked, never coached toward different labels).
"""

from __future__ import annotations

import json
from pathlib import Path

from capability_compiler import (
    PROMPTS_DIR,
    _extract_json,
    _fill,
    _references_block,
    load_skill_bundle,
)

GROUNDING_VERSION = "0.1.0"

# Fields whose synthesized_fields flag exempts them from the strict support gate.
_FIELD_OF_PATH = {
    "applicability.when": "applicability.when",
    "applicability.avoid_when": "applicability.avoid_when",
    "procedure.steps": "procedure.steps",  # not synthesizable — always strict
    "validation": "validation",
    "recovery": "recovery",
    "expected_outcomes": "expected_outcomes",
    "requires": "requires.runtime",  # any requires.* flag exempts the joint item
    "produces": "produces",
    "summary": "summary",  # not synthesizable — always strict
}


def _base_field(path: str) -> str:
    return path.split("[", 1)[0].strip()


def _is_flagged(unit: dict, claim_path: str) -> bool:
    flags = set(unit.get("synthesized_fields") or [])
    base = _base_field(claim_path)
    if base == "requires":
        return any(f.startswith("requires.") for f in flags)
    return _FIELD_OF_PATH.get(base, base) in flags or base in flags


def audit_manifest(bundle: dict, manifest: dict, llm, max_parse_retries: int = 1) -> dict:
    """One audit call per skill; returns the parsed audit (units -> claims)."""
    template = (PROMPTS_DIR / "critique_capabilities.txt").read_text()
    prompt = _fill(
        template,
        skill_name=bundle["skill_name"],
        skill_md=bundle["skill_md"][:30000],
        references_block=_references_block(bundle),
        manifest_json=json.dumps(
            {"capabilities": manifest.get("capabilities", [])}, indent=2
        ),
    )
    last_err = None
    for _ in range(1 + max_parse_retries):
        raw = llm(prompt, max_tokens=16000, temperature=0.0)
        try:
            audit = _extract_json(raw)
            if isinstance(audit.get("units"), list):
                return audit
            last_err = "audit JSON missing 'units' list"
        except (ValueError, json.JSONDecodeError) as e:
            last_err = str(e)[:200]
    raise RuntimeError(f"grounding audit unparseable after retries: {last_err}")


def score_audit(manifest: dict, audit: dict) -> dict:
    """Map audit labels onto the E1 metrics. Pure function, no LLM."""
    units_by_id = {u["id"]: u for u in manifest.get("capabilities", [])}
    executable_kinds = ("procedure", "workflow")

    total = strict_total = 0
    supported = strict_supported = 0
    fabrications = []  # unsupported claims in UNflagged fields
    flagged_unsupported = []  # unsupported even as inference, in flagged fields
    label_hist = {"supported": 0, "entailed": 0, "unsupported": 0}
    applicability_ok = applicability_n = 0
    validation_ok = validation_n = 0
    locators_bad = []

    for au in audit.get("units", []):
        unit = units_by_id.get(au.get("id"))
        if unit is None:
            continue
        for claim in au.get("claims", []):
            label = claim.get("label")
            if label not in label_hist:
                continue
            label_hist[label] += 1
            total += 1
            ok = label in ("supported", "entailed")
            supported += ok
            flagged = _is_flagged(unit, claim.get("path", ""))
            if not flagged:
                strict_total += 1
                strict_supported += ok
                if not ok:
                    fabrications.append(
                        {"unit": au["id"], "path": claim.get("path"),
                         "note": claim.get("note", "")}
                    )
            elif not ok:
                flagged_unsupported.append(
                    {"unit": au["id"], "path": claim.get("path"),
                     "note": claim.get("note", "")}
                )
        if isinstance(au.get("applicability_correct"), bool):
            applicability_n += 1
            applicability_ok += au["applicability_correct"]
        if unit.get("kind") in executable_kinds and isinstance(
            au.get("validation_usable"), bool
        ):
            validation_n += 1
            validation_ok += au["validation_usable"]
        if au.get("source_locators_resolve") is False:
            locators_bad.append(au["id"])

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "claims_audited": total,
        "label_histogram": label_hist,
        "support_rate_all": rate(supported, total),
        "support_rate_strict": rate(strict_supported, strict_total),
        "strict_claims": strict_total,
        "fabrications": fabrications,
        "flagged_but_unsupported": flagged_unsupported,
        "applicability_correct_rate": rate(applicability_ok, applicability_n),
        "validation_usable_rate": rate(validation_ok, validation_n),
        "units_with_bad_locators": locators_bad,
    }


def run_audit(skill_dir: Path, manifest_path: Path, llm) -> dict:
    bundle = load_skill_bundle(skill_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    audit = audit_manifest(bundle, manifest, llm)
    scores = score_audit(manifest, audit)
    return {
        "grounding_version": GROUNDING_VERSION,
        "skill_name": bundle["skill_name"],
        "scores": scores,
        "audit": audit,
    }
