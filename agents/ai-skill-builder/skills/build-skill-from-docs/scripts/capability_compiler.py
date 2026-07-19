"""Capability compiler — skill bundle -> capabilities.json (schema 0.2).

Experimental stage of the capability-linker MVP (W3 in
docs/skill-builder/capability-linker-mvp-plan.md). Two entry points, both in
skill_builder.py: `--emit-capabilities` on a normal build, and the
`compile-existing <skill-dir>` subcommand for packages the builder did not
write (e.g. the AI-Research-SKILLs corpus).

Design constraints:
- The LLM transport is injected as a callable (skill_builder passes its
  `_llm_call`), so this module has no transport code and no circular import.
- The schema/validator single source lives in the MLEvolve sidecar
  (infra/agents/mlevolve/mlevolve_sidecar/capability_schema.py); it is located
  by walking up to the repo root, overridable via $CAPABILITY_SCHEMA_PATH for
  standalone (~/.openclaw) deployments.
- Validation errors drive a bounded repair loop: the model gets its own JSON
  back with the validator errors and must return a complete corrected object.
  Attempts are recorded in the compile report — the repair-loop convergence is
  E1's material-correction proxy.
- Compile failure never raises out of compile_capabilities(); callers decide
  whether a missing manifest fails anything (per plan §4.1 it must not fail an
  otherwise valid build).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPTS_DIR / "prompts"

COMPILER_VERSION = "0.1.0"

MAX_SKILL_MD_CHARS = 30000
MAX_REF_FILE_CHARS = 6000
MAX_UNITS = 7
MIN_UNITS_SOFT = 3
LLM_MAX_TOKENS = 16000


# ── schema module discovery ──────────────────────────────────────────────────


def _find_schema_module_path() -> Path:
    env = os.environ.get("CAPABILITY_SCHEMA_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"$CAPABILITY_SCHEMA_PATH points at a missing file: {p}")
    rel = Path("infra/agents/mlevolve/mlevolve_sidecar/capability_schema.py")
    for parent in (SCRIPTS_DIR, *SCRIPTS_DIR.parents):
        cand = parent / rel
        if cand.is_file():
            return cand
    local = SCRIPTS_DIR / "capability_schema.py"
    if local.is_file():
        return local
    raise FileNotFoundError(
        "capability_schema.py not found (repo layout or co-located copy); "
        "set CAPABILITY_SCHEMA_PATH for standalone deployments"
    )


def load_schema_module():
    path = _find_schema_module_path()
    spec = importlib.util.spec_from_file_location("capability_schema", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── bundle loading (compile-existing path) ───────────────────────────────────


def _frontmatter_name(skill_md: str) -> str:
    lines = skill_md.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:40]:
        if line.strip() == "---":
            break
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def load_skill_bundle(skill_dir: Path) -> dict:
    """Read SKILL.md + references/*.md (+ scripts listing) from a skill dir."""
    skill_dir = Path(skill_dir)
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.is_file():
        raise FileNotFoundError(f"no SKILL.md under {skill_dir}")
    skill_md = skill_md_path.read_text(encoding="utf-8", errors="replace")
    references: dict[str, str] = {}
    refs_dir = skill_dir / "references"
    if refs_dir.is_dir():
        for f in sorted(refs_dir.glob("*.md")):
            references[f"references/{f.name}"] = f.read_text(
                encoding="utf-8", errors="replace"
            )
    scripts: list[str] = []
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        scripts = sorted(
            str(p.relative_to(skill_dir)) for p in scripts_dir.iterdir() if p.is_file()
        )
    return {
        "skill_name": _frontmatter_name(skill_md) or skill_dir.name,
        "skill_md": skill_md,
        "references": references,
        "scripts": scripts,
    }


def build_source_snapshot(skill_dir: Path, url: str = "", kind: str = "skill-package") -> dict:
    """Snapshot the bundle content (hash over SKILL.md + references)."""
    bundle = load_skill_bundle(skill_dir)
    h = hashlib.sha256(bundle["skill_md"].encode("utf-8"))
    for rel in sorted(bundle["references"]):
        h.update(rel.encode("utf-8"))
        h.update(bundle["references"][rel].encode("utf-8"))
    return {
        "kind": kind,
        "url": url or f"file://{Path(skill_dir).resolve()}",
        "content_sha256": h.hexdigest(),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── prompt assembly ──────────────────────────────────────────────────────────


def _fill(template: str, **vars: str) -> str:
    out = template
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def _references_block(bundle: dict) -> str:
    parts = []
    for relpath in sorted(bundle["references"]):
        text = bundle["references"][relpath]
        clipped = text[:MAX_REF_FILE_CHARS]
        suffix = "\n[... truncated ...]" if len(text) > MAX_REF_FILE_CHARS else ""
        parts.append(f"--- {relpath} ---\n{clipped}{suffix}")
    if bundle["scripts"]:
        parts.append(
            "--- bundled script paths (contents omitted) ---\n"
            + "\n".join(bundle["scripts"])
        )
    return "\n\n".join(parts) if parts else "(no reference files in this bundle)"


def _extract_json(raw: str) -> dict:
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in output: {t[:200]!r}")
    return json.loads(t[start : end + 1])


def _repair_block(errors: list, previous_json: str) -> str:
    bullet = "\n".join(f"- {e}" for e in errors)
    return (
        "\nPREVIOUS ATTEMPT FAILED VALIDATION\n"
        "=================================\n"
        "Your previous output:\n"
        f"{previous_json}\n\n"
        "Deterministic validator errors:\n"
        f"{bullet}\n\n"
        "Fix every error and return the corrected COMPLETE JSON object "
        "(all units — not a diff, not commentary).\n"
    )


# ── stats for the compile report ─────────────────────────────────────────────


def manifest_stats(manifest: dict) -> dict:
    units = manifest.get("capabilities", [])
    kinds: dict[str, int] = {}
    synthesized: dict[str, int] = {}
    for u in units:
        kinds[u.get("kind", "?")] = kinds.get(u.get("kind", "?"), 0) + 1
        for f in u.get("synthesized_fields", []) or []:
            synthesized[f] = synthesized.get(f, 0) + 1
    return {
        "units": len(units),
        "kinds": kinds,
        "units_with_avoid_when": sum(
            1 for u in units if (u.get("applicability") or {}).get("avoid_when")
        ),
        "units_with_depends_on": sum(1 for u in units if u.get("depends_on")),
        "synthesized_field_instances": synthesized,
    }


# ── the compiler ─────────────────────────────────────────────────────────────


def compile_capabilities(
    bundle: dict,
    source_snapshot: dict,
    llm,
    schema_mod=None,
    skill_dir: Path | None = None,
    max_repair_rounds: int = 2,
) -> tuple[dict | None, dict]:
    """One structured compiler call + bounded validator-driven repair loop.

    Returns (manifest | None, report). Never raises on model/validation
    failure — transport-level exceptions from `llm` do propagate.
    """
    schema_mod = schema_mod or load_schema_module()
    template = (PROMPTS_DIR / "compile_capabilities.txt").read_text()
    base_vars = dict(
        skill_name=bundle["skill_name"],
        schema_version=schema_mod.SCHEMA_VERSION,
        kinds=", ".join(schema_mod.KINDS),
        stages=", ".join(schema_mod.CANONICAL_STAGES),
        runtime_vocab=", ".join(schema_mod.RUNTIME_VOCAB),
        synthesizable_fields=", ".join(schema_mod.SYNTHESIZABLE_FIELDS),
        max_units=str(MAX_UNITS),
        skill_md=bundle["skill_md"][:MAX_SKILL_MD_CHARS],
        references_block=_references_block(bundle),
    )
    report = {
        "skill_name": bundle["skill_name"],
        "compiler_version": COMPILER_VERSION,
        "schema_version": schema_mod.SCHEMA_VERSION,
        "transport": os.environ.get("MLEVAL_LLM_TRANSPORT", "claude"),
        "model_override": os.environ.get("MLEVAL_LLM_MODEL", ""),
        "status": "failed",
        "attempts": [],
    }

    repair = ""
    manifest = None
    for attempt in range(1, max_repair_rounds + 2):
        prompt = _fill(template, repair_block=repair, **base_vars)
        raw = llm(prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.2)
        rec: dict = {
            "attempt": attempt,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
        }
        try:
            candidate = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            rec["parse_error"] = str(e)[:300]
            report["attempts"].append(rec)
            repair = _repair_block(
                [f"output was not a parseable JSON object: {e}"], raw[:6000]
            )
            continue

        # Identity and provenance are stamped deterministically — the model is
        # never trusted with them (it could only fabricate).
        candidate["schema_version"] = schema_mod.SCHEMA_VERSION
        candidate["skill_name"] = bundle["skill_name"]
        candidate["source_snapshot"] = source_snapshot

        errors, warnings = schema_mod.validate_manifest(candidate, skill_dir=skill_dir)
        n_units = len(candidate.get("capabilities") or [])
        if n_units > MAX_UNITS:
            errors.append(
                f"manifest has {n_units} units; emit at most {MAX_UNITS} — "
                "keep the operationally most valuable, merge or drop the rest"
            )
        rec["units"] = n_units
        rec["validator_errors"] = errors
        rec["validator_warning_count"] = len(warnings)
        report["attempts"].append(rec)

        if not errors:
            manifest = candidate
            report["status"] = "ok"
            report["validator_warnings"] = warnings
            report["stats"] = manifest_stats(manifest)
            if n_units < MIN_UNITS_SOFT:
                report["note"] = (
                    f"only {n_units} units (soft floor {MIN_UNITS_SOFT}) — "
                    "acceptable if the source is thin, review in E1"
                )
            break
        repair = _repair_block(errors, json.dumps(candidate, indent=2))

    return manifest, report


def write_outputs(out_dir: Path, manifest: dict | None, report: dict) -> list:
    """Write capabilities.json (+ compile report). Returns relative paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if manifest is not None:
        (out_dir / "capabilities.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        written.append("capabilities.json")
    (out_dir / "capability_compile_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    written.append("capability_compile_report.json")
    return written
