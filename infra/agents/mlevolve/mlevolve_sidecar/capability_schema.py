"""Capability manifest schema 0.2 — definition + deterministic validator.

Single source of truth for ``capabilities.json`` (see
docs/skill-builder/capability-linker-mvp-plan.md §4.2 and the HLD §7).
Used at build time (builder gate) and at runtime (loader); therefore
stdlib-only, no LLM, no third-party deps — same constraint as the rest of
the sidecar.

Validation contract (HLD §8.3): hard errors make a manifest unloadable in
capability modes; warnings are recorded but do not block. Security scanning
of text content is the builder's job (it reuses the scout scanner); this
module only runs a cheap injection-marker screen so a hand-dropped manifest
cannot bypass every check.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = "0.2"

KINDS = ("reference", "procedure", "diagnostic", "workflow")

# Canonical, agent-agnostic stage classes. Adapters map native vocabulary
# onto these (MLEvolve: draft->generate, improve->refine, debug->debug,
# evolution->explore).
CANONICAL_STAGES = ("generate", "refine", "debug", "explore")

# Runtime facts the harness can actually know (HLD §7.4).
RUNTIME_VOCAB = (
    "python",
    "gpu",
    "multi-gpu",
    "multi-node",
    "network",
    "persistent-service",
    "credentials",
)

SOURCE_KINDS = ("skill-package", "upstream-doc")

HARDWARE_BASIS = ("explicit-section", "inferred")

# Field paths a compiler may declare as synthesized (authored without direct
# textual support in the source). Audited at a stricter bar in E1.
SYNTHESIZABLE_FIELDS = (
    "applicability.avoid_when",
    "validation",
    "recovery",
    "expected_outcomes",
    "produces",
    "requires.runtime",
    "requires.hardware",
    "requires.artifacts",
    "depends_on",
    "delivery.stages",
)

_ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_ARTIFACT_RE = re.compile(r"^[a-z0-9_-]+:[a-z0-9._/-]+$")

# Cheap screen only; the builder's real IPI/security scanner runs at build.
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your instructions",
    "you are now",
)


def _str_list(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(x, str) and x.strip() for x in value
    )


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_strings(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _walk_strings(item)


def _check_cycles(dep_graph: dict) -> list:
    """Return one representative cycle path per detected cycle."""
    cycles = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in dep_graph}

    def dfs(node, path):
        color[node] = GRAY
        for dep in dep_graph.get(node, ()):
            if dep not in color:
                continue  # unknown id: reported separately
            if color[dep] == GRAY:
                cycles.append(path + [dep])
            elif color[dep] == WHITE:
                dfs(dep, path + [dep])
        color[node] = BLACK

    for node in dep_graph:
        if color[node] == WHITE:
            dfs(node, [node])
    return cycles


def _validate_unit(unit, index, ids, skill_dir, errors, warnings):
    where = f"capabilities[{index}]"
    if not isinstance(unit, dict):
        errors.append(f"{where}: unit must be an object")
        return

    uid = unit.get("id")
    where = f"capabilities[{index}] ({uid or '?'})"

    # --- identity ---
    if not isinstance(uid, str) or not _ID_RE.match(uid or ""):
        errors.append(f"{where}: 'id' must be kebab-case ([a-z0-9-])")
    kind = unit.get("kind")
    if kind not in KINDS:
        errors.append(f"{where}: 'kind' must be one of {KINDS}, got {kind!r}")
    for field in ("title", "summary"):
        if not isinstance(unit.get(field), str) or not unit[field].strip():
            errors.append(f"{where}: '{field}' is required and must be non-empty")

    # --- applicability ---
    applicability = unit.get("applicability")
    if not isinstance(applicability, dict) or not _str_list(
        applicability.get("when")
    ) or not applicability.get("when"):
        errors.append(f"{where}: 'applicability.when' requires >=1 condition")
    else:
        avoid = applicability.get("avoid_when")
        if avoid is not None and not _str_list(avoid):
            errors.append(f"{where}: 'applicability.avoid_when' must be a string list")

    # --- requires ---
    requires = unit.get("requires", {})
    if requires and not isinstance(requires, dict):
        errors.append(f"{where}: 'requires' must be an object")
        requires = {}
    runtime = requires.get("runtime", [])
    if runtime:
        if not _str_list(runtime):
            errors.append(f"{where}: 'requires.runtime' must be a string list")
        else:
            for entry in runtime:
                if entry not in RUNTIME_VOCAB:
                    errors.append(
                        f"{where}: unknown runtime capability {entry!r} "
                        f"(vocab: {RUNTIME_VOCAB})"
                    )
    hardware = requires.get("hardware", {})
    if hardware:
        if not isinstance(hardware, dict):
            errors.append(f"{where}: 'requires.hardware' must be an object")
        else:
            mgc = hardware.get("min_gpu_count")
            if mgc is not None and (not isinstance(mgc, int) or mgc < 1):
                errors.append(f"{where}: 'hardware.min_gpu_count' must be int >= 1")
            vram = hardware.get("vram_gb")
            if vram is not None:
                if not isinstance(vram, (int, float)) or vram <= 0:
                    errors.append(f"{where}: 'hardware.vram_gb' must be > 0")
                basis = hardware.get("basis")
                if basis not in HARDWARE_BASIS:
                    errors.append(
                        f"{where}: 'hardware.vram_gb' requires 'basis' in "
                        f"{HARDWARE_BASIS}"
                    )
                elif basis == "inferred":
                    warnings.append(
                        f"{where}: vram_gb has basis=inferred — linker treats it "
                        "as advisory, never a hard filter"
                    )
    for field in ("artifacts",):
        arts = requires.get(field, [])
        if arts:
            if not _str_list(arts):
                errors.append(f"{where}: 'requires.{field}' must be a string list")
            else:
                for art in arts:
                    if not _ARTIFACT_RE.match(art):
                        errors.append(
                            f"{where}: malformed artifact {art!r} "
                            "(expected 'namespace:name')"
                        )
                    elif not art.startswith("ml:"):
                        warnings.append(
                            f"{where}: library-specific artifact namespace {art!r}"
                        )

    # --- produces ---
    produces = unit.get("produces", [])
    if produces:
        if not _str_list(produces):
            errors.append(f"{where}: 'produces' must be a string list")
        else:
            for art in produces:
                if not _ARTIFACT_RE.match(art):
                    errors.append(
                        f"{where}: malformed produced artifact {art!r} "
                        "(expected 'namespace:name')"
                    )
    if kind == "diagnostic" and not produces:
        errors.append(
            f"{where}: diagnostic units must produce an observable "
            "output/report artifact"
        )

    # --- procedure / reference pointer ---
    procedure = unit.get("procedure", {})
    if procedure and not isinstance(procedure, dict):
        errors.append(f"{where}: 'procedure' must be an object")
        procedure = {}
    steps = procedure.get("steps", [])
    ref_files = procedure.get("reference_files", [])
    scripts = procedure.get("scripts", [])
    for name, val in (("steps", steps), ("reference_files", ref_files),
                      ("scripts", scripts)):
        if val and not _str_list(val):
            errors.append(f"{where}: 'procedure.{name}' must be a string list")
    if not steps and not ref_files:
        errors.append(
            f"{where}: needs inline 'procedure.steps' or a "
            "'procedure.reference_files' pointer"
        )
    if skill_dir is not None:
        for rel in list(ref_files) + list(scripts):
            if isinstance(rel, str) and not (Path(skill_dir) / rel).is_file():
                errors.append(f"{where}: missing file {rel!r} under {skill_dir}")

    # --- validation / recovery ---
    validation = unit.get("validation", [])
    if validation and not _str_list(validation):
        errors.append(f"{where}: 'validation' must be a string list")
    if kind in ("procedure", "workflow") and not validation:
        errors.append(f"{where}: {kind} units require >=1 validation condition")
    recovery = unit.get("recovery", [])
    if recovery and not _str_list(recovery):
        errors.append(f"{where}: 'recovery' must be a string list")
    if kind in ("procedure", "workflow") and not recovery:
        warnings.append(f"{where}: no recovery guidance")

    # --- delivery ---
    delivery = unit.get("delivery", {})
    if delivery:
        stages = delivery.get("stages", []) if isinstance(delivery, dict) else None
        if stages is None or not _str_list(stages):
            errors.append(f"{where}: 'delivery.stages' must be a string list")
        else:
            for stage in stages:
                if stage not in CANONICAL_STAGES:
                    errors.append(
                        f"{where}: unknown stage {stage!r} "
                        f"(canonical: {CANONICAL_STAGES})"
                    )

    # --- sources ---
    sources = unit.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append(f"{where}: >=1 source locator required")
    else:
        for j, src in enumerate(sources):
            if not isinstance(src, dict):
                errors.append(f"{where}: sources[{j}] must be an object")
                continue
            if src.get("kind") not in SOURCE_KINDS:
                errors.append(
                    f"{where}: sources[{j}].kind must be in {SOURCE_KINDS}"
                )
            if not isinstance(src.get("locator"), str) or not src["locator"].strip():
                errors.append(f"{where}: sources[{j}].locator is required")
            if src.get("kind") == "upstream-doc" and "upstream_verified" not in src:
                warnings.append(
                    f"{where}: sources[{j}] upstream-doc without "
                    "'upstream_verified' flag"
                )

    # --- synthesized_fields ---
    synthesized = unit.get("synthesized_fields", [])
    if synthesized:
        if not _str_list(synthesized):
            errors.append(f"{where}: 'synthesized_fields' must be a string list")
        else:
            for field in synthesized:
                if field not in SYNTHESIZABLE_FIELDS:
                    errors.append(
                        f"{where}: {field!r} is not a synthesizable field "
                        f"(allowed: {SYNTHESIZABLE_FIELDS})"
                    )

    # --- depends_on shape (graph checks happen at manifest level) ---
    deps = unit.get("depends_on", [])
    if deps and not _str_list(deps):
        errors.append(f"{where}: 'depends_on' must be a string list of unit ids")


def validate_manifest(manifest, skill_dir=None):
    """Validate a parsed capabilities.json object.

    Returns (errors, warnings): two lists of strings. Empty errors == valid.
    ``skill_dir`` (path-like) enables reference/script file-existence checks.
    """
    errors, warnings = [], []

    if not isinstance(manifest, dict):
        return (["manifest must be a JSON object"], warnings)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("skill_name"), str) or not manifest.get(
        "skill_name", ""
    ).strip():
        errors.append("'skill_name' is required")

    snapshot = manifest.get("source_snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("url"):
        errors.append("'source_snapshot' with a 'url' is required")
    elif snapshot.get("kind") not in SOURCE_KINDS:
        errors.append(f"'source_snapshot.kind' must be in {SOURCE_KINDS}")

    units = manifest.get("capabilities")
    if not isinstance(units, list) or not units:
        errors.append("'capabilities' must be a non-empty list")
        return (errors, warnings)

    ids = [u.get("id") for u in units if isinstance(u, dict)]
    for uid in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate capability id {uid!r}")

    known = set(ids)
    for i, unit in enumerate(units):
        _validate_unit(unit, i, known, skill_dir, errors, warnings)
        for dep in (unit.get("depends_on", []) if isinstance(unit, dict) else []):
            if dep not in known:
                errors.append(
                    f"capabilities[{i}]: depends_on unknown capability {dep!r}"
                )

    dep_graph = {
        u["id"]: [d for d in u.get("depends_on", []) if d in known]
        for u in units
        if isinstance(u, dict) and isinstance(u.get("id"), str)
    }
    for cycle in _check_cycles(dep_graph):
        errors.append("dependency cycle: " + " -> ".join(cycle))

    lowered = [(s, s.lower()) for s in _walk_strings(manifest)]
    for original, low in lowered:
        for marker in _INJECTION_MARKERS:
            if marker in low:
                errors.append(
                    f"injection marker {marker!r} found in manifest text: "
                    f"{original[:80]!r}"
                )

    return (errors, warnings)


def load_and_validate(path, skill_dir=None):
    """Parse + validate a capabilities.json file. Returns (manifest|None, errors, warnings)."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (None, [f"unreadable manifest: {exc}"], [])
    errors, warnings = validate_manifest(manifest, skill_dir=skill_dir)
    return (manifest if not errors else None, errors, warnings)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    skill_dir = None
    if "--skill-dir" in argv:
        i = argv.index("--skill-dir")
        skill_dir = argv[i + 1]
        del argv[i : i + 2]
    if not argv:
        print("usage: capability_schema.py <capabilities.json>... [--skill-dir DIR]")
        return 2
    exit_code = 0
    for path in argv:
        manifest, errors, warnings = load_and_validate(path, skill_dir=skill_dir)
        n_units = len(manifest["capabilities"]) if manifest else 0
        status = "VALID" if not errors else "INVALID"
        print(f"{path}: {status} ({n_units} units)")
        for err in errors:
            print(f"  ERROR: {err}")
            exit_code = 1
        for warn in warnings:
            print(f"  warning: {warn}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
