"""Capability linker — agent-generic state-aware selection of capability units.

This is the runtime half of the capability-linker MVP (HLD §9-11, plan §4.3/§4.6).
Given a ``NodeProfile`` (the code-generation node's current search state) and a
set of loaded skills that carry a validated ``capabilities.json`` manifest, it
runs the four-stage linking pipeline and renders a compact "## Linked ML
Capabilities" brief:

    A. deterministic hard filter (three-valued: false=reject, true=keep,
       unknown=pass to selector) — HLD §10.2-A / §10.3
    B. temperature-zero LLM semantic selection (select_capabilities func-call)
    C. deterministic dependency closure + topological order — HLD §10.2-C
    D. budgeted rendering (≤ max_units before closure, ≤ 1 reference file) — §10.2-D

AGENT-GENERIC BY CONSTRUCTION: this module imports nothing from ``mlevolve`` /
``agents`` / ``llm``-at-module-scope. ``capability_schema`` (sibling, stdlib-only)
and ``llm.FunctionSpec`` are imported lazily *inside* functions so any agent that
can build a ``NodeProfile`` and supply an ``llm_query`` callable can link. The
MLEvolve-specific glue (NodeProfile construction from an agent, the impl_guideline
seam, the llm.query closure) lives in ``skill_injector.py`` — never here.

Hard invariant: ``link()`` MUST NOT raise. Any internal error returns an empty
``LinkResult`` with an ``"error"`` telemetry field, so a broken manifest or a
selector transport failure can never break the host agent's code generation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# MLEvolve stage vocabulary -> canonical schema stage classes (schema §7.3 /
# capability_schema.CANONICAL_STAGES). Adapters for other agents supply their own
# mapping; this constant is the MLEvolve pilot's, kept here so the injector can
# reuse it without re-declaring the mapping.
STAGE_MAP = {
    "draft": "generate",
    "improve": "refine",
    "debug": "debug",
    "evolution": "explore",
}

# Runtime capabilities whose UNKNOWN status excludes the unit (HLD §10.3 security
# exception: "Security, credentials, and destructive-operation requirements ...
# unknown means the unit must not be represented as currently executable").
#
# Scoped to ``credentials`` deliberately. The W5 contract elaborates the exception
# as "credentials (or names a destructive/network-privileged need)", but the
# authoritative HLD §10.3 lists *security / credentials / destructive-operation* —
# not plain ``network`` or ``persistent-service``. In this schema's runtime vocab
# there is no "destructive" cap, and ``network`` denotes ordinary egress (a model
# download, an HF hub fetch) that legitimate units depend on — e.g.
# transformer-lens ``load-hooked-transformer`` (network) is the root of the entire
# interpretability chain, and ``vllm serve`` needs network+persistent-service.
# Auto-excluding those on the merely-unknown network fact would prune relevant
# procedures and fail the E2 "retain relevant procedures" gate. So ``network`` /
# ``persistent-service`` / ``multi-node`` are treated as ordinary three-valued
# unknowns (passed to the selector), and only ``credentials`` reject-on-unknown.
_SECURITY_RUNTIME = ("credentials",)

# Context caps for the selector prompt (mirror skill_injector's routing caps).
_TASK_CHARS = 6000
_PARENT_CHARS = 1500
_ANALYSIS_CHARS = 500


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeProfile:
    """The code-gen node's current search state, agent-agnostic.

    ``stage`` is a canonical stage (one of capability_schema.CANONICAL_STAGES).
    ``hardware`` is the parse_hardware() dict. ``runtime_capabilities`` is the set
    of runtime caps KNOWN to be PRESENT — absence from the set means UNKNOWN (NOT
    known-absent), so the hard filter can reason three-valued.
    """

    stage: str
    task_text: str = ""
    parent_code: str | None = None
    parent_error: str | None = None
    parent_analysis: str | None = None
    hardware: dict = field(default_factory=dict)
    runtime_capabilities: set = field(default_factory=set)


@dataclass
class LinkBudget:
    """Rendering budget (HLD §10.2-D). ``max_units`` is applied BEFORE dependency
    closure; ``max_reference_files`` caps full reference bodies across the whole
    brief; ``max_chars`` is an optional hard ceiling."""

    max_units: int = 3
    max_reference_files: int = 1
    max_chars: int | None = None


@dataclass
class LinkResult:
    """Linker output. ``rendered`` is the brief ("" when nothing injected);
    ``selected_ids`` are qualified ids ("skill/unit") in final topological order;
    ``telemetry`` carries the HLD §11 fields."""

    rendered: str = ""
    selected_ids: list = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hardware parsing (plan §4.3) — deterministic, never guesses
# ---------------------------------------------------------------------------

def parse_hardware(text: str) -> dict:
    """Parse the free-text ``MLEVAL_HARDWARE`` string into structured facts.

    Returns ``{"gpu_count": int|None, "vram_gb": float|None, "gpu_name": str|None,
    "raw": text}``. On any parse miss the field is ``None`` (UNKNOWN — never
    guessed, per HLD §10.3 three-valued logic). A recognisably CPU-only string
    ("... (no GPU)", "CPU only", "a CPU") yields ``gpu_count == 0``. Fully
    unparseable text yields all ``None``.

    Handles the MLE-bench mirror format ("1 NVIDIA RTX A6000 GPU (48 GB VRAM),
    8 CPUs, 32 GB RAM"), multi-GPU counts ("4x", "2 x", "8 x"), bare VRAM ("80GB"),
    and CPU-only markers. System "RAM" is never mistaken for VRAM.
    """
    result = {"gpu_count": None, "vram_gb": None, "gpu_name": None, "raw": ""}
    if not isinstance(text, str):
        return result
    result["raw"] = text
    if not text.strip():
        return result
    low = text.lower()

    # --- explicit CPU-only markers ---
    if re.search(r"\bno\s+gpus?\b", low) or "cpu-only" in low or "cpu only" in low:
        result["gpu_count"] = 0
        return result

    # --- GPU count ---
    count = None
    # multiplier form: "4x", "2 x", "8×" (followed by a letter/name)
    m = re.search(r"(?<![\w.])(\d+)\s*[x×](?=\s*[A-Za-z])", text)
    if m:
        count = int(m.group(1))
    else:
        # leading-count form: "1 NVIDIA ... GPU" — a number, then a name, then
        # "GPU", all within the same comma-delimited clause (so a trailing
        # "8 CPUs" clause is never taken as the GPU count).
        m = re.search(
            r"(?<![\w.])(\d+)\s+(?=[A-Za-z][^,]*?\bgpus?\b)", text, re.IGNORECASE
        )
        if m:
            count = int(m.group(1))

    # --- GPU name ---
    name = None
    m = re.search(
        r"([A-Za-z][A-Za-z0-9][A-Za-z0-9 .\-]*?)\s+gpus?\b", text, re.IGNORECASE
    )
    if m:
        name = m.group(1).strip()
    else:
        # "Nx <name>" without a literal "GPU" token: "4x A100 (40GB)".
        m = re.search(
            r"(?<![\w.])\d+\s*[x×]\s*([A-Za-z][A-Za-z0-9 .\-]*?)(?=\s*[(,]|$)",
            text,
        )
        if m:
            name = m.group(1).strip()
    if name:
        # strip a trailing "<num>GB" that got swept into the name ("A100 80GB").
        name = re.sub(r"\s*\d+(?:\.\d+)?\s*g(?:i)?b\b.*$", "", name, flags=re.IGNORECASE)
        name = name.strip() or None

    gpu_present = (
        (count is not None and count > 0)
        or name is not None
        or bool(re.search(r"\bgpus?\b", low))
    )

    if not gpu_present:
        # No GPU signal at all. A CPU mention => known CPU-only (0); else unknown.
        if "cpu" in low:
            result["gpu_count"] = 0
        return result

    result["gpu_count"] = count
    result["gpu_name"] = name

    # --- VRAM (only when a GPU is present, so system RAM is never captured) ---
    vram = None
    for pat in (
        r"(\d+(?:\.\d+)?)\s*g(?:i)?b\s*(?:of\s+)?vram",  # "48 GB VRAM", "80GB VRAM"
        r"vram[:=]?\s*(\d+(?:\.\d+)?)\s*g(?:i)?b",        # "VRAM: 48GB"
        r"\(\s*(\d+(?:\.\d+)?)\s*g(?:i)?b\b",             # "(80GB)" / "(48 GB VRAM)"
    ):
        mm = re.search(pat, low)
        if mm:
            vram = float(mm.group(1))
            break
    if vram is None:
        # bare "<num>GB" adjacent to the GPU, but never a system-RAM figure.
        mm = re.search(r"(\d+(?:\.\d+)?)\s*g(?:i)?b\b(?!\s*(?:ram|of\s+ram))", low)
        if mm:
            vram = float(mm.group(1))
    result["vram_gb"] = vram
    return result


# ---------------------------------------------------------------------------
# Unit registry
# ---------------------------------------------------------------------------

def _build_registry(cap_skills):
    """Flatten cap_skills into ordered unit records + a qid->record index.

    Each record: {qid, skill, skill_name, unit, local_id, dep_qids}. depends_on
    ids are LOCAL to a manifest (schema validation guarantees this), so they are
    qualified within the same skill.
    """
    units = []
    by_qid = {}
    for skill in cap_skills:
        sname = skill.get("name") or ""
        for unit in skill.get("capabilities") or []:
            if not isinstance(unit, dict):
                continue
            uid = unit.get("id")
            if not isinstance(uid, str):
                continue
            qid = f"{sname}/{uid}"
            rec = {
                "qid": qid,
                "skill": skill,
                "skill_name": sname,
                "unit": unit,
                "local_id": uid,
                "dep_qids": [f"{sname}/{d}" for d in (unit.get("depends_on") or [])],
            }
            units.append(rec)
            by_qid[qid] = rec
    return units, by_qid


# ---------------------------------------------------------------------------
# Stage A — hard compatibility filter (three-valued)
# ---------------------------------------------------------------------------

def _intrinsic_reason(rec, profile):
    """Return a machine reason code if this unit is KNOWN-incompatible on its own
    facts (ignoring dependencies), else None. Three-valued: only KNOWN-false facts
    reject; UNKNOWN never rejects (except the credentials security exception)."""
    unit = rec["unit"]
    stage = profile.stage
    hw = profile.hardware or {}
    gpu_count = hw.get("gpu_count")
    vram = hw.get("vram_gb")
    known_present = profile.runtime_capabilities or set()

    # stage allowlist (non-empty and excludes the current stage)
    stages = (unit.get("delivery") or {}).get("stages") or []
    if stages and stage not in stages:
        return "stage_excluded"

    requires = unit.get("requires") or {}
    runtime = requires.get("runtime") or []
    req_hw = requires.get("hardware") or {}

    # security exception: credentials (etc.) that are UNKNOWN => not executable
    for cap in runtime:
        if cap in _SECURITY_RUNTIME and cap not in known_present:
            return "security_unknown"

    min_gpu = req_hw.get("min_gpu_count")
    requires_gpu = (
        "gpu" in runtime
        or "multi-gpu" in runtime
        or (isinstance(min_gpu, int) and min_gpu >= 1)
        or (req_hw.get("vram_gb") is not None)
    )

    # a GPU need on a KNOWN CPU-only node
    if gpu_count == 0 and requires_gpu:
        return "cpu_only"

    # multi-gpu need on a KNOWN single-GPU node
    if "multi-gpu" in runtime and gpu_count is not None and gpu_count <= 1:
        return "cpu_only" if gpu_count == 0 else "gpu_count"

    # explicit min_gpu_count above the KNOWN count
    if isinstance(min_gpu, int) and gpu_count is not None and min_gpu > gpu_count:
        return "gpu_count"

    # VRAM: ONLY hard-filters from an explicit-section basis (plan §3.3). Advisory
    # / inferred VRAM never rejects.
    req_vram = req_hw.get("vram_gb")
    if (
        req_vram is not None
        and req_hw.get("basis") == "explicit-section"
        and vram is not None
        and req_vram > vram
    ):
        return "vram_gb"

    # other runtime caps: only reject when the fact is KNOWN-absent. We KNOW the
    # status of "python" (always present) and the gpu family (from hardware);
    # everything else (network / persistent-service / multi-node) is UNKNOWN here
    # and passes to the selector.
    known_runtime = {"python"}  # gpu/multi-gpu are handled above (the continue below)
    for cap in runtime:
        if cap in ("gpu", "multi-gpu") or cap in _SECURITY_RUNTIME:
            continue  # handled above
        if cap in known_runtime and cap not in known_present:
            return "runtime_absent"
    return None


def _hard_filter(units, by_qid, profile):
    """Return (survivors, rejected) where rejected: qid -> reason code.

    Intrinsic rejections first, then dependency propagation to a fixpoint: a unit
    whose required dependency is missing or itself hard-incompatible is rejected
    (dep_missing / dep_incompatible)."""
    rejected = {}
    for rec in units:
        reason = _intrinsic_reason(rec, profile)
        if reason:
            rejected[rec["qid"]] = reason

    changed = True
    while changed:
        changed = False
        for rec in units:
            qid = rec["qid"]
            if qid in rejected:
                continue
            for dep in rec["dep_qids"]:
                if dep not in by_qid:
                    rejected[qid] = "dep_missing"
                    changed = True
                    break
                if dep in rejected:
                    rejected[qid] = "dep_incompatible"
                    changed = True
                    break

    survivors = [rec for rec in units if rec["qid"] not in rejected]
    return survivors, rejected


# ---------------------------------------------------------------------------
# Stage B — semantic selection
# ---------------------------------------------------------------------------

def _build_selector_spec():
    """Build the select_capabilities FunctionSpec, or None if llm is unavailable
    (offline/replay stubs ignore the spec argument)."""
    try:
        try:
            from llm import FunctionSpec
        except ImportError:
            from llm.gemini import FunctionSpec
    except Exception:  # noqa: BLE001 — no llm available (offline): stub ignores spec
        return None
    return FunctionSpec(
        name="select_capabilities",
        description=(
            "Choose which capability units are relevant to the CURRENT coding "
            "sub-task and should be linked into the coder's context. Give a "
            "one-line reason per chosen unit. Return an empty list (with a "
            "decline_reason) if none apply."
        ),
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "capability_id": {"type": "string"},
                            "reason": {"type": "string"},
                            "reference_files": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["capability_id", "reason", "reference_files"],
                    },
                },
                "decline_reason": {"type": "string"},
            },
            "required": ["selections", "decline_reason"],
        },
    )


def _selector_catalog(survivors) -> str:
    lines = []
    for rec in survivors:
        u = rec["unit"]
        lines.append(f"- {rec['qid']}: {u.get('title', '')}")
        summ = u.get("summary")
        if summ:
            lines.append(f"    {summ}")
        appl = u.get("applicability") or {}
        when = appl.get("when") or []
        if when:
            lines.append("    apply when: " + "; ".join(when))
        avoid = appl.get("avoid_when") or []
        if avoid:
            lines.append("    avoid when: " + "; ".join(avoid))
    return "\n".join(lines)


def _selector_system(survivors) -> str:
    return (
        "You are a capability linker for an ML-engineering coding agent. Given "
        "the current coding sub-task and node state, choose which of the "
        "available capability units are RELEVANT and should be linked into the "
        "coder's context. Be selective: pick a unit only if it clearly helps the "
        "current sub-task, and respect each unit's 'avoid when' conditions. For "
        "each chosen unit give a one-line reason and optionally the reference "
        "files to load. Return an empty list — with a decline_reason — if none "
        "apply.\n\nAvailable capability units:\n" + _selector_catalog(survivors)
    )


def _selector_user(profile) -> str:
    parts = [f"Stage: {profile.stage or 'unknown'}"]
    hw = profile.hardware or {}
    if hw.get("raw"):
        parts.append(f"Compute: {hw['raw']}.")
    task = profile.task_text or ""
    parts.append(f"Task:\n{task[:_TASK_CHARS]}")
    if profile.stage == "debug":
        if profile.parent_error:
            parts.append(f"Error output (tail):\n{profile.parent_error[-_PARENT_CHARS:]}")
        if profile.parent_analysis:
            parts.append(f"Root-cause analysis:\n{profile.parent_analysis[:_ANALYSIS_CHARS]}")
    else:
        if profile.parent_code:
            parts.append(f"Current solution code (head):\n{profile.parent_code[:_PARENT_CHARS]}")
    return "\n\n".join(parts)


def _run_selection(profile, survivors, llm_query):
    """Call the selector (retry once on exception, then decline). Returns
    (selected, sel_refs, sel_reasons, decline_reason, selector_error).

    ``selected`` is an ordered list of survivor qids the selector picked;
    ``sel_refs`` maps qid -> selector-chosen reference filenames; ``sel_reasons``
    maps qid -> the selector's one-line reason. llm_query is None => skip (empty).
    NEVER inject-all on failure."""
    if llm_query is None:
        return [], {}, {}, "selector_skipped", None
    valid = {rec["qid"] for rec in survivors}
    system = _selector_system(survivors)
    user = _selector_user(profile)
    func_spec = _build_selector_spec()

    out = None
    last_exc = None
    for attempt in (1, 2):
        try:
            out = llm_query(system, user, func_spec)
            break
        except Exception as e:  # noqa: BLE001 — retry once, then decline (never all)
            last_exc = e
            logger.warning("[capability_linker] selector attempt %d failed: %s", attempt, e)
    if out is None:
        if last_exc is not None:  # both attempts raised — a genuine transport error
            return [], {}, {}, None, f"{type(last_exc).__name__}: {last_exc}"
        # selector returned None (not a dict) without raising — treat as a decline,
        # not a transport error (avoids a mislabeled "NoneType: None" in telemetry).
        return [], {}, {}, "selector_returned_none", None

    selections = out.get("selections", []) if isinstance(out, dict) else []
    decline_reason = out.get("decline_reason") if isinstance(out, dict) else None
    selected = []
    sel_refs = {}
    sel_reasons = {}
    seen = set()
    for sel in selections or []:
        if not isinstance(sel, dict):
            continue
        qid = sel.get("capability_id")
        if qid in valid and qid not in seen:
            seen.add(qid)
            selected.append(qid)
            refs = sel.get("reference_files")
            sel_refs[qid] = refs if isinstance(refs, list) else []
            sel_reasons[qid] = sel.get("reason") or ""
    return selected, sel_refs, sel_reasons, decline_reason, None


# ---------------------------------------------------------------------------
# Stage C — dependency closure + topological order
# ---------------------------------------------------------------------------

def _deps_resolvable(qid, by_qid, rejected, seen):
    """True iff every transitive dependency of qid is present and not hard-rejected.
    Accumulates the transitive deps into ``seen``."""
    for dep in by_qid[qid]["dep_qids"]:
        if dep not in by_qid or dep in rejected:
            return False
        if dep in seen:
            continue
        seen.add(dep)
        if not _deps_resolvable(dep, by_qid, rejected, seen):
            return False
    return True


def _dep_closure(roots, by_qid, rejected):
    """Return (order, dep_added, dep_dropped). Drops a root whose dependency is
    missing/incompatible; otherwise adds its deps and emits a deterministic
    topological (dependency-first) order."""
    root_set = set(roots)
    include = set()
    kept_roots = []
    dep_dropped = []
    for r in roots:
        seen = set()
        if _deps_resolvable(r, by_qid, rejected, seen):
            kept_roots.append(r)
            include.add(r)
            include |= seen
        else:
            dep_dropped.append(r)

    order = []
    placed = set()

    def visit(q):
        if q in placed:
            return
        placed.add(q)
        for dep in by_qid[q]["dep_qids"]:
            if dep in include:
                visit(dep)
        order.append(q)

    for r in kept_roots:
        visit(r)

    dep_added = [q for q in order if q not in root_set]
    return order, dep_added, dep_dropped


# ---------------------------------------------------------------------------
# Stage D — budgeted rendering
# ---------------------------------------------------------------------------

def _requirements_line(unit) -> str:
    requires = unit.get("requires") or {}
    parts = []
    runtime = requires.get("runtime") or []
    if runtime:
        parts.append("runtime: " + ", ".join(runtime))
    hw = requires.get("hardware") or {}
    hw_bits = []
    if hw.get("min_gpu_count"):
        hw_bits.append(f"min_gpu_count={hw['min_gpu_count']}")
    if hw.get("vram_gb") is not None:
        advisory = "" if hw.get("basis") == "explicit-section" else " (advisory)"
        hw_bits.append(f"vram_gb>={hw['vram_gb']}{advisory}")
    if hw_bits:
        parts.append("hardware: " + ", ".join(hw_bits))
    arts = requires.get("artifacts") or []
    if arts:
        parts.append("artifacts: " + ", ".join(arts))
    return "; ".join(parts)


def _render_unit_block(qid, unit, reason) -> str:
    lines = [f"### {qid}"]
    if reason:
        lines.append(f"Why linked: {reason}")
    appl = unit.get("applicability") or {}
    when = appl.get("when") or []
    if when:
        lines.append("Apply when: " + "; ".join(when))
    avoid = appl.get("avoid_when") or []
    if avoid:
        lines.append("Do not apply when: " + "; ".join(avoid))
    reqline = _requirements_line(unit)
    if reqline:
        lines.append("Requirements: " + reqline)
    steps = (unit.get("procedure") or {}).get("steps") or []
    if steps:
        lines.append("Procedure:")
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s}")
    validation = unit.get("validation") or []
    if validation:
        lines.append("Validate:")
        for v in validation:
            lines.append(f"- {v}")
    recovery = unit.get("recovery") or []
    if recovery:
        lines.append("Recovery:")
        for r in recovery:
            lines.append(f"- {r}")
    sources = unit.get("sources") or []
    if sources and isinstance(sources[0], dict) and sources[0].get("locator"):
        lines.append(f"Source: {sources[0]['locator']}")
    return "\n".join(lines)


def _gather_references(order, by_qid, sel_refs, max_refs):
    """Pick up to ``max_refs`` reference-file bodies total, in rendered order.

    Two global passes so the selector's EXPLICIT choice wins over a unit's
    incidental ``procedure.reference_files`` (including a pulled-in dependency's):
    first collect selector-chosen files across all units, then fall back to
    procedure files. Returns a list of (qid, filename, body)."""
    if max_refs <= 0:
        return []
    chosen = []
    picked = set()  # (qid, basename) already taken

    def _collect(getter):
        for qid in order:
            if len(chosen) >= max_refs:
                return
            rec = by_qid[qid]
            refsmap = rec["skill"].get("references") or {}
            for fn in getter(rec):
                base = str(fn).split("/")[-1]
                key = (qid, base)
                if base in refsmap and key not in picked:
                    picked.add(key)
                    chosen.append((qid, base, refsmap[base]))
                    break  # at most one file per unit

    # pass 1: selector-chosen reference files (strong signal)
    _collect(lambda rec: sel_refs.get(rec["qid"], []))
    # pass 2: unit procedure.reference_files (fallback)
    _collect(lambda rec: (rec["unit"].get("procedure") or {}).get("reference_files") or [])
    return chosen[:max_refs]


def _render_brief(order, reasons, refs, by_qid) -> str:
    if not order:
        return ""
    blocks = ["## Linked ML Capabilities"]
    for qid in order:
        blocks.append(_render_unit_block(qid, by_qid[qid]["unit"], reasons.get(qid, "")))
    for qid, fn, body in refs:
        blocks.append(f"#### Reference ({qid}): references/{fn}\n\n{body}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def link(profile, cap_skills, llm_query, budget=None) -> LinkResult:
    """Run the four-stage linker for one code-generation node.

    See module docstring. NEVER raises: on any internal error returns an empty
    LinkResult carrying a telemetry ``"error"`` field.
    """
    if budget is None:
        budget = LinkBudget()
    try:
        return _link_impl(profile, cap_skills, llm_query, budget)
    except Exception as e:  # noqa: BLE001 — a capability arm must never break codegen
        logger.warning("[capability_linker] link failed: %s", e)
        try:
            schema_version = _schema_version()
        except Exception:  # noqa: BLE001
            schema_version = None
        return LinkResult(
            rendered="",
            selected_ids=[],
            telemetry={
                "error": f"{type(e).__name__}: {e}",
                "capability_schema_version": schema_version,
                "legacy_fallback": False,
            },
        )


def _schema_version():
    from .capability_schema import SCHEMA_VERSION
    return SCHEMA_VERSION


def _link_impl(profile, cap_skills, llm_query, budget) -> LinkResult:
    cap_skills = list(cap_skills or [])
    units, by_qid = _build_registry(cap_skills)

    telemetry = {
        "capability_schema_version": _schema_version(),
        "stage": profile.stage,
        "hardware": dict(profile.hardware or {}),
        "runtime_capabilities": sorted(profile.runtime_capabilities or set()),
        "loaded_skill_count": len(cap_skills),
        "loaded_capability_count": len(units),
        "candidate_ids": [],
        "hard_filtered": [],
        "selector_selected_ids": [],
        "decline_reason": None,
        "selector_error": None,
        "dep_added_ids": [],
        "dep_dropped_ids": [],
        "rendered_order": [],
        "selected_reference_files": [],
        "injected_chars": 0,
        "truncated": False,
        "legacy_fallback": False,
    }

    if not units:
        return LinkResult(rendered="", selected_ids=[], telemetry=telemetry)

    # --- A. hard filter ---
    survivors, rejected = _hard_filter(units, by_qid, profile)
    telemetry["candidate_ids"] = [rec["qid"] for rec in survivors]
    telemetry["hard_filtered"] = [
        {"id": rec["qid"], "reason": rejected[rec["qid"]]}
        for rec in units
        if rec["qid"] in rejected
    ]
    if not survivors:
        telemetry["decline_reason"] = "no_candidates"
        return LinkResult(rendered="", selected_ids=[], telemetry=telemetry)

    # --- B. semantic selection ---
    selected, sel_refs, sel_reasons, decline_reason, selector_error = _run_selection(
        profile, survivors, llm_query
    )
    telemetry["selector_selected_ids"] = list(selected)
    telemetry["decline_reason"] = decline_reason
    telemetry["selector_error"] = selector_error
    if not selected:
        return LinkResult(rendered="", selected_ids=[], telemetry=telemetry)

    # --- D(pre). budget max_units BEFORE closure ---
    roots = list(selected)
    if len(roots) > budget.max_units:
        roots = roots[: budget.max_units]
        telemetry["truncated"] = True

    # --- C. dependency closure ---
    order, dep_added, dep_dropped = _dep_closure(roots, by_qid, rejected)
    telemetry["dep_added_ids"] = list(dep_added)
    telemetry["dep_dropped_ids"] = list(dep_dropped)

    def _reason_for(qid):
        # Selector reason for a chosen root; a generic marker for a pulled-in dep.
        return sel_reasons.get(qid) or "Required as a dependency of a linked capability."

    # --- D. budgeted rendering with max_chars enforcement ---
    include_ref = budget.max_reference_files > 0
    rendered = ""
    final_order = order
    selected_refs_meta = []
    while True:
        reasons = {qid: _reason_for(qid) for qid in final_order}
        refs = (
            _gather_references(final_order, by_qid, sel_refs, budget.max_reference_files)
            if include_ref
            else []
        )
        rendered = _render_brief(final_order, reasons, refs, by_qid)
        selected_refs_meta = [f"{qid}#{fn}" for qid, fn, _ in refs]
        if budget.max_chars is None or len(rendered) <= budget.max_chars or not final_order:
            break
        # over budget: drop the optional reference first, then lowest-ranked root.
        if include_ref and refs:
            include_ref = False
            telemetry["truncated"] = True
            continue
        telemetry["truncated"] = True
        if len(roots) <= 1:
            roots = []
            final_order = []
            selected_refs_meta = []
            rendered = ""
            break
        roots = roots[:-1]
        final_order, dep_added, _ = _dep_closure(roots, by_qid, rejected)
        telemetry["dep_added_ids"] = list(dep_added)

    telemetry["rendered_order"] = list(final_order)
    telemetry["selected_reference_files"] = selected_refs_meta
    telemetry["injected_chars"] = len(rendered)
    return LinkResult(rendered=rendered, selected_ids=list(final_order), telemetry=telemetry)


__all__ = [
    "STAGE_MAP",
    "NodeProfile",
    "LinkBudget",
    "LinkResult",
    "parse_hardware",
    "link",
]
