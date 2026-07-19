"""Tests for mlevolve_sidecar/capability_linker.py (the runtime linker, HLD §9-11).

``capability_linker`` is agent-generic and imports nothing from MLEvolve at
module scope, but its ``_schema_version()`` does a *relative* import
(``from .capability_schema import SCHEMA_VERSION``). So it is loaded under a
lightweight stub ``mlevolve_sidecar`` package that only carries a ``__path__``
(pointing at the real sidecar dir) — enough for intra-package relative imports to
resolve to the real, stdlib-only ``capability_schema.py``. No upstream MLEvolve
stub is needed for the linker core (unlike the injector).

Coverage: parse_hardware · STAGE_MAP · three-valued hard filter (every reason
code) · dependency closure (topo order + drops) · budgeted rendering (max_units
before closure, ≤1 reference file, section format, avoid-when omission) · link()
with a stub selector (select / decline / retry-then-decline / None / bad-unit /
never-raises) · a real E1 manifest smoke across all four canonical stages.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SIDECAR = _REPO / "infra/agents/mlevolve/mlevolve_sidecar"
_E1 = _REPO / "docs/skill-builder/capability-mvp/e1"


def _load(sub):
    """Load a sidecar submodule by file path under a stub parent package so that
    relative imports (``from .capability_schema import ...``) resolve."""
    pkg = "mlevolve_sidecar"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(_SIDECAR)]
        sys.modules[pkg] = m
    full = f"{pkg}.{sub}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _SIDECAR / f"{sub}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load("capability_linker")
cs = _load("capability_schema")
CANONICAL_STAGES = cs.CANONICAL_STAGES


# --------------------------------------------------------------------------- #
# Fixture builders — retriever-shaped skill dicts and units.
# --------------------------------------------------------------------------- #

def _unit(uid, **over):
    u = {
        "id": uid,
        "kind": "reference",
        "title": f"Title {uid}",
        "summary": f"Summary of {uid}",
        "applicability": {"when": [f"when {uid} applies"]},
        "procedure": {"steps": [f"do {uid}"]},
        "sources": [{"kind": "skill-package", "locator": f"SKILL.md#{uid}"}],
    }
    u.update(over)
    return u


def _skill(name, units, references=None, reference_files=None):
    return {
        "name": name,
        "body": "SKILL BODY - must never appear in a capability brief",
        "references": references or {},
        "reference_files": reference_files or [],
        "capabilities": list(units),
    }


def _profile(stage="generate", gpu_count=1, vram=48.0, caps=None):
    caps = {"python", "gpu"} if caps is None else caps
    hw = {
        "gpu_count": gpu_count,
        "vram_gb": vram,
        "gpu_name": "A6000",
        "raw": "1 NVIDIA RTX A6000 GPU (48 GB VRAM)",
    }
    return cl.NodeProfile(
        stage=stage,
        task_text="Fine-tune a causal LM with LoRA on GSM8K.",
        hardware=hw,
        runtime_capabilities=set(caps),
    )


def _stub(*qids, decline=""):
    """A selector stub: returns the given qids (in order) as selections."""
    def q(system_message, user_message, func_spec):
        return {
            "selections": [
                {"capability_id": qid, "reason": f"reason for {qid}", "reference_files": []}
                for qid in qids
            ],
            "decline_reason": decline,
        }
    return q


def _filter(profile, *units, name="s"):
    recs, by = cl._build_registry([_skill(name, list(units))])
    survivors, rejected = cl._hard_filter(recs, by, profile)
    return {r["qid"] for r in survivors}, rejected


def _reg(*units, name="s"):
    return cl._build_registry([_skill(name, list(units))])


def _candidates(profile, skills):
    recs, by = cl._build_registry(skills)
    survivors, _ = cl._hard_filter(recs, by, profile)
    return [r["qid"] for r in survivors]


# --------------------------------------------------------------------------- #
# parse_hardware
# --------------------------------------------------------------------------- #

def test_parse_hardware_a6000():
    hw = cl.parse_hardware("1 NVIDIA RTX A6000 GPU (48 GB VRAM), 8 CPUs, 32 GB RAM")
    assert hw["gpu_count"] == 1
    assert hw["vram_gb"] == 48.0
    assert hw["gpu_name"] and "A6000" in hw["gpu_name"]
    assert hw["raw"].startswith("1 NVIDIA")


def test_parse_hardware_multi_gpu():
    hw = cl.parse_hardware("4x A100 (80 GB)")
    assert hw["gpu_count"] == 4
    assert hw["vram_gb"] == 80.0
    assert hw["gpu_name"] == "A100"  # trailing GB never swept into the name


def test_parse_hardware_cpu_only():
    for text in ("16 CPUs, 64 GB RAM (no GPU)", "CPU only", "cpu-only, 32 GB RAM"):
        hw = cl.parse_hardware(text)
        assert hw["gpu_count"] == 0, text
        assert hw["gpu_name"] is None


def test_parse_hardware_garbage_all_none():
    hw = cl.parse_hardware("lorem ipsum dolor sit amet")
    assert hw["gpu_count"] is None
    assert hw["vram_gb"] is None
    assert hw["gpu_name"] is None
    assert hw["raw"] == "lorem ipsum dolor sit amet"


def test_parse_hardware_empty_and_non_string():
    for bad in ("", "   ", None, 123):
        hw = cl.parse_hardware(bad)
        assert hw["gpu_count"] is None and hw["vram_gb"] is None and hw["gpu_name"] is None


# --------------------------------------------------------------------------- #
# STAGE_MAP
# --------------------------------------------------------------------------- #

def test_stage_map():
    assert cl.STAGE_MAP == {
        "draft": "generate",
        "improve": "refine",
        "debug": "debug",
        "evolution": "explore",
    }
    for v in cl.STAGE_MAP.values():
        assert v in CANONICAL_STAGES


# --------------------------------------------------------------------------- #
# Stage A — three-valued hard filter (every reason code)
# --------------------------------------------------------------------------- #

def test_hard_filter_stage_excluded():
    p = _profile(stage="generate")
    surv, rej = _filter(p, _unit("deb", delivery={"stages": ["debug"]}))
    assert "s/deb" not in surv
    assert rej["s/deb"] == "stage_excluded"


def test_hard_filter_no_stage_allowlist_is_unknown_keep():
    p = _profile(stage="generate")
    surv, rej = _filter(p, _unit("any"))  # no delivery.stages -> all stages allowed
    assert "s/any" in surv and "s/any" not in rej


def test_hard_filter_min_gpu_count_reject_only_when_known():
    u = _unit("big", requires={"hardware": {"min_gpu_count": 4}})
    # known single-GPU node -> reject
    _, rej = _filter(_profile(gpu_count=1), u)
    assert rej["s/big"] == "gpu_count"
    # unknown gpu_count -> do NOT hard-filter
    surv, rej2 = _filter(_profile(gpu_count=None, caps={"python"}), u)
    assert "s/big" in surv and "s/big" not in rej2


def test_hard_filter_vram_rejects_only_explicit_section():
    p = _profile(gpu_count=1, vram=48.0)
    # explicit-section basis + over budget -> reject
    _, rej = _filter(p, _unit("ve", requires={"hardware": {"vram_gb": 80, "basis": "explicit-section"}}))
    assert rej["s/ve"] == "vram_gb"
    # inferred basis -> advisory, never rejects
    surv, rej2 = _filter(p, _unit("vi", requires={"hardware": {"vram_gb": 80, "basis": "inferred"}}))
    assert "s/vi" in surv and "s/vi" not in rej2
    # no basis -> advisory, never rejects
    surv2, rej3 = _filter(p, _unit("vn", requires={"hardware": {"vram_gb": 80}}))
    assert "s/vn" in surv2 and "s/vn" not in rej3
    # unknown vram on the node -> do not reject even with explicit-section
    surv3, rej4 = _filter(_profile(gpu_count=1, vram=None), _unit("ve2", requires={"hardware": {"vram_gb": 80, "basis": "explicit-section"}}))
    assert "s/ve2" in surv3 and "s/ve2" not in rej4


def test_hard_filter_unknown_runtime_does_not_reject():
    p = _profile(gpu_count=1, caps={"python", "gpu"})
    surv, rej = _filter(p, _unit("net", requires={"runtime": ["python", "network"]}))
    assert "s/net" in surv and "s/net" not in rej


def test_hard_filter_known_absent_runtime_rejects():
    # python is a KNOWN cap; a profile that does not carry it -> known-absent reject.
    p = cl.NodeProfile(stage="generate", hardware={"gpu_count": 1}, runtime_capabilities=set())
    _, rej = _filter(p, _unit("needpy", requires={"runtime": ["python"]}))
    assert rej["s/needpy"] == "runtime_absent"


def test_hard_filter_cpu_only_rejects_gpu_unit():
    p = _profile(gpu_count=0, vram=None, caps={"python"})
    _, rej = _filter(p, _unit("g", requires={"runtime": ["python", "gpu"]}))
    assert rej["s/g"] == "cpu_only"


def test_hard_filter_multi_gpu_on_single_gpu():
    p = _profile(gpu_count=1, caps={"python", "gpu"})
    _, rej = _filter(p, _unit("mg", requires={"runtime": ["python", "multi-gpu"]}))
    assert rej["s/mg"] == "gpu_count"


def test_hard_filter_credentials_unknown_security_exception():
    p = _profile(gpu_count=1, caps={"python", "gpu"})
    u = _unit("cred", requires={"runtime": ["python", "credentials"]})
    _, rej = _filter(p, u)
    assert rej["s/cred"] == "security_unknown"
    # KNOWN-present credentials -> the unit is executable, keep it
    p2 = _profile(gpu_count=1, caps={"python", "gpu", "credentials"})
    surv2, rej2 = _filter(p2, u)
    assert "s/cred" in surv2 and "s/cred" not in rej2


def test_hard_filter_dep_missing_and_incompatible():
    # dependency absent from the library
    _, rej = _filter(_profile(), _unit("root", depends_on=["ghost"]))
    assert rej["s/root"] == "dep_missing"
    # dependency present but itself hard-incompatible (stage-excluded)
    p = _profile(stage="generate")
    _, rej2 = _filter(p, _unit("r2", depends_on=["child"]), _unit("child", delivery={"stages": ["debug"]}))
    assert rej2["s/child"] == "stage_excluded"
    assert rej2["s/r2"] == "dep_incompatible"


# --------------------------------------------------------------------------- #
# Stage C — dependency closure
# --------------------------------------------------------------------------- #

def test_dep_closure_adds_dep_topo_order():
    _, by = _reg(_unit("a", depends_on=["b"]), _unit("b"))
    order, added, dropped = cl._dep_closure(["s/a"], by, {})
    assert order == ["s/b", "s/a"]  # dependency before dependent
    assert added == ["s/b"] and dropped == []


def test_dep_closure_drops_on_missing_dep():
    _, by = _reg(_unit("a", depends_on=["missing"]))
    order, added, dropped = cl._dep_closure(["s/a"], by, {})
    assert order == [] and added == [] and dropped == ["s/a"]


def test_dep_closure_drops_on_incompatible_dep():
    _, by = _reg(_unit("a", depends_on=["b"]), _unit("b"))
    order, added, dropped = cl._dep_closure(["s/a"], by, {"s/b": "stage_excluded"})
    assert order == [] and dropped == ["s/a"]


# --------------------------------------------------------------------------- #
# Stage D — budgeted rendering
# --------------------------------------------------------------------------- #

def test_budget_max_units_before_closure():
    skill = _skill("s", [_unit(f"u{i}") for i in range(3)])
    res = cl.link(_profile(), [skill], _stub("s/u0", "s/u1", "s/u2"), cl.LinkBudget(max_units=2))
    assert res.selected_ids == ["s/u0", "s/u1"]  # capped to 2 roots, no deps
    assert res.telemetry["truncated"] is True


def test_at_most_one_reference_file():
    u1 = _unit("one", procedure={"steps": ["s1"], "reference_files": ["references/one.md"]})
    u2 = _unit("two", procedure={"steps": ["s2"], "reference_files": ["references/two.md"]})
    skill = _skill("s", [u1, u2], references={"one.md": "ONE-BODY", "two.md": "TWO-BODY"})
    res = cl.link(_profile(), [skill], _stub("s/one", "s/two"),
                  cl.LinkBudget(max_units=3, max_reference_files=1))
    assert res.rendered.count("#### Reference") == 1
    assert len(res.telemetry["selected_reference_files"]) == 1
    assert ("ONE-BODY" in res.rendered) ^ ("TWO-BODY" in res.rendered)
    assert "SKILL BODY" not in res.rendered  # never dumps the SKILL.md body


def test_render_format_has_all_headed_sections():
    u = _unit(
        "full",
        kind="procedure",
        applicability={"when": ["cond A"], "avoid_when": ["cond B"]},
        requires={"runtime": ["python", "gpu"], "hardware": {"min_gpu_count": 1}, "artifacts": ["ml:dataset"]},
        procedure={"steps": ["step one", "step two"]},
        validation=["check X"],
        recovery=["do Y"],
    )
    res = cl.link(_profile(), [_skill("s", [u])], _stub("s/full"))
    r = res.rendered
    for token in (
        "## Linked ML Capabilities",
        "### s/full",
        "Why linked:",
        "Apply when: cond A",
        "Do not apply when: cond B",
        "Requirements:",
        "Procedure:",
        "1. step one",
        "2. step two",
        "Validate:",
        "- check X",
        "Recovery:",
        "- do Y",
        "Source: SKILL.md#full",
    ):
        assert token in r, f"missing: {token!r}"


def test_avoid_when_line_omitted_when_empty():
    u = _unit("noavoid", applicability={"when": ["only condition"]})
    block = cl._render_unit_block("s/noavoid", u, "because reasons")
    assert "Apply when: only condition" in block
    assert "Do not apply when:" not in block


# --------------------------------------------------------------------------- #
# link() with a stub selector
# --------------------------------------------------------------------------- #

def test_link_selects_and_renders():
    res = cl.link(_profile(), [_skill("s", [_unit("pick")])], _stub("s/pick"))
    assert res.rendered.startswith("## Linked ML Capabilities")
    assert "s/pick" in res.rendered
    assert res.selected_ids == ["s/pick"]
    assert res.telemetry["selector_selected_ids"] == ["s/pick"]
    assert res.telemetry["rendered_order"] == ["s/pick"]
    assert res.telemetry["legacy_fallback"] is False
    assert res.telemetry["injected_chars"] == len(res.rendered)


def test_link_pulls_dependency_into_render():
    skill = _skill("s", [_unit("a", depends_on=["b"]), _unit("b")])
    res = cl.link(_profile(), [skill], _stub("s/a"))
    assert res.selected_ids == ["s/b", "s/a"]  # dep prepended, topo order
    assert res.telemetry["dep_added_ids"] == ["s/b"]


def test_link_decline_renders_nothing():
    res = cl.link(_profile(), [_skill("s", [_unit("u")])], _stub(decline="nothing relevant"))
    assert res.rendered == "" and res.selected_ids == []
    assert res.telemetry["decline_reason"] == "nothing relevant"
    assert res.telemetry["legacy_fallback"] is False


def test_link_none_llm_query_skips_selection():
    res = cl.link(_profile(), [_skill("s", [_unit("u")])], None)
    assert res.rendered == "" and res.selected_ids == []
    assert res.telemetry["decline_reason"] == "selector_skipped"


def test_link_selector_retries_once_then_declines():
    calls = {"n": 0}

    def boom(system_message, user_message, func_spec):
        calls["n"] += 1
        raise RuntimeError("transport down")

    res = cl.link(_profile(), [_skill("s", [_unit("u")])], boom)
    assert calls["n"] == 2  # one retry then decline, NEVER inject-all
    assert res.rendered == "" and res.selected_ids == []
    assert res.telemetry["selector_error"] and "RuntimeError" in res.telemetry["selector_error"]
    assert res.telemetry["legacy_fallback"] is False


def test_link_empty_library_is_baseline():
    res = cl.link(_profile(), [], _stub("x"))
    assert res.rendered == "" and res.selected_ids == []
    assert res.telemetry["loaded_capability_count"] == 0
    assert res.telemetry["legacy_fallback"] is False


def test_link_never_raises_on_bad_cap_skills():
    # a None skill entry blows up _build_registry internally; link() must swallow it.
    res = cl.link(_profile(), [None], _stub("x"))
    assert res.rendered == "" and res.selected_ids == []
    assert "error" in res.telemetry
    assert res.telemetry["legacy_fallback"] is False


def test_link_skips_non_dict_units():
    skill = {
        "name": "s",
        "body": "B",
        "references": {},
        "reference_files": [],
        "capabilities": ["not-a-dict", _unit("good"), {"no_id": True}],
    }
    res = cl.link(_profile(), [skill], _stub("s/good"))
    assert res.selected_ids == ["s/good"]  # malformed units ignored, good one links


# --------------------------------------------------------------------------- #
# Real E1 manifest smoke — link() across all four canonical stages.
# --------------------------------------------------------------------------- #

def _peft_skill():
    manifest = json.loads((_E1 / "peft" / "capabilities.json").read_text())
    return {
        "name": manifest["skill_name"],
        "body": "PEFT SKILL BODY - must not leak into a brief",
        "references": {"advanced-usage.md": "ADV REFERENCE", "troubleshooting.md": "TS REFERENCE"},
        "reference_files": ["advanced-usage.md", "troubleshooting.md"],
        "capabilities": manifest["capabilities"],
    }


def test_peft_manifest_validates():
    manifest = json.loads((_E1 / "peft" / "capabilities.json").read_text())
    errors, _ = cs.validate_manifest(manifest)
    assert errors == [], errors


@pytest.mark.parametrize("stage", ["generate", "refine", "debug", "explore"])
def test_smoke_link_real_manifest_each_stage(stage):
    skill = _peft_skill()
    p = _profile(stage=stage)
    cands = _candidates(p, [skill])
    assert cands, f"no surviving candidates at stage {stage!r}"

    res = cl.link(p, [skill], _stub(cands[0]))
    assert res.rendered.startswith("## Linked ML Capabilities")
    assert cands[0] in res.selected_ids
    assert "PEFT SKILL BODY" not in res.rendered
    assert res.telemetry["legacy_fallback"] is False


# --------------------------------------------------------------------------- #
# Stage D — max_chars budget trimming (drop reference first, then roots)
# --------------------------------------------------------------------------- #

def test_max_chars_drops_reference_first():
    # Two small units + one HUGE reference; a budget that fits both unit blocks but
    # not the reference must drop the reference and keep both roots.
    u0 = _unit("u0", procedure={"steps": ["s0"], "reference_files": ["references/big.md"]})
    u1 = _unit("u1")
    skill = _skill("s", [u0, u1], references={"big.md": "R" * 5000})
    two_unit_len = len(cl.link(_profile(), [skill], _stub("s/u0", "s/u1"),
                               cl.LinkBudget(max_reference_files=0)).rendered)
    res = cl.link(_profile(), [skill], _stub("s/u0", "s/u1"),
                  cl.LinkBudget(max_units=3, max_reference_files=1, max_chars=two_unit_len + 50))
    assert "#### Reference" not in res.rendered          # reference dropped first
    assert res.telemetry["selected_reference_files"] == []
    assert res.selected_ids == ["s/u0", "s/u1"]          # both roots survive
    assert res.telemetry["truncated"] is True
    assert len(res.rendered) <= two_unit_len + 50


def test_max_chars_drops_lowest_ranked_root():
    # No references; a budget too small for two unit blocks trims the lowest-ranked root.
    skill = _skill("s", [_unit("u0"), _unit("u1")])
    one_unit_len = len(cl.link(_profile(), [skill], _stub("s/u0"),
                               cl.LinkBudget(max_reference_files=0)).rendered)
    res = cl.link(_profile(), [skill], _stub("s/u0", "s/u1"),
                  cl.LinkBudget(max_units=3, max_reference_files=0, max_chars=one_unit_len + 20))
    assert res.selected_ids == ["s/u0"]                  # lowest-ranked (u1) dropped
    assert res.telemetry["truncated"] is True
    assert len(res.rendered) <= one_unit_len + 20


# --------------------------------------------------------------------------- #
# Stage C — deeper topological order (chain + diamond)
# --------------------------------------------------------------------------- #

def test_dep_closure_deep_chain():
    _, by = _reg(_unit("a", depends_on=["b"]), _unit("b", depends_on=["c"]), _unit("c"))
    order, added, dropped = cl._dep_closure(["s/a"], by, {})
    assert order == ["s/c", "s/b", "s/a"]  # strict dependency-first chain
    assert set(added) == {"s/b", "s/c"} and dropped == []


def test_dep_closure_diamond():
    _, by = _reg(
        _unit("a", depends_on=["b", "c"]),
        _unit("b", depends_on=["d"]),
        _unit("c", depends_on=["d"]),
        _unit("d"),
    )
    order, added, dropped = cl._dep_closure(["s/a"], by, {})
    assert order.count("s/d") == 1                       # shared dep appears once
    assert order.index("s/d") < order.index("s/b")       # d before its dependents
    assert order.index("s/d") < order.index("s/c")
    assert order.index("s/b") < order.index("s/a")       # a is last
    assert order.index("s/c") < order.index("s/a")
    assert dropped == []


# --------------------------------------------------------------------------- #
# Stage D — selector-chosen reference wins over procedure.reference_files
# --------------------------------------------------------------------------- #

def test_selector_chosen_reference_precedence():
    u = _unit("u", procedure={"steps": ["s"], "reference_files": ["references/other.md"]})
    skill = _skill("s", [u], references={"other.md": "OTHER-BODY", "picked.md": "PICKED-BODY"})

    def stub(system_message, user_message, func_spec):
        return {"selections": [{"capability_id": "s/u", "reason": "r",
                                "reference_files": ["picked.md"]}], "decline_reason": ""}

    res = cl.link(_profile(), [skill], stub, cl.LinkBudget(max_reference_files=1))
    assert "PICKED-BODY" in res.rendered                 # selector choice wins
    assert "OTHER-BODY" not in res.rendered              # procedure fallback not used
    assert res.telemetry["selected_reference_files"] == ["s/u#picked.md"]
    assert res.telemetry["capability_schema_version"] == "0.2"
    for key in (
        "capability_schema_version", "stage", "hardware", "runtime_capabilities",
        "loaded_skill_count", "loaded_capability_count", "candidate_ids",
        "hard_filtered", "selector_selected_ids", "decline_reason", "selector_error",
        "dep_added_ids", "dep_dropped_ids", "rendered_order",
        "selected_reference_files", "injected_chars", "truncated", "legacy_fallback",
    ):
        assert key in res.telemetry, f"telemetry missing {key!r} at stage {stage}"


def test_smoke_link_real_manifest_decline_is_clean():
    skill = _peft_skill()
    res = cl.link(_profile(stage="generate"), [skill], _stub(decline="none apply"))
    assert res.rendered == "" and res.selected_ids == []
    assert res.telemetry["decline_reason"] == "none apply"
    assert res.telemetry["legacy_fallback"] is False
