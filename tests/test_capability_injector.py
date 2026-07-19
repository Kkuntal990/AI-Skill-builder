"""Tests for the capability delivery-mode dispatch in skill_injector.py (W5).

Covers the MLEvolve-specific glue added by the capability-linker MVP (HLD §10):
  - ``_capability_mode`` env read (default legacy; unrecognised → legacy);
  - dispatch routing (legacy vs capability_task / capability_node);
  - ``_build_node_profile`` adapter (STAGE_MAP, harness-strip, parent fields,
    hardware parse, conservative runtime capabilities);
  - ``_inject_capability`` end-to-end against the REAL ``capability_linker`` and a
    stubbed ``llm`` transport: empty library == no-skill baseline, brief lands in
    the guideline, telemetry emitted with ``legacy_fallback`` False, capability_task
    caches the first-node brief and reuses it, and an error never falls back to
    legacy content.

Loaded WITHOUT executing the package ``__init__`` (which imports upstream
MLEvolve) — the same stub-package pattern as test_skill_selector_context.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SIDECAR = _REPO / "infra/agents/mlevolve/mlevolve_sidecar"
_E1 = _REPO / "docs/skill-builder/capability-mvp/e1"

# Route the import-time cell_init write off the working tree.
os.environ.setdefault(
    "MLEVAL_SELECTION_LOG", str(Path(__file__).parent / "__pycache__" / "_cap_test_events.jsonl")
)


def _load_sidecar_module(sub: str):
    """Load a sidecar submodule by file path under a stub parent package.

    The stub gives ``mlevolve_sidecar`` a ``__path__`` so intra-package imports
    (``from . import capability_linker`` etc.) resolve without running the real
    ``__init__`` (which imports MLEvolve). Loads any not-yet-present dependency
    submodules the same way.
    """
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
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # pragma: no cover — needs upstream MLEvolve
        del sys.modules[full]
        pytest.skip(f"{sub} import needs upstream MLEvolve: {e}")
    return mod


@pytest.fixture()
def inj():
    # skill_retriever + selection_logger + capability_linker load lazily via the
    # package path; import the injector last (registers its meta_path hook — inert
    # here since no agent modules load).
    _load_sidecar_module("skill_retriever")
    _load_sidecar_module("selection_logger")
    _load_sidecar_module("capability_linker")
    return _load_sidecar_module("skill_injector")


# --------------------------------------------------------------------------- #
# Fixtures: fake agent, a real capability skill, and a stubbed llm transport.
# --------------------------------------------------------------------------- #

class _FakeParent:
    code = "import torch\nmodel = load()\n"
    term_out = "Traceback (most recent call last):\nRuntimeError: CUDA out of memory"
    analysis = "Batch size too large for the available VRAM."


class _FakeFeedback:
    model = "test/selector-model"


class _FakeACfg:
    feedback = _FakeFeedback()


class _FakeAgent:
    def __init__(self, stage="draft", parent=None):
        self.task_desc = "## Description\nFine-tune a causal LM with LoRA on GSM8K.\n"
        self.acfg = _FakeACfg()
        self.cfg = object()
        self._mleval_stage = stage
        self._mleval_parent = parent


def _real_cap_skill():
    """Build a retriever-shaped cap_skill dict from a real E1 manifest (peft)."""
    manifest = json.loads((_E1 / "peft" / "capabilities.json").read_text())
    return {
        "name": manifest["skill_name"],
        "body": "SKILL BODY (must never appear in a capability brief)",
        "references": {"lora-methods.md": "LORA REFERENCE BODY"},
        "reference_files": ["lora-methods.md"],
        "capabilities": manifest["capabilities"],
    }


def _first_generate_qid(cap_skill):
    """Qualified id of a unit that admits the 'generate' stage (or has no stage
    allowlist), so it survives the hard filter at a draft node."""
    name = cap_skill["name"]
    for u in cap_skill["capabilities"]:
        stages = (u.get("delivery") or {}).get("stages") or []
        if not stages or "generate" in stages:
            return f"{name}/{u['id']}"
    raise AssertionError("no generate-stage unit in the peft fixture")


class _FakeFunctionSpec:
    def __init__(self, name, description, json_schema):
        self.name = name


def _install_fake_llm(select_qid, calls):
    """Register a fake ``llm`` module whose query() selects one capability id."""
    fake = types.ModuleType("llm")
    fake.FunctionSpec = _FakeFunctionSpec

    def _query(system_message, user_message, func_spec, model, temperature, cfg):
        calls.append({"model": model, "temperature": temperature})
        return {
            "selections": [
                {"capability_id": select_qid, "reason": "needed for the draft", "reference_files": []}
            ],
            "decline_reason": "",
        }

    fake.query = _query
    sys.modules["llm"] = fake
    return fake


@pytest.fixture(autouse=True)
def _clean_env_and_hardware(inj):
    """Fresh hardware cache + a known GPU line + isolated selection log per test."""
    eh = sys.modules["mlevolve_sidecar.eval_harness"]
    eh._HARDWARE_CACHE = None
    prev = {k: os.environ.get(k) for k in
            ("MLEVAL_HARDWARE", "MLEVAL_SKILL_DELIVERY_MODE", "MLEVAL_SELECTION_LOG")}
    os.environ["MLEVAL_HARDWARE"] = "1 NVIDIA RTX A6000 GPU (48 GB VRAM), 8 CPUs, 32 GB RAM"
    yield
    eh._HARDWARE_CACHE = None
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    sys.modules.pop("llm", None)


# --------------------------------------------------------------------------- #
# _capability_mode
# --------------------------------------------------------------------------- #

def test_default_mode_is_legacy(inj):
    os.environ.pop("MLEVAL_SKILL_DELIVERY_MODE", None)
    assert inj._capability_mode() == "legacy"
    assert "legacy" not in inj._CAPABILITY_MODES


def test_capability_modes_recognised(inj):
    for m in ("capability_task", "capability_node"):
        os.environ["MLEVAL_SKILL_DELIVERY_MODE"] = m
        assert inj._capability_mode() == m
        assert m in inj._CAPABILITY_MODES


def test_unrecognised_mode_is_not_capability(inj):
    os.environ["MLEVAL_SKILL_DELIVERY_MODE"] = "capabilty_typo"
    assert inj._capability_mode() not in inj._CAPABILITY_MODES  # → legacy path


def test_blank_mode_is_legacy(inj):
    os.environ["MLEVAL_SKILL_DELIVERY_MODE"] = "   "
    assert inj._capability_mode() == "legacy"


# --------------------------------------------------------------------------- #
# _build_node_profile adapter
# --------------------------------------------------------------------------- #

def test_build_node_profile_maps_stage_and_hardware(inj):
    agent = _FakeAgent(stage="debug", parent=_FakeParent())
    p = inj._build_node_profile(agent)
    assert p.stage == "debug"  # debug → debug
    assert p.parent_error and "CUDA out of memory" in p.parent_error
    assert p.parent_code and "import torch" in p.parent_code
    assert p.parent_analysis and "Batch size" in p.parent_analysis
    assert p.hardware["gpu_count"] == 1
    assert p.hardware["vram_gb"] == 48.0
    # conservative runtime caps: python always; gpu since count>0; NOT multi-gpu.
    assert p.runtime_capabilities == {"python", "gpu"}


def test_build_node_profile_stage_map_and_task_strip(inj):
    agent = _FakeAgent(stage="draft", parent=None)
    p = inj._build_node_profile(agent)
    assert p.stage == "generate"  # draft → generate via STAGE_MAP
    assert p.parent_code is None and p.parent_error is None
    assert "Fine-tune a causal LM" in p.task_text


def test_build_node_profile_multi_gpu_and_cpu_only(inj):
    eh = sys.modules["mlevolve_sidecar.eval_harness"]
    eh._HARDWARE_CACHE = None
    os.environ["MLEVAL_HARDWARE"] = "4x NVIDIA A100 (80GB)"
    p = inj._build_node_profile(_FakeAgent(stage="improve"))
    assert p.stage == "refine"
    assert p.runtime_capabilities == {"python", "gpu", "multi-gpu"}

    eh._HARDWARE_CACHE = None
    os.environ["MLEVAL_HARDWARE"] = "8 CPUs, 32 GB RAM (no GPU)"
    p2 = inj._build_node_profile(_FakeAgent(stage="draft"))
    assert p2.hardware["gpu_count"] == 0
    assert p2.runtime_capabilities == {"python"}  # gpu NOT claimed on a CPU-only node


# --------------------------------------------------------------------------- #
# Dispatch routing
# --------------------------------------------------------------------------- #

def test_dispatch_routes_by_mode(inj, monkeypatch):
    routed = {"legacy": 0, "cap": 0}
    monkeypatch.setattr(inj, "_inject_legacy", lambda a, r: routed.__setitem__("legacy", routed["legacy"] + 1))
    monkeypatch.setattr(inj, "_inject_capability", lambda a, r, m: routed.__setitem__("cap", routed["cap"] + 1))

    def orig_fn(agent):
        return {"Implementation guideline": ["base"]}

    wrapped = inj._wrap_impl_guideline(orig_fn)
    agent = _FakeAgent()

    os.environ.pop("MLEVAL_SKILL_DELIVERY_MODE", None)
    wrapped(agent)
    os.environ["MLEVAL_SKILL_DELIVERY_MODE"] = "typo-mode"
    wrapped(agent)
    assert routed == {"legacy": 2, "cap": 0}

    os.environ["MLEVAL_SKILL_DELIVERY_MODE"] = "capability_node"
    wrapped(agent)
    os.environ["MLEVAL_SKILL_DELIVERY_MODE"] = "capability_task"
    wrapped(agent)
    assert routed == {"legacy": 2, "cap": 2}


def test_legacy_dispatch_appends_catalog_not_capabilities(inj, monkeypatch):
    # Non-empty legacy library, empty selection (declined) → catalog only.
    sr = inj.skill_retriever
    monkeypatch.setattr(sr, "loaded_skills", lambda: [
        {"name": "demo", "description": "d", "body": "B", "references": {}, "reference_files": []}
    ])
    monkeypatch.setattr(sr, "catalog_text", lambda: "- demo: d")
    monkeypatch.setattr(inj, "_ensure_selection", lambda a, s: ([], {"fallback_mode": "catalog_only"}))
    monkeypatch.setattr(inj, "_log_node_selection", lambda *a, **k: None)
    os.environ.pop("MLEVAL_SKILL_DELIVERY_MODE", None)

    result = {"Implementation guideline": ["base"]}
    inj._wrap_impl_guideline(lambda agent: result)(_FakeAgent())
    gl = result["Implementation guideline"]
    assert any("## Available Skills (catalog)" in l for l in gl)
    assert not any("## Linked ML Capabilities" in l for l in gl)


# --------------------------------------------------------------------------- #
# _inject_capability
# --------------------------------------------------------------------------- #

def test_empty_capability_library_injects_nothing(inj, monkeypatch):
    monkeypatch.setattr(inj.skill_retriever, "loaded_capability_skills", lambda: [])
    result = {"Implementation guideline": ["base"]}
    inj._inject_capability(_FakeAgent(), result, "capability_node")
    assert result["Implementation guideline"] == ["base"]  # no-skill baseline


def _read_events(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_capability_node_injects_brief_and_telemetry(inj, monkeypatch, tmp_path):
    cap = _real_cap_skill()
    qid = _first_generate_qid(cap)
    monkeypatch.setattr(inj.skill_retriever, "loaded_capability_skills", lambda: [cap])
    calls = []
    _install_fake_llm(qid, calls)
    log = tmp_path / "events.jsonl"
    os.environ["MLEVAL_SELECTION_LOG"] = str(log)

    result = {"Implementation guideline": ["base"]}
    inj._inject_capability(_FakeAgent(stage="draft"), result, "capability_node")

    gl = result["Implementation guideline"]
    assert any("## Linked ML Capabilities" in l for l in gl)
    brief = next(l for l in gl if "## Linked ML Capabilities" in l)
    assert qid in brief
    assert "SKILL BODY" not in brief  # never dumps the SKILL.md body
    assert len(calls) == 1  # selector fired exactly once

    events = _read_events(log)
    cap_events = [e for e in events if e.get("event") == "capability_node"]
    assert len(cap_events) == 1
    ev = cap_events[0]
    assert ev["delivery_mode"] == "capability_node"
    assert ev["legacy_fallback"] is False
    assert ev["node_stage"] == "generate"
    assert ev["cache_reused"] is False
    assert qid in ev["rendered_order"]
    assert ev["sidecar_version"]  # stamped by selection_logger._ctx


def test_capability_task_caches_first_node(inj, monkeypatch, tmp_path):
    cap = _real_cap_skill()
    qid = _first_generate_qid(cap)
    monkeypatch.setattr(inj.skill_retriever, "loaded_capability_skills", lambda: [cap])
    calls = []
    _install_fake_llm(qid, calls)
    os.environ["MLEVAL_SELECTION_LOG"] = str(tmp_path / "events.jsonl")

    agent = _FakeAgent(stage="draft")
    r1 = {"Implementation guideline": ["base"]}
    inj._inject_capability(agent, r1, "capability_task")
    # Later node: different stage, same agent — reuse the cached brief verbatim.
    agent._mleval_stage = "improve"
    agent._mleval_parent = _FakeParent()
    r2 = {"Implementation guideline": ["base"]}
    inj._inject_capability(agent, r2, "capability_task")

    brief1 = next(l for l in r1["Implementation guideline"] if "## Linked ML Capabilities" in l)
    brief2 = next(l for l in r2["Implementation guideline"] if "## Linked ML Capabilities" in l)
    assert brief1 == brief2                     # reused verbatim
    assert len(calls) == 1                      # linker/selector ran ONCE
    assert hasattr(agent, "_mleval_cap_brief")  # cached on the shared agent

    events = [e for e in _read_events(tmp_path / "events.jsonl") if e.get("event") == "capability_node"]
    assert len(events) == 2                      # one event per node
    assert events[0]["cache_reused"] is False
    assert events[1]["cache_reused"] is True
    assert events[1]["node_stage"] == "refine"   # live stage, not the frozen draft stage


def test_capability_task_recomputes_after_reset(inj, monkeypatch, tmp_path):
    """A fresh agent (new trajectory) with no cache links again (no cross-agent leak)."""
    cap = _real_cap_skill()
    qid = _first_generate_qid(cap)
    monkeypatch.setattr(inj.skill_retriever, "loaded_capability_skills", lambda: [cap])
    calls = []
    _install_fake_llm(qid, calls)
    os.environ["MLEVAL_SELECTION_LOG"] = str(tmp_path / "e.jsonl")
    inj._inject_capability(_FakeAgent(stage="draft"), {"Implementation guideline": []}, "capability_task")
    inj._inject_capability(_FakeAgent(stage="draft"), {"Implementation guideline": []}, "capability_task")
    assert len(calls) == 2  # two independent agents → two links


def test_capability_never_breaks_or_falls_back_on_llm_error(inj, monkeypatch, tmp_path):
    cap = _real_cap_skill()
    monkeypatch.setattr(inj.skill_retriever, "loaded_capability_skills", lambda: [cap])
    # llm.query raises → linker retries once then declines → inject nothing.
    fake = types.ModuleType("llm")
    fake.FunctionSpec = _FakeFunctionSpec

    def _boom(**kwargs):
        raise RuntimeError("transport down")

    fake.query = _boom
    sys.modules["llm"] = fake
    os.environ["MLEVAL_SELECTION_LOG"] = str(tmp_path / "e.jsonl")

    result = {"Implementation guideline": ["base"]}
    inj._inject_capability(_FakeAgent(stage="draft"), result, "capability_node")
    gl = result["Implementation guideline"]
    assert gl == ["base"]                                   # nothing injected
    assert not any("## Loaded Skill Content" in l for l in gl)  # NO legacy fallback

    events = [e for e in _read_events(tmp_path / "e.jsonl") if e.get("event") == "capability_node"]
    assert len(events) == 1
    assert events[0]["legacy_fallback"] is False
    assert events[0]["selector_error"]  # the error is recorded, not silently dropped


def test_capability_missing_llm_module_injects_nothing(inj, monkeypatch, tmp_path):
    cap = _real_cap_skill()
    monkeypatch.setattr(inj.skill_retriever, "loaded_capability_skills", lambda: [cap])
    sys.modules.pop("llm", None)
    # Force `import llm` to fail so the whole branch is caught and injects nothing.
    monkeypatch.setitem(sys.modules, "llm", None)
    os.environ["MLEVAL_SELECTION_LOG"] = str(tmp_path / "e.jsonl")
    result = {"Implementation guideline": ["base"]}
    inj._inject_capability(_FakeAgent(stage="draft"), result, "capability_node")
    assert result["Implementation guideline"] == ["base"]
