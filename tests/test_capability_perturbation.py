"""Perturbation-pair tests for the capability linker — W6 / E2 mechanism evidence.

This is the PRIMARY deterministic evidence for HLD §12-E2: paired ``NodeProfile``
fixtures (a base + a perturbed variant) each carrying the EXPECTED linker response,
so we can assert the deterministic hard filter (``capability_linker._hard_filter`` /
``_intrinsic_reason``) *changes its admit/reject decision only on the perturbations
it is supposed to react to*, and is INVARIANT to the ones it must ignore.

Everything here is DETERMINISTIC and zero-GPU: no LLM calls, no ``link()`` selector.
The seven pairs (contract §"Perturbation fixtures"):

  1. one GPU vs multiple GPUs — a ``min_gpu_count>1`` unit rejects on 1-GPU, admits
     on multi-GPU (reason code ``gpu_count``).
  2. sufficient vs insufficient VRAM — an ``explicit-section`` VRAM unit rejects when
     ``vram < req`` and admits when ``vram >= req``; an ``inferred``-basis VRAM unit
     is NEVER hard-filtered (advisory).
  3. generate vs debug stage — a ``delivery.stages=["debug"]`` unit admits at debug,
     rejects at generate (``stage_excluded``); a stage-agnostic unit admits at both.
  4. training-error vs serving-error wording — an IRRELEVANT wording perturbation of
     the same profile shape must NOT change the hard-filter set (invariance).
  5. checkpoint absent/present · tokenizer-mismatch/unrelated error — SEMANTIC content
     perturbations must leave the DETERMINISTIC hard-filter set invariant (the semantic
     selection response is the LLM-pass follow-up, out of scope for the deterministic
     path).
  6. network / persistent-service available vs unavailable — three-valued UNKNOWN by
     design: these runtime caps must NEVER hard-filter regardless of presence.
  7. credentials-required unit — EXCLUDED (``security_unknown``) whenever the cap is
     not KNOWN-present, regardless of any other perturbation.

Plus ~3 reconstruction tests over ``scripts/replay_capability_linker.py``: a draft /
debug / improve node parsed from a tiny synthetic prompts.jsonl, asserting stage
mapping, parent-field slicing, and unknown-hardware three-valued reconstruction.

Both modules are imported by file path (``spec_from_file_location``) exactly like
``tests/test_capability_linker.py`` — ``capability_linker`` under a stub
``mlevolve_sidecar`` package so its relative ``capability_schema`` import resolves,
and ``replay_capability_linker`` as a top-level script module.
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
_SCRIPTS = _REPO / "scripts"


def _load_sidecar(sub):
    """Load a sidecar submodule under a stub ``mlevolve_sidecar`` package so that its
    relative imports (``from .capability_schema import ...``) resolve — same pattern
    as tests/test_capability_linker.py."""
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


def _load_script(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load_sidecar("capability_linker")
_load_sidecar("capability_schema")  # pre-resolve the relative-import target
# replay_capability_linker's module body calls its own _load_sidecar("capability_linker")
# which finds the entry we just registered — so rcl.NodeProfile IS cl.NodeProfile.
rcl = _load_script("replay_capability_linker", _SCRIPTS / "replay_capability_linker.py")


# --------------------------------------------------------------------------- #
# Fixture builders — minimal valid capability units + NodeProfiles, inline.
# --------------------------------------------------------------------------- #

def _unit(uid, **over):
    """A minimal-but-valid capability unit. ``requires`` / ``delivery`` overrides drive
    the hard filter; the rest keeps it renderable."""
    u = {
        "id": uid,
        "kind": "reference",
        "title": f"Title {uid}",
        "summary": f"Summary of {uid}",
        "applicability": {"when": [f"apply {uid}"]},
        "procedure": {"steps": [f"do {uid}"]},
        "sources": [{"kind": "skill-package", "locator": f"SKILL.md#{uid}"}],
    }
    u.update(over)
    return u


def _profile(stage, *, gpu_count=1, vram=48.0, gpu_name="A6000", caps=None,
             task_text="Fine-tune a causal LM with LoRA on GSM8K.",
             parent_error=None, parent_analysis=None, parent_code=None):
    """Build a NodeProfile. ``caps`` defaults to a conservative KNOWN-present set
    derived from the hardware (python always; gpu iff a GPU is present; multi-gpu iff
    count>1) — the same three-valued convention the spike-018 reconstruction uses."""
    if caps is None:
        caps = {"python"}
        if isinstance(gpu_count, int) and gpu_count > 0:
            caps.add("gpu")
        if isinstance(gpu_count, int) and gpu_count > 1:
            caps.add("multi-gpu")
    hw = {
        "gpu_count": gpu_count,
        "vram_gb": vram,
        "gpu_name": gpu_name,
        "raw": f"{gpu_count} x {gpu_name} ({vram} GB VRAM)" if gpu_name else "",
    }
    return cl.NodeProfile(
        stage=stage,
        task_text=task_text,
        parent_error=parent_error,
        parent_analysis=parent_analysis,
        parent_code=parent_code,
        hardware=hw,
        runtime_capabilities=set(caps),
    )


def _hard(profile, units, name="cap"):
    """Run the deterministic hard filter over ``units`` for ``profile``.
    Returns (survivor_qids:set, rejected:{qid->reason})."""
    skill = {"name": name, "body": "BODY", "references": {}, "capabilities": list(units)}
    recs, by = cl._build_registry([skill])
    survivors, rejected = cl._hard_filter(recs, by, profile)
    return {r["qid"] for r in survivors}, dict(rejected)


# =========================================================================== #
# Sanity: the replay module shares the same linker object we import directly.
# =========================================================================== #

def test_replay_module_shares_linker_object():
    assert rcl.NodeProfile is cl.NodeProfile
    assert rcl.STAGE_MAP is cl.STAGE_MAP


# =========================================================================== #
# Pair 1 — one GPU vs multiple GPUs (min_gpu_count > 1).
# =========================================================================== #

def test_pair1_min_gpu_count_single_vs_multi():
    distributed = _unit(
        "distributed",
        requires={"runtime": ["python", "gpu"], "hardware": {"min_gpu_count": 2}},
    )
    control = _unit("single_gpu")  # stage/hardware-agnostic; must admit on both

    # base: one GPU -> the min_gpu_count>1 unit is KNOWN-incompatible.
    surv1, rej1 = _hard(_profile("generate", gpu_count=1), [distributed, control])
    assert "cap/distributed" not in surv1
    assert rej1["cap/distributed"] == "gpu_count"       # EXPECTED: rejected on 1-GPU
    assert "cap/single_gpu" in surv1                      # invariant control admitted

    # perturbed: four GPUs -> the same unit is now admissible.
    surv4, rej4 = _hard(_profile("generate", gpu_count=4), [distributed, control])
    assert "cap/distributed" in surv4                     # EXPECTED: admitted on multi-GPU
    assert "cap/distributed" not in rej4
    assert "cap/single_gpu" in surv4                      # control still admitted

    # UNKNOWN gpu_count must NOT hard-filter (three-valued): documents the boundary.
    survU, rejU = _hard(
        _profile("generate", gpu_count=None, vram=None, gpu_name=None, caps={"python"}),
        [distributed, control],
    )
    assert "cap/distributed" in survU and "cap/distributed" not in rejU


# =========================================================================== #
# Pair 2 — sufficient vs insufficient VRAM (explicit-section vs inferred basis).
# =========================================================================== #

def test_pair2_vram_explicit_rejects_inferred_never():
    explicit = _unit(
        "bigmem_explicit",
        requires={"runtime": ["python", "gpu"],
                  "hardware": {"vram_gb": 80, "basis": "explicit-section"}},
    )
    inferred = _unit(
        "bigmem_inferred",
        requires={"runtime": ["python", "gpu"],
                  "hardware": {"vram_gb": 80, "basis": "inferred"}},
    )

    # base: 48 GB < 80 GB required.
    surv_lo, rej_lo = _hard(_profile("generate", vram=48.0), [explicit, inferred])
    assert rej_lo["cap/bigmem_explicit"] == "vram_gb"     # EXPECTED: explicit rejects
    assert "cap/bigmem_explicit" not in surv_lo
    assert "cap/bigmem_inferred" in surv_lo               # inferred NEVER filtered
    assert "cap/bigmem_inferred" not in rej_lo

    # perturbed: 80 GB >= 80 GB required.
    surv_hi, rej_hi = _hard(_profile("generate", vram=80.0), [explicit, inferred])
    assert "cap/bigmem_explicit" in surv_hi               # EXPECTED: explicit admits
    assert "cap/bigmem_explicit" not in rej_hi
    assert "cap/bigmem_inferred" in surv_hi               # inferred still admitted

    # the inferred unit is INVARIANT to the VRAM perturbation (advisory only).
    assert ("cap/bigmem_inferred" in surv_lo) == ("cap/bigmem_inferred" in surv_hi)


# =========================================================================== #
# Pair 3 — generate vs debug stage (delivery.stages allowlist).
# =========================================================================== #

def test_pair3_stage_allowlist_debug_only_vs_agnostic():
    debug_only = _unit("dbg_only", delivery={"stages": ["debug"]})
    agnostic = _unit("stage_agnostic")  # no delivery.stages -> every stage allowed

    # base: generate -> the debug-gated unit is excluded.
    surv_g, rej_g = _hard(_profile("generate"), [debug_only, agnostic])
    assert rej_g["cap/dbg_only"] == "stage_excluded"      # EXPECTED: rejected at generate
    assert "cap/dbg_only" not in surv_g
    assert "cap/stage_agnostic" in surv_g                 # agnostic admitted

    # perturbed: debug -> the debug-gated unit is admitted.
    surv_d, rej_d = _hard(_profile("debug"), [debug_only, agnostic])
    assert "cap/dbg_only" in surv_d                        # EXPECTED: admitted at debug
    assert "cap/dbg_only" not in rej_d
    assert "cap/stage_agnostic" in surv_d                 # agnostic still admitted

    # the stage-agnostic unit is INVARIANT to the stage perturbation.
    assert ("cap/stage_agnostic" in surv_g) == ("cap/stage_agnostic" in surv_d)


# A small mixed library reused by the invariance pairs (4 & 5): one agnostic unit
# (always admits), one debug-gated unit, one 2-GPU unit (rejects on 1-GPU).
_MIXED = [
    _unit("train_lora"),
    _unit("dbg_only", delivery={"stages": ["debug"]}),
    _unit("multi_gpu",
          requires={"runtime": ["python", "gpu"], "hardware": {"min_gpu_count": 2}}),
]


# =========================================================================== #
# Pair 4 — training-error vs serving-error wording (IRRELEVANT -> invariance).
# =========================================================================== #

def test_pair4_error_wording_invariance():
    base = _profile(
        "debug", gpu_count=1,
        task_text="Fine-tune a causal LM with LoRA and write predictions.csv.",
        parent_error="RuntimeError: CUDA out of memory during the training loop.",
        parent_analysis="OOM while backpropagating; batch too large for training.",
    )
    perturbed = _profile(
        "debug", gpu_count=1,
        task_text="Serve a causal LM and write predictions.csv for the test split.",
        parent_error="RuntimeError: CUDA out of memory during inference / serving.",
        parent_analysis="OOM while generating; KV-cache too large for serving.",
    )
    surv_b, rej_b = _hard(base, _MIXED)
    surv_p, rej_p = _hard(perturbed, _MIXED)

    # The hard filter reads stage / hardware / runtime ONLY — never the free text.
    assert surv_b == surv_p                                # invariance of the admit set
    assert rej_b == rej_p                                  # invariance of reject reasons
    # And the expected shape at debug/1-GPU: multi_gpu is the only reject.
    assert surv_b == {"cap/train_lora", "cap/dbg_only"}
    assert rej_b == {"cap/multi_gpu": "gpu_count"}


# =========================================================================== #
# Pair 5 — checkpoint absent/present · tokenizer-mismatch/unrelated error.
#          SEMANTIC content -> the DETERMINISTIC filter must be invariant.
#          (The semantic-selection response is the LLM-pass follow-up.)
# =========================================================================== #

def test_pair5_semantic_content_invariance():
    def _dbg(**kw):
        return _profile("debug", gpu_count=1, **kw)

    surv_ref, rej_ref = _hard(_dbg(), _MIXED)

    variants = [
        # checkpoint absent vs present
        _dbg(parent_error="FileNotFoundError: checkpoints/adapter_model.bin is missing."),
        _dbg(parent_error="Loaded checkpoint checkpoints/adapter_model.bin OK; metric low."),
        # tokenizer-mismatch vs an unrelated python error
        _dbg(parent_analysis="Tokenizer/model vocab mismatch: embeddings not resized."),
        _dbg(parent_error="TypeError: unsupported operand type(s) for +: 'int' and 'str'."),
    ]
    for v in variants:
        surv_v, rej_v = _hard(v, _MIXED)
        assert surv_v == surv_ref, "hard filter changed on a purely-semantic perturbation"
        assert rej_v == rej_ref, "reject reasons changed on a purely-semantic perturbation"


# =========================================================================== #
# Pair 6 — network / persistent-service available vs unavailable.
#          Three-valued UNKNOWN by design -> must NEVER hard-filter.
# =========================================================================== #

def test_pair6_network_and_persistent_service_never_hard_filter():
    net = _unit("download", requires={"runtime": ["python", "network"]})
    serve = _unit(
        "serve_endpoint",
        requires={"runtime": ["python", "persistent-service", "gpu"]},
    )

    # available: the caps are declared KNOWN-present.
    surv_av, rej_av = _hard(
        _profile("generate", caps={"python", "gpu", "network", "persistent-service"}),
        [net, serve],
    )
    assert "cap/download" in surv_av and "cap/download" not in rej_av
    assert "cap/serve_endpoint" in surv_av and "cap/serve_endpoint" not in rej_av

    # unavailable / unknown: the caps are simply absent (UNKNOWN, not known-false).
    surv_un, rej_un = _hard(
        _profile("generate", caps={"python", "gpu"}), [net, serve],
    )
    assert "cap/download" in surv_un and "cap/download" not in rej_un
    assert "cap/serve_endpoint" in surv_un and "cap/serve_endpoint" not in rej_un

    # INVARIANCE: network/persistent-service presence does not move the filter set.
    assert surv_av == surv_un and rej_av == rej_un


# =========================================================================== #
# Pair 7 — credentials-required unit -> EXCLUDED regardless of perturbation.
# =========================================================================== #

def test_pair7_credentials_excluded_regardless_of_perturbation():
    cred = _unit("gated_hub", requires={"runtime": ["python", "credentials"]})

    # Perturb an IRRELEVANT axis (hardware) while credentials stay UNKNOWN in both:
    # the security exception must exclude the unit in every case.
    for prof in (
        _profile("generate", gpu_count=1),
        _profile("generate", gpu_count=4),
        _profile("debug", gpu_count=1),
    ):
        surv, rej = _hard(prof, [cred])
        assert "cap/gated_hub" not in surv
        assert rej["cap/gated_hub"] == "security_unknown"  # EXPECTED: excluded

    # Documents the exception's precondition: only KNOWN-present credentials admit it
    # (the deterministic spike-018 reconstruction never marks credentials present, so
    # a credentials unit is always excluded there).
    surv_ok, rej_ok = _hard(
        _profile("generate", caps={"python", "gpu", "credentials"}), [cred],
    )
    assert "cap/gated_hub" in surv_ok and "cap/gated_hub" not in rej_ok


# =========================================================================== #
# Aggregate deterministic gate — 0 hard-constraint violations across every pair.
# =========================================================================== #

def test_no_hard_constraint_violations_across_all_perturbations():
    """The E2 deterministic gate: every ADMITTED unit re-passes its own intrinsic
    hard constraints against the profile it was admitted under (~0 by construction).
    Runs the replay module's deterministic_link — the same layer the harness uses —
    over every perturbation profile against a library of every perturbation unit."""
    all_units = _MIXED + [
        _unit("distributed",
              requires={"runtime": ["python", "gpu"], "hardware": {"min_gpu_count": 2}}),
        _unit("bigmem_explicit",
              requires={"runtime": ["python", "gpu"],
                        "hardware": {"vram_gb": 80, "basis": "explicit-section"}}),
        _unit("bigmem_inferred",
              requires={"runtime": ["python", "gpu"],
                        "hardware": {"vram_gb": 80, "basis": "inferred"}}),
        _unit("download", requires={"runtime": ["python", "network"]}),
        _unit("gated_hub", requires={"runtime": ["python", "credentials"]}),
    ]
    cap_skills = [{"name": "cap", "body": "B", "references": {}, "capabilities": all_units}]
    budget = cl.LinkBudget(max_units=3)

    profiles = [
        _profile("generate", gpu_count=1, vram=48.0),
        _profile("generate", gpu_count=4, vram=80.0),
        _profile("debug", gpu_count=1, vram=48.0),
        _profile("refine", gpu_count=2, vram=48.0),
        _profile("explore", gpu_count=1, vram=24.0),
        _profile("generate", gpu_count=None, vram=None, gpu_name=None, caps={"python"}),
    ]
    total_admitted = 0
    for prof in profiles:
        det = rcl.deterministic_link(prof, cap_skills, budget)
        total_admitted += det["cap_upper_bound"]
        assert det["violations"] == [], (
            f"filter defect: admitted units violate constraints at stage {prof.stage}: "
            f"{det['violations']}"
        )
    assert total_admitted > 0  # the harness actually admitted something to validate


# =========================================================================== #
# replay_capability_linker — NodeProfile reconstruction from a synthetic jsonl.
# =========================================================================== #

def _draft_msg():
    return (
        "Stage: draft\n\n"
        "Task:\n"
        "Fine-tune a causal LM with LoRA on GSM8K and write predictions.csv."
    )


def _debug_msg():
    return (
        "Stage: debug\n\n"
        "Task:\n"
        "Fine-tune a causal LM with LoRA on GSM8K.\n\n"
        "Error output (tail):\n"
        "Traceback (most recent call last):\n"
        "  File \"runfile_0.py\", line 42, in <module>\n"
        "RuntimeError: CUDA out of memory."
    )


def _improve_msg():
    return (
        "Stage: improve\n\n"
        "Task:\n"
        "Fine-tune a causal LM with LoRA on GSM8K.\n\n"
        "Current solution code (head):\n"
        "import torch\n"
        "model = load_model()\n"
        "trainer.train()"
    )


def test_reconstruct_draft_node_unknown_hardware():
    stage_raw, prof = rcl.parse_node_profile(_draft_msg())
    assert stage_raw == "draft"
    assert prof.stage == "generate"                        # STAGE_MAP: draft -> generate
    assert "predictions.csv" in prof.task_text
    assert prof.parent_error is None
    assert prof.parent_analysis is None
    assert prof.parent_code is None
    # No Compute line in spike-018 -> hardware is all-None UNKNOWN (correct).
    assert prof.hardware == {"gpu_count": None, "vram_gb": None, "gpu_name": None, "raw": ""}
    # Conservative KNOWN-present set: python only when hardware is unknown (no gpu).
    assert prof.runtime_capabilities == {"python"}


def test_reconstruct_debug_node_error_tail():
    stage_raw, prof = rcl.parse_node_profile(_debug_msg())
    assert stage_raw == "debug"
    assert prof.stage == "debug"                            # STAGE_MAP identity
    assert prof.task_text.startswith("Fine-tune a causal LM")
    assert "Error output" not in prof.task_text            # task slice stops at next header
    assert prof.parent_error is not None
    assert "CUDA out of memory" in prof.parent_error
    assert "Traceback" in prof.parent_error
    assert prof.parent_code is None and prof.parent_analysis is None
    assert prof.runtime_capabilities == {"python"}         # unknown hardware


def test_reconstruct_improve_node_code():
    stage_raw, prof = rcl.parse_node_profile(_improve_msg())
    assert stage_raw == "improve"
    assert prof.stage == "refine"                           # STAGE_MAP: improve -> refine
    assert prof.parent_code is not None
    assert "import torch" in prof.parent_code
    assert "trainer.train()" in prof.parent_code
    assert prof.parent_error is None and prof.parent_analysis is None
    assert prof.task_text.startswith("Fine-tune a causal LM")


def test_reconstruct_profiles_reads_jsonl(tmp_path):
    """reconstruct_profiles() reads a whole prompts.jsonl: keeps select_skills asks,
    maps stages, and recovers the legacy recorded pick for the exposure baseline."""
    records = [
        {"func_spec_name": "select_skills", "user_message": _draft_msg(),
         "output": json.dumps({"selections": [
             {"skill_name": "peft-tuning", "references": []}]})},
        {"func_spec_name": "select_skills", "user_message": _debug_msg(),
         "output": json.dumps({"selections": []})},                 # legacy declined
        {"func_spec_name": "select_skills", "user_message": _improve_msg()},  # no output
        # a non-selector record must be skipped entirely.
        {"func_spec_name": "generate", "user_message": "Stage: draft\n\nTask:\nnope"},
    ]
    (tmp_path / "prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")

    nodes = rcl.reconstruct_profiles(tmp_path)
    assert len(nodes) == 3                                  # the "generate" record dropped
    assert [n["stage_raw"] for n in nodes] == ["draft", "debug", "improve"]
    assert [n["stage"] for n in nodes] == ["generate", "debug", "refine"]
    # legacy recorded selection: first node picked peft-tuning; the rest declined.
    assert nodes[0]["recorded_skills"] == ["peft-tuning"]
    assert nodes[1]["recorded_skills"] == []
    assert nodes[2]["recorded_skills"] == []
    # every reconstructed profile is a real NodeProfile with unknown hardware here.
    for n in nodes:
        assert isinstance(n["profile"], cl.NodeProfile)
        assert n["profile"].hardware["gpu_count"] is None


def test_reconstruct_profiles_with_compute_line_marks_gpu(tmp_path):
    """Guards the Compute-line branch (ABSENT in spike-018 but part of the contract):
    a present ``Compute:`` line parses hardware and adds gpu to the KNOWN caps."""
    msg = (
        "Stage: draft\n\n"
        "Compute: 4 x A100 (80 GB VRAM), 32 CPUs, 512 GB RAM.\n\n"
        "Task:\nTrain a big model."
    )
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"func_spec_name": "select_skills", "user_message": msg}) + "\n")
    nodes = rcl.reconstruct_profiles(tmp_path)
    assert len(nodes) == 1
    prof = nodes[0]["profile"]
    assert prof.hardware["gpu_count"] == 4
    assert prof.hardware["vram_gb"] == 80.0
    # KNOWN hardware -> gpu and multi-gpu enter the runtime-capability set.
    assert {"python", "gpu", "multi-gpu"} <= prof.runtime_capabilities


# --------------------------------------------------------------------------- #
# selector_health validity guard (W6+): a silent claude-CLI failure yields 0
# injected units and is indistinguishable from a real abstention at the exposure
# level — which would spuriously INFLATE the --with-selector exposure reduction.
# The guard MUST bucket masked failures (transport_error + parse_failure)
# separately from genuine declines, and only flag the run trustworthy at masked==0.
# --------------------------------------------------------------------------- #

def test_selector_health_separates_masked_failure_from_genuine_decline():
    rows = [
        {"selector_selected_ids": ["peft/a"], "decline_reason": None, "selector_error": None},
        {"selector_selected_ids": [], "decline_reason": "nothing relevant for a debug node",
         "selector_error": None},
        {"selector_selected_ids": [], "decline_reason": "no_candidates", "selector_error": None},
        {"selector_selected_ids": [], "decline_reason": "selector returned no JSON",
         "selector_error": None},
        {"selector_selected_ids": [], "decline_reason": None,
         "selector_error": "RuntimeError: claude transport failed"},
    ]
    h = rcl._selector_health(rows)
    assert h["selected"] == 1
    assert h["genuine_decline"] == 1          # the real "nothing relevant" decline
    assert h["no_candidates"] == 1            # deterministic filter emptied — not a failure
    assert h["parse_failure"] == 1            # ran but unparseable -> masked
    assert h["transport_error"] == 1          # both attempts raised -> masked
    assert h["masked_failure"] == 2
    assert h["trustworthy"] is False


def test_selector_health_all_clean_is_trustworthy():
    rows = [
        {"selector_selected_ids": ["peft/a", "vllm/b"], "decline_reason": None, "selector_error": None},
        {"selector_selected_ids": [], "decline_reason": "irrelevant to this explore step",
         "selector_error": None},
    ]
    h = rcl._selector_health(rows)
    assert h["masked_failure"] == 0 and h["trustworthy"] is True
    assert h["selected"] == 1 and h["genuine_decline"] == 1
