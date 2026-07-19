"""Coverage-recall audit for compiled capability manifests (the RECALL axis of E1).

capability_grounding.py measures PRECISION: "is every compiled unit backed by the
source?" (no fabrication). It says NOTHING about RECALL: "is every solvable issue in
the source captured as a *selectable* unit?" A skill can contain the fix for an issue
in prose (e.g. gradient checkpointing for a training OOM) while the compiler never
lifts it into a unit — so a per-node selector correctly declines against an incomplete
library and the help never reaches the agent. That false-negative is invisible to a
precision-only audit and to exposure metrics. This module measures it.

Method (two-stage, source-anchored, deliberately anti-circular):
  1. EXTRACT solvable issues from the source ALONE (SKILL.md + references), blind to
     the compiled units, each with a VERBATIM source quote.  (extract_solvable_issues)
  2. MAP each issue to a compiled unit, this stage seeing the unit catalog.  (map_issue_coverage)
  3. Deterministic guards (no LLM): every issue quote must verify against the real
     source (drops hallucinated issues); every claimed covering unit-id must exist in
     the manifest (drops hallucinated coverage). coverage_rate + gaps are computed in
     code from the surviving, grounded issues.

Like the compiler/grounding critic, the LLM transport is injected. The CLI ships a
claude-CLI-ONLY transport that RAISES on failure (no OpenRouter fallback) so a silent
transport drop can never masquerade as "0 issues / full coverage".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from capability_compiler import (  # reuse the compiler's bundle + prompt machinery
    PROMPTS_DIR,
    _extract_json,
    _fill,
    _references_block,
    load_skill_bundle,
)

COVERAGE_VERSION = "0.1.0"

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Whitespace/case-normalize for robust verbatim-quote matching."""
    return _WS.sub(" ", (s or "")).strip().lower()


def _unit_stages(unit: dict) -> list:
    return (unit.get("delivery") or {}).get("stages") or unit.get("stages") or []


# ── LLM stages ────────────────────────────────────────────────────────────────


def extract_issues(bundle: dict, llm, max_parse_retries: int = 1) -> dict:
    """Stage 1: enumerate source-anchored solvable issues, blind to the units."""
    template = (PROMPTS_DIR / "extract_solvable_issues.txt").read_text()
    prompt = _fill(
        template,
        skill_name=bundle["skill_name"],
        skill_md=bundle["skill_md"][:30000],
        references_block=_references_block(bundle),
    )
    last_err = None
    for _ in range(1 + max_parse_retries):
        raw = llm(prompt, max_tokens=16000, temperature=0.0)
        try:
            out = _extract_json(raw)
            if isinstance(out.get("issues"), list):
                return out
            last_err = "extract JSON missing 'issues' list"
        except (ValueError, json.JSONDecodeError) as e:
            last_err = str(e)[:200]
    raise RuntimeError(f"issue extraction unparseable after retries: {last_err}")


def _units_catalog(manifest: dict) -> str:
    cat = []
    for u in manifest.get("capabilities", []):
        ap = u.get("applicability") or {}
        cat.append({
            "id": u.get("id"),
            "kind": u.get("kind"),
            "title": u.get("title"),
            "summary": u.get("summary"),
            "when": ap.get("when"),
            "avoid_when": ap.get("avoid_when"),
            "stages": _unit_stages(u),
        })
    return json.dumps(cat, indent=2)


def map_coverage(bundle: dict, issues: list, manifest: dict, llm,
                 max_parse_retries: int = 1) -> dict:
    """Stage 2: for each issue, decide whether a compiled unit would be delivered."""
    template = (PROMPTS_DIR / "map_issue_coverage.txt").read_text()
    prompt = _fill(
        template,
        skill_name=bundle["skill_name"],
        units_catalog_json=_units_catalog(manifest),
        issues_json=json.dumps(issues, indent=2),
    )
    last_err = None
    for _ in range(1 + max_parse_retries):
        raw = llm(prompt, max_tokens=8000, temperature=0.0)
        try:
            out = _extract_json(raw)
            if isinstance(out.get("mappings"), list):
                return out
            last_err = "coverage JSON missing 'mappings' list"
        except (ValueError, json.JSONDecodeError) as e:
            last_err = str(e)[:200]
    raise RuntimeError(f"coverage mapping unparseable after retries: {last_err}")


# ── deterministic scoring ───────────────────────────────────────────────────────


def score_coverage(bundle: dict, manifest: dict, issues: list, mappings: list) -> dict:
    """Join issues↔mappings deterministically; verify quotes against source and
    covering-unit-ids against the manifest; compute coverage_rate + gaps. No LLM."""
    source_norm = _norm(bundle["skill_md"] + "\n" +
                        "\n".join(bundle["references"].values()))
    unit_ids = {u.get("id") for u in manifest.get("capabilities", [])}
    map_by_id = {m.get("issue_id"): m for m in mappings}

    grounded, ungrounded = [], []          # ungrounded = quote not found in source
    for iss in issues:
        q = _norm(iss.get("quote", ""))
        rec = dict(iss)
        rec["quote_verified"] = bool(q) and q in source_norm
        (grounded if rec["quote_verified"] else ungrounded).append(rec)

    covered, gaps = [], []
    for iss in grounded:
        m = map_by_id.get(iss["id"]) or {}
        cby = m.get("covered_by")
        # covered only if the LLM said so AND the named unit really exists.
        real_unit = cby in unit_ids and cby not in (None, "NONE", "")
        is_covered = bool(m.get("covered")) and real_unit
        row = {
            "id": iss["id"],
            "statement": iss.get("statement"),
            "source_fix": iss.get("source_fix"),
            "quote": iss.get("quote"),
            "source_file": iss.get("source_file"),
            "stages": iss.get("stages") or [],
            "severity": iss.get("severity"),
            "covered_by": cby if real_unit else "NONE",
            "stage_gated": bool(m.get("stage_gated")),
            "reason": m.get("reason"),
            "unit_id_hallucinated": bool(cby not in (None, "NONE", "") and cby not in unit_ids),
        }
        (covered if is_covered else gaps).append(row)

    def rate(n, d):
        return round(n / d, 4) if d else None

    def _by(key, rows):
        out = {}
        for r in rows:
            for v in (r.get(key) if isinstance(r.get(key), list) else [r.get(key)]):
                out[v] = out.get(v, 0) + 1
        return out

    n_grounded = len(grounded)
    high_gaps = [g for g in gaps if g["severity"] == "high"]
    return {
        "issues_extracted": len(issues),
        "issues_grounded": n_grounded,
        "issues_ungrounded_dropped": len(ungrounded),
        "ungrounded_issue_ids": [i["id"] for i in ungrounded],
        "covered": len(covered),
        "gaps": len(gaps),
        "coverage_rate": rate(len(covered), n_grounded),
        "high_severity_gaps": len(high_gaps),
        "gaps_by_stage": _by("stages", gaps),
        "gaps_by_severity": _by("severity", gaps),
        "stage_gated_gaps": sum(1 for g in gaps if g["stage_gated"]),
        "hallucinated_unit_ids": sum(1 for g in gaps if g["unit_id_hallucinated"]),
        "gap_detail": sorted(gaps, key=lambda g: {"high": 0, "medium": 1, "low": 2}
                             .get(g["severity"], 3)),
        "covered_detail": covered,
    }


def _progress(msg: str) -> None:
    print(f"[coverage-audit] {msg}", file=sys.stderr, flush=True)


def run_coverage_audit(skill_dir: Path, manifest_path: Path, llm,
                       cache_path: Path | None = None, chunk_size: int = 12) -> dict:
    bundle = load_skill_bundle(skill_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    # Stage 1 — extract issues (the expensive call). Cache it so a later map-stage
    # failure never discards it; a re-run resumes from the cache.
    if cache_path and Path(cache_path).is_file():
        ex = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        _progress(f"{bundle['skill_name']}: loaded {len(ex['issues'])} cached issues")
    else:
        _progress(f"{bundle['skill_name']}: extracting solvable issues from source ...")
        ex = extract_issues(bundle, llm)
        if cache_path:
            Path(cache_path).write_text(json.dumps(ex, indent=2), encoding="utf-8")
        _progress(f"{bundle['skill_name']}: extracted {len(ex['issues'])} issues (cached)")

    # Stage 2 — map coverage in SMALL BATCHES so each claude call is bounded
    # (mapping all ~50 issues in one call overran the transport timeout).
    issues = ex["issues"]
    mappings = []
    n_chunks = (len(issues) + chunk_size - 1) // chunk_size
    for ci in range(0, len(issues), chunk_size):
        chunk = issues[ci:ci + chunk_size]
        _progress(f"{bundle['skill_name']}: mapping chunk {ci // chunk_size + 1}/{n_chunks} "
                  f"({len(chunk)} issues) ...")
        cov = map_coverage(bundle, chunk, manifest, llm)
        mappings.extend(cov.get("mappings", []))
    _progress(f"{bundle['skill_name']}: mapped {len(mappings)}; scoring ...")
    scores = score_coverage(bundle, manifest, issues, mappings)
    return {
        "coverage_version": COVERAGE_VERSION,
        "skill_name": bundle["skill_name"],
        "manifest": str(manifest_path),
        "units_in_manifest": len(manifest.get("capabilities", [])),
        "scores": scores,
    }


# ── claude-CLI-only transport (raises loudly; no OpenRouter masking) ─────────────


def _claude_only(prompt: str, max_tokens: int = 8000, temperature: float = 0.0) -> str:  # noqa: ARG001
    import shutil
    claude = shutil.which("claude")
    if not claude:
        raise FileNotFoundError("`claude` CLI not on PATH")
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    model = os.environ.get("MLEVAL_LLM_MODEL", "").strip()
    cmd = [claude, "-p", prompt, "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): "
                           f"{(proc.stderr or '')[:300]}")
    if not (proc.stdout or "").strip():
        raise RuntimeError("claude CLI returned empty output")
    return proc.stdout


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("skill_dir", type=Path, help="skill bundle dir (has SKILL.md)")
    p.add_argument("manifest", type=Path, help="compiled capabilities.json")
    p.add_argument("--out", type=Path, help="write the full report JSON here")
    p.add_argument("--issues-cache", type=Path,
                   help="cache file for extracted issues (default: <out>.issues.json); "
                        "resumes extraction across re-runs")
    p.add_argument("--chunk-size", type=int, default=12,
                   help="issues per coverage-mapping claude call (default 12)")
    p.add_argument("--json", action="store_true", help="print the full report JSON")
    args = p.parse_args(argv)

    cache = args.issues_cache
    if cache is None and args.out is not None:
        cache = args.out.with_suffix(".issues.json")
    report = run_coverage_audit(args.skill_dir, args.manifest, _claude_only,
                                cache_path=cache, chunk_size=args.chunk_size)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        s = report["scores"]
        print(f"\n=== coverage-recall audit: {report['skill_name']} "
              f"({report['units_in_manifest']} units) ===")
        print(f"issues: {s['issues_extracted']} extracted "
              f"({s['issues_grounded']} grounded, {s['issues_ungrounded_dropped']} dropped-ungrounded)")
        print(f"COVERAGE_RATE = {s['coverage_rate']}  "
              f"({s['covered']} covered / {s['gaps']} gaps; "
              f"{s['high_severity_gaps']} high-severity)")
        print(f"gaps by stage: {s['gaps_by_stage']} | stage-gated gaps: {s['stage_gated_gaps']}")
        print("\nTOP GAPS (severity-ordered):")
        for g in s["gap_detail"][:12]:
            print(f"  [{g['severity']:6}] {g['id']}  (stages={g['stages']})")
            print(f"           issue: {g['statement']}")
            print(f"           source has: {g['source_fix']}  [{g['source_file']}]")
            print(f"           why gap: {g['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
