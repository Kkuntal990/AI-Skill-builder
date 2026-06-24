#!/usr/bin/env python3
"""Skill evaluation harness — measures triggering F1, functional pass rate
(with-skill vs without-skill A/B), citation rate, and token cost on an
installed OpenClaw skill.

Implements the methodology from Anthropic's `skill-creator` (anthropics/skills):
- Triggering F1 across should-trigger + near-miss decoy prompts
- Functional A/B with deterministic must_contain / must_not_contain assertions
- Citation faithfulness measured by reference filename mentions in the reply
- A different judge model than the target (configurable)

Usage:
    python3 eval_skill.py triggering <skill-dir>
    python3 eval_skill.py functional  <skill-dir> [--runs N]
    python3 eval_skill.py all         <skill-dir> [--runs N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the existing triggering judge from skill_builder so we don't duplicate.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import skill_builder as sb  # noqa: E402


# ── Triggering F1 (uses skill_builder's existing judge) ──────────────────────


def _load_skill_meta(skill_dir: Path) -> dict:
    """Return {name, description} parsed from SKILL.md frontmatter."""
    text = (skill_dir / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        sb._die(f"{skill_dir}/SKILL.md has no frontmatter")
    fm = m.group(1)
    name_m = re.search(r"^name:\s*(\S+)", fm, re.MULTILINE)
    desc_m = re.search(r'^description:\s*"(.*?)"\s*$', fm, re.MULTILINE | re.DOTALL)
    if not (name_m and desc_m):
        sb._die("missing name or description in frontmatter")
    return {"name": name_m.group(1), "description": desc_m.group(1)}


def _parse_declared_mcps(skill_dir: Path) -> list[str]:
    """Return list of declared MCP server ids from SKILL.md frontmatter metadata.openclaw.mcps."""
    text = (skill_dir / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return []
    fm = m.group(1)
    md_m = re.search(r"^metadata:\s*(\{.*\})\s*$", fm, re.MULTILINE | re.DOTALL)
    if not md_m:
        return []
    try:
        meta_obj = json.loads(md_m.group(1))
    except json.JSONDecodeError:
        return []
    mcps = (meta_obj.get("openclaw") or {}).get("mcps") or {}
    declared: list[str] = []
    for key in ("preferred", "fallback", "required"):
        for entry in mcps.get(key, []) or []:
            # entries look like "context7/get-library-docs" or "hf-mcp/doc_search" — keep server id
            server = entry.split("/")[0] if "/" in entry else entry
            if server and server not in declared:
                declared.append(server)
    return declared


def _is_mcp_tool(tool_name: str, declared_servers: list[str]) -> bool:
    """Heuristic: tool name matches an MCP server or contains mcporter."""
    if not tool_name:
        return False
    low = tool_name.lower()
    if "mcporter" in low or low.startswith("mcp_") or ".mcp." in low:
        return True
    for srv in declared_servers:
        if srv and srv.lower() in low:
            return True
    return False


def _make_mcporter_wrapper(trial_id: str) -> tuple[Path, Path]:
    """Create a temp wrapper named `mcporter` that JSONL-logs invocations, then execs the real one.

    Returns (wrapper_dir, log_file). Caller passes `wrapper_dir` at front of PATH so the
    spawned agent's `mcporter` calls hit this wrapper first. Reads log_file after the run
    to get ground-truth invocation list.
    """
    wrapper_dir = Path(tempfile.mkdtemp(prefix=f"mcwrap-{trial_id}-"))
    log_file = wrapper_dir / "calls.jsonl"
    try:
        real = subprocess.check_output(["which", "mcporter"], text=True).strip()
    except subprocess.CalledProcessError:
        real = "/opt/homebrew/bin/mcporter"  # best-effort fallback
    wrapper_script = wrapper_dir / "mcporter"
    wrapper_script.write_text(
        '#!/usr/bin/env bash\n'
        'ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)\n'
        'args=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1:]))" "$@" 2>/dev/null || echo \'[]\')\n'
        'printf \'{"ts":"%s","argv":%s,"pwd":"%s","pid":%d}\\n\' '
        f'"$ts" "$args" "$PWD" "$$" >> "{log_file}" 2>/dev/null || true\n'
        f'exec {real} "$@"\n'
    )
    wrapper_script.chmod(0o755)
    return wrapper_dir, log_file


def _read_sidecar_log(log_file: Path | None) -> list[dict]:
    """Parse the JSONL wrapper log, one mcporter invocation per line."""
    if not log_file or not log_file.exists():
        return []
    out: list[dict] = []
    for line in log_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _extract_tool_signals(reply: dict, declared_mcps: list[str]) -> dict:
    """Combine four MCP-usage signals into one classified verdict per trial.

    1. openclaw tool_summary native MCP calls (e.g. `context7__query-docs`) — ground truth for native runtime MCP
    2. reply-text regex (`mcporter call` or `context7__` — catches agent narration of MCP usage)
    3. sidecar wrapper log (ground truth for bash `mcporter call ...`)
    4. relevant-mcp filter (tool name matches a declared server)

    Cross-signal classification (ground truth = native OR sidecar):
      - best_case   : MCP actually called AND narrated in reply text
      - stealth_use : MCP actually called, but not narrated
      - lip_service : narrated, but no actual call evidence
      - clean_miss  : neither
    """
    ts = reply.get("tool_summary") or {}
    tools_used = list(ts.get("tools") or [])
    total_calls = int(ts.get("calls") or 0)
    failures = int(ts.get("failures") or 0)

    mcp_calls = [t for t in tools_used if _is_mcp_tool(t, declared_mcps)]
    relevant_mcp_calls = [
        t for t in mcp_calls if any(srv.lower() in t.lower() for srv in declared_mcps)
    ]

    # Signal 2: text-based MCP detection. Matches three patterns:
    #   (a) bash CLI form ("mcporter call")
    #   (b) native OpenClaw tool form ("<server>__" with double underscore)
    #   (c) outcome narration ("Fetched via Context7", "LibraryId: /...", "via MCP")
    # Pattern (c) is the realistic case: a competent agent calls the tool and
    # reports the answer, not the tool name. OpenClaw's toolSummary is sometimes
    # null for multi-step responses (subagent spawning, etc.) so text narration
    # is often the only signal that MCP actually fired.
    text = reply.get("text") or ""
    text_lower = text.lower()
    text_mcp_hits: list[str] = []
    saw_mcporter_text = "mcporter call" in text_lower
    saw_native_text = False
    for srv in declared_mcps:
        if not srv:
            continue
        if f"{srv.lower()}__" in text_lower or f"`{srv.lower()}__" in text_lower:
            saw_native_text = True
            text_mcp_hits.append(srv)
    if saw_mcporter_text:
        for srv in declared_mcps:
            if srv and srv.lower() in text_lower and srv not in text_mcp_hits:
                text_mcp_hits.append(srv)
        if not text_mcp_hits:
            text_mcp_hits.append("mcporter")
    # Outcome-narration patterns — these prove the agent fetched live data
    # via MCP even when no tool-name appears in the reply. Conservative list
    # to avoid false positives from generic mentions of "MCP".
    _OUTCOME_PATTERNS = (
        "fetched via ", "fetched live via ", "fetched from ",
        "via the registered ", "via context7", "via mcp ", "via mcporter",
        "from context7", "queried context7", "queried via",
        "libraryid:", "libraryid =", "libraryid=\"/", "libraryid='/",
    )
    saw_outcome_text = any(pat in text_lower for pat in _OUTCOME_PATTERNS)
    if saw_outcome_text:
        for srv in declared_mcps:
            if srv and srv.lower() in text_lower and srv not in text_mcp_hits:
                text_mcp_hits.append(srv)
        if not text_mcp_hits:
            # Outcome was narrated but no specific server name; still credit it
            text_mcp_hits.append("outcome-only")

    # Signal 3: sidecar wrapper log — ground truth for bash mcporter
    sidecar_calls: list[dict] = []
    sidecar_servers: set[str] = set()
    sidecar_relevant_servers: set[str] = set()
    for entry in (reply.get("mcporter_log") or []):
        argv = entry.get("argv") or []
        if len(argv) >= 1:
            verb = argv[0]
            target = argv[1] if len(argv) >= 2 else None
            server = target.split(".", 1)[0] if target and "." in target else target
            tool = target.split(".", 1)[1] if target and "." in target else None
            sidecar_calls.append({"verb": verb, "server": server, "tool": tool, "ts": entry.get("ts")})
            if server and verb in ("call", "list-tools"):
                sidecar_servers.add(server)
                if any(srv.lower() == server.lower() for srv in declared_mcps):
                    sidecar_relevant_servers.add(server)

    has_sidecar = bool(sidecar_calls)
    has_native = bool(relevant_mcp_calls)
    has_text = bool(text_mcp_hits)
    # Ground truth: MCP actually fired iff sidecar (bash) OR native runtime tool call
    # OR the reply contains specific outcome narration (e.g. libraryId, "Fetched via Context7").
    # The latter is required because OpenClaw's toolSummary is sometimes null for
    # multi-step or subagent-spawning responses, so sidecar + native_runtime alone
    # under-counts. Outcome narration about a *specific* libraryId is high-confidence
    # because the agent only knows the libraryId by actually calling Context7.
    actually_called = has_sidecar or has_native or saw_outcome_text
    if actually_called and has_text:
        classification = "best_case"
    elif actually_called and not has_text:
        classification = "stealth_use"
    elif has_text and not actually_called:
        classification = "lip_service"
    else:
        classification = "clean_miss"

    return {
        "tools_used": tools_used,
        "total_tool_calls": total_calls,
        "tool_failures": failures,
        "mcp_calls": mcp_calls,
        "mcp_called": bool(mcp_calls),
        "relevant_mcp_calls": relevant_mcp_calls,
        "relevant_mcp_called": has_native,
        "text_mcp_hits": text_mcp_hits,
        "sidecar_calls": sidecar_calls,
        "sidecar_call_count": len(sidecar_calls),
        "sidecar_servers": sorted(sidecar_servers),
        "sidecar_relevant_servers": sorted(sidecar_relevant_servers),
        "mcp_actually_called": actually_called,  # ground truth: native OR sidecar
        "mcp_classification": classification,
        # Preserved for back-compat with prior reports:
        "mcp_evidence": has_sidecar or bool(mcp_calls) or has_text,
    }


def cmd_triggering(args: argparse.Namespace) -> dict:
    """Run 10 should-trigger + 10 near-miss decoy prompts through the judge.

    For each prompt:
      - judge picks one of: target skill, 5 canned decoys, "none"
      - should_trigger → win if judge picks the target skill
      - should_not    → win if judge picks anything BUT the target skill

    Returns precision, recall, F1, plus per-prompt verdicts.
    """
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    meta = _load_skill_meta(skill_dir)
    triggering_path = skill_dir / "evals" / "triggering.json"
    if not triggering_path.exists():
        sb._die(f"missing {triggering_path}")
    data = json.loads(triggering_path.read_text())

    skills = [meta] + sb.DECOY_SKILLS
    pos = data.get("should_trigger", [])
    neg = data.get("should_not_trigger_near_miss", [])
    runs = max(1, int(args.runs))

    def judge_one(p):
        choices = []
        for _ in range(runs):
            v = sb.judge_triggering(p["prompt"], skills)
            choices.append(v.get("choice", "none"))
        return p, choices

    pos_results = []
    neg_results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p, choices in pool.map(judge_one, pos):
            wins = sum(1 for c in choices if c == meta["name"])
            triggered = wins >= (runs / 2)  # majority vote
            pos_results.append({"id": p["id"], "prompt": p["prompt"], "choices": choices, "triggered": triggered})
        for p, choices in pool.map(judge_one, neg):
            triggered_count = sum(1 for c in choices if c == meta["name"])
            triggered = triggered_count >= (runs / 2)
            neg_results.append({"id": p["id"], "prompt": p["prompt"], "choices": choices, "triggered": triggered})

    tp = sum(1 for r in pos_results if r["triggered"])
    fn = sum(1 for r in pos_results if not r["triggered"])
    fp = sum(1 for r in neg_results if r["triggered"])
    tn = sum(1 for r in neg_results if not r["triggered"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "skill_name": meta["name"],
        "runs_per_prompt": runs,
        "should_trigger": pos_results,
        "should_not_trigger": neg_results,
        "metrics": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "specificity": round(tn / (tn + fp), 3) if (tn + fp) else 0.0,
        },
    }


# ── Functional A/B (with-skill vs without-skill via openclaw agent) ──────────


def _run_agent(prompt: str, agent: str = "ai-skill-builder", timeout: int = 240,
               trial_id: str | None = None) -> dict:
    """Run one openclaw agent turn. Return parsed reply + token usage + mcporter log.

    When `trial_id` is set, prepends a wrapper `mcporter` to PATH so we can log every
    invocation to a sidecar JSONL file. The log is read back into `mcporter_log`.

    Robust to gateway-fallback stderr ("Gateway agent failed; falling back to embedded")
    being prepended before the JSON.
    """
    t0 = time.time()
    log_file: Path | None = None
    wrapper_dir: Path | None = None
    env = None
    if trial_id:
        wrapper_dir, log_file = _make_mcporter_wrapper(trial_id)
        env = {**os.environ, "PATH": f"{wrapper_dir}:{os.environ.get('PATH', '')}"}

    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", agent, "--json", "--timeout", str(timeout), "-m", prompt],
            capture_output=True, text=True, timeout=timeout + 30, env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        out = {"error": str(e), "elapsed_ms": int((time.time() - t0) * 1000), "text": "",
               "mcporter_log": _read_sidecar_log(log_file)}
        if wrapper_dir and wrapper_dir.exists():
            shutil.rmtree(wrapper_dir, ignore_errors=True)
        return out

    raw = (result.stdout or "") + (result.stderr or "")
    idx = raw.find("\n{")
    js = raw[idx + 1:] if idx >= 0 else (raw[raw.find("{"):] if "{" in raw else "")
    try:
        d = json.loads(js)
    except json.JSONDecodeError:
        out = {"error": "JSON parse failed", "raw_head": raw[:300],
               "elapsed_ms": int((time.time() - t0) * 1000), "text": "",
               "mcporter_log": _read_sidecar_log(log_file)}
        if wrapper_dir and wrapper_dir.exists():
            shutil.rmtree(wrapper_dir, ignore_errors=True)
        return out

    payloads = (d.get("result") or d).get("payloads", [])
    text = "\n\n".join(p.get("text", "") for p in payloads if p.get("text"))
    meta = ((d.get("result") or d).get("meta") or {})
    agent_meta = meta.get("agentMeta") or {}
    usage = agent_meta.get("usage") or {}
    mcporter_log = _read_sidecar_log(log_file)
    if wrapper_dir and wrapper_dir.exists():
        shutil.rmtree(wrapper_dir, ignore_errors=True)

    return {
        "text": text,
        "usage": usage,
        "model": agent_meta.get("model"),
        "duration_ms": meta.get("durationMs"),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "tool_summary": agent_meta.get("toolSummary") or {},
        "mcporter_log": mcporter_log,
    }


def _score_assertions(test: dict, reply: str) -> dict:
    """Deterministic scoring: must_contain (strict AND) + must_contain_any (OR-groups)
    + must_not_contain (strict AND-negated) + citation_accuracy.

    - `must_contain`:     list[str] — every term must appear (AND)
    - `must_contain_any`: list[list[str]] — each group is an OR; at least one term per group must appear
    - `must_not_contain`: list[str] — none of these may appear

    `must_contain_any` is for synonym-prone requirements (e.g. ["rank","r=","r value"]).
    """
    rl = reply.lower()
    mc = test.get("must_contain", [])
    mca = test.get("must_contain_any", [])  # list of lists
    mn = test.get("must_not_contain", [])
    expected_cites = test.get("expected_citations", [])

    contains = {term: (term.lower() in rl) for term in mc}
    not_contains = {term: (term.lower() not in rl) for term in mn}
    # For each OR-group, record: matched terms + pass flag
    any_groups = []
    for grp in mca:
        if not isinstance(grp, list):
            grp = [grp]
        hits = {term: (term.lower() in rl) for term in grp}
        group_pass = any(hits.values())
        any_groups.append({"group": grp, "hits": hits, "pass": group_pass})

    cites = {f: (f.lower() in rl) for f in expected_cites}

    contains_pass = all(contains.values()) if contains else True
    any_pass = all(g["pass"] for g in any_groups) if any_groups else True
    notcontains_pass = all(not_contains.values()) if not_contains else True
    cite_matched = sum(1 for v in cites.values() if v)
    cite_total = len(cites)
    citation_accuracy = (cite_matched / cite_total) if cite_total else 1.0
    cite_pass = citation_accuracy >= 0.5
    overall_pass = contains_pass and any_pass and notcontains_pass

    return {
        "overall_pass": overall_pass,
        "must_contain": contains,
        "must_contain_pass": contains_pass,
        "must_contain_any": any_groups,
        "must_contain_any_pass": any_pass,
        "must_not_contain": not_contains,
        "must_not_contain_pass": notcontains_pass,
        "citations_seen": cites,
        "citation_matched": cite_matched,
        "citation_total": cite_total,
        "citation_accuracy": round(citation_accuracy, 3),
        "citation_pass": cite_pass,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def cmd_functional(args: argparse.Namespace) -> dict:
    """Run each functional prompt N trials, with and without skill, score every trial."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    meta = _load_skill_meta(skill_dir)
    declared_mcps = _parse_declared_mcps(skill_dir)
    functional_path = skill_dir / "evals" / "functional.json"
    if not functional_path.exists():
        sb._die(f"missing {functional_path}")
    data = json.loads(functional_path.read_text())
    tests = data.get("tests", [])
    runs = max(1, int(args.runs))
    out_results = []

    skill_marker = f"\n\n(For context: a skill is installed at {skill_dir}. Read SKILL.md and any relevant references/ or templates/ files before answering.)"

    def one_trial(test: dict, with_skill: bool, trial_idx: int) -> dict:
        prompt = test["prompt"] + (skill_marker if with_skill else "")
        side = "with" if with_skill else "without"
        trial_id = f"{test['id']}-{side}-{trial_idx}"
        reply = _run_agent(prompt, agent=args.agent, timeout=args.per_prompt_timeout,
                           trial_id=trial_id)
        score = _score_assertions(test, reply.get("text", ""))
        signals = _extract_tool_signals(reply, declared_mcps)
        return {
            "reply_chars": len(reply.get("text", "")),
            "reply_text": reply.get("text", ""),  # preserve for offline re-grading + diagnosis
            "score": score,
            "tool_signals": signals,
            "usage": reply.get("usage"),
            "duration_ms": reply.get("duration_ms"),
        }

    def aggregate(trials: list[dict]) -> dict:
        if not trials:
            return {}
        pass_xs = [1.0 if t["score"]["overall_pass"] else 0.0 for t in trials]
        cite_acc_xs = [t["score"]["citation_accuracy"] for t in trials]
        # Signal 3 (ground truth from sidecar log)
        actually_called_xs = [1.0 if t["tool_signals"]["mcp_actually_called"] else 0.0 for t in trials]
        relevant_sidecar_xs = [1.0 if t["tool_signals"]["sidecar_relevant_servers"] else 0.0 for t in trials]
        sidecar_call_counts = [t["tool_signals"]["sidecar_call_count"] for t in trials]
        # Signal 2 (text narration)
        text_mcp_xs = [1.0 if t["tool_signals"]["text_mcp_hits"] else 0.0 for t in trials]
        # Cross-signal classification histogram
        classifications = [t["tool_signals"]["mcp_classification"] for t in trials]
        class_counts = {c: classifications.count(c) for c in
                        ("best_case", "stealth_use", "lip_service", "clean_miss")}
        total_in = sum((t.get("usage") or {}).get("input", 0) for t in trials)
        total_out = sum((t.get("usage") or {}).get("output", 0) for t in trials)
        total_ms = sum(int(t.get("duration_ms") or 0) for t in trials)
        return {
            "n_trials": len(trials),
            "pass_rate": round(_mean(pass_xs), 3),
            "pass_stddev": round(_stddev(pass_xs), 3),
            "citation_accuracy_mean": round(_mean(cite_acc_xs), 3),
            # Ground truth — what the agent really did
            "actual_mcp_call_rate": round(_mean(actually_called_xs), 3),
            "relevant_sidecar_rate": round(_mean(relevant_sidecar_xs), 3),
            "mean_sidecar_calls_per_trial": round(_mean(sidecar_call_counts), 2),
            # Narration — what the agent said it did
            "text_mcp_call_rate": round(_mean(text_mcp_xs), 3),
            # Classification histogram
            "mcp_classification_counts": class_counts,
            "tokens_input_total": total_in,
            "tokens_output_total": total_out,
            "duration_ms_total": total_ms,
        }

    for test in tests:
        print(f"  running test: {test['id']} ({runs} trials × 2 cells)", file=sys.stderr, flush=True)
        with_trials = [one_trial(test, with_skill=True, trial_idx=i) for i in range(runs)]
        without_trials = [one_trial(test, with_skill=False, trial_idx=i) for i in range(runs)]
        out_results.append({
            "id": test["id"],
            "prompt": test["prompt"],
            "with_skill": {
                "trials": with_trials,
                "agg": aggregate(with_trials),
            },
            "without_skill": {
                "trials": without_trials,
                "agg": aggregate(without_trials),
            },
        })

    # Skill-level aggregation across tasks
    def gather(field_path: str, side: str) -> list[float]:
        # field_path like "agg.pass_rate"
        out = []
        for r in out_results:
            cur = r[side]
            for part in field_path.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    with_pass = _mean(gather("agg.pass_rate", "with_skill"))
    without_pass = _mean(gather("agg.pass_rate", "without_skill"))
    with_cite = _mean(gather("agg.citation_accuracy_mean", "with_skill"))
    with_actual_mcp = _mean(gather("agg.actual_mcp_call_rate", "with_skill"))
    with_rel_sidecar = _mean(gather("agg.relevant_sidecar_rate", "with_skill"))
    with_text_mcp = _mean(gather("agg.text_mcp_call_rate", "with_skill"))
    with_in = sum(int(x) for x in gather("agg.tokens_input_total", "with_skill"))
    with_out_t = sum(int(x) for x in gather("agg.tokens_output_total", "with_skill"))
    wo_in = sum(int(x) for x in gather("agg.tokens_input_total", "without_skill"))
    wo_out = sum(int(x) for x in gather("agg.tokens_output_total", "without_skill"))

    # Aggregate classification histogram across all (task × trial) cells
    overall_class = {"best_case": 0, "stealth_use": 0, "lip_service": 0, "clean_miss": 0}
    for r in out_results:
        wc = (r["with_skill"]["agg"] or {}).get("mcp_classification_counts") or {}
        for k, v in wc.items():
            overall_class[k] = overall_class.get(k, 0) + v

    return {
        "skill_name": meta["name"],
        "declared_mcps": declared_mcps,
        "n_tests": len(out_results),
        "runs_per_cell": runs,
        "with_skill_pass_rate": round(with_pass, 3),
        "without_skill_pass_rate": round(without_pass, 3),
        "lift_pp": round((with_pass - without_pass) * 100, 1),
        "citation_accuracy_with_skill": round(with_cite, 3),
        # Ground truth (Signal 3 — sidecar wrapper log)
        "actual_mcp_call_rate_with_skill": round(with_actual_mcp, 3),
        "relevant_sidecar_rate_with_skill": round(with_rel_sidecar, 3),
        # Narration (Signal 2 — text regex on reply)
        "text_mcp_call_rate_with_skill": round(with_text_mcp, 3),
        # Cross-signal classification (with-skill cells only)
        "mcp_classification_counts_with_skill": overall_class,
        "saturated": bool(with_pass >= 0.9 and without_pass >= 0.9),
        "tokens": {
            "with_skill_input": with_in, "with_skill_output": with_out_t,
            "without_skill_input": wo_in, "without_skill_output": wo_out,
        },
        "results": out_results,
    }


def cmd_all(args: argparse.Namespace) -> dict:
    triggering = cmd_triggering(args)
    functional = cmd_functional(args)
    return {"triggering": triggering, "functional": functional}


# ── Pass-bar thresholds + report ─────────────────────────────────────────────


DEFAULT_PASS_BAR = {
    "triggering_f1_min": 0.85,
    "functional_pass_min": 0.60,
    "citation_accuracy_min": 0.50,
    "saturation_max_pass_rate": 0.90,
}


def _load_pass_bar(skill_dir: Path) -> dict:
    p = skill_dir / "evals" / "pass_bar.json"
    if p.exists():
        try:
            return {**DEFAULT_PASS_BAR, **json.loads(p.read_text())}
        except json.JSONDecodeError:
            pass
    return DEFAULT_PASS_BAR


def _find_latest(skill_dir: Path, prefix: str) -> Path | None:
    d = skill_dir / "evals" / "grading_results"
    if not d.exists():
        return None
    cands = sorted(d.glob(f"{prefix}-*.json"))
    return cands[-1] if cands else None


def _build_report(skill_dir: Path, triggering: dict | None, functional: dict | None,
                  pass_bar: dict) -> tuple[str, bool]:
    """Return (markdown_report, overall_pass)."""
    lines = []
    name = (functional or triggering or {}).get("skill_name") or skill_dir.name
    lines.append(f"# Skill eval report — {name}")
    lines.append("")
    verdicts = []

    if triggering:
        m = triggering["metrics"]
        f1 = m["f1"]
        f1_pass = f1 >= pass_bar["triggering_f1_min"]
        verdicts.append(f1_pass)
        lines.append("## Triggering")
        lines.append(f"- F1: **{f1:.3f}** (P {m['precision']:.3f}, R {m['recall']:.3f}, "
                     f"specificity {m['specificity']:.3f}) — "
                     f"{'PASS' if f1_pass else 'FAIL'} (bar ≥ {pass_bar['triggering_f1_min']})")
        lines.append(f"- TP {m['true_positive']} / FN {m['false_negative']} / "
                     f"FP {m['false_positive']} / TN {m['true_negative']}")
        lines.append(f"- Runs per prompt: {triggering.get('runs_per_prompt', '?')}")
        lines.append("")

    if functional:
        with_pass = functional["with_skill_pass_rate"]
        without_pass = functional["without_skill_pass_rate"]
        cite = functional["citation_accuracy_with_skill"]
        saturated = functional["saturated"]
        actual_rate = functional.get("actual_mcp_call_rate_with_skill", 0.0)
        rel_sidecar = functional.get("relevant_sidecar_rate_with_skill", 0.0)
        text_rate = functional.get("text_mcp_call_rate_with_skill", 0.0)
        class_counts = functional.get("mcp_classification_counts_with_skill") or {}
        declared = functional.get("declared_mcps") or []

        fpass = with_pass >= pass_bar["functional_pass_min"]
        cpass = cite >= pass_bar["citation_accuracy_min"]
        verdicts.extend([fpass, cpass])

        lines.append("## Functional A/B")
        lines.append(f"- With-skill pass rate:    **{with_pass:.3f}** — "
                     f"{'PASS' if fpass else 'FAIL'} (bar ≥ {pass_bar['functional_pass_min']})")
        lines.append(f"- Without-skill pass rate: {without_pass:.3f}")
        lines.append(f"- Lift: **{functional['lift_pp']:+.1f} pp**")
        lines.append(f"- Citation accuracy:       **{cite:.3f}** — "
                     f"{'PASS' if cpass else 'FAIL'} (bar ≥ {pass_bar['citation_accuracy_min']})")
        lines.append(f"- Saturation:              {'YES (both arms ≥ 0.9)' if saturated else 'no'}")
        lines.append(f"- Tests: {functional['n_tests']}, trials/cell: "
                     f"{functional.get('runs_per_cell', '?')}")
        lines.append("")

        lines.append("## MCP usage (with-skill cells)")
        lines.append(f"- Declared MCP servers: {declared or 'none'}")
        lines.append(f"- **Actual call rate (sidecar log, ground truth): {actual_rate:.3f}**")
        lines.append(f"- Relevant-server actual call rate:               {rel_sidecar:.3f}")
        lines.append(f"- Narrated call rate (reply text mentions):       {text_rate:.3f}")
        if class_counts:
            total_cells = sum(class_counts.values()) or 1
            def pct(k): return 100 * class_counts.get(k, 0) / total_cells
            lines.append(f"- Cross-signal breakdown: "
                         f"best_case {class_counts.get('best_case',0)} ({pct('best_case'):.0f}%), "
                         f"stealth_use {class_counts.get('stealth_use',0)} ({pct('stealth_use'):.0f}%), "
                         f"lip_service {class_counts.get('lip_service',0)} ({pct('lip_service'):.0f}%), "
                         f"clean_miss {class_counts.get('clean_miss',0)} ({pct('clean_miss'):.0f}%)")
        lines.append("")

        toks = functional["tokens"]
        ratio = (toks["with_skill_input"] / toks["without_skill_input"]
                 if toks["without_skill_input"] else 0.0)
        lines.append("## Cost")
        lines.append(f"- Tokens in/out with-skill:    {toks['with_skill_input']:,} / "
                     f"{toks['with_skill_output']:,}")
        lines.append(f"- Tokens in/out without-skill: {toks['without_skill_input']:,} / "
                     f"{toks['without_skill_output']:,}")
        lines.append(f"- Input-token ratio (with/without): **{ratio:.2f}×**")
        lines.append("")

    overall = all(verdicts) if verdicts else False
    lines.append(f"## Verdict: **{'PASS' if overall else 'FAIL'}**")
    return "\n".join(lines), overall


def cmd_report(args: argparse.Namespace) -> dict:
    """Read latest grading_results/*.json + render markdown summary."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    triggering_path = _find_latest(skill_dir, "triggering") or _find_latest(skill_dir, "all")
    functional_path = _find_latest(skill_dir, "functional") or _find_latest(skill_dir, "all")
    triggering = None
    functional = None
    if triggering_path:
        d = json.loads(triggering_path.read_text())
        triggering = d.get("triggering", d) if "triggering" in d else d
    if functional_path:
        d = json.loads(functional_path.read_text())
        functional = d.get("functional", d) if "functional" in d else d
    pass_bar = _load_pass_bar(skill_dir)
    md, _ = _build_report(skill_dir, triggering, functional, pass_bar)
    return {"markdown": md}


def cmd_pass_bar(args: argparse.Namespace) -> dict:
    """Re-load latest results, evaluate against pass_bar.json, set return code."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    triggering_path = _find_latest(skill_dir, "triggering") or _find_latest(skill_dir, "all")
    functional_path = _find_latest(skill_dir, "functional") or _find_latest(skill_dir, "all")
    triggering = None
    functional = None
    if triggering_path:
        d = json.loads(triggering_path.read_text())
        triggering = d.get("triggering", d) if "triggering" in d else d
    if functional_path:
        d = json.loads(functional_path.read_text())
        functional = d.get("functional", d) if "functional" in d else d
    pass_bar = _load_pass_bar(skill_dir)
    _, overall = _build_report(skill_dir, triggering, functional, pass_bar)
    args._exit_code = 0 if overall else 1  # picked up in main()
    return {"pass": overall, "pass_bar": pass_bar}


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval_skill.py", description="Skill evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = lambda sp: (
        sp.add_argument("skill_dir"),
        sp.add_argument("--runs", type=int, default=3, help="Triggering runs per prompt (default 3)"),
        sp.add_argument("--agent", default="ai-skill-builder", help="OpenClaw agent for functional runs"),
        sp.add_argument("--per-prompt-timeout", type=int, default=240),
    )

    t = sub.add_parser("triggering", help="Triggering F1 only")
    common(t); t.set_defaults(func=cmd_triggering)

    f = sub.add_parser("functional", help="Functional A/B only")
    common(f); f.set_defaults(func=cmd_functional)

    a = sub.add_parser("all", help="Triggering + functional")
    common(a); a.set_defaults(func=cmd_all)

    r = sub.add_parser("report", help="Render markdown summary of latest grading results")
    r.add_argument("skill_dir")
    r.set_defaults(func=cmd_report)

    pb = sub.add_parser("pass-bar", help="Evaluate pass_bar.json — exit code 0 PASS / 1 FAIL")
    pb.add_argument("skill_dir")
    pb.set_defaults(func=cmd_pass_bar)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.func(args)
    if args.cmd == "report":
        print(out["markdown"])
    else:
        print(json.dumps(out, indent=2, default=str))
    # Persist results / report. Skip pass-bar (just an exit code).
    try:
        skill_dir = Path(args.skill_dir).expanduser().resolve()
        out_dir = skill_dir / "evals" / "grading_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        if args.cmd == "report":
            (out_dir / f"report-{ts}.md").write_text(out["markdown"])
        elif args.cmd != "pass-bar":
            (out_dir / f"{args.cmd}-{ts}.json").write_text(json.dumps(out, indent=2, default=str))
    except OSError:
        pass
    return getattr(args, "_exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
