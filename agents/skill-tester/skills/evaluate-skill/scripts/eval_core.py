#!/usr/bin/env python3
"""Transport-agnostic skill-eval primitives, shared by `eval_skill.py` (the CLI
harness) and `skill_builder.py` (the build-time eval gate).

M0 refactor (Skills-phase3): these functions used to live in `eval_skill.py`,
which imports `skill_builder`. Moving them here — with the triggering judge
*injected* as a parameter rather than imported — lets `skill_builder` call the
eval primitives for its ship-gate without a circular import. `eval_core` imports
nothing from `skill_builder`.

Public API (used by both callers):
    run_triggering(skill_dir, judge_fn, decoys, *, runs=3) -> dict
    run_functional(skill_dir, *, agent="ai-skill-builder", runs=3,
                   per_prompt_timeout=240) -> dict
    run_activation(skill_dir, *, model="", runs=3, per_prompt_timeout=180,
                   max_concurrency=3) -> dict
    build_report(skill_dir, triggering, functional, pass_bar, activation=None) -> (md, passed)
    load_pass_bar(skill_dir) -> dict
    find_latest(skill_dir, prefix) -> Path | None
    load_skill_meta(skill_dir) -> {name, description}
"""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _die(msg: str):
    """Print an error and exit. Callers that must not abort (the build gate)
    should guard their inputs (e.g. check the evals file exists) before calling."""
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# ── Skill metadata / MCP declaration parsing ─────────────────────────────────


def load_skill_meta(skill_dir: Path) -> dict:
    """Return {name, description} parsed from SKILL.md frontmatter."""
    text = (skill_dir / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        _die(f"{skill_dir}/SKILL.md has no frontmatter")
    fm = m.group(1)
    name_m = re.search(r"^name:\s*(\S+)", fm, re.MULTILINE)
    desc_m = re.search(r'^description:\s*"(.*?)"\s*$', fm, re.MULTILINE | re.DOTALL)
    if not (name_m and desc_m):
        _die("missing name or description in frontmatter")
    return {"name": name_m.group(1), "description": desc_m.group(1)}


def parse_declared_mcps(skill_dir: Path) -> list[str]:
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
    """Create a temp wrapper named `mcporter` that JSONL-logs invocations, then execs the real one."""
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
            text_mcp_hits.append("outcome-only")

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
        "mcp_actually_called": actually_called,
        "mcp_classification": classification,
        "mcp_evidence": has_sidecar or bool(mcp_calls) or has_text,
    }


# ── claude -p organic-activation executor (Skills-3.0 Phase 3-3) ─────────────

SKILLBUILD_LLM_MODEL = os.environ.get("SKILLBUILD_LLM_MODEL", "opus").strip() or "opus"

# Mirrors agents/skill-tester/{SOUL,AGENTS}.md — a clean baseline executor.
_SKILLTESTER_SYSTEM = (
    "You are a bare evaluation-target agent. No personality, no greetings. Answer the "
    "user's prompt directly and concisely. If one of your available skills is genuinely "
    "relevant, use it; otherwise answer without it. Never force skill usage."
)


def _run_claude_executor(prompt: str, skill_dir: "Path | None" = None,
                         timeout: int = 180, model: str = "") -> dict:
    """Run one eval turn on `claude -p --model <opus>` (Claude subscription).

    If `skill_dir` is given, the skill is copied into a temp `.claude/skills/<name>/`
    catalog and made AVAILABLE (not force-read), so activation is ORGANIC. We detect the
    Skill/Read tool-call from stream-json events and capture the final reply text.
    """
    model = model or SKILLBUILD_LLM_MODEL
    workdir = Path(tempfile.mkdtemp(prefix="skilltester-"))
    skill_key = ""
    t0 = time.time()
    try:
        if skill_dir is not None:
            skill_key = f"{skill_dir.name}-{uuid.uuid4().hex[:8]}"
            dest = workdir / ".claude" / "skills" / skill_key
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_dir / "SKILL.md", dest / "SKILL.md")
            for sub in ("references", "scripts", "templates"):
                if (skill_dir / sub).is_dir():
                    shutil.copytree(skill_dir / sub, dest / sub)
        cmd = ["claude", "-p", prompt, "--model", model,
               "--output-format", "stream-json", "--verbose",
               "--include-partial-messages",
               "--append-system-prompt", _SKILLTESTER_SYSTEM]
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                cwd=str(workdir), env=env)
        activated = False
        reply = ""
        buffer = ""

        def _match(name: str, blob: str) -> bool:
            if name not in ("Skill", "Read"):
                return False
            return (not skill_key) or (skill_key in blob) or (skill_dir and skill_dir.name in blob)

        try:
            while time.time() - t0 < timeout:
                done = proc.poll() is not None
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if ready:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if chunk:
                        buffer += chunk.decode("utf-8", "replace")
                elif done:
                    break
                elif not ready:
                    continue
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("type")
                    if et == "assistant":
                        for ci in ev.get("message", {}).get("content", []):
                            if ci.get("type") == "tool_use" and _match(
                                    ci.get("name", ""), json.dumps(ci.get("input", {}))):
                                activated = True
                    elif et == "stream_event":
                        se = ev.get("event", {})
                        if se.get("type") == "content_block_start":
                            cb = se.get("content_block", {})
                            if cb.get("type") == "tool_use" and cb.get("name") in ("Skill", "Read"):
                                activated = activated or (not skill_key)
                    elif et == "result":
                        r = ev.get("result")
                        if isinstance(r, str):
                            reply = r
                if done and "\n" not in buffer:
                    break
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        return {"activated": activated, "reply": reply.strip(), "model": model,
                "elapsed_ms": int((time.time() - t0) * 1000)}
    except (FileNotFoundError, OSError) as e:
        return {"activated": False, "reply": "", "model": model,
                "elapsed_ms": int((time.time() - t0) * 1000), "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── openclaw agent executor (with/without functional A/B) ────────────────────


def _run_agent(prompt: str, agent: str = "ai-skill-builder", timeout: int = 240,
               trial_id: str | None = None) -> dict:
    """Run one openclaw agent turn. Return parsed reply + token usage + mcporter log."""
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


def _score_assertions(test: dict, reply: str, grader_fn=None) -> dict:
    """Deterministic scoring: must_contain (strict AND) + must_contain_any (OR-groups)
    + must_not_contain (strict AND-negated) + citation_accuracy. Optional LLM grader
    tier (P1.5): `test["judge"]` assertions are graded by grader_fn(reply, text) ->
    {passed, evidence}; skipped (no effect on overall_pass) when grader_fn is None, so
    deterministic behavior is preserved for skills without judge assertions."""
    rl = reply.lower()
    mc = test.get("must_contain", [])
    mca = test.get("must_contain_any", [])  # list of lists
    mn = test.get("must_not_contain", [])
    expected_cites = test.get("expected_citations", [])

    contains = {term: (term.lower() in rl) for term in mc}
    not_contains = {term: (term.lower() not in rl) for term in mn}
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

    # P1.5 LLM grader tier — judge-type assertions substrings can't check. Skipped
    # (no effect on overall_pass) when grader_fn is None → deterministic behavior kept.
    judge_results = []
    judge_pass = True
    for spec in (test.get("judge") or []):
        txt = spec if isinstance(spec, str) else (spec or {}).get("text", "")
        if not txt:
            continue
        if grader_fn is not None:
            v = grader_fn(reply, txt) or {}
            passed = bool(v.get("passed"))
            judge_results.append({"text": txt, "passed": passed, "evidence": v.get("evidence", "")})
            judge_pass = judge_pass and passed
        else:
            judge_results.append({"text": txt, "passed": None, "evidence": "", "skipped": "no grader"})

    overall_pass = contains_pass and any_pass and notcontains_pass and judge_pass

    return {
        "overall_pass": overall_pass,
        "judge": judge_results,
        "judge_pass": judge_pass,
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


# ── P1.4 assertion analyzer (meta-evaluation) ────────────────────────────────


def analyze(functional_result: dict, *, flaky_stddev: float = 0.3,
            token_ratio: float = 1.5) -> dict:
    """Flag eval-quality problems across the with/without cells (skill-creator's
    analyzer pass). Does NOT propose skill fixes — that's the repair step.

    - non_discriminating: a test whose pass_rate is identical with and without the
      skill doesn't measure skill value → down-weight it in the gate (M5).
    - flaky: a cell whose pass_stddev exceeds flaky_stddev is high-variance/unreliable.
    - time_token: with-cell costs >token_ratio× the tokens for no pass gain.
    """
    tests = functional_result.get("results", []) or []
    non_discriminating, flaky, time_token = [], [], []
    for r in tests:
        w = (r.get("with_skill") or {}).get("agg") or {}
        wo = (r.get("without_skill") or {}).get("agg") or {}
        wp, op = w.get("pass_rate"), wo.get("pass_rate")
        if wp is not None and op is not None and wp == op:
            non_discriminating.append(r.get("id"))
        for cell, label in ((w, "with"), (wo, "without")):
            sd = cell.get("pass_stddev") or 0.0
            if sd > flaky_stddev:
                flaky.append({"id": r.get("id"), "cell": label, "stddev": sd})
        if wp is not None and op is not None and wp <= op:
            wt = (w.get("tokens_input_total", 0) or 0) + (w.get("tokens_output_total", 0) or 0)
            ot = (wo.get("tokens_input_total", 0) or 0) + (wo.get("tokens_output_total", 0) or 0)
            if ot and wt > token_ratio * ot:
                time_token.append({"id": r.get("id"), "with_tokens": wt, "without_tokens": ot})
    return {
        "n_tests": len(tests),
        "non_discriminating": non_discriminating,
        "n_non_discriminating": len(non_discriminating),
        "flaky": flaky,
        "time_token": time_token,
        "note": ("non-discriminating tests pass/fail identically with and without the skill — "
                 "they don't measure skill value; down-weight them in the gate"),
    }


# ── P1.5 LLM grader (claude -p) — optional judge for non-substring assertions ──

_GRADE_PROMPT = (
    "You are grading whether a REPLY satisfies an ASSERTION about it. Be strict and "
    "literal. Reply with ONLY JSON: {{\"passed\": true|false, \"evidence\": \"<short quote "
    "or one-line reason>\"}}.\n\nASSERTION: {a}\n\nREPLY:\n{r}"
)


def _claude_text(prompt: str, *, model: str = "", timeout: int = 120) -> str:
    """Minimal `claude -p ... --output-format text` call (Claude subscription).
    Best-effort: returns "" on any failure."""
    model = model or SKILLBUILD_LLM_MODEL
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        r = subprocess.run(["claude", "-p", prompt, "--model", model, "--output-format", "text"],
                           capture_output=True, text=True, timeout=timeout, env=env)
        return (r.stdout or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def make_claude_grader(*, model: str = "", timeout: int = 120):
    """Return a grader_fn(reply, assertion_text) -> {passed, evidence} backed by
    `claude -p`. Used to grade `test["judge"]` assertions in run_functional."""
    model = model or SKILLBUILD_LLM_MODEL

    def grader(reply: str, assertion_text: str) -> dict:
        out = _claude_text(_GRADE_PROMPT.format(a=assertion_text, r=(reply or "")[:6000]),
                           model=model, timeout=timeout)
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if not m:
            return {"passed": False, "evidence": "grader returned no JSON"}
        try:
            d = json.loads(m.group(0))
            return {"passed": bool(d.get("passed")), "evidence": str(d.get("evidence", ""))[:300]}
        except json.JSONDecodeError:
            return {"passed": False, "evidence": "grader JSON parse failed"}

    return grader


# ── Triggering judge + description optimizer (M0-relocate: were in skill_builder) ──
# These moved here so eval_skill.py needs NOTHING from skill_builder — the whole
# behavioral-eval stack is self-contained and can live inside the skill-tester agent.
# LLM calls go through _claude_text (claude -p / subscription), not skill_builder._llm_call.


def _strip_fences(text: str) -> str:
    """Remove surrounding ```lang fences if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
    return t


DECOY_SKILLS = [
    {"name": "data-preprocessing",
     "description": "Clean, transform, and prepare tabular or text data for ML models. Use when the user needs to handle missing values, tokenize text, normalize features, split datasets, or convert between data formats (pandas, parquet, arrow)."},
    {"name": "model-evaluation",
     "description": "Compute evaluation metrics for ML models. Use when the user wants to measure accuracy, F1, AUC, perplexity, BLEU, ROUGE, or compare model performance across runs."},
    {"name": "experiment-tracking",
     "description": "Log, compare, and visualize ML experiments. Use when the user mentions tracking runs, comparing hyperparameters, viewing loss curves, or integrating with MLflow, W&B, or TensorBoard."},
    {"name": "vector-retrieval",
     "description": "Build and query vector indexes for semantic search or RAG. Use when the user wants to embed documents, set up a vector database (FAISS, Chroma, Qdrant, Pinecone), or implement retrieval-augmented generation."},
    {"name": "deployment-serving",
     "description": "Deploy and serve ML models in production. Use when the user asks about model serving, containerization, inference endpoints, autoscaling, or integrating with FastAPI, TorchServe, vLLM, or TGI."},
]

_JUDGE_PROMPT = """You are simulating which skill an agent would pick for a user message.

Given a USER MESSAGE and a list of SKILL DESCRIPTIONS, decide which skill (if
any) would fire. Consider triggering keywords, task type, and fit.

Rules:
- Return ONLY a JSON object. No prose.
- If no skill fits well, return `"choice": "none"`.
- The `confidence` field is 0.0-1.0.
- Be honest: if the target skill's description is vague, don't pick it.

Return ONLY JSON:

{
  "choice": "<skill_name or 'none'>",
  "reason": "<one short sentence>",
  "confidence": <0.0-1.0>
}

---
USER MESSAGE:
{{user_message}}

AVAILABLE SKILLS:
{{skills_list}}
"""

_IMPROVE_PROMPT = """You are improving an OpenClaw skill description to make it trigger more reliably.

The current description FAILED to trigger on these user messages (an LLM judge
picked a different skill or "none"). Rewrite the description so the right
messages trigger it, without becoming so broad it triggers on unrelated ones.

Rules:
- Keep it ONE paragraph (3-5 sentences).
- Write in THIRD PERSON (no "I"/"you"/"we").
- Start with an action verb ("Train...", "Apply...", "Generate...").
- State BOTH what the skill does AND when to use it. Include a "Use when..." clause with
  concrete scenarios from the failing messages, plus the key terms a user would mention.
- Be a little "pushy" — agents tend to UNDER-trigger skills.
- Cut BOTH failure modes: rewrite so failing messages now trigger (recall) WITHOUT
  becoming so broad it fires on unrelated tasks (precision). If the judge picked another
  skill because the description over-claimed its territory, narrow that claim.
- Name specific capabilities (class/algorithm names) — not generic terms; no comma keyword lists.
- Don't claim capabilities the skill doesn't have.
- Keep it under ~900 characters (hard cap 1024).

Return ONLY the new description as a single plain text paragraph. No quotes, no markdown.

---
SKILL NAME: {{skill_name}}

CURRENT SKILL.md BODY (for capability reference):
{{skill_body}}

CURRENT DESCRIPTION:
{{current_description}}

FAILING USER MESSAGES (these SHOULD have triggered the skill):
{{failing_prompts}}

JUDGE'S REASONING (why other skills won):
{{judge_reasons}}
"""


def judge_triggering(user_message: str, skills: list[dict]) -> dict:
    """Ask claude -p which skill it would pick. Returns {choice, reason, confidence}."""
    skills_text = "\n".join(f"- `{s['name']}`: {s['description']}" for s in skills)
    prompt = _JUDGE_PROMPT.replace("{{user_message}}", user_message).replace("{{skills_list}}", skills_text)
    raw = _strip_fences(_claude_text(prompt, timeout=120))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"choice": "none", "reason": "judge returned no JSON", "confidence": 0.0}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"choice": "none", "reason": "judge JSON parse failed", "confidence": 0.0}


def improve_description(skill_name: str, skill_body: str, current_description: str,
                        failing: list[dict]) -> str:
    """Rewrite the description from failing triggering prompts (claude -p)."""
    if not failing:
        return current_description
    failing_prompts = "\n".join(f"- {f.get('prompt', '')}" for f in failing)
    judge_reasons = "\n".join(
        f"- (picked `{f.get('judge_choice', '?')}`) {f.get('judge_reason', '')}" for f in failing)
    prompt = (_IMPROVE_PROMPT
              .replace("{{skill_name}}", skill_name)
              .replace("{{skill_body}}", (skill_body or "")[:4000])
              .replace("{{current_description}}", current_description)
              .replace("{{failing_prompts}}", failing_prompts)
              .replace("{{judge_reasons}}", judge_reasons))
    out = _strip_fences(_claude_text(prompt, timeout=120)).strip()
    return out or current_description


def load_sibling_descriptions(siblings_dir: str, exclude_name: str = "") -> list[dict]:
    """Load {name, description} for each SKILL.md under siblings_dir (one level of
    subdirs + the dir itself), excluding the skill under test. Lets triggering compete
    against REAL co-resident skills instead of canned decoys."""
    out: list[dict] = []
    base = Path(siblings_dir).expanduser()
    if not base.exists():
        return out
    candidates = sorted(base.glob("*/SKILL.md"))
    if (base / "SKILL.md").exists():
        candidates.append(base / "SKILL.md")
    for p in candidates:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        fm = text[3:end] if end != -1 else text
        nm = re.search(r"^name:\s*(.+?)\s*$", fm, re.MULTILINE)
        dm = re.search(r"^description:\s*(.+?)\s*$", fm, re.MULTILINE)
        name = nm.group(1).strip().strip("\"'") if nm else p.parent.name
        descr = dm.group(1).strip().strip("\"'") if dm else ""
        if name and name != exclude_name and descr:
            out.append({"name": name, "description": descr})
    return out


# ── Eval cores (importable; take explicit params, no argparse / no skill_builder) ──


def run_triggering(skill_dir: Path, judge_fn, decoys: list[dict], *, runs: int = 3,
                   competitors_label: str = "decoys") -> dict:
    """Triggering F1 over should-trigger + near-miss prompts.

    `judge_fn(prompt, skills) -> {"choice": <name|"none">, ...}` is injected by the caller
    (typically this module's own `judge_triggering`); the injection seam is kept so callers
    can supply an alternate judge. `decoys` is the competitor skill list (real siblings or canned);
    `competitors_label` records which ("siblings" | "decoys") for the report.
    """
    decoys = list(decoys)
    meta = load_skill_meta(skill_dir)
    triggering_path = skill_dir / "evals" / "triggering.json"
    if not triggering_path.exists():
        _die(f"missing {triggering_path}")
    data = json.loads(triggering_path.read_text())

    skills = [meta] + list(decoys)
    pos = data.get("should_trigger", [])
    neg = data.get("should_not_trigger_near_miss", [])
    runs = max(1, int(runs))

    def judge_one(p):
        choices = []
        for _ in range(runs):
            v = judge_fn(p["prompt"], skills)
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
        "competitors": competitors_label,
        "n_competitors": len(decoys),
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


def run_functional(skill_dir: Path, *, agent: str = "ai-skill-builder", runs: int = 3,
                   per_prompt_timeout: int = 240, grader_fn=None) -> dict:
    """Run each functional prompt N trials, with and without skill, score every trial.
    `grader_fn` (P1.5, optional) grades any `test["judge"]` assertions."""
    meta = load_skill_meta(skill_dir)
    declared_mcps = parse_declared_mcps(skill_dir)
    functional_path = skill_dir / "evals" / "functional.json"
    if not functional_path.exists():
        _die(f"missing {functional_path}")
    data = json.loads(functional_path.read_text())
    tests = data.get("tests", [])
    runs = max(1, int(runs))
    out_results = []

    skill_marker = f"\n\n(For context: a skill is installed at {skill_dir}. Read SKILL.md and any relevant references/ or templates/ files before answering.)"

    def one_trial(test: dict, with_skill: bool, trial_idx: int) -> dict:
        prompt = test["prompt"] + (skill_marker if with_skill else "")
        side = "with" if with_skill else "without"
        trial_id = f"{test['id']}-{side}-{trial_idx}"
        reply = _run_agent(prompt, agent=agent, timeout=per_prompt_timeout,
                           trial_id=trial_id)
        score = _score_assertions(test, reply.get("text", ""), grader_fn=grader_fn)
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
        actually_called_xs = [1.0 if t["tool_signals"]["mcp_actually_called"] else 0.0 for t in trials]
        relevant_sidecar_xs = [1.0 if t["tool_signals"]["sidecar_relevant_servers"] else 0.0 for t in trials]
        sidecar_call_counts = [t["tool_signals"]["sidecar_call_count"] for t in trials]
        text_mcp_xs = [1.0 if t["tool_signals"]["text_mcp_hits"] else 0.0 for t in trials]
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
            "actual_mcp_call_rate": round(_mean(actually_called_xs), 3),
            "relevant_sidecar_rate": round(_mean(relevant_sidecar_xs), 3),
            "mean_sidecar_calls_per_trial": round(_mean(sidecar_call_counts), 2),
            "text_mcp_call_rate": round(_mean(text_mcp_xs), 3),
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
            "with_skill": {"trials": with_trials, "agg": aggregate(with_trials)},
            "without_skill": {"trials": without_trials, "agg": aggregate(without_trials)},
        })

    def gather(field_path: str, side: str) -> list[float]:
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

    overall_class = {"best_case": 0, "stealth_use": 0, "lip_service": 0, "clean_miss": 0}
    for r in out_results:
        wc = (r["with_skill"]["agg"] or {}).get("mcp_classification_counts") or {}
        for k, v in wc.items():
            overall_class[k] = overall_class.get(k, 0) + v

    result = {
        "skill_name": meta["name"],
        "declared_mcps": declared_mcps,
        "n_tests": len(out_results),
        "runs_per_cell": runs,
        "with_skill_pass_rate": round(with_pass, 3),
        "without_skill_pass_rate": round(without_pass, 3),
        "lift_pp": round((with_pass - without_pass) * 100, 1),
        "citation_accuracy_with_skill": round(with_cite, 3),
        "actual_mcp_call_rate_with_skill": round(with_actual_mcp, 3),
        "relevant_sidecar_rate_with_skill": round(with_rel_sidecar, 3),
        "text_mcp_call_rate_with_skill": round(with_text_mcp, 3),
        "mcp_classification_counts_with_skill": overall_class,
        "saturated": bool(with_pass >= 0.9 and without_pass >= 0.9),
        "tokens": {
            "with_skill_input": with_in, "with_skill_output": with_out_t,
            "without_skill_input": wo_in, "without_skill_output": wo_out,
        },
        "results": out_results,
    }
    result["analysis"] = analyze(result)  # P1.4 meta-eval attached to every run
    return result


def run_activation(skill_dir: Path, *, model: str = "", runs: int = 3,
                   per_prompt_timeout: int = 180, max_concurrency: int = 3) -> dict:
    """Organic-activation eval (Skills-3.0 Phase 3-3): does `claude -p` reach for the
    skill unprompted on should-trigger vs near-miss prompts (available-not-forced)."""
    meta = load_skill_meta(skill_dir)
    triggering_path = skill_dir / "evals" / "triggering.json"
    if not triggering_path.exists():
        _die(f"missing {triggering_path}")
    data = json.loads(triggering_path.read_text())
    pos = data.get("should_trigger", [])
    neg = data.get("should_not_trigger_near_miss", [])
    runs = max(1, int(runs))
    model = model or SKILLBUILD_LLM_MODEL
    conc = max(1, int(max_concurrency))
    errors: list[str] = []

    def activate_one(p: dict) -> tuple[dict, list[bool], list[dict]]:
        acts: list[bool] = []
        trials: list[dict] = []
        for _ in range(runs):
            r = _run_claude_executor(p["prompt"], skill_dir=skill_dir,
                                     timeout=per_prompt_timeout, model=model)
            if r.get("error"):
                errors.append(str(r["error"]))
            acts.append(bool(r.get("activated")))
            trials.append({
                "activated": bool(r.get("activated")),
                "elapsed_ms": r.get("elapsed_ms"),
                "reply_chars": len(r.get("reply", "")),
                "error": r.get("error"),
            })
        return p, acts, trials

    pos_results: list[dict] = []
    neg_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=conc) as pool:
        for p, acts, trials in pool.map(activate_one, pos):
            activated = sum(acts) >= (runs / 2)  # majority vote
            print(f"  [activation] {p['id']}: {sum(acts)}/{runs} activated",
                  file=sys.stderr, flush=True)
            pos_results.append({"id": p["id"], "prompt": p["prompt"],
                                "activations": acts, "activated": activated, "trials": trials})
        for p, acts, trials in pool.map(activate_one, neg):
            activated = sum(acts) >= (runs / 2)
            print(f"  [activation:near-miss] {p['id']}: {sum(acts)}/{runs} activated",
                  file=sys.stderr, flush=True)
            neg_results.append({"id": p["id"], "prompt": p["prompt"],
                                "activations": acts, "activated": activated, "trials": trials})

    tp = sum(1 for r in pos_results if r["activated"])
    fn = sum(1 for r in pos_results if not r["activated"])
    fp = sum(1 for r in neg_results if r["activated"])
    tn = sum(1 for r in neg_results if not r["activated"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "skill_name": meta["name"],
        "executor": "claude -p (organic activation, available-not-forced)",
        "model": model,
        "runs_per_prompt": runs,
        "should_trigger": pos_results,
        "should_not_trigger": neg_results,
        "errors": sorted(set(errors)),
        "metrics": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "activation_precision": round(precision, 3),
            "activation_recall": round(recall, 3),
            "activation_f1": round(f1, 3),
            "false_activation_rate": round(fp / (fp + tn), 3) if (fp + tn) else 0.0,
        },
    }


# ── P1.3 held-out description optimization (skill-creator run_loop.py) ───────


def _score_description(description: str, skill_name: str, pos: list, neg: list,
                       judge_fn, decoys: list, runs: int) -> dict:
    """F1 of a candidate description over labeled queries, via the injected judge."""
    skills = [{"name": skill_name, "description": description}] + list(decoys)

    def majority(q) -> bool:
        wins = sum(1 for _ in range(runs)
                   if (judge_fn(q["prompt"], skills) or {}).get("choice") == skill_name)
        return wins >= (runs / 2)

    pos_hit = [(q, majority(q)) for q in pos]
    neg_hit = [(q, majority(q)) for q in neg]
    tp = sum(1 for _, t in pos_hit if t)
    fn = len(pos) - tp
    fp = sum(1 for _, t in neg_hit if t)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": round(f1, 3), "precision": round(prec, 3), "recall": round(rec, 3),
            "failing": [q for q, t in pos_hit if not t]}


def optimize_description(skill_dir: Path, judge_fn, decoys: list, improve_fn, *,
                         holdout: float = 0.4, runs: int = 3, max_iters: int = 5) -> dict:
    """Optimize the SKILL.md description on a 60/40 held-out split of the triggering set.

    Iterates (≤max_iters) proposing rewrites from TRAIN failures only (improve_fn is blind
    to the test split → anti-overfit), then selects best_description by TEST F1. Mirrors
    skill-creator's run_loop.py. judge_fn + improve_fn injected (no skill_builder import)."""
    meta = load_skill_meta(skill_dir)
    tp_path = skill_dir / "evals" / "triggering.json"
    if not tp_path.exists():
        _die(f"missing {tp_path}")
    data = json.loads(tp_path.read_text())
    pos_all = data.get("should_trigger", [])
    neg_all = data.get("should_not_trigger_near_miss", [])
    test_start = int(round((1 - holdout) * 5))  # holdout=0.4 → i%5 in {3,4} = test (40%)

    def split(items):
        train, test = [], []
        for i, it in enumerate(sorted(items, key=lambda x: str(x.get("id")))):
            (test if (i % 5) >= test_start else train).append(it)
        return train, test

    pos_tr, pos_te = split(pos_all)
    neg_tr, neg_te = split(neg_all)
    name, cur = meta["name"], meta["description"]

    def score(desc, p, n):
        return _score_description(desc, name, p, n, judge_fn, decoys, runs)

    init_tr, init_te = score(cur, pos_tr, neg_tr), score(cur, pos_te, neg_te)
    candidates = [{"description": cur, "train_f1": init_tr["f1"], "test_f1": init_te["f1"], "source": "initial"}]
    history = []
    best_train_desc, best_train = cur, init_tr
    for it in range(max_iters):
        if not best_train["failing"]:
            break
        new_desc = improve_fn(name, "", best_train_desc, best_train["failing"])  # blind to test
        if not new_desc or new_desc.strip() == best_train_desc.strip():
            break
        tr, te = score(new_desc, pos_tr, neg_tr), score(new_desc, pos_te, neg_te)
        candidates.append({"description": new_desc, "train_f1": tr["f1"], "test_f1": te["f1"], "source": f"iter{it + 1}"})
        history.append({"iter": it + 1, "train_f1": tr["f1"], "test_f1": te["f1"]})
        if tr["f1"] > best_train["f1"]:
            best_train_desc, best_train = new_desc, tr
        else:
            break  # no train improvement → stop
    best = max(candidates, key=lambda c: c["test_f1"])  # SELECT BY TEST (anti-overfit)
    return {
        "skill_name": name,
        "split": {"pos_train": len(pos_tr), "pos_test": len(pos_te),
                  "neg_train": len(neg_tr), "neg_test": len(neg_te)},
        "initial": {"train_f1": init_tr["f1"], "test_f1": init_te["f1"]},
        "candidates": candidates,
        "history": history,
        "best_description": best["description"],
        "best_test_f1": best["test_f1"],
        "selected_by": "test",
        "improved": best["description"].strip() != cur.strip(),
    }


# ── P1.9 reconstruction / round-trip check (MIND-Skill-inspired; P2) ─────────
# Inspired by / adapted from MIND-Skill (Li et al. 2026, arXiv:2605.08670): a frozen
# agent reconstructs from the skill ALONE (no source); gaps vs source-derived gold notes
# = knowledge the skill failed to carry. Adapted here to a doc-grounded one-shot QA check
# (see docs/skill-builder/eval-gate-plan.md P1.9 for the four divergences from the paper).


def reconstruction_check(skill_dir: Path, source_text: str, *, model: str = "",
                         timeout: int = 180, llm=None, reconstruct=None) -> dict:
    """Round-trip completeness: reconstruct workflows/preconditions/API usage from the
    skill ALONE and diff against gold notes distilled from the source. Returns
    {missing, passed}. `llm(prompt)->text` and `reconstruct()->text` are injectable for
    testing; default to claude -p (gold/compare) + the skill-available executor
    (reconstruction). Best-effort."""
    llm = llm or (lambda p: _claude_text(p, model=model, timeout=timeout))
    if reconstruct is None:
        def reconstruct():
            r = _run_claude_executor(
                "Using ONLY your available skill, reconstruct its key workflows, "
                "preconditions, and API usage as a concise bullet list.",
                skill_dir=skill_dir, timeout=timeout, model=model)
            return r.get("reply", "")
    gold = llm("From these docs, list the KEY workflows, preconditions, and API calls a "
               "skill about this library MUST convey. Terse bullets.\n\nDOCS:\n"
               + (source_text or "")[:6000])
    recon = reconstruct()
    verdict = llm("GOLD NOTES (from source):\n" + (gold or "")
                  + "\n\nRECONSTRUCTION (from the skill alone):\n" + (recon or "")
                  + "\n\nList as terse bullets what is present in GOLD but MISSING or WRONG in "
                    "the RECONSTRUCTION (knowledge the skill failed to carry). If nothing is "
                    "missing, reply exactly: NONE")
    v = (verdict or "").strip()
    missing = [] if (not v or v.upper().startswith("NONE")) else [
        ln.strip("-* ").strip() for ln in v.splitlines() if ln.strip()]
    return {
        "skill_name": skill_dir.name,
        "gold_notes": gold,
        "reconstruction": recon,
        "missing": missing,
        "n_missing": len(missing),
        "passed": not missing,
        "citation": "inspired by MIND-Skill (arXiv:2605.08670)",
    }


# ── Pass-bar thresholds + report ─────────────────────────────────────────────


DEFAULT_PASS_BAR = {
    "triggering_f1_min": 0.85,
    "functional_pass_min": 0.60,
    "citation_accuracy_min": 0.50,
    "saturation_max_pass_rate": 0.90,
    "activation_recall_min": 0.50,  # organic-activation recall (only enforced when activation was run)
}


def load_pass_bar(skill_dir: Path) -> dict:
    p = skill_dir / "evals" / "pass_bar.json"
    if p.exists():
        try:
            return {**DEFAULT_PASS_BAR, **json.loads(p.read_text())}
        except json.JSONDecodeError:
            pass
    return DEFAULT_PASS_BAR


def find_latest(skill_dir: Path, prefix: str) -> Path | None:
    d = skill_dir / "evals" / "grading_results"
    if not d.exists():
        return None
    cands = sorted(d.glob(f"{prefix}-*.json"))
    return cands[-1] if cands else None


def build_report(skill_dir: Path, triggering: dict | None, functional: dict | None,
                 pass_bar: dict, activation: dict | None = None) -> tuple[str, bool]:
    """Return (markdown_report, overall_pass)."""
    lines = []
    name = (functional or triggering or activation or {}).get("skill_name") or skill_dir.name
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

    if activation:
        am = activation["metrics"]
        arec = am["activation_recall"]
        apass = arec >= pass_bar["activation_recall_min"]
        verdicts.append(apass)
        lines.append("## Organic activation (`claude -p`, available-not-forced)")
        lines.append(f"- Executor: {activation.get('executor', 'claude -p')} · model {activation.get('model', '?')}")
        lines.append(f"- Activation recall: **{arec:.3f}** — "
                     f"{'PASS' if apass else 'FAIL'} (bar ≥ {pass_bar['activation_recall_min']})")
        lines.append(f"- Activation precision: {am['activation_precision']:.3f} · "
                     f"F1 {am['activation_f1']:.3f}")
        lines.append(f"- False-activation rate (near-miss): {am['false_activation_rate']:.3f}")
        lines.append(f"- TP {am['true_positive']} / FN {am['false_negative']} / "
                     f"FP {am['false_positive']} / TN {am['true_negative']}")
        if activation.get("errors"):
            lines.append(f"- ⚠️ executor errors: {activation['errors']}")
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

    analysis = (functional or {}).get("analysis") if functional else None
    if analysis:
        nd = analysis.get("non_discriminating") or []
        lines.append("## Eval quality (analyzer)")
        lines.append(f"- Non-discriminating tests (same pass/fail with & without skill): "
                     f"{len(nd)}/{analysis.get('n_tests', '?')}"
                     f"{' — ' + ', '.join(str(x) for x in nd) if nd else ''}")
        if analysis.get("flaky"):
            lines.append(f"- Flaky (high-variance) cells: {analysis['flaky']}")
        if analysis.get("time_token"):
            lines.append(f"- Cost-inefficient (with-cell pricier, no pass gain): {analysis['time_token']}")
        lines.append("")

    overall = all(verdicts) if verdicts else False
    lines.append(f"## Verdict: **{'PASS' if overall else 'FAIL'}**")
    return "\n".join(lines), overall


# ── Ship-gate verdict + baseline probe (tester-owned; the builder delegates here) ──


def run_gate(skill_dir, profile: str = "smoke", *, siblings_dir: str = "") -> dict:
    """Behavioral ship-gate verdict. Reads evals/triggering.json in skill_dir, runs
    triggering (judge vs real siblings) + activation (profile=='full'), scores against
    pass_bar.json. Returns the BEHAVIORAL verdict only — the artifact critic's
    quality_gate is the author's concern and is folded in caller-side. Never raises.

    profile: 'smoke' = triggering runs=1, no activation, no functional; 'full' = triggering
    runs=3 + activation + the functional with/without-skill A/B (ADVISORY — reported with an
    analyzer pass, never hard-blocks, because the assertions are self-authored).
    """
    skill_dir = Path(skill_dir).expanduser().resolve()
    pass_bar = load_pass_bar(skill_dir)
    meta = load_skill_meta(skill_dir)
    gate: dict = {"profile": profile, "ran": [], "skipped": []}
    triggering = activation = functional = fanalysis = None

    sib_dir = siblings_dir or str(skill_dir.parent)
    sibs = load_sibling_descriptions(sib_dir, exclude_name=meta["name"])
    decoys, label = (sibs, "siblings") if len(sibs) >= 2 else (DECOY_SKILLS, "decoys")
    runs = 1 if profile == "smoke" else 3
    try:
        triggering = run_triggering(skill_dir, judge_triggering, decoys,
                                    runs=runs, competitors_label=label)
        gate["ran"].append("triggering")
    except SystemExit:
        gate["skipped"].append("triggering (no eval set)")
    if profile == "full":
        try:
            activation = run_activation(skill_dir, runs=1, per_prompt_timeout=180)
            gate["ran"].append("activation")
        except SystemExit:
            gate["skipped"].append("activation")
        # Functional with/without-skill A/B (ADVISORY). Clean baseline executor = `main`
        # (never skill-tester — recursion; and ai-skill-builder's bundled skills confound
        # the without-cell). analyze() flags non-discriminating / flaky self-authored tests.
        if (skill_dir / "evals" / "functional.json").exists():
            try:
                functional = run_functional(skill_dir, agent="main", runs=2, per_prompt_timeout=240)
                fanalysis = analyze(functional)
                gate["ran"].append("functional (advisory)")
            except SystemExit:
                gate["skipped"].append("functional (no eval set)")
        else:
            gate["skipped"].append("functional (no functional.json)")
    else:
        gate["skipped"].append("functional (smoke profile skips it)")

    reasons: list[str] = []
    if triggering and triggering["metrics"]["f1"] < pass_bar["triggering_f1_min"]:
        reasons.append(f"triggering F1 {triggering['metrics']['f1']:.2f} < bar {pass_bar['triggering_f1_min']}")
    if activation and activation["metrics"]["activation_recall"] < pass_bar["activation_recall_min"]:
        reasons.append(f"activation recall {activation['metrics']['activation_recall']:.2f} "
                       f"< bar {pass_bar['activation_recall_min']}")

    failing_pos = []
    if triggering:
        for p in triggering.get("should_trigger", []):
            if not p.get("triggered"):
                choices = p.get("choices") or []
                failing_pos.append({"id": p.get("id"), "prompt": p.get("prompt"),
                                    "judge_choice": (choices[0] if choices else "none"),
                                    "judge_reason": ""})

    md, _ = build_report(skill_dir, triggering, functional, pass_bar, activation)
    # Functional is advisory: surface pass-rate + lift + the analyzer flags, but keep it OUT
    # of `reasons` so it never blocks the ship (self-authored assertions).
    functional_metrics = None
    if functional:
        wp = functional.get("with_skill_pass_rate")
        wo = functional.get("without_skill_pass_rate")
        functional_metrics = {
            "with_skill_pass_rate": wp,
            "without_skill_pass_rate": wo,
            "lift": (round(wp - wo, 3) if (wp is not None and wo is not None) else None),
            "saturated": functional.get("saturated"),
            "advisory": True,
        }
    gate.update({
        "passed": not reasons,
        "reasons": reasons,
        "report": md,
        "triggering_metrics": (triggering or {}).get("metrics"),
        "activation_metrics": (activation or {}).get("metrics"),
        "functional_metrics": functional_metrics,
        "functional_analysis": fanalysis,
        "failing_positives": failing_pos,
    })
    return gate


def baseline_probe(intent_brief: str, doc_text: str = "", *, agent: str = "main",
                   n_questions: int = 2, timeout: int = 180) -> dict:
    """P0.3 baseline-first probe (tester-owned): generate intent-derived questions,
    run the base model (no skill) on them via `agent`, summarize where it falls short.
    Returns {gap_notes: str} — "" when the baseline already answers well. Best-effort;
    never raises. LLM calls use claude -p; base runs use the OpenClaw `agent`.

    Recursion guard: default executor is `main`, never `skill-tester`, so a tester
    that hosts this can't spawn itself.
    """
    out = {"gap_notes": ""}
    if not intent_brief:
        return out
    try:
        q_prompt = (
            f"Given this skill INTENT and docs, list {n_questions} short, realistic user "
            "questions whose answers the skill must get right. One per line, no numbering, "
            "no preamble.\n\nINTENT:\n" + intent_brief
            + "\n\nDOCS (excerpt):\n" + (doc_text or "")[:3000]
        )
        raw = _claude_text(q_prompt, timeout=120)
        questions = [ln.strip("-*0123456789. ").strip() for ln in raw.splitlines() if ln.strip()][:n_questions]
        if not questions:
            return out
        pairs = []
        for q in questions:
            r = _run_agent(q, agent=agent, timeout=timeout)
            pairs.append((q, (r.get("text") or "")[:1500]))
        joined = "\n\n".join(f"Q: {q}\nBASELINE ANSWER (no skill): {a}" for q, a in pairs)
        g_prompt = (
            "Below are questions and the base model's answers WITHOUT the skill. In 3-6 terse "
            "bullets, list what the baseline got WRONG, vague, or MISSED that a good skill must "
            "nail — these become the skill's priorities. If the baseline already answers "
            "everything well, reply exactly: NONE\n\n" + joined
        )
        gaps = _claude_text(g_prompt, timeout=120).strip()
        if not gaps or gaps.upper().startswith("NONE") or len(gaps) < 10:
            return out
        out["gap_notes"] = gaps[:2000]
        return out
    except (RuntimeError, OSError, KeyError, ValueError):
        return out
