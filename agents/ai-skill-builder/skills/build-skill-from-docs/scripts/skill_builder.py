#!/usr/bin/env python3
"""AI Skill Builder -- generate an OpenClaw SKILL.md from a package's documentation.

Usage:
    python3 skill_builder.py build https://huggingface.co/docs/trl/index
    python3 skill_builder.py preview <url>
    python3 skill_builder.py plan <url>
    python3 skill_builder.py sources <owner/repo>
    python3 skill_builder.py built

Flags for `build`:
    --name X              Override skill name (default: derived from repo)
    --with-pitfalls       Mine closed bug issues into references/pitfalls.md
    --with-version-notes  Include changelog highlights in SKILL.md
    --no-evals            Skip evals/evals.json
    --force               Overwrite existing skill at the target path
    --out <dir>           Write to <dir> instead of ~/.openclaw/workspace/skills/
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Import scanner from sibling skill-scout agent ─────────────────────────────

_SCOUT_SCRIPTS = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "ai-skill-scout" / "skills" / "find-ai-skill" / "scripts"
)
if str(_SCOUT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCOUT_SCRIPTS))
try:
    import skill_scout  # type: ignore
    BLOCK_PATTERNS = skill_scout.BLOCK_PATTERNS
    CAUTION_PATTERNS = skill_scout.CAUTION_PATTERNS
except (ImportError, AttributeError):
    BLOCK_PATTERNS = []
    CAUTION_PATTERNS = []


def _load_openrouter_key() -> str:
    """Load OpenRouter API key from env, then from OpenClaw auth profiles."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    for agent_dir in ("ai-skill-builder", "ai-skill-scout-opus", "ai-skill-scout", "main"):
        p = Path.home() / ".openclaw" / "agents" / agent_dir / "agent" / "auth-profiles.json"
        if not p.exists():
            continue
        try:
            auth = json.loads(p.read_text())
            prof = auth.get("profiles", {}).get("openrouter:default", {})
            k = prof.get("key") or prof.get("apiKey") or ""
            if k:
                return k
        except (json.JSONDecodeError, OSError):
            continue
    return ""


# ── Constants ─────────────────────────────────────────────────────────────────

SKILLS_DIR = Path.home() / ".openclaw" / "workspace" / "skills"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CACHE_TTL = 3600
MAX_SKILL_LINES = 500
# Target range: 60–150 lines covers the expanded structure (workflows, when-to-use,
# hardware, templates sections). Hard cap stays 500 per Anthropic guideline.
TARGET_SKILL_LINES = (60, 150)
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"
HTTP_TIMEOUT = 30
LLM_TIMEOUT = 120
USER_AGENT = "ai-skill-builder/1.0 (+https://github.com/Kkuntal990/AI-Skill-builder)"
BUILDER_VERSION = "1.3.0"

# Runtime MCP declarations. Skills *declare* expected MCPs in frontmatter; they
# don't auto-install. The agent runtime decides whether to invoke them.
_MCP_HF = {
    "preferred": ["hf-mcp/doc_search", "hf-mcp/doc_fetch"],
    "fallback": ["context7/get-library-docs"],
}
_MCP_GENERIC = {
    "preferred": [],
    "fallback": ["context7/get-library-docs"],
}


# ── Small utilities ───────────────────────────────────────────────────────────


def _die(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _ensure_data_dir() -> "None":
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _fill_prompt(template: str, **vars: str) -> str:
    """Replace `{{key}}` placeholders. Safer than .format() with JSON-heavy templates."""
    out = template
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def _audit(event: str, payload: dict) -> "None":
    _ensure_data_dir()
    line = json.dumps({"ts": int(time.time()), "event": event, **payload})
    with (DATA_DIR / "skill-builder-audit.log").open("a") as f:
        f.write(line + "\n")


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        _die(f"URL must be http(s): {url!r}")
    if len(url) > 2048:
        _die("URL too long (>2048 chars)")
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


# ── URL → owner/repo resolution ───────────────────────────────────────────────

_HF_DOCS_MAP = {
    # Hugging Face docs sites map to known repos
    "huggingface.co": {
        "trl": "huggingface/trl",
        "transformers": "huggingface/transformers",
        "accelerate": "huggingface/accelerate",
        "peft": "huggingface/peft",
        "datasets": "huggingface/datasets",
        "diffusers": "huggingface/diffusers",
        "tokenizers": "huggingface/tokenizers",
        "evaluate": "huggingface/evaluate",
        "safetensors": "huggingface/safetensors",
        "hub": "huggingface/huggingface_hub",
    },
}

# Custom docs subdomains for ML packages that don't live on huggingface.co or
# *.readthedocs.io. Static map; expand as needed. Cheaper than scraping each
# page for its canonical repo link.
_DOCS_DOMAIN_MAP = {
    "docs.vllm.ai": "vllm-project/vllm",
    "docs.unsloth.ai": "unslothai/unsloth",
    "docs.litellm.ai": "BerriAI/litellm",
    "docs.langchain.com": "langchain-ai/langchain",
    "docs.ray.io": "ray-project/ray",
    "lightning.ai": "Lightning-AI/pytorch-lightning",
    "docs.bentoml.com": "bentoml/BentoML",
    "docs.langgraph.dev": "langchain-ai/langgraph",
    "docs.crewai.com": "crewAIInc/crewAI",
    "docs.dspy.ai": "stanfordnlp/dspy",
    "docs.smolagents.com": "huggingface/smolagents",
    "docs.haystack.deepset.ai": "deepset-ai/haystack",
    "docs.guardrailsai.com": "guardrails-ai/guardrails",
    "docs.lmdeploy.com": "InternLM/lmdeploy",
}


def _url_parts(url: str) -> tuple[str, list[str]]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host, [p for p in parsed.path.split("/") if p]


def is_github_repo_root(url: str) -> bool:
    """True for https://github.com/<owner>/<repo> (optionally trailing /).

    Excludes blob/tree/issues/pull/etc. subpaths.
    """
    host, parts = _url_parts(url)
    return host == "github.com" and len(parts) == 2


def resolve_repo(url: str) -> str | None:
    """Return owner/repo if derivable from URL, else None."""
    host, path_parts = _url_parts(url)

    if host == "github.com" and len(path_parts) >= 2:
        return f"{path_parts[0]}/{path_parts[1]}"

    # Custom docs subdomains (vllm, unsloth, dspy, …)
    if host in _DOCS_DOMAIN_MAP:
        return _DOCS_DOMAIN_MAP[host]

    if host in _HF_DOCS_MAP and path_parts and path_parts[0] == "docs":
        if len(path_parts) >= 2:
            pkg = path_parts[1]
            # Some URLs have a locale between docs and package (e.g. /docs/en/trl)
            if pkg in ("en", "fr", "zh", "ko") and len(path_parts) >= 3:
                pkg = path_parts[2]
            if pkg in _HF_DOCS_MAP[host]:
                return _HF_DOCS_MAP[host][pkg]

    # readthedocs: project.readthedocs.io -> try project/project
    if host.endswith(".readthedocs.io"):
        proj = host.split(".")[0]
        return f"{proj}/{proj}"

    return None


# ── HTTP / HTML ingestion ─────────────────────────────────────────────────────


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        _die(f"failed to fetch {url}: {e}")
    return ""


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_RE = re.compile(r"<script.*?</script>", re.DOTALL | re.IGNORECASE)
_HTML_STYLE_RE = re.compile(r"<style.*?</style>", re.DOTALL | re.IGNORECASE)
_HTML_NBSP_RE = re.compile(r"&nbsp;")
_HTML_AMP_RE = re.compile(r"&amp;")
_HTML_LT_RE = re.compile(r"&lt;")
_HTML_GT_RE = re.compile(r"&gt;")
_HTML_QUOT_RE = re.compile(r"&quot;|&#39;")
_WS_RUN_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Strip HTML to readable text. Simple and deterministic, no deps."""
    s = _HTML_SCRIPT_RE.sub("", html)
    s = _HTML_STYLE_RE.sub("", s)
    # Preserve code blocks: <pre> and <code> → fenced
    s = re.sub(r"<pre[^>]*>", "\n```\n", s)
    s = re.sub(r"</pre>", "\n```\n", s)
    s = re.sub(r"<code[^>]*>", "`", s)
    s = re.sub(r"</code>", "`", s)
    # Headings
    for level in range(1, 7):
        s = re.sub(rf"<h{level}[^>]*>", "\n" + "#" * level + " ", s, flags=re.IGNORECASE)
        s = re.sub(rf"</h{level}>", "\n", s, flags=re.IGNORECASE)
    # Lists
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.IGNORECASE)
    # Line breaks and paragraphs
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</p>", "\n\n", s, flags=re.IGNORECASE)
    # Strip remaining tags
    s = _HTML_TAG_RE.sub("", s)
    # Entities
    s = _HTML_NBSP_RE.sub(" ", s)
    s = _HTML_AMP_RE.sub("&", s)
    s = _HTML_LT_RE.sub("<", s)
    s = _HTML_GT_RE.sub(">", s)
    s = _HTML_QUOT_RE.sub('"', s)
    # Collapse runs
    s = _WS_RUN_RE.sub("\n\n", s)
    return s.strip()


def fetch_doc(url: str) -> str:
    """Fetch a doc page and return its text content (HTML stripped)."""
    html = _http_get(url)
    return html_to_text(html)


# ── GitHub ingestion via `gh` ─────────────────────────────────────────────────


def _gh(*args: str, check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        _die("gh CLI not installed or not on PATH")
    except subprocess.TimeoutExpired:
        return ""
    if check and result.returncode != 0:
        return ""
    return result.stdout


def fetch_readme(repo: str) -> str:
    """Return the repo's README.md as markdown, or '' if unavailable."""
    out = _gh("api", f"repos/{repo}/readme", "--jq", ".content", check=False)
    if not out.strip():
        return ""
    try:
        return base64.b64decode(out.strip()).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return ""


def fetch_examples_listing(repo: str) -> list[str]:
    """Return a list of example filenames under common examples/ paths."""
    names = []
    for path in ("examples", "example", "samples"):
        out = _gh("api", f"repos/{repo}/contents/{path}", check=False)
        if not out.strip():
            continue
        try:
            items = json.loads(out)
            if isinstance(items, list):
                names.extend(
                    f"{path}/{i['name']}" for i in items[:20] if isinstance(i, dict)
                )
        except json.JSONDecodeError:
            continue
        if names:
            break
    return names


def fetch_bug_issues(repo: str, limit: int = 20) -> list[dict]:
    """Return closed bug-labeled issues sorted by reactions (high first)."""
    out = _gh(
        "issue", "list", "--repo", repo,
        "--state", "closed", "--label", "bug",
        "--limit", str(limit),
        "--json", "number,title,reactionGroups,body,url",
        check=False,
    )
    if not out.strip():
        return []
    try:
        issues = json.loads(out)
    except json.JSONDecodeError:
        return []

    def _reactions(issue: dict) -> int:
        total = 0
        for g in issue.get("reactionGroups") or []:
            if g.get("content") in ("THUMBS_UP", "HEART", "HOORAY"):
                total += g.get("users", {}).get("totalCount", 0)
        return total

    issues.sort(key=_reactions, reverse=True)
    return issues


def fetch_open_issues(repo: str, limit: int = 20, min_reactions: int = 3) -> list[dict]:
    """Return open issues sorted by reactions, filtered by `min_reactions`.

    These represent unresolved gotchas — users are stuck, not yet fixed.
    """
    out = _gh(
        "issue", "list", "--repo", repo,
        "--state", "open",
        "--limit", str(limit * 3),  # fetch extra; we'll filter below
        "--json", "number,title,reactionGroups,body,url,labels",
        check=False,
    )
    if not out.strip():
        return []
    try:
        issues = json.loads(out)
    except json.JSONDecodeError:
        return []

    def _reactions(issue: dict) -> int:
        total = 0
        for g in issue.get("reactionGroups") or []:
            if g.get("content") in ("THUMBS_UP", "HEART", "HOORAY"):
                total += g.get("users", {}).get("totalCount", 0)
        return total

    scored = [(i, _reactions(i)) for i in issues]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [i for i, r in scored if r >= min_reactions][:limit]


# Stack-trace / error-message extraction
_TRACEBACK_RE = re.compile(
    r"(?:Traceback \(most recent call last\):(?:\n.{0,500}){1,15})", re.DOTALL
)
_ERROR_LINE_RE = re.compile(
    r"^[A-Z]\w*(?:Error|Exception|Warning):\s+.{5,200}$", re.MULTILINE
)
_OOM_RE = re.compile(r"(CUDA out of memory|OutOfMemoryError|RuntimeError: .{10,200})")


def extract_stack_traces(issues: list[dict], max_per_issue: int = 2) -> list[dict]:
    """Pull tracebacks and error lines out of issue bodies.

    Returns list of {"issue": number, "url": url, "trace": str} dicts.
    One issue can yield multiple entries up to `max_per_issue`.
    """
    out = []
    for issue in issues:
        body = (issue.get("body") or "")[:8000]
        if not body:
            continue
        traces = []
        for m in _TRACEBACK_RE.finditer(body):
            traces.append(m.group(0).strip())
            if len(traces) >= max_per_issue:
                break
        if len(traces) < max_per_issue:
            for m in _ERROR_LINE_RE.finditer(body):
                traces.append(m.group(0).strip())
                if len(traces) >= max_per_issue:
                    break
        for t in traces[:max_per_issue]:
            out.append({
                "issue": issue.get("number"),
                "url": issue.get("url", ""),
                "trace": t[:1200],
            })
    return out


def fetch_changelog(repo: str) -> str:
    """Fetch CHANGELOG.md or latest release notes, whichever exists."""
    for path in ("CHANGELOG.md", "CHANGELOG", "HISTORY.md", "RELEASES.md"):
        out = _gh("api", f"repos/{repo}/contents/{path}", "--jq", ".content", check=False)
        if out.strip():
            try:
                return base64.b64decode(out.strip()).decode("utf-8", errors="replace")[:8000]
            except (ValueError, UnicodeDecodeError):
                continue
    # Fall back to latest release body
    out = _gh("api", f"repos/{repo}/releases/latest", "--jq", ".body", check=False)
    return out.strip()[:8000] if out.strip() else ""


def fetch_question_issues(repo: str, limit: int = 15) -> list[dict]:
    """Closed `question`-labeled issues sorted by reactions. Used by community-gotchas."""
    out = _gh(
        "issue", "list", "--repo", repo,
        "--state", "closed", "--label", "question",
        "--limit", str(limit),
        "--json", "number,title,reactionGroups,body,url",
        check=False,
    )
    if not out.strip():
        return []
    try:
        issues = json.loads(out)
    except json.JSONDecodeError:
        return []

    def _reactions(issue: dict) -> int:
        total = 0
        for g in issue.get("reactionGroups") or []:
            if g.get("content") in ("THUMBS_UP", "HEART", "HOORAY"):
                total += g.get("users", {}).get("totalCount", 0)
        return total

    issues.sort(key=_reactions, reverse=True)
    return issues


# ── Stack Exchange community substrate (build-time MCP seam ②) ───────────────


def _se_get(path: str, params: dict) -> dict:
    """GET a Stack Exchange API endpoint. Public read, no auth needed for top-N."""
    qs = urllib.parse.urlencode({**params, "site": "stackoverflow"})
    url = f"https://api.stackexchange.com/2.3/{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read()
            # Stack Exchange responses are gzip-encoded by spec.
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}


def fetch_stackexchange_qas(tag: str, limit: int = 15, min_score: int = 2) -> list[dict]:
    """Top-voted Q&As on Stack Overflow tagged `tag`, with accepted-answer body.

    Returns list of {"question_id", "title", "body", "url", "score",
    "answer_body", "answer_score", "tags"}. Filtered by min_score; HTML stripped;
    IPI-scanned via `_run_scanner` — entries with BLOCK hits are dropped.
    """
    # The "filter" query param controls which fields are returned; this one
    # includes question + answer bodies. It's a stable Stack Exchange filter ID
    # documented at api.stackexchange.com/docs/filters.
    questions = _se_get("questions", {
        "tagged": tag,
        "sort": "votes",
        "order": "desc",
        "pagesize": min(limit * 2, 30),
        "filter": "withbody",
    }).get("items", [])
    if not questions:
        return []
    out = []
    for q in questions:
        if q.get("score", 0) < min_score:
            continue
        if not q.get("is_answered"):
            continue
        accepted_id = q.get("accepted_answer_id")
        if not accepted_id:
            continue
        ans_resp = _se_get(f"answers/{accepted_id}", {"filter": "withbody"})
        ans_items = ans_resp.get("items", [])
        if not ans_items:
            continue
        ans = ans_items[0]
        # Strip HTML; SO bodies are HTML by default.
        q_text = html_to_text(q.get("body", ""))[:3000]
        a_text = html_to_text(ans.get("body", ""))[:3000]
        # IPI guard: drop the entry if either side trips a BLOCK pattern.
        combined = q_text + "\n" + a_text
        blocks, _ = _run_scanner(combined)
        if blocks:
            _audit("community.dropped_ipi", {
                "source": "stackoverflow",
                "question_id": q.get("question_id"),
                "blocks": blocks,
            })
            continue
        out.append({
            "question_id": q.get("question_id"),
            "title": q.get("title", ""),
            "body": q_text,
            "url": q.get("link", ""),
            "score": q.get("score", 0),
            "answer_body": a_text,
            "answer_score": ans.get("score", 0),
            "tags": q.get("tags", []),
        })
        if len(out) >= limit:
            break
    return out


# ── README install parser (deterministic) ─────────────────────────────────────

_INSTALL_HEADING_RE = re.compile(r"^#+\s*install", re.IGNORECASE | re.MULTILINE)
_PIP_RE = re.compile(r"pip\s+install\s+([^\n`]+)", re.IGNORECASE)
_CONDA_RE = re.compile(r"conda\s+install\s+([^\n`]+)", re.IGNORECASE)


def extract_install_section(readme: str) -> str:
    """Return the first ~40 lines after an 'install' heading, or empty."""
    if not readme:
        return ""
    m = _INSTALL_HEADING_RE.search(readme)
    if not m:
        return ""
    start = m.start()
    chunk = readme[start:start + 3000]
    lines = chunk.split("\n")
    # Stop at next heading of same or shallower level
    out_lines = [lines[0]]
    for ln in lines[1:]:
        if re.match(r"^#+\s", ln) and not ln.startswith("##"):
            break
        out_lines.append(ln)
        if len(out_lines) >= 40:
            break
    return "\n".join(out_lines)


def detect_install_commands(readme: str) -> list[dict]:
    """Return structured install metadata entries derived from README."""
    out = []
    section = extract_install_section(readme) or readme[:2000]
    for m in _PIP_RE.finditer(section):
        pkgs = m.group(1).strip().split()
        pkgs = [p for p in pkgs if not p.startswith("-")]
        if pkgs:
            out.append({
                "id": f"pip-{pkgs[0]}",
                "kind": "pip",
                "packages": pkgs,
                "label": f"pip install {' '.join(pkgs)}",
            })
            break  # first pip command is the canonical one
    return out


# ── OpenRouter client ─────────────────────────────────────────────────────────


def _openrouter_call(prompt: str, max_tokens: int = 4000, temperature: float = 0.3) -> str:
    key = _load_openrouter_key()
    if not key:
        _die(
            "OPENROUTER_API_KEY not set and no openrouter:default profile in OpenClaw "
            "auth store. Building a skill without an LLM is not useful; aborting."
        )
    body = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Kkuntal990/AI-Skill-builder",
            "X-Title": "AI Skill Builder",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        _die(f"OpenRouter request failed: {e}")
        return ""
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        _die(f"OpenRouter returned unexpected payload: {data}")
        return ""


def _strip_fences(text: str) -> str:
    """Remove surrounding ```markdown or ```json fences if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if len(lines) >= 2:
            # drop first ```lang line and trailing ``` if present
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines)
    return t.strip()


# ── LLM stages ────────────────────────────────────────────────────────────────


def plan_structure(
    doc_text: str,
    with_pitfalls: bool,
    with_evals: bool,
    with_templates: bool = False,
) -> dict:
    tpl = _read_prompt("plan_structure.txt")
    prompt = _fill_prompt(
        tpl,
        doc_text=doc_text[:20000],
        with_pitfalls=str(with_pitfalls).lower(),
        with_evals=str(with_evals).lower(),
        with_templates=str(with_templates).lower(),
    )
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=2500, temperature=0.2))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        _die(f"plan_structure returned non-JSON: {raw[:300]}")
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        _die(f"plan_structure JSON parse failed: {e}\n{raw[:300]}")
    for key in ("skill_name", "one_line_purpose", "references"):
        if key not in plan:
            _die(f"plan_structure missing '{key}': {plan}")
    if len(plan["references"]) > 5:
        plan["references"] = plan["references"][:5]
    # Defaults for new fields (backward compat if LLM omits them)
    plan.setdefault("workflows", [])
    plan.setdefault("templates", [])
    plan.setdefault("include_hardware_section", False)
    plan.setdefault("include_when_to_use_section", True)
    # Cap sizes defensively
    plan["workflows"] = plan["workflows"][:4]
    plan["templates"] = plan["templates"][:3] if with_templates else []
    # Enforce checklist shape
    cleaned_wfs = []
    for wf in plan["workflows"]:
        if isinstance(wf, dict) and "title" in wf and "checklist" in wf:
            wf["checklist"] = [str(s) for s in wf["checklist"][:6]]
            cleaned_wfs.append(wf)
    plan["workflows"] = cleaned_wfs
    return plan


# Hardware requirement extraction — regex side-channel for the body prompt.
_HW_MARKERS = re.compile(
    r"\b(?:VRAM|GPU memory|GPU RAM|memory footprint|gradient checkpoint"
    r"|mixed precision|bf16|fp16|bfloat16|float16|quantiz|QLoRA|A100|H100"
    r"|RTX [234]\d{3}|consumer GPU|TPU|single GPU|multi-GPU|multi-node"
    r"|DeepSpeed (?:ZeRO[- ]?[0-3]|stage [0-3])|FSDP"
    r"|\d+\s*GB(?:\s*(?:VRAM|memory|of RAM|GPU))?"
    r"|\d+\s*B parameters?"
    r")\b",
    re.IGNORECASE,
)


def extract_hardware_hints(*texts: str, max_sentences: int = 12) -> str:
    """Grep provided texts for hardware-relevant sentences; return a joined list.

    Returns empty string if no hints found. The output is fed to the body
    prompt as a side-channel so the LLM can emit concrete VRAM guidance
    instead of hand-waving.
    """
    picked = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            s = sentence.strip()
            if not s or len(s) < 20 or len(s) > 400:
                continue
            if _HW_MARKERS.search(s) is None:
                continue
            key = s[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= max_sentences:
                return "\n".join(f"- {p}" for p in picked)
    return "\n".join(f"- {p}" for p in picked)


def write_body(
    skill_name: str,
    references: list[dict],
    workflows: list[dict],
    templates: list[dict],
    include_when_to_use: bool,
    include_hardware: bool,
    hardware_hints: str,
    readme_install: str,
    doc_text: str,
) -> str:
    tpl = _read_prompt("write_skill_body.txt")
    refs_list = "\n".join(
        f"- `references/{r['filename']}` — {r['covers']}" for r in references
    ) or "(no reference files)"
    workflows_list = "\n".join(
        f"- **{w['title']}** — checklist: {w['checklist']}" for w in workflows
    ) or "(no workflows)"
    templates_list = "\n".join(
        f"- `templates/{t['filename']}` — {t['covers']}" for t in templates
    ) or "(no templates)"
    prompt = _fill_prompt(
        tpl,
        skill_name=skill_name,
        references_list=refs_list,
        workflows_list=workflows_list,
        templates_list=templates_list,
        hardware_hints=hardware_hints or "(no hardware hints extracted)",
        include_when_to_use=str(include_when_to_use).lower(),
        include_hardware=str(include_hardware and bool(hardware_hints)).lower(),
        readme_install=readme_install[:3000] or "(README install section not found)",
        doc_text=doc_text[:15000],
    )
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=4000, temperature=0.3))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


def write_template(
    filename: str,
    covers: str,
    package_name: str,
    readme_install: str,
    doc_text: str,
    examples_text: str,
) -> str:
    """Generate a runnable Python script via LLM. Returns raw Python source."""
    tpl = _read_prompt("write_template.txt")
    prompt = _fill_prompt(
        tpl,
        filename=filename,
        covers=covers,
        package_name=package_name,
        readme_install=readme_install[:2000] or "(not available)",
        doc_text=doc_text[:14000],
        examples_text=examples_text[:2000] or "(not available)",
    )
    raw = _openrouter_call(prompt, max_tokens=4000, temperature=0.2)
    # The LLM may still wrap in ```python fences despite instructions — strip them.
    raw = _strip_fences(raw)
    return raw.strip() + "\n"


def validate_templates(templates: dict[str, str]) -> list[dict]:
    """py_compile each template. Returns list of validation issues."""
    import tempfile as _tempfile
    issues = []
    for name, src in templates.items():
        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(src)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                issues.append({
                    "severity": "warning",
                    "where": f"templates/{name}",
                    "message": f"py_compile failed: {err[:300]}",
                })
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
    return issues


def write_reference(
    filename: str,
    topic_covers: str,
    doc_text: str,
    readme_text: str,
    examples_text: str,
) -> str:
    tpl = _read_prompt("write_reference.txt")
    prompt = _fill_prompt(
        tpl,
        filename=filename,
        topic_covers=topic_covers,
        doc_text=doc_text[:18000],
        readme_text=readme_text[:3000],
        examples_text=examples_text[:2000],
    )
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=5000, temperature=0.3))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


def write_evals(skill_name: str, skill_body: str) -> dict:
    tpl = _read_prompt("write_evals.txt")
    prompt = _fill_prompt(tpl, skill_name=skill_name, skill_body=skill_body[:4000])
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=1000, temperature=0.5))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"skill_name": skill_name, "prompts": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"skill_name": skill_name, "prompts": []}


def distill_pitfalls(package_name: str, issues: list[dict]) -> str:
    tpl = _read_prompt("distill_pitfalls.txt")
    lines = []
    for i in issues:
        body = (i.get("body") or "")[:500].replace("\n", " ")
        lines.append(f"- #{i['number']} {i['title']}\n  {body}\n  ({i['url']})")
    prompt = _fill_prompt(tpl, package_name=package_name, issues_text="\n".join(lines))
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=3000, temperature=0.3))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


def _issue_snippet(issue: dict, body_chars: int = 400) -> str:
    body = (issue.get("body") or "")[:body_chars].replace("\n", " ")
    return f"- #{issue.get('number')} {issue.get('title', '')}\n  {body}\n  ({issue.get('url', '')})"


def write_community_gotchas(
    package_name: str,
    qas: list[dict],
    closed_issues: list[dict],
) -> str:
    """Synthesize references/community-gotchas.md from SE Q&As + closed issues."""
    if not (qas or closed_issues):
        return ""
    tpl = _read_prompt("distill_community_gotchas.txt")
    qa_lines = []
    for q in qas[:15]:
        body = (q.get("body") or "")[:500].replace("\n", " ")
        ans = (q.get("answer_body") or "")[:600].replace("\n", " ")
        qa_lines.append(
            f"- Q#{q.get('question_id')} score={q.get('score', 0)} "
            f"answer_score={q.get('answer_score', 0)}\n"
            f"  Title: {q.get('title', '')}\n"
            f"  Body: {body}\n"
            f"  Accepted answer: {ans}\n"
            f"  URL: {q.get('url', '')}"
        )
    qa_text = "\n".join(qa_lines) or "(none)"
    issue_text = "\n".join(_issue_snippet(i) for i in closed_issues[:15]) or "(none)"
    prompt = _fill_prompt(
        tpl,
        package_name=package_name,
        stackexchange_text=qa_text,
        github_issues_text=issue_text,
    )
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=4000, temperature=0.3))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


def write_troubleshooting(
    package_name: str,
    repo: str,
    closed_issues: list[dict],
    open_issues: list[dict],
    stack_traces: list[dict],
) -> str:
    if not (closed_issues or open_issues or stack_traces):
        return ""
    tpl = _read_prompt("write_troubleshooting.txt")
    closed_text = "\n".join(_issue_snippet(i) for i in closed_issues[:20]) or "(none)"
    open_text = "\n".join(_issue_snippet(i) for i in open_issues[:15]) or "(none)"
    traces_text = "\n\n".join(
        f"from #{t['issue']} ({t['url']}):\n```\n{t['trace']}\n```"
        for t in stack_traces[:20]
    ) or "(none extracted)"
    prompt = _fill_prompt(
        tpl,
        package_name=package_name,
        repo_url=f"https://github.com/{repo}" if repo else "",
        closed_issues_text=closed_text,
        open_issues_text=open_text,
        stack_traces_text=traces_text,
    )
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=4000, temperature=0.3))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


# ── Feature 2: triggering eval + description optimizer ───────────────────────

DECOY_SKILLS = [
    {
        "name": "data-preprocessing",
        "description": "Clean, transform, and prepare tabular or text data for ML models. Use when the user needs to handle missing values, tokenize text, normalize features, split datasets, or convert between data formats (pandas, parquet, arrow).",
    },
    {
        "name": "model-evaluation",
        "description": "Compute evaluation metrics for ML models. Use when the user wants to measure accuracy, F1, AUC, perplexity, BLEU, ROUGE, or compare model performance across runs.",
    },
    {
        "name": "experiment-tracking",
        "description": "Log, compare, and visualize ML experiments. Use when the user mentions tracking runs, comparing hyperparameters, viewing loss curves, or integrating with MLflow, W&B, or TensorBoard.",
    },
    {
        "name": "vector-retrieval",
        "description": "Build and query vector indexes for semantic search or RAG. Use when the user wants to embed documents, set up a vector database (FAISS, Chroma, Qdrant, Pinecone), or implement retrieval-augmented generation.",
    },
    {
        "name": "deployment-serving",
        "description": "Deploy and serve ML models in production. Use when the user asks about model serving, containerization, inference endpoints, autoscaling, or integrating with FastAPI, TorchServe, vLLM, or TGI.",
    },
]


def judge_triggering(user_message: str, skills: list[dict]) -> dict:
    """Ask an LLM which skill it would pick for a user message.

    Returns {"choice": str, "reason": str, "confidence": float}.
    """
    tpl = _read_prompt("judge_triggering.txt")
    skills_text = "\n".join(
        f"- `{s['name']}`: {s['description']}" for s in skills
    )
    prompt = _fill_prompt(tpl, user_message=user_message, skills_list=skills_text)
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=400, temperature=0.0))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"choice": "none", "reason": "judge returned no JSON", "confidence": 0.0}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"choice": "none", "reason": "judge JSON parse failed", "confidence": 0.0}


def evaluate_triggering(
    skill_name: str, skill_description: str, eval_prompts: list[dict]
) -> dict:
    """Run triggering judge over each eval prompt. Returns per-prompt results + win rate."""
    if not eval_prompts:
        return {"win_rate": None, "results": [], "failing": []}
    skills = [{"name": skill_name, "description": skill_description}] + DECOY_SKILLS
    results = []
    failing = []
    for p in eval_prompts:
        msg = p.get("prompt", "")
        if not msg:
            continue
        verdict = judge_triggering(msg, skills)
        won = verdict.get("choice") == skill_name
        entry = {
            "prompt_id": p.get("id"),
            "prompt": msg,
            "won": won,
            "judge_choice": verdict.get("choice"),
            "judge_reason": verdict.get("reason", ""),
        }
        results.append(entry)
        if not won:
            failing.append(entry)
    total = len(results)
    wins = sum(1 for r in results if r["won"])
    return {
        "win_rate": wins / total if total else None,
        "wins": wins,
        "total": total,
        "results": results,
        "failing": failing,
    }


def improve_description(
    skill_name: str,
    skill_body: str,
    current_description: str,
    failing: list[dict],
) -> str:
    """Ask an LLM to rewrite the description based on failing eval prompts."""
    if not failing:
        return current_description
    tpl = _read_prompt("improve_description.txt")
    failing_prompts = "\n".join(f"- {f['prompt']}" for f in failing)
    judge_reasons = "\n".join(
        f"- (picked `{f['judge_choice']}`) {f['judge_reason']}" for f in failing
    )
    prompt = _fill_prompt(
        tpl,
        skill_name=skill_name,
        skill_body=skill_body[:4000],
        current_description=current_description,
        failing_prompts=failing_prompts,
        judge_reasons=judge_reasons,
    )
    raw = _strip_fences(_openrouter_call(prompt, max_tokens=800, temperature=0.4))
    # Take the first non-empty paragraph
    for para in raw.split("\n\n"):
        p = para.strip()
        if len(p) > 50 and not p.startswith("#"):
            return p
    return raw.strip() or current_description


# ── Frontmatter assembly (deterministic) ──────────────────────────────────────


def _extract_description(body: str) -> str:
    """Pull the first non-heading paragraph from SKILL.md body as the description."""
    paragraphs = []
    current = []
    for line in body.split("\n"):
        if line.startswith("#"):
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current).strip())
    for p in paragraphs:
        if len(p) > 40:
            return p
    return paragraphs[0] if paragraphs else ""


def _mcp_defaults_for(repo: str | None, url: str) -> dict:
    """Pick runtime-MCP declarations based on the package family.

    HF-hosted packages get HF Documentation Semantic Search as preferred,
    Context7 as universal fallback. Anything else gets Context7-only.
    """
    host, _ = _url_parts(url)
    if (repo and repo.startswith("huggingface/")) or host == "huggingface.co":
        return _MCP_HF
    return _MCP_GENERIC


def assemble_frontmatter(
    skill_name: str,
    description: str,
    install_entries: list[dict],
    bins: list[str],
    emoji: str = "🤖",
    mcps: dict | None = None,
    provenance: dict | None = None,
    coverage: list[str] | None = None,
) -> str:
    """Build the YAML+JSON frontmatter deterministically.

    `mcps`, `provenance`, and `coverage` are Phase 1.3 additions. Skills declare
    runtime MCPs (Seam ③) without auto-installing; `provenance` carries source
    metadata so downstream consumers can detect drift; `coverage` lists the
    build-time substrates (build-time MCP seams ①, ②) used.
    """
    meta = {
        "openclaw": {
            "emoji": emoji,
            "requires": {"bins": bins},
        }
    }
    if install_entries:
        meta["openclaw"]["install"] = install_entries
    if mcps and (mcps.get("preferred") or mcps.get("fallback")):
        meta["openclaw"]["mcps"] = mcps
    if provenance:
        meta["openclaw"]["source"] = provenance
    if coverage:
        meta["openclaw"]["coverage"] = coverage
    # JSON for metadata; ensure_ascii=False keeps emoji as-is instead of \uXXXX
    # (which would otherwise trip the obfuscation scanner on our own frontmatter).
    desc_escaped = description.replace('"', '\\"')
    fm = (
        "---\n"
        f"name: {skill_name}\n"
        f'description: "{desc_escaped}"\n'
        f"metadata: {json.dumps(meta, ensure_ascii=False)}\n"
        "---\n\n"
    )
    return fm


# ── Validation ────────────────────────────────────────────────────────────────


# ML-safe method-call patterns that Scout's scanner flags as injection.
# These are method calls on objects (e.g. `model.eval()`), not the dangerous
# Python builtins. If EVERY occurrence of the matched pattern is a method call,
# drop the false positive.
_ML_SAFE_METHOD_CALL = re.compile(
    r"(?P<pre>[\.\w])\s*(?P<fn>eval|exec)\s*\("
)
_BUILTIN_CALL = re.compile(
    r"(?<![\w.])(?P<fn>eval|exec)\s*\("
)


def _is_ml_safe(pattern_desc: str, text: str) -> bool:
    """Return True if a block hit is a known false positive for ML content.

    Scout's `injection: eval()` / `exec()` patterns match Python's dangerous
    builtins. But `model.eval()` (set model to eval mode) and `trainer.eval()`
    are ubiquitous in ML code and harmless. Only treat as injection if we find
    at least one call that ISN'T a method call.
    """
    if pattern_desc not in ("injection: eval()", "injection: exec()"):
        return False
    fn = "eval" if "eval" in pattern_desc else "exec"
    # Does any match NOT have a dot or word char immediately preceding it?
    for m in _BUILTIN_CALL.finditer(text):
        if m.group("fn") == fn:
            return False  # real builtin call found → not safe
    return True  # every match was a method call


def _run_scanner(text: str) -> tuple[list[str], list[str]]:
    """Return (blocks, cautions) found in text. Filters ML-safe false positives."""
    blocks, cautions = [], []
    for pat, desc in BLOCK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            if _is_ml_safe(desc, text):
                continue
            blocks.append(desc)
    for pat, desc in CAUTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            cautions.append(desc)
    return blocks, cautions


_SHELL_CMD_RE = re.compile(r"^[a-z][a-z0-9_\-./]*$")


def _extract_shell_commands(text: str) -> set[str]:
    """Extract shell commands (first tokens) from explicit bash/sh fences.

    Filters out markdown table rows, prose sentences, and separators so we
    don't flag "Pass", "|---|", or "After" as fabricated commands.
    """
    cmds = set()
    for m in re.finditer(
        r"```(?:bash|sh|shell|console|zsh)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE
    ):
        for line in m.group(1).split("\n"):
            ln = line.strip().lstrip("$").strip()
            if not ln or ln.startswith("#") or ln.startswith("|"):
                continue
            tokens = ln.split()
            if not tokens:
                continue
            first = tokens[0]
            # Only accept things that look like actual shell commands
            if _SHELL_CMD_RE.match(first):
                cmds.add(first)
    return cmds


def validate_skill(
    skill_md: str, references: dict[str, str], sources: dict[str, str]
) -> list[dict]:
    """Return list of validation warnings/errors. Empty list = clean."""
    issues = []
    # Line count
    body_lines = len(skill_md.split("\n"))
    if body_lines > MAX_SKILL_LINES:
        issues.append({
            "severity": "error",
            "where": "SKILL.md",
            "message": f"SKILL.md is {body_lines} lines; max is {MAX_SKILL_LINES}",
        })
    elif body_lines > TARGET_SKILL_LINES[1]:
        issues.append({
            "severity": "warning",
            "where": "SKILL.md",
            "message": f"SKILL.md is {body_lines} lines; target is ≤{TARGET_SKILL_LINES[1]}",
        })
    # Frontmatter check
    if not skill_md.startswith("---\n"):
        issues.append({
            "severity": "error",
            "where": "SKILL.md",
            "message": "missing YAML frontmatter",
        })
    else:
        end = skill_md.find("\n---\n", 4)
        if end == -1:
            issues.append({
                "severity": "error",
                "where": "SKILL.md",
                "message": "unterminated frontmatter",
            })
        else:
            fm = skill_md[4:end]
            if not re.search(r"^name:\s*\S", fm, re.MULTILINE):
                issues.append({
                    "severity": "error",
                    "where": "frontmatter",
                    "message": "missing name",
                })
            if not re.search(r"^description:\s*\S", fm, re.MULTILINE):
                issues.append({
                    "severity": "error",
                    "where": "frontmatter",
                    "message": "missing description",
                })
    # Security scan across all content
    combined = skill_md + "\n" + "\n".join(references.values())
    blocks, cautions = _run_scanner(combined)
    for b in blocks:
        issues.append({"severity": "error", "where": "security-scan", "message": b})
    for c in cautions:
        issues.append({"severity": "warning", "where": "security-scan", "message": c})
    # Shell command fabrication check
    gen_cmds = _extract_shell_commands(combined)
    src_text = "\n".join(sources.values()).lower()
    fabricated = []
    skip_verbs = {"cd", "ls", "cat", "echo", "git", "make", "python", "python3",
                  "pip", "conda", "curl", "export", "source"}
    for cmd in gen_cmds:
        if cmd in skip_verbs:
            continue
        if cmd.lower() not in src_text:
            fabricated.append(cmd)
    if fabricated:
        issues.append({
            "severity": "warning",
            "where": "shell-commands",
            "message": f"possibly fabricated commands (not found in sources): {fabricated[:5]}",
        })
    return issues


def openclaw_skills_check(skill_dir: Path) -> tuple[bool, str]:
    """Run `openclaw skills check`. The CLI takes no args; it scans the workspace.

    If `skill_dir` isn't inside the default workspace, we skip — the check
    wouldn't see the skill anyway.
    """
    try:
        skill_dir.resolve().relative_to(SKILLS_DIR.resolve())
    except ValueError:
        return (True, f"skipped: {skill_dir} is outside {SKILLS_DIR}")
    try:
        result = subprocess.run(
            ["openclaw", "skills", "check"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return (True, "openclaw CLI unavailable; skipped")
    return (result.returncode == 0, (result.stdout + result.stderr).strip())


# ── Write to workspace ────────────────────────────────────────────────────────


def write_skill(
    out_dir: Path,
    skill_name: str,
    skill_md: str,
    references: dict[str, str],
    evals: dict | None,
    templates: dict[str, str] | None,
    force: bool,
) -> Path:
    target = out_dir / skill_name
    if target.exists() and not force:
        _die(f"target {target} already exists; pass --force to overwrite")
    if target.exists():
        import shutil
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(skill_md)
    if references:
        (target / "references").mkdir(exist_ok=True)
        for name, content in references.items():
            (target / "references" / name).write_text(content)
    if evals and evals.get("prompts"):
        (target / "evals").mkdir(exist_ok=True)
        (target / "evals" / "evals.json").write_text(json.dumps(evals, indent=2))
    if templates:
        (target / "templates").mkdir(exist_ok=True)
        for name, src in templates.items():
            (target / "templates" / name).write_text(src)
    return target


def update_lockfile(skill_name: str, source_url: str, skill_md: str, path: Path) -> "None":
    _ensure_data_dir()
    lockfile = DATA_DIR / "built-skills.json"
    if lockfile.exists():
        try:
            data = json.loads(lockfile.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data[skill_name] = {
        "source_url": source_url,
        "path": str(path),
        "hash": hashlib.sha256(skill_md.encode("utf-8")).hexdigest()[:16],
        "built_at": int(time.time()),
    }
    lockfile.write_text(json.dumps(data, indent=2))


# ── Pipelines ─────────────────────────────────────────────────────────────────


def _se_tag_candidates(repo: str | None) -> list[str]:
    """Return Stack Overflow tag candidates ordered most-specific first.

    SE tags don't always match repo names. e.g. `huggingface/trl` is tagged
    `huggingface-trl` on SO, not `trl`. Try a few shapes; the first one that
    returns hits wins. Empty list if no repo.
    """
    if not repo or "/" not in repo:
        return []
    owner, short = repo.split("/", 1)
    short = short.lower()
    owner = owner.lower()
    candidates = [short]
    if owner != short:
        candidates.append(f"{owner}-{short}")
    # Don't fall back to broad org-wide tags (e.g. `huggingface-transformers`
    # for `huggingface/trl`) — it pollutes trl-training with transformers Q&As.
    # If a package has no specific SE tag, return empty — `--with-troubleshooting`
    # remains the community signal channel.
    seen = set()
    out = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_stackexchange_qas_for_repo(repo: str | None, limit: int = 15) -> list[dict]:
    """Probe candidate tags in order; return Q&As from the first non-empty one."""
    for tag in _se_tag_candidates(repo):
        qas = fetch_stackexchange_qas(tag, limit=limit)
        if qas:
            return qas
    return []


def _gather_sources(
    url: str,
    with_pitfalls: bool,
    with_version_notes: bool,
    with_troubleshooting: bool = False,
    with_community: bool = False,
) -> dict:
    repo = resolve_repo(url)
    # Optimization: if the user passed a GitHub repo root, skip the HTML fetch
    # (which would pull mostly GitHub chrome) and use the README via `gh api`
    # as both the doc and readme source. Single fetch, clean markdown.
    if repo and is_github_repo_root(url):
        readme = fetch_readme(repo)
        doc_text = readme
    else:
        doc_text = fetch_doc(url)
        readme = fetch_readme(repo) if repo else ""
    sources = {"doc": doc_text}
    if repo:
        sources["repo"] = repo
        sources["readme"] = readme
        sources["readme_install"] = extract_install_section(readme)
        sources["examples"] = "\n".join(fetch_examples_listing(repo))
    else:
        sources["repo"] = ""
        sources["readme"] = ""
        sources["readme_install"] = ""
        sources["examples"] = ""
    # Pitfalls uses closed bug issues only (already-resolved failure modes).
    if with_pitfalls and repo:
        sources["issues"] = fetch_bug_issues(repo)
    else:
        sources["issues"] = []
    # Troubleshooting uses BOTH closed bugs (for fixes) AND open issues (for
    # unresolved gotchas), plus extracted stack traces from their bodies.
    if with_troubleshooting and repo:
        closed = sources["issues"] or fetch_bug_issues(repo)
        open_issues = fetch_open_issues(repo, limit=20, min_reactions=3)
        sources["closed_bug_issues"] = closed
        sources["open_issues"] = open_issues
        sources["stack_traces"] = extract_stack_traces(closed + open_issues)
    else:
        sources["closed_bug_issues"] = []
        sources["open_issues"] = []
        sources["stack_traces"] = []
    if with_version_notes and repo:
        sources["changelog"] = fetch_changelog(repo)
    else:
        sources["changelog"] = ""
    # Community substrate: Stack Exchange + closed `question`-labeled issues.
    # IPI scan happens inside fetch_stackexchange_qas; closed issues are vetted
    # human-moderated content, lower risk.
    if with_community and repo:
        sources["stackexchange_qas"] = fetch_stackexchange_qas_for_repo(repo, limit=15)
        sources["question_issues"] = fetch_question_issues(repo, limit=15)
    else:
        sources["stackexchange_qas"] = []
        sources["question_issues"] = []
    return sources


def _pipeline(
    url: str, args: argparse.Namespace
) -> dict:
    with_troubleshooting = getattr(args, "with_troubleshooting", False)
    with_templates = getattr(args, "with_templates", False)
    with_community = getattr(args, "with_community", False)
    sources = _gather_sources(
        url,
        args.with_pitfalls,
        args.with_version_notes,
        with_troubleshooting,
        with_community,
    )
    include_evals = not args.no_evals
    run_eval_loop = include_evals and not getattr(args, "no_eval_triggering", False)

    plan = plan_structure(
        doc_text=sources["doc"],
        with_pitfalls=args.with_pitfalls,
        with_evals=include_evals,
        with_templates=with_templates,
    )
    skill_name = args.name or plan["skill_name"]

    # Hardware hints — regex side-channel from doc + README
    hardware_hints = ""
    if plan.get("include_hardware_section"):
        hardware_hints = extract_hardware_hints(sources["doc"], sources["readme"])
    include_hardware = bool(hardware_hints) and plan.get("include_hardware_section", False)

    body = write_body(
        skill_name=skill_name,
        references=plan["references"],
        workflows=plan.get("workflows", []),
        templates=plan.get("templates", []),
        include_when_to_use=plan.get("include_when_to_use_section", True),
        include_hardware=include_hardware,
        hardware_hints=hardware_hints,
        readme_install=sources["readme_install"],
        doc_text=sources["doc"],
    )

    refs_content: dict[str, str] = {}
    templates_content: dict[str, str] = {}

    # References + templates synthesized in parallel. All use the LLM; batch them together.
    with ThreadPoolExecutor(max_workers=6) as pool:
        ref_futures = {
            pool.submit(
                write_reference,
                r["filename"], r["covers"], sources["doc"],
                sources["readme"], sources["examples"],
            ): r["filename"] for r in plan["references"]
        }
        tpl_futures = {}
        if with_templates and plan.get("templates"):
            pkg = sources["repo"].split("/")[-1] if sources["repo"] else skill_name
            tpl_futures = {
                pool.submit(
                    write_template,
                    t["filename"], t["covers"], pkg,
                    sources["readme_install"], sources["doc"], sources["examples"],
                ): t["filename"] for t in plan["templates"]
            }
        for fut in as_completed({**ref_futures, **tpl_futures}):
            if fut in ref_futures:
                refs_content[ref_futures[fut]] = fut.result()
            else:
                templates_content[tpl_futures[fut]] = fut.result()

    if args.with_pitfalls and sources["issues"]:
        pkg = sources["repo"].split("/")[-1] if sources["repo"] else skill_name
        refs_content["pitfalls.md"] = distill_pitfalls(pkg, sources["issues"])

    if with_troubleshooting and (
        sources["closed_bug_issues"] or sources["open_issues"] or sources["stack_traces"]
    ):
        pkg = sources["repo"].split("/")[-1] if sources["repo"] else skill_name
        troubleshooting_md = write_troubleshooting(
            package_name=pkg,
            repo=sources["repo"],
            closed_issues=sources["closed_bug_issues"],
            open_issues=sources["open_issues"],
            stack_traces=sources["stack_traces"],
        )
        if troubleshooting_md:
            refs_content["troubleshooting.md"] = troubleshooting_md

    if with_community and (sources["stackexchange_qas"] or sources["question_issues"]):
        pkg = sources["repo"].split("/")[-1] if sources["repo"] else skill_name
        community_md = write_community_gotchas(
            package_name=pkg,
            qas=sources["stackexchange_qas"],
            closed_issues=sources["question_issues"],
        )
        if community_md:
            refs_content["community-gotchas.md"] = community_md

    evals_doc = write_evals(skill_name, body) if include_evals else None

    description = _extract_description(body)
    triggering_report = None
    if run_eval_loop and evals_doc and evals_doc.get("prompts"):
        initial = evaluate_triggering(skill_name, description, evals_doc["prompts"])
        triggering_report = {"initial": initial}
        if initial.get("win_rate") is not None and initial["win_rate"] < 1.0 and initial["failing"]:
            improved = improve_description(
                skill_name=skill_name,
                skill_body=body,
                current_description=description,
                failing=initial["failing"],
            )
            if improved and improved != description:
                revised = evaluate_triggering(skill_name, improved, evals_doc["prompts"])
                if revised.get("win_rate", 0) > initial.get("win_rate", 0):
                    description = improved
                    triggering_report["revised"] = revised
                    triggering_report["description_updated"] = True
                else:
                    triggering_report["revised"] = revised
                    triggering_report["description_updated"] = False

    install_entries = detect_install_commands(sources["readme"])
    bins = ["python3"]

    # Provenance: SHA-256 of the synthesized body so drift can be detected later
    # without re-running the pipeline. URL + repo + fetched_at are the audit trail.
    provenance = {
        "url": url,
        "repo": sources["repo"] or "",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "builder_version": BUILDER_VERSION,
    }
    # Coverage: which substrates contributed to this build.
    coverage = ["html"]  # always have the doc page
    if sources["repo"]:
        coverage.append("gh-readme")
    if args.with_pitfalls and sources.get("issues"):
        coverage.append("gh-issues-bug-closed")
    if with_troubleshooting and (sources.get("open_issues") or sources.get("stack_traces")):
        coverage.append("gh-issues-open")
    if with_community and sources.get("stackexchange_qas"):
        coverage.append("stackexchange")
    if with_community and sources.get("question_issues"):
        coverage.append("gh-issues-question-closed")
    if args.with_version_notes and sources.get("changelog"):
        coverage.append("changelog")

    mcps = _mcp_defaults_for(sources["repo"] or None, url)

    frontmatter = assemble_frontmatter(
        skill_name=skill_name,
        description=description,
        install_entries=install_entries,
        bins=bins,
        mcps=mcps,
        provenance=provenance,
        coverage=coverage,
    )
    skill_md = frontmatter + body

    validation_issues = validate_skill(
        skill_md=skill_md,
        references=refs_content,
        sources={
            "doc": sources["doc"],
            "readme": sources["readme"],
        },
    )
    # Templates validated separately via py_compile
    if templates_content:
        validation_issues.extend(validate_templates(templates_content))

    return {
        "skill_name": skill_name,
        "plan": plan,
        "skill_md": skill_md,
        "references": refs_content,
        "templates": templates_content,
        "evals": evals_doc,
        "triggering_report": triggering_report,
        "validation": validation_issues,
        "source_url": url,
        "repo": sources["repo"],
        "hardware_hints_used": bool(hardware_hints),
    }


# ── MCP server: serve a skill's references over stdio ────────────────────────
# Minimal MCP stdio server implementing the 2025-03-26 spec. Stdlib-only — no
# external `mcp` Python SDK dependency, since this only needs initialize +
# tools/list + tools/call. Each line on stdin is one JSON-RPC request; each
# line on stdout is one JSON-RPC response.


def _mcp_search_refs(skill_dir: Path, query: str, max_hits: int = 5) -> str:
    """Return matching paragraphs from the skill's references/ tree.

    Plain substring + token-overlap scoring — no embeddings. Stdlib-only.
    Each hit is the section heading + ~400 char paragraph + source path.
    """
    refs = skill_dir / "references"
    if not refs.exists():
        return f"(no references directory at {refs})"
    q_tokens = {t.lower() for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", query) if len(t) > 2}
    if not q_tokens:
        return "(empty query)"
    scored: list[tuple[float, str, str, str]] = []  # (score, file, heading, snippet)
    for path in sorted(refs.glob("*.md")):
        text = path.read_text(errors="replace")
        # Split by markdown headings (## / ###)
        sections = re.split(r"(?m)^(#{2,3}\s.+)$", text)
        # sections is [pre, h1, body1, h2, body2, ...]
        cur_heading = path.name
        for i in range(0, len(sections), 2):
            if i == 0:
                body = sections[0]
            else:
                cur_heading = sections[i].lstrip("# ").strip()
                body = sections[i + 1] if i + 1 < len(sections) else ""
            body_tokens = {t.lower() for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", body) if len(t) > 2}
            if not body_tokens:
                continue
            overlap = len(q_tokens & body_tokens)
            if overlap == 0:
                continue
            score = overlap / max(1, len(q_tokens)) + (overlap / max(1, len(body_tokens))) * 0.5
            snippet = body.strip()[:600]
            scored.append((score, path.name, cur_heading, snippet))
    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        return f"(no matches for query: {query!r})"
    out = []
    for score, fname, heading, snippet in scored[:max_hits]:
        out.append(f"## from `references/{fname}` — {heading}  (score={score:.2f})\n\n{snippet}")
    return "\n\n---\n\n".join(out)


def _mcp_send(payload: dict) -> "None":
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _mcp_handle(skill_dir: Path, req: dict) -> dict | None:
    """Dispatch one JSON-RPC request. Returns response dict, or None for notifications."""
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": f"skill-{skill_dir.name}", "version": BUILDER_VERSION},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "tools": [
                    {
                        "name": "search_skill_refs",
                        "description": (
                            f"Search the {skill_dir.name} skill's references/ directory. "
                            "Returns matching sections from references/*.md ranked by token overlap. "
                            "Use this when SKILL.md / loaded references don't fully answer a question."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "max_hits": {"type": "integer", "default": 5},
                            },
                            "required": ["query"],
                        },
                    }
                ]
            },
        }
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "search_skill_refs":
            text = _mcp_search_refs(skill_dir, args.get("query", ""), int(args.get("max_hits", 5)))
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": f"unknown tool: {name}"},
        }
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def cmd_serve(args: argparse.Namespace) -> dict:
    """Run the skill-as-MCP stdio server. Blocks until stdin closes."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    if not (skill_dir / "SKILL.md").exists():
        _die(f"no SKILL.md at {skill_dir}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _mcp_handle(skill_dir, req)
        if resp is not None:
            _mcp_send(resp)
    return {"served": str(skill_dir)}


# ── Subcommand handlers ───────────────────────────────────────────────────────


def cmd_sources(args: argparse.Namespace) -> dict:
    repo = args.repo
    if "/" not in repo:
        _die("expected owner/repo format")
    readme = fetch_readme(repo)
    examples = fetch_examples_listing(repo)
    issues = fetch_bug_issues(repo, limit=5)
    changelog = fetch_changelog(repo)
    return {
        "repo": repo,
        "readme_len": len(readme),
        "readme_install_len": len(extract_install_section(readme)),
        "examples_count": len(examples),
        "examples_sample": examples[:5],
        "bug_issues_count": len(issues),
        "issues_sample": [{"number": i["number"], "title": i["title"]} for i in issues[:3]],
        "changelog_len": len(changelog),
    }


def cmd_plan(args: argparse.Namespace) -> dict:
    url = _normalize_url(args.url)
    doc = fetch_doc(url)
    plan = plan_structure(
        doc,
        args.with_pitfalls,
        not args.no_evals,
        with_templates=getattr(args, "with_templates", False),
    )
    plan["source_url"] = url
    plan["doc_chars"] = len(doc)
    return plan


def cmd_preview(args: argparse.Namespace) -> dict:
    url = _normalize_url(args.url)
    result = _pipeline(url, args)
    return {
        "skill_name": result["skill_name"],
        "source_url": url,
        "validation": result["validation"],
        "SKILL.md": result["skill_md"],
        "references": result["references"],
        "templates": result.get("templates", {}),
        "evals": result["evals"],
        "triggering_report": result.get("triggering_report"),
        "hardware_hints_used": result.get("hardware_hints_used", False),
    }


def cmd_build(args: argparse.Namespace) -> dict:
    url = _normalize_url(args.url)
    result = _pipeline(url, args)
    # Fail on errors, pass through warnings
    errors = [i for i in result["validation"] if i["severity"] == "error"]
    if errors:
        _audit("build.rejected", {
            "url": url, "skill_name": result["skill_name"],
            "errors": errors,
        })
        return {
            "status": "rejected",
            "reason": "validation errors",
            "validation": result["validation"],
            "skill_name": result["skill_name"],
            "source_url": url,
        }

    out_dir = Path(args.out) if args.out else SKILLS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = write_skill(
        out_dir=out_dir,
        skill_name=result["skill_name"],
        skill_md=result["skill_md"],
        references=result["references"],
        evals=result["evals"],
        templates=result.get("templates"),
        force=args.force,
    )
    ok, check_out = openclaw_skills_check(target)
    update_lockfile(result["skill_name"], url, result["skill_md"], target)
    _audit("build.completed", {
        "url": url, "skill_name": result["skill_name"],
        "path": str(target), "openclaw_check_ok": ok,
        "warnings": [i for i in result["validation"] if i["severity"] == "warning"],
    })
    return {
        "status": "ok",
        "skill_name": result["skill_name"],
        "path": str(target),
        "source_url": url,
        "repo": result["repo"],
        "files_written": [
            "SKILL.md",
            *(f"references/{k}" for k in result["references"]),
            *(f"templates/{k}" for k in (result.get("templates") or {})),
            *(["evals/evals.json"] if result["evals"] else []),
        ],
        "validation_warnings": [i for i in result["validation"] if i["severity"] == "warning"],
        "openclaw_skills_check": {"ok": ok, "output": check_out[:1000]},
        "triggering_report": result.get("triggering_report"),
        "hardware_hints_used": result.get("hardware_hints_used", False),
    }


def cmd_built(_args: argparse.Namespace) -> dict:
    lockfile = DATA_DIR / "built-skills.json"
    if not lockfile.exists():
        return {"built": {}}
    try:
        return {"built": json.loads(lockfile.read_text())}
    except json.JSONDecodeError:
        return {"built": {}}


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skill_builder.py", description="Build OpenClaw skills from docs")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_build_flags(sp: argparse.ArgumentParser) -> "None":
        sp.add_argument("--name", default=None)
        sp.add_argument("--with-pitfalls", action="store_true",
                        help="Closed bug issues distilled into pitfalls.md")
        sp.add_argument("--with-troubleshooting", action="store_true",
                        help="Open + closed issues + stack traces distilled into troubleshooting.md")
        sp.add_argument("--with-version-notes", action="store_true")
        sp.add_argument("--with-templates", action="store_true",
                        help="Generate 1-3 runnable Python scripts in templates/ (py_compile validated)")
        sp.add_argument("--with-community", action="store_true",
                        help="Curated Stack Exchange Q&As + closed `question` issues distilled into "
                             "references/community-gotchas.md (CC BY-SA attributed; IPI-scanned)")
        sp.add_argument("--no-evals", action="store_true")
        sp.add_argument("--no-eval-triggering", action="store_true",
                        help="Skip the triggering judge + description optimizer loop")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--out", default=None)

    build = sub.add_parser("build", help="Full pipeline: fetch, synthesize, validate, write")
    build.add_argument("url")
    _add_build_flags(build)
    build.set_defaults(func=cmd_build)

    preview = sub.add_parser("preview", help="Synthesize but print to stdout, don't write")
    preview.add_argument("url")
    _add_build_flags(preview)
    preview.set_defaults(func=cmd_preview)

    plan = sub.add_parser("plan", help="Only run PLAN STRUCTURE; show file decisions")
    plan.add_argument("url")
    plan.add_argument("--with-pitfalls", action="store_true")
    plan.add_argument("--with-templates", action="store_true")
    plan.add_argument("--no-evals", action="store_true")
    plan.set_defaults(func=cmd_plan)

    sources = sub.add_parser("sources", help="Dry-run: show what would be fetched")
    sources.add_argument("repo", help="owner/repo")
    sources.set_defaults(func=cmd_sources)

    built = sub.add_parser("built", help="List skills this agent has generated")
    built.set_defaults(func=cmd_built)

    serve = sub.add_parser(
        "serve",
        help="Run an MCP stdio server exposing search_skill_refs over a skill's references/",
    )
    serve.add_argument("skill_dir", help="Path to a skill directory containing SKILL.md")
    serve.set_defaults(func=cmd_serve)

    return p


def _emit_result(args: argparse.Namespace, result: dict) -> "None":
    """Print JSON unless this is `serve` (which streams to stdout itself)."""
    if getattr(args, "cmd", None) == "serve":
        return
    print(json.dumps(result, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    _emit_result(args, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
