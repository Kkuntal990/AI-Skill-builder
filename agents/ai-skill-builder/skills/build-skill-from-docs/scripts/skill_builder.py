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
import shutil
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
# Max generate→critic→repair rounds before shipping with residual findings as
# warnings (ship-with-warning, not hard-reject — a usable skill with a flagged
# residual beats no skill; the residual is surfaced via quality_gate + report).
MAX_REPAIR_ROUNDS = 3
# Skills with more than this many DOMAIN reference modules are flagged for review:
# SkillsBench (arXiv:2602.12670) finds focused skills (≤3 modules) outperform
# larger bundles. Soft signal (warning), not a hard cap.
MAX_FOCUSED_MODULES = 3
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"
HTTP_TIMEOUT = 30
LLM_TIMEOUT = 120
USER_AGENT = "ai-skill-builder/1.0 (+https://github.com/Kkuntal990/AI-Skill-builder)"
BUILDER_VERSION = "2.1.0"

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


def _claude_cli_call(prompt: str, max_tokens: int = 4000, temperature: float = 0.3) -> str:  # noqa: ARG001
    """One-shot inference via the local Claude Code CLI (`claude -p`).

    Routes through the user's Claude SUBSCRIPTION (no per-token API credit) using
    Claude models (the CLI's configured default — typically Opus — unless
    MLEVAL_LLM_MODEL overrides via `--model`). `claude -p` does not expose
    temperature/max_tokens, so those are accepted for signature parity and ignored.
    The prompt fits well under ARG_MAX (doc text is capped at ~15k chars), so it is
    passed as an argument, matching the verified smoke test.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise FileNotFoundError("`claude` CLI not found on PATH")
    cmd = [claude_bin, "-p", prompt, "--output-format", "text"]
    model = os.environ.get("MLEVAL_LLM_MODEL", "").strip()
    if model:
        cmd += ["--model", model]
    timeout_s = max(LLM_TIMEOUT * 3, 300)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"claude CLI call timed out after {timeout_s}s") from e
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {(proc.stderr or '')[:300]}")
    return proc.stdout


def _llm_call(prompt: str, max_tokens: int = 4000, temperature: float = 0.3) -> str:
    """Dispatch one LLM call to the configured transport.

    MLEVAL_LLM_TRANSPORT selects the backend:
      - "claude" (default): local Claude Code CLI -> Claude SUBSCRIPTION (no API
        credit), Claude models. Falls back to OpenRouter if the CLI is missing/fails
        and an OpenRouter key is configured.
      - "openrouter": direct OpenRouter API -> PAID credit. Needs OPENROUTER_API_KEY
        (or the openrouter:default OpenClaw auth profile); model = OPENROUTER_MODEL.

    All synthesis steps (plan/body/reference/critique/repair/triggering/evals/...)
    funnel through here, so the transport is a single switch.
    """
    transport = os.environ.get("MLEVAL_LLM_TRANSPORT", "claude").strip().lower()
    if transport == "openrouter":
        return _openrouter_call(prompt, max_tokens=max_tokens, temperature=temperature)
    # default: Claude subscription via CLI, with OpenRouter as the paid fallback.
    if shutil.which("claude"):
        try:
            return _claude_cli_call(prompt, max_tokens=max_tokens, temperature=temperature)
        except (FileNotFoundError, RuntimeError) as e:
            if _load_openrouter_key():
                return _openrouter_call(prompt, max_tokens=max_tokens, temperature=temperature)
            _die(f"claude CLI transport failed and no OpenRouter fallback configured: {e}")
            return ""
    if _load_openrouter_key():
        return _openrouter_call(prompt, max_tokens=max_tokens, temperature=temperature)
    _die("No LLM transport available: `claude` CLI not on PATH and no OpenRouter key "
         "(set MLEVAL_LLM_TRANSPORT=openrouter + OPENROUTER_API_KEY, or install/login `claude`).")
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
    with_scripts: bool = False,
    with_version_notes: bool = False,
    intent: str = "",
) -> dict:
    tpl = _read_prompt("plan_structure.txt")
    prompt = _fill_prompt(
        tpl,
        doc_text=doc_text[:20000],
        with_pitfalls=str(with_pitfalls).lower(),
        with_evals=str(with_evals).lower(),
        with_templates=str(with_templates).lower(),
        with_scripts=str(with_scripts).lower(),
        with_version_notes=str(with_version_notes).lower(),
        intent=intent or "(none provided — infer sensible, cheapest-that-works defaults)",
    )
    raw = _strip_fences(_llm_call(prompt, max_tokens=3500, temperature=0.2))
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
    plan.setdefault("mcp_workflow_triggers", [])
    plan.setdefault("decision_tree", [])
    plan.setdefault("templates", [])
    plan.setdefault("scripts", [])
    plan.setdefault("old_patterns", [])
    plan.setdefault("include_hardware_section", False)
    plan.setdefault("include_when_to_use_section", True)
    # Cap sizes defensively
    plan["workflows"] = plan["workflows"][:4]
    plan["templates"] = plan["templates"][:3] if with_templates else []
    plan["scripts"] = plan["scripts"][:3] if with_scripts else []
    plan["decision_tree"] = plan["decision_tree"][:6]
    plan["old_patterns"] = plan["old_patterns"][:6] if with_version_notes else []
    # Trim MCP triggers to one-per-workflow alignment; pad with empty string if LLM under-emitted
    triggers = [str(t) for t in plan["mcp_workflow_triggers"][: len(plan["workflows"])]]
    while len(triggers) < len(plan["workflows"]):
        triggers.append("")
    plan["mcp_workflow_triggers"] = triggers
    # Enforce checklist shape
    cleaned_wfs = []
    for wf in plan["workflows"]:
        if isinstance(wf, dict) and "title" in wf and "checklist" in wf:
            wf["checklist"] = [str(s) for s in wf["checklist"][:6]]
            cleaned_wfs.append(wf)
    plan["workflows"] = cleaned_wfs
    # Enforce decision_tree row shape
    cleaned_dt = []
    for row in plan["decision_tree"]:
        if isinstance(row, dict) and {"trigger", "decide", "refer_to"} <= set(row):
            cleaned_dt.append({k: str(row[k]) for k in ("trigger", "decide", "refer_to")})
    plan["decision_tree"] = cleaned_dt
    # Enforce script row shape
    cleaned_scripts = []
    for s in plan["scripts"]:
        if isinstance(s, dict) and {"filename", "lang", "purpose"} <= set(s):
            if s["lang"] not in ("bash", "python"):
                continue
            s.setdefault("executes_what", "")
            cleaned_scripts.append({k: str(s[k]) for k in ("filename", "lang", "purpose", "executes_what")})
    plan["scripts"] = cleaned_scripts
    return plan


def build_contract(plan: dict, intent: str = "") -> str:
    """Format the skill's gates into a compact "Skill Contract" that travels into
    every reference/script/template so a precondition stays WITH its action.

    Phase 3.0-1 (Skills 3.0). The gates already exist in ``plan["decision_tree"]``
    (each row = {trigger, decide, refer_to}); the QLoRA gate-split happened because
    they were never passed to ``write_reference`` — the reference author was blind to
    the body's gates and reproduced the source docs' ungated framing. Threading this
    Contract in mirrors Anthropic's ``pdf/forms.md`` discipline (a fallback path opens
    with "Use this when …"). Returns "" when there are no gates/purpose to carry.
    """
    lines: list[str] = []
    purpose = str(plan.get("one_line_purpose", "")).strip()
    if purpose:
        lines.append(f"PURPOSE: {purpose}")
    if intent.strip():
        lines.append(f"INTENT / TARGET ENVIRONMENT: {intent.strip()}")
    dt = plan.get("decision_tree") or []
    gate_rows = [r for r in dt if isinstance(r, dict) and r.get("trigger")]
    if gate_rows:
        lines.append(
            "GATES (each row is a precondition that MUST travel with its action — "
            "if THIS file documents the action, restate the 'when' at its point of use):"
        )
        for row in gate_rows:
            lines.append(
                f"  - WHEN {row.get('trigger','')} → {row.get('decide','')}"
                f"  (see {row.get('refer_to','')})"
            )
    return "\n".join(lines)


def infer_intent(doc_text: str) -> str:
    """Infer a short intent brief {purpose · target environment · success criteria} from the
    docs when the user didn't pass one (Phase 3.0-2 "infer-and-state-assumptions" mode).

    Grounded in Anthropic skill-creator's 4 scoping questions. Best-effort: returns "" if the
    prompt is missing or the LLM call fails — the pipeline then proceeds with no intent, so
    inference never blocks a build. The result is recorded as an ASSUMPTION in provenance.
    """
    try:
        tpl = _read_prompt("intent_capture.txt")
    except FileNotFoundError:
        return ""
    try:
        raw = _strip_fences(_llm_call(_fill_prompt(tpl, doc_text=doc_text[:12000]),
                                      max_tokens=400, temperature=0.2)).strip()
    except (RuntimeError, OSError):
        return ""
    return raw[:1200]


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
    mcp_workflow_triggers: list[str] | None = None,
    decision_tree: list[dict] | None = None,
    scripts: list[dict] | None = None,
    old_patterns: list[dict] | None = None,
    intent: str = "",
) -> str:
    tpl = _read_prompt("write_skill_body.txt")
    mcp_workflow_triggers = mcp_workflow_triggers or []
    decision_tree = decision_tree or []
    scripts = scripts or []
    old_patterns = old_patterns or []
    refs_list = "\n".join(
        f"- `references/{r['filename']}` — {r['covers']}" for r in references
    ) or "(no reference files)"
    workflows_list = "\n".join(
        f"- **{w['title']}** — checklist: {w['checklist']}" for w in workflows
    ) or "(no workflows)"
    if mcp_workflow_triggers and workflows:
        mcp_triggers_text = "\n".join(
            f"- For workflow '{w['title']}': {t or '(no MCP trigger — skip the MCP step in this workflow)'}"
            for w, t in zip(workflows, mcp_workflow_triggers)
        )
    else:
        mcp_triggers_text = "(no MCP triggers — omit per-workflow MCP steps AND the tail '## Looking things up live' section)"
    if decision_tree:
        decision_tree_text = "\n".join(
            f"- trigger: '{r['trigger']}' → decide: '{r['decide']}' → refer_to: '{r['refer_to']}'"
            for r in decision_tree
        )
    else:
        decision_tree_text = "(no decision tree — skip the `## Decision Tree` section entirely)"
    templates_list = "\n".join(
        f"- `templates/{t['filename']}` — {t['covers']}" for t in templates
    ) or "(no templates — skip the `## Templates` section entirely)"
    if scripts:
        scripts_list = "\n".join(
            f"- `scripts/{s['filename']}` ({s['lang']}) — {s['purpose']}. Run when: {s.get('executes_what','')}"
            for s in scripts
        )
    else:
        scripts_list = "(no scripts — skip the `## Scripts` section entirely)"
    if old_patterns:
        old_patterns_list = "\n".join(
            f"- {p['name']} (deprecated in {p.get('deprecated_in','?')}) → use {p.get('replacement','?')}"
            for p in old_patterns
        )
    else:
        old_patterns_list = "(no old patterns — skip the `## Old Patterns` section entirely)"
    prompt = _fill_prompt(
        tpl,
        skill_name=skill_name,
        references_list=refs_list,
        workflows_list=workflows_list,
        mcp_workflow_triggers_list=mcp_triggers_text,
        decision_tree_list=decision_tree_text,
        templates_list=templates_list,
        scripts_list=scripts_list,
        old_patterns_list=old_patterns_list,
        hardware_hints=hardware_hints or "(no hardware hints extracted)",
        include_when_to_use=str(include_when_to_use).lower(),
        include_hardware=str(include_hardware and bool(hardware_hints)).lower(),
        readme_install=readme_install[:3000] or "(README install section not found)",
        doc_text=doc_text[:15000],
        intent=intent or "(none provided — default to the cheapest approach that works)",
    )
    raw = _strip_fences(_llm_call(prompt, max_tokens=5000, temperature=0.3))
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
    raw = _llm_call(prompt, max_tokens=4000, temperature=0.2)
    # The LLM may still wrap in ```python fences despite instructions — strip them.
    raw = _strip_fences(raw)
    return raw.strip() + "\n"


def write_utility_script(
    filename: str,
    lang: str,
    purpose: str,
    executes_what: str,
    readme_text: str,
    doc_text: str,
) -> str:
    """Generate a short utility script (bash or python) for scripts/. Returns raw source."""
    tpl = _read_prompt("write_scripts.txt")
    prompt = _fill_prompt(
        tpl,
        filename=filename,
        lang=lang,
        purpose=purpose,
        executes_what=executes_what or "(no specific trigger)",
        readme_text=readme_text[:2000] or "(not available)",
        doc_text=doc_text[:12000],
    )
    raw = _llm_call(prompt, max_tokens=2500, temperature=0.2)
    raw = _strip_fences(raw)
    return raw.strip() + "\n"


_TOC_PATTERN = re.compile(r"^##\s+Contents\b", re.MULTILINE | re.IGNORECASE)
_H2_PATTERN = re.compile(r"^##\s+(?!Contents\b)(.+?)\s*$", re.MULTILINE)


def _inject_toc_if_long(content: str, min_lines: int = 100) -> str:
    """Prepend a `## Contents` ToC to references over `min_lines` if the LLM didn't write one.

    Idempotent: skips files that already have a `## Contents` section.
    Inserts the ToC between the title's intro paragraph and the first `##` heading.
    """
    if not content or content.count("\n") + 1 < min_lines:
        return content
    if _TOC_PATTERN.search(content):
        return content
    headings = _H2_PATTERN.findall(content)
    if len(headings) < 3:
        # Few sections — ToC adds no value
        return content
    toc_lines = ["## Contents", ""]
    toc_lines.extend(f"- {h.strip()}" for h in headings)
    toc_lines.append("")
    toc_block = "\n".join(toc_lines)
    # Insert before the first `## ` heading
    first_h2 = re.search(r"^##\s+", content, re.MULTILINE)
    if not first_h2:
        return content
    return content[: first_h2.start()].rstrip() + "\n\n" + toc_block + "\n" + content[first_h2.start() :]


def validate_scripts(scripts: dict[str, str]) -> list[dict]:
    """Light validation: Python scripts py_compile; bash scripts shebang + bash -n.

    Returns a list of validation issues (same shape as validate_templates).
    """
    import tempfile as _tempfile
    issues = []
    for name, src in scripts.items():
        is_py = name.endswith(".py")
        is_sh = name.endswith(".sh")
        if not (is_py or is_sh):
            continue
        suffix = ".py" if is_py else ".sh"
        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(src)
            tmp_path = f.name
        try:
            if is_py:
                cmd = [sys.executable, "-m", "py_compile", tmp_path]
            else:
                cmd = ["bash", "-n", tmp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                issues.append({
                    "kind": "script_syntax",
                    "file": f"scripts/{name}",
                    "detail": err[:500],
                })
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            issues.append({
                "kind": "script_check_failed",
                "file": f"scripts/{name}",
                "detail": str(e),
            })
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
    return issues


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
    contract: str = "",
) -> str:
    tpl = _read_prompt("write_reference.txt")
    prompt = _fill_prompt(
        tpl,
        filename=filename,
        topic_covers=topic_covers,
        doc_text=doc_text[:18000],
        readme_text=readme_text[:3000],
        examples_text=examples_text[:2000],
        contract=contract or "(no gates declared)",
    )
    raw = _strip_fences(_llm_call(prompt, max_tokens=5000, temperature=0.3))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


def write_evals(skill_name: str, skill_body: str) -> dict:
    tpl = _read_prompt("write_evals.txt")
    prompt = _fill_prompt(tpl, skill_name=skill_name, skill_body=skill_body[:4000])
    raw = _strip_fences(_llm_call(prompt, max_tokens=1000, temperature=0.5))
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
    raw = _strip_fences(_llm_call(prompt, max_tokens=3000, temperature=0.3))
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
    raw = _strip_fences(_llm_call(prompt, max_tokens=4000, temperature=0.3))
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
    raw = _strip_fences(_llm_call(prompt, max_tokens=4000, temperature=0.3))
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
    raw = _strip_fences(_llm_call(prompt, max_tokens=400, temperature=0.0))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"choice": "none", "reason": "judge returned no JSON", "confidence": 0.0}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"choice": "none", "reason": "judge JSON parse failed", "confidence": 0.0}


def evaluate_triggering(
    skill_name: str,
    skill_description: str,
    eval_prompts: list[dict],
    siblings: list[dict] | None = None,
    negative_prompts: list[dict] | None = None,
) -> dict:
    """Run the triggering judge over each eval prompt. Returns per-prompt results + win rate.

    `siblings`: real co-resident skill descriptions to compete against instead of
    the canned `DECOY_SKILLS` (set via `--siblings`). Competing against the actual
    sibling set is a far harder, more realistic precision test than synthetic decoys.

    `negative_prompts`: should-NOT-trigger near-misses. If the target wins one, that
    is a false positive (over-triggering). Makes the eval bidirectional — it catches
    over-selection (the other half of the triggering failure mode), not just misses.
    """
    if not eval_prompts:
        return {"win_rate": None, "results": [], "failing": []}
    competitors = siblings if siblings else DECOY_SKILLS
    skills = [{"name": skill_name, "description": skill_description}] + competitors
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
    # Bidirectional: should-NOT-trigger near-misses. Target winning = false positive.
    false_positives = []
    for p in (negative_prompts or []):
        msg = p.get("prompt", "")
        if not msg:
            continue
        verdict = judge_triggering(msg, skills)
        if verdict.get("choice") == skill_name:
            false_positives.append({
                "prompt_id": p.get("id"),
                "prompt": msg,
                "judge_reason": verdict.get("reason", ""),
            })
    return {
        "win_rate": wins / total if total else None,
        "wins": wins,
        "total": total,
        "results": results,
        "failing": failing,
        "competitors": "siblings" if siblings else "decoys",
        "n_competitors": len(competitors),
        "false_positives": false_positives,
        "fp_total": len(negative_prompts or []),
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
    raw = _strip_fences(_llm_call(prompt, max_tokens=800, temperature=0.4))
    # Take the first non-empty paragraph
    for para in raw.split("\n\n"):
        p = para.strip()
        if len(p) > 50 and not p.startswith("#"):
            return p
    return raw.strip() or current_description


# ── Quality critic + bounded repair (P1–P4 reliability checklist) ─────────────
# critique_skill mirrors _run_scanner's BLOCK/WARN shape: deterministic regex for
# the mechanical anti-patterns (P3) + structural checks (P1/P2), plus one LLM call
# for the judgment dimensions (P4 scope-honesty, P1 semantic). The LLM critic is
# told to abstain when unsure (demystifying-evals judge hygiene) so it adds recall
# without inventing findings. "block" findings drive the repair loop in _pipeline.

# All-caps directive words used as standalone rigid commands (P3 anti-pattern).
_CRIT_ALLCAPS = re.compile(r"(?<![A-Za-z])(ALWAYS|NEVER|MUST NOT|MUST|DO NOT|DON'T)(?![A-Za-z])")
# Windows-style paths: drive-letter (C:\) or UNC/backslash separators.
_CRIT_WIN_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\[A-Za-z]|[A-Za-z]+\\[A-Za-z]+\\)")
# Time-sensitive phrasing tied to a month/year ("after July 2026", "as of 2026").
_CRIT_TIME = re.compile(
    r"\b(?:after|before|as of|since|starting|until)\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December|"
    r"(?:19|20)\d{2})",  # year-like only (avoids "after 1000 steps")
    re.IGNORECASE,
)
# First/second-person pronouns in a description paragraph (P1 — must be third person).
_CRIT_FIRST_PERSON = re.compile(r"(?<![A-Za-z])(I|we|our|us|my|you|your)(?![A-Za-z])", re.IGNORECASE)


def critique_skill(
    body: str, skill_name: str, module_count: int = 0, run_llm: bool = True
) -> list[dict]:
    """Critique a SKILL.md body against the P1–P4 reliability checklist.

    Returns findings shaped like _run_scanner output:
    {"severity": "block"|"warn", "where": "<dimension>", "message": str}.
    "block" findings drive a repair round; "warn" findings ship as report
    warnings. Deterministic regex covers P3 anti-patterns + mechanical P1/P2;
    one LLM call (skippable via run_llm=False) covers P4 scope-honesty + semantic P1.
    """
    findings: list[dict] = []

    # ── P3 — content anti-patterns (deterministic) ──
    allcaps = sorted({m.upper() for m in _CRIT_ALLCAPS.findall(body)})
    if allcaps:
        findings.append({
            "severity": "warn", "where": "P3-allcaps",
            "message": f"rigid all-caps directive(s) {allcaps} — state the constraint once and explain why instead",
        })
    if _CRIT_WIN_PATH.search(body):
        findings.append({
            "severity": "warn", "where": "P3-windows-path",
            "message": "Windows-style backslash path — use forward slashes for portability",
        })
    if _CRIT_TIME.search(body):
        findings.append({
            "severity": "warn", "where": "P3-time-sensitive",
            "message": "time-sensitive phrasing tied to a date — move legacy info to an `## Old Patterns` section",
        })

    # ── P1 — description mechanics (deterministic) ──
    desc = _extract_description(body)
    if desc:
        if _CRIT_FIRST_PERSON.search(desc):
            findings.append({
                "severity": "warn", "where": "P1-person",
                "message": "description uses first/second person — write in third person (it is injected into the skill-selection system prompt)",
            })
        if not re.search(r"\b(?:use|invoke)\b[^.]*\bwhen(?:ever)?\b", desc, re.IGNORECASE):
            findings.append({
                "severity": "warn", "where": "P1-when-clause",
                "message": "description has no 'Use when...' trigger clause — agents under-trigger without explicit when-conditions",
            })

    # ── P2 — focus / anti-bloat (deterministic; SkillsBench focused-skill finding) ──
    if module_count > MAX_FOCUSED_MODULES:
        findings.append({
            "severity": "warn", "where": "P2-bloat",
            "message": f"{module_count} domain reference modules; focused skills (≤{MAX_FOCUSED_MODULES}) outperform large bundles (SkillsBench) — consider grouping or splitting",
        })

    # ── P4 + semantic P1 — LLM judgment (abstain-when-unsure) ──
    if run_llm:
        try:
            tpl = _read_prompt("critique_skill.txt")
            prompt = _fill_prompt(tpl, skill_name=skill_name, skill_body=body[:12000])
            raw = _strip_fences(_llm_call(prompt, max_tokens=1200, temperature=0.0))
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match is None:
                raise ValueError("critic returned no JSON array")
            llm_findings = json.loads(match.group(0))
            # Map each LLM critic dimension to its reliability-checklist priority.
            _dim_to_p = {
                "scope-honesty": "P4",
                "conditional-gating": "P3",
                "description": "P1",
            }
            for f in llm_findings:
                sev = "block" if f.get("severity") == "block" else "warn"
                dim = f.get("dimension", "scope-honesty")
                span = (f.get("offending_span") or "").strip()
                reason = (f.get("reason") or "").strip()
                msg = reason
                if span:
                    msg = f"{reason} (offending text: {span[:160]!r})"
                findings.append({
                    "severity": sev,
                    "where": f"{_dim_to_p.get(dim, 'P1')}-{dim}",
                    "message": msg or f"{dim} violation",
                })
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            # Critic LLM returned unparseable output — fail open (no block) so a
            # flaky judge can't wedge the build; deterministic checks still apply.
            findings.append({
                "severity": "warn", "where": "P4-critic-error",
                "message": "scope-honesty critic returned unparseable output; deterministic checks applied only",
            })
    return findings


# Reference-scan (Phase 3.0-1): a resource-heavy / conditional action documented in a
# reference file must restate its precondition at the point of use (Anthropic pdf/forms.md
# discipline). The body critic above only sees the body, so it cannot catch a gate that
# lives in SKILL.md but is dropped from the reference — exactly the QLoRA gate-split.
_REF_HEAVY_ACTION = re.compile(
    r"\b(?:load_in_4bit|load_in_8bit|BitsAndBytesConfig|bnb_4bit|nf4"
    r"|prepare_model_for_kbit_training|use_dora\s*=\s*True|DeepSpeed|ZeRO-?[0-3]"
    r"|FSDP|GPTQConfig)\b",
    re.IGNORECASE,
)
_REF_PRECONDITION = re.compile(
    r"(?:use\s+this\s+only\s+(?:when|if)|only\s+when|only\s+if|use\s+only\s+(?:when|if)"
    r"|does\s*n[o']?t\s+fit|doesn'?t\s+fit|won'?t\s+fit|too\s+large\s+(?:for|to)"
    r"|with\s+memory\s+headroom|otherwise\s+use|prefer\b[^.\n]*\bunless"
    r"|<\s*\d+\s*GB|memory[- ]constrained)",
    re.IGNORECASE,
)


def critique_references(references: "dict[str, str] | None", contract: str = "") -> list[dict]:
    """Scan reference files for a resource-heavy/conditional action documented WITHOUT
    restating its precondition (Phase 3.0-1). Returns block findings keyed
    ``P3-ungated-reference``. Deterministic, no LLM call.

    Only fires when the skill actually declares gates (non-empty ``contract``) — with no
    declared gate there is no precondition to demand. This is the detector that would have
    caught ``peft-tuning/references/quantization.md`` (full QLoRA recipe, no "only when").
    """
    out: list[dict] = []
    if not references or not (contract or "").strip():
        return out
    for name, content in references.items():
        c = content or ""
        heavy = sorted({m.group(0).lower() for m in _REF_HEAVY_ACTION.finditer(c)})
        if heavy and not _REF_PRECONDITION.search(c):
            out.append({
                "severity": "block",
                "where": "P3-ungated-reference",
                "message": (
                    f"references/{name} documents resource-heavy/conditional action(s) {heavy} "
                    f"but restates no precondition (\"use this only when …\"); the gate lives only "
                    f"in SKILL.md and won't travel with the action (the QLoRA gate-split). Add the "
                    f"WHEN clause at the recipe's point of use."
                ),
            })
    return out


def repair_skill_body(body: str, findings: list[dict], skill_name: str) -> str:
    """Re-generate the full body fixing the listed findings.

    Full regeneration, NOT a diff-patch: targeted patching is what corrupts
    bodies (mid-diff truncation → SyntaxError); full regen with a "fix only these"
    instruction is robust.
    """
    tpl = _read_prompt("repair_skill_body.txt")
    findings_text = "\n".join(
        f"- [{f['severity']}] {f['where']}: {f['message']}" for f in findings
    ) or "(no findings)"
    prompt = _fill_prompt(
        tpl, skill_name=skill_name, findings=findings_text, skill_body=body,
    )
    raw = _strip_fences(_llm_call(prompt, max_tokens=5000, temperature=0.2))
    idx = raw.find("# ")
    if idx > 0:
        raw = raw[idx:]
    return raw.strip() + "\n"


def _load_sibling_descriptions(siblings_dir: str, exclude_name: str = "") -> list[dict]:
    """Load {name, description} for each SKILL.md under siblings_dir (one level of
    subdirs, plus the dir itself), excluding the skill being built. Lets the
    triggering eval compete against REAL co-resident skills instead of canned decoys."""
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


_API_KEY_VALUE = re.compile(
    r"(?i)(?:OPENAI|ANTHROPIC)_API_KEY\s*=\s*['\"]?sk-[\w-]{20,}"
)

# Patterns inherent to HTTP/REST API + distributed-systems documentation.
# Skill-scout flags these for low-trust skill discovery (where any network egress
# is suspect); skill-builder works from high-trust official docs where these are
# the *canonical examples* — every API doc has curl POST, every distributed
# system doc has connectivity tests. Demoted from BLOCK to warning so reviewers
# still see them but the build isn't gated.
_DEMOTE_TO_CAUTION = frozenset({
    "exfiltration: curl pipe",
    "exfiltration: curl POST",
    "exfiltration: curl data send",
    "exfiltration: HTTP POST via requests",
    "exfiltration: netcat",       # `nc -zv host port` for connectivity tests
    "exfiltration: raw socket connect",  # `socket.connect()` in client examples
})


def _is_ml_safe(pattern_desc: str, text: str) -> bool:
    """Return True if a block hit is a known false positive for ML content.

    Scout's `injection: eval()` / `exec()` patterns match Python's dangerous
    builtins. But `model.eval()` (set model to eval mode) and `trainer.eval()`
    are ubiquitous in ML code and harmless. Only treat as injection if we find
    at least one call that ISN'T a method call.

    `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are required env var names in
    legitimate docs (e.g. vLLM's OpenAI-compatible server). Safe unless an
    actual key value (sk-...) follows.
    """
    if pattern_desc in ("injection: eval()", "injection: exec()"):
        fn = "eval" if "eval" in pattern_desc else "exec"
        # Does any match NOT have a dot or word char immediately preceding it?
        for m in _BUILTIN_CALL.finditer(text):
            if m.group("fn") == fn:
                return False  # real builtin call found → not safe
        return True  # every match was a method call
    if pattern_desc in (
        "credential: OpenAI key reference",
        "credential: Anthropic key reference",
    ):
        # Safe iff no actual key value (sk-...) appears in the text.
        return _API_KEY_VALUE.search(text) is None
    return False


def _run_scanner(
    text: str, sources: dict[str, str] | None = None  # noqa: ARG001
) -> tuple[list[str], list[str]]:
    """Return (blocks, cautions) found in text. Filters ML-safe false positives
    and demotes HTTP/networking patterns from BLOCK to CAUTION (still surfaced
    to the reviewer, but no longer gates the build).
    """
    blocks, cautions = [], []
    for pat, desc in BLOCK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            if _is_ml_safe(desc, text):
                continue
            if desc in _DEMOTE_TO_CAUTION:
                cautions.append(desc)
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
    skill_md: str, references: dict[str, str], sources: dict[str, str],
    templates: dict[str, str] | None = None, scripts: dict[str, str] | None = None,
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
            # name: present + lowercase/hyphen charset, <=64 chars, no reserved words.
            # (The charset rule also guarantees "no XML tags" for the name.)
            name_m = re.search(r"^name:\s*(.+?)\s*$", fm, re.MULTILINE)
            name_val = name_m.group(1).strip().strip("\"'") if name_m else ""
            if not name_val:
                issues.append({
                    "severity": "error", "where": "frontmatter", "message": "missing name",
                })
            else:
                if not re.fullmatch(r"[a-z0-9-]{1,64}", name_val):
                    issues.append({
                        "severity": "error", "where": "frontmatter",
                        "message": f"name {name_val!r} must be lowercase letters/numbers/hyphens, <=64 chars",
                    })
                if "anthropic" in name_val.lower() or "claude" in name_val.lower():
                    issues.append({
                        "severity": "error", "where": "frontmatter",
                        "message": f"name {name_val!r} contains a reserved word (anthropic/claude)",
                    })
            # description: present, non-empty, <=1024 chars, no XML-like tags.
            desc_m = re.search(r"^description:\s*(.+?)\s*$", fm, re.MULTILINE)
            desc_val = desc_m.group(1).strip().strip("\"'") if desc_m else ""
            if not desc_val:
                issues.append({
                    "severity": "error", "where": "frontmatter", "message": "missing description",
                })
            else:
                if len(desc_val) > 1024:
                    issues.append({
                        "severity": "error", "where": "frontmatter",
                        "message": f"description is {len(desc_val)} chars; max is 1024",
                    })
                if re.search(r"<\/?[a-zA-Z]", desc_val):
                    issues.append({
                        "severity": "error", "where": "frontmatter",
                        "message": "description contains XML-like tags (not allowed in frontmatter)",
                    })
    # Security scan across all content. Pass sources so HTTP-API patterns the LLM
    # transcribed from official docs are recognized as legitimate, not flagged as exfil.
    combined = skill_md + "\n" + "\n".join(references.values())
    blocks, cautions = _run_scanner(combined, sources=sources)
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
    # Dead-pointer check: every references/templates/scripts file named in the body must
    # actually be bundled, else the skill ships with instructions pointing at files the
    # agent will never find (e.g. a `## Templates` list when --with-templates was off).
    manifest = {
        "references": set(references or {}),
        "templates": set(templates or {}),
        "scripts": set(scripts or {}),
    }
    missing = set()
    for sub, fname in re.findall(
        r"\b(references|templates|scripts)/([A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z0-9]+)", skill_md
    ):
        if fname not in manifest[sub]:
            missing.add(f"{sub}/{fname}")
    for path in sorted(missing):
        issues.append({
            "severity": "error",
            "where": "dead-pointer",
            "message": f"SKILL.md references `{path}` but no such file is bundled",
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
    scripts: dict[str, str] | None,
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
    if scripts:
        scripts_dir = target / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        for name, src in scripts.items():
            script_path = scripts_dir / name
            script_path.write_text(src)
            # Mark as executable so the agent can run them directly per Anthropic's "execute, don't read" pattern
            script_path.chmod(0o755)
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
    with_scripts = getattr(args, "with_scripts", False)
    with_community = getattr(args, "with_community", False)
    with_version_notes = getattr(args, "with_version_notes", False)
    sources = _gather_sources(
        url,
        args.with_pitfalls,
        with_version_notes,
        with_troubleshooting,
        with_community,
    )
    include_evals = not args.no_evals
    run_eval_loop = include_evals and not getattr(args, "no_eval_triggering", False)

    # Intent brief (Phase 3.0-2): what the skill is FOR + target environment + success
    # criteria. Explicit `--intent` (or `--intent @file`) wins; otherwise infer from the
    # docs and record it as an assumption. Gates advanced features to the environment so
    # the skill defaults lean (e.g. "48GB GPU" ⇒ plain fp16 LoRA, QLoRA gated).
    raw_intent = (getattr(args, "intent", None) or "").strip()
    intent_brief, intent_source = "", "none"
    if raw_intent.startswith("@"):
        try:
            intent_brief = Path(raw_intent[1:]).read_text().strip()
            intent_source = "file"
        except OSError:
            _die(f"--intent @file not readable: {raw_intent[1:]}")
    elif raw_intent:
        intent_brief, intent_source = raw_intent, "explicit"
    elif not getattr(args, "no_intent_inference", False):
        intent_brief = infer_intent(sources["doc"])
        intent_source = "inferred" if intent_brief else "none"
    args.intent_brief = intent_brief  # consumed by build_contract below

    plan = plan_structure(
        doc_text=sources["doc"],
        with_pitfalls=args.with_pitfalls,
        with_evals=include_evals,
        with_templates=with_templates,
        with_scripts=with_scripts,
        with_version_notes=with_version_notes,
        intent=intent_brief,
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
        mcp_workflow_triggers=plan.get("mcp_workflow_triggers", []),
        decision_tree=plan.get("decision_tree", []),
        scripts=plan.get("scripts", []),
        old_patterns=plan.get("old_patterns", []),
        intent=intent_brief,
    )

    # ── Quality critic + bounded repair loop (Phase C/D) ──
    # generate → critique (P1–P4) → if any "block" finding, repair → re-critique,
    # bounded to MAX_REPAIR_ROUNDS. The body is settled here, BEFORE the description
    # is extracted (below) and before refs synthesis, so all downstream stages see
    # the repaired body. Residual findings ship as warnings (folded into validation
    # below) — the critic never hard-rejects; the P0 gates remain the only hard gate.
    critic_report = None
    if not getattr(args, "no_critic", False):
        module_count = len(plan.get("references", []))
        critic_findings = critique_skill(body, skill_name, module_count=module_count)
        initial_findings = list(critic_findings)
        rounds = 0
        while rounds < MAX_REPAIR_ROUNDS and any(f["severity"] == "block" for f in critic_findings):
            blocking = [f for f in critic_findings if f["severity"] == "block"]
            body = repair_skill_body(body, blocking, skill_name)
            rounds += 1
            critic_findings = critique_skill(body, skill_name, module_count=module_count)
        residual_blocks = [f for f in critic_findings if f["severity"] == "block"]
        critic_report = {
            "rounds": rounds,
            "repaired": rounds > 0,
            "initial_findings": initial_findings,
            "final_findings": critic_findings,
            "quality_gate": "failed" if residual_blocks else "passed",
        }

    refs_content: dict[str, str] = {}
    templates_content: dict[str, str] = {}
    scripts_content: dict[str, str] = {}

    # Skill Contract (Phase 3.0-1): the body's gates travel into every reference so a
    # precondition stays with its action (fixes the QLoRA gate-split). Built once here.
    contract = build_contract(plan, intent=getattr(args, "intent_brief", "") or "")

    # References + templates + scripts synthesized in parallel. All use the LLM; batch them together.
    with ThreadPoolExecutor(max_workers=6) as pool:
        ref_futures = {
            pool.submit(
                write_reference,
                r["filename"], r["covers"], sources["doc"],
                sources["readme"], sources["examples"],
                contract,
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
        script_futures = {}
        if with_scripts and plan.get("scripts"):
            script_futures = {
                pool.submit(
                    write_utility_script,
                    s["filename"], s["lang"], s["purpose"], s.get("executes_what", ""),
                    sources["readme"], sources["doc"],
                ): s["filename"] for s in plan["scripts"]
            }
        for fut in as_completed({**ref_futures, **tpl_futures, **script_futures}):
            if fut in ref_futures:
                refs_content[ref_futures[fut]] = fut.result()
            elif fut in tpl_futures:
                templates_content[tpl_futures[fut]] = fut.result()
            else:
                scripts_content[script_futures[fut]] = fut.result()

    # Auto-inject `## Contents` ToC into any reference >= 100 lines that doesn't already have one.
    # Per Anthropic best-practices: long references previewed with `head -N` miss content
    # past the cutoff. ToC lets Claude see the full scope even on a partial read.
    refs_content = {
        name: _inject_toc_if_long(content) for name, content in refs_content.items()
    }

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

    # Reference-scan critic (Phase 3.0-1): catch a gated action reproduced ungated in a
    # reference — the body critic above only sees the body. Folded into the same critic
    # report; it flips quality_gate to "failed" (the hard ship-gate is Phase 3.0-6).
    if not getattr(args, "no_critic", False):
        ref_findings = critique_references(refs_content, contract)
        if ref_findings:
            if critic_report is None:
                critic_report = {"rounds": 0, "repaired": False,
                                 "initial_findings": [], "final_findings": [],
                                 "quality_gate": "passed"}
            critic_report["final_findings"] = list(critic_report.get("final_findings") or []) + ref_findings
            if any(f["severity"] == "block" for f in ref_findings):
                critic_report["quality_gate"] = "failed"

    evals_doc = write_evals(skill_name, body) if include_evals else None

    description = _extract_description(body)
    # Real co-resident siblings (--siblings) replace the canned decoys in the
    # triggering eval — a harder, realistic precision test. negative_prompts (when
    # the eval doc carries them) make it bidirectional (catch over-triggering too).
    siblings = (
        _load_sibling_descriptions(args.siblings, exclude_name=skill_name)
        if getattr(args, "siblings", None) else None
    ) or None
    negative_prompts = evals_doc.get("negative_prompts") if evals_doc else None
    triggering_report = None
    if run_eval_loop and evals_doc and evals_doc.get("prompts"):
        initial = evaluate_triggering(
            skill_name, description, evals_doc["prompts"],
            siblings=siblings, negative_prompts=negative_prompts,
        )
        triggering_report = {"initial": initial}
        if initial.get("win_rate") is not None and initial["win_rate"] < 1.0 and initial["failing"]:
            improved = improve_description(
                skill_name=skill_name,
                skill_body=body,
                current_description=description,
                failing=initial["failing"],
            )
            if improved and improved != description:
                revised = evaluate_triggering(
                    skill_name, improved, evals_doc["prompts"],
                    siblings=siblings, negative_prompts=negative_prompts,
                )
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
        templates=templates_content,
        scripts=scripts_content,
    )
    # Templates validated separately via py_compile
    if templates_content:
        validation_issues.extend(validate_templates(templates_content))
    # Scripts validated: py_compile for .py, bash -n for .sh
    if scripts_content:
        validation_issues.extend(validate_scripts(scripts_content))
    # Fold residual critic findings into validation as WARNINGS (ship-with-warning —
    # the critic does not hard-reject). A "block" that survived the repair loop is
    # tagged UNRESOLVED; the quality_gate field (below) carries the pass/fail signal.
    if critic_report:
        for f in critic_report["final_findings"]:
            note = " [UNRESOLVED after repair]" if f["severity"] == "block" else ""
            validation_issues.append({
                "severity": "warning",
                "where": f"critic:{f['where']}",
                "message": f["message"] + note,
            })

    return {
        "skill_name": skill_name,
        "plan": plan,
        "skill_md": skill_md,
        "references": refs_content,
        "templates": templates_content,
        "scripts": scripts_content,
        "evals": evals_doc,
        "triggering_report": triggering_report,
        "critic": critic_report,
        "quality_gate": (critic_report or {}).get("quality_gate", "skipped"),
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
        with_scripts=getattr(args, "with_scripts", False),
        with_version_notes=getattr(args, "with_version_notes", False),
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
        "scripts": result.get("scripts", {}),
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
        scripts=result.get("scripts"),
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
            *(f"scripts/{k}" for k in (result.get("scripts") or {})),
            *(["evals/evals.json"] if result["evals"] else []),
        ],
        "validation_warnings": [i for i in result["validation"] if i["severity"] == "warning"],
        "openclaw_skills_check": {"ok": ok, "output": check_out[:1000]},
        "triggering_report": result.get("triggering_report"),
        "quality_gate": result.get("quality_gate", "skipped"),
        "critic_rounds": (result.get("critic") or {}).get("rounds", 0),
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
        sp.add_argument("--with-scripts", action="store_true",
                        help="Generate 1-3 SHORT utility scripts in scripts/ (executed, not read) — "
                             "bash/python health checks, validators, probes. Validated via py_compile / bash -n.")
        sp.add_argument("--with-community", action="store_true",
                        help="Curated Stack Exchange Q&As + closed `question` issues distilled into "
                             "references/community-gotchas.md (CC BY-SA attributed; IPI-scanned)")
        sp.add_argument("--no-evals", action="store_true")
        sp.add_argument("--no-eval-triggering", action="store_true",
                        help="Skip the triggering judge + description optimizer loop")
        sp.add_argument("--no-critic", action="store_true",
                        help="Skip the quality critic + bounded repair loop (P1–P4 reliability checklist)")
        sp.add_argument("--siblings", default=None,
                        help="Directory of co-resident skills (each <dir>/<name>/SKILL.md) to use as "
                             "REAL competitors in the triggering eval instead of the canned decoys")
        sp.add_argument("--intent", default=None,
                        help="Intent brief: what the skill is FOR + target environment + success "
                             "criteria; gates advanced features to the environment so the skill "
                             "defaults lean. Pass a string, or '@path' to read from a file. If "
                             "omitted, it is inferred from the docs and recorded as an assumption.")
        sp.add_argument("--no-intent-inference", action="store_true",
                        help="With no --intent, skip doc-based intent inference (build with no intent brief)")
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
    plan.add_argument("--with-scripts", action="store_true")
    plan.add_argument("--with-version-notes", action="store_true")
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
