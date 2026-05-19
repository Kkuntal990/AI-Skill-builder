#!/usr/bin/env python3
"""AI Skill Scout -- search, evaluate, and install OpenClaw skills from GitHub.

Usage:
    python3 skill_scout.py search "fine-tuning"
    python3 skill_scout.py install oracle/accelerated-data-science skills/aqua-cli
    python3 skill_scout.py installed
    python3 skill_scout.py gaps
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────

SKILLS_DIR = Path.home() / ".openclaw" / "workspace" / "skills"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
QUARANTINE_DIR = Path(tempfile.gettempdir()) / "skill-quarantine"
CACHE_TTL = 3600  # 1 hour

KNOWN_ORGS = frozenset({
    "huggingface", "anthropics", "oracle", "google", "microsoft",
    "meta-llama", "nvidia", "aws", "deepmind", "openai",
    "NousResearch", "mistralai", "databricks",
})

STAR_THRESHOLD = 500

# ── Security Scanner Patterns ──────────────────────────────────────────

BLOCK_PATTERNS = [
    # Exfiltration (10)
    (r"curl\s+.*\|", "exfiltration: curl pipe"),
    (r"wget\s+.*--post", "exfiltration: wget POST"),
    (r"curl\s+.*-X\s*POST", "exfiltration: curl POST"),
    (r"curl\s+.*-d\s", "exfiltration: curl data send"),
    (r"nc\s+-", "exfiltration: netcat"),
    (r"\bscp\b.*@", "exfiltration: scp to remote"),
    (r"\brsync\b.*@", "exfiltration: rsync to remote"),
    (r"\bsftp\b.*@", "exfiltration: sftp to remote"),
    (r"socket\.connect\s*\(", "exfiltration: raw socket connect"),
    (r"requests\.post\s*\(", "exfiltration: HTTP POST via requests"),
    # Injection (8)
    (r"\beval\s*\(", "injection: eval()"),
    (r"\bexec\s*\(", "injection: exec()"),
    (r"os\.system\s*\(", "injection: os.system()"),
    (r"subprocess\.call\s*\(.*shell\s*=\s*True", "injection: subprocess shell=True"),
    (r"subprocess\.Popen\s*\(.*shell\s*=\s*True", "injection: subprocess.Popen shell=True"),
    (r"__import__\s*\(", "injection: dynamic import"),
    (r"compile\s*\(.*exec", "injection: compile+exec"),
    (r"os\.popen\s*\(", "injection: os.popen()"),
    # Destructive (8)
    (r"rm\s+-rf\s+[/~]", "destructive: rm -rf on root/home"),
    (r"rm\s+-rf\s+\*", "destructive: rm -rf wildcard"),
    (r"\bDROP\s+TABLE\b", "destructive: DROP TABLE"),
    (r"\bmkfs\b", "destructive: mkfs"),
    (r"\bdd\s+if=.*of=/dev/", "destructive: dd to device"),
    (r"chmod\s+-R\s+777\s+/", "destructive: chmod 777 on root"),
    (r"chown\s+-R\s+.*\s+/[^t]", "destructive: chown on system path"),
    (r"truncate\s+.*-s\s*0", "destructive: truncate to zero"),
    # Obfuscation (8)
    (r"base64\s+(-d|--decode)\s*.*\|\s*(bash|sh|python)", "obfuscation: base64 to shell"),
    (r"echo\s+.*\|\s*base64\s+-d", "obfuscation: echo base64 decode"),
    (r"\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}", "obfuscation: hex encoding"),
    (r"chr\s*\(\s*\d+\s*\)\s*\+\s*chr", "obfuscation: chr() concatenation"),
    (r"codecs\.decode\s*\(.*rot", "obfuscation: rot13/codec decode"),
    (r"\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}", "obfuscation: unicode escape"),
    (r"bytes\.fromhex\s*\(", "obfuscation: bytes.fromhex()"),
    (r"zlib\.decompress\s*\(.*base64", "obfuscation: zlib+base64"),
    # Credential access (10)
    (r"cat\s+~/\.ssh", "credential: SSH key access"),
    (r"\bAWS_SECRET", "credential: AWS secret reference"),
    (r"cat\s+.*\.env\b", "credential: .env file read"),
    (r"/etc/shadow", "credential: shadow file access"),
    (r"cat\s+.*credentials", "credential: credentials file read"),
    (r"ANTHROPIC_API_KEY", "credential: Anthropic key reference"),
    (r"OPENAI_API_KEY", "credential: OpenAI key reference"),
    (r"cat\s+.*\.netrc", "credential: .netrc file read"),
    (r"cat\s+.*\.npmrc", "credential: .npmrc file read"),
    (r"cat\s+.*token", "credential: token file read"),
    # Privilege escalation (4)
    (r"\bsudo\s+", "privilege: sudo usage"),
    (r"setuid\s*\(", "privilege: setuid call"),
    (r"os\.setuid\s*\(", "privilege: os.setuid()"),
    (r"chmod\s+[u+]*s\s+", "privilege: setuid bit"),
    # Reverse shell (4)
    (r"bash\s+-i\s+>&\s*/dev/tcp", "revshell: bash /dev/tcp"),
    (r"python.*socket.*connect.*dup2", "revshell: python socket dup2"),
    (r"php\s+-r\s+.*fsockopen", "revshell: php fsockopen"),
    (r"mkfifo\s+.*nc\s+", "revshell: mkfifo+netcat"),
]

CAUTION_PATTERNS = [
    (r"https?://[^\s\"']+", "external URL reference"),
    (r"pip\s+install\s+(?!-r)", "pip install"),
    (r"npm\s+install", "npm install"),
    (r"curl\s+", "curl usage"),
    (r"wget\s+", "wget usage"),
    (r"git\s+clone\s+", "git clone from external"),
    (r"docker\s+run\s+", "docker container execution"),
    (r"chmod\s+\+x\s+", "making file executable"),
]

# ── Data Helpers ───────────────────────────────────────────────────────


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(path)


def _append_log(msg: str) -> None:
    log_path = DATA_DIR / "skill-audit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


# ── Step 1: CHECK ──────────────────────────────────────────────────────


def check_installed(query: str) -> list[str]:
    """Return names of installed skills matching the query."""
    matches = []
    q = query.lower()
    if SKILLS_DIR.exists():
        for skill_dir in SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists() and q in skill_md.read_text().lower():
                    matches.append(skill_dir.name)
    lockfile = _load_json(DATA_DIR / "installed-skills.json")
    for name in lockfile:
        if q in name.lower() and name not in matches:
            matches.append(name)
    return matches


# ── Query Expansion (LLM) ──────────────────────────────────────────────


def _load_openrouter_key() -> str:
    """Load OpenRouter API key from env or OpenClaw auth store."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    # Fallback: read from OpenClaw's auth-profiles.json
    for agent_dir in ("main", "ai-skill-scout"):
        p = Path.home() / ".openclaw" / "agents" / agent_dir / "agent" / "auth-profiles.json"
        if p.exists():
            try:
                auth = json.loads(p.read_text())
                prof = auth.get("profiles", {}).get("openrouter:default", {})
                k = prof.get("key") or prof.get("apiKey") or ""
                if k:
                    return k
            except (json.JSONDecodeError, OSError):
                continue
    return ""


def expand_query(query: str, model: str = "anthropic/claude-opus-4.6") -> list[str]:
    """Expand a user query into 3-6 search variants using an LLM via OpenRouter.

    Returns a list starting with the original query, followed by synonyms and
    related terms. On failure (no key, network error), returns [query] only so
    the caller still gets a result.
    """
    key = _load_openrouter_key()
    if not key:
        return [query]

    prompt = (
        "You expand user queries into search terms for finding OpenClaw/Claude skills "
        "on GitHub. Given a single query, return ONLY a JSON array of 3-6 related "
        "terms (including the original). Each term should be a single word or short "
        "phrase that would appear in an AI/ML skill name or description.\n\n"
        f'Query: "{query}"\n\n'
        'Example input: "fine-tuning"\n'
        'Example output: ["fine-tuning", "lora", "qlora", "sft", "peft", "instruction-tuning"]\n\n'
        "Return only the JSON array, no other text."
    )

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Kkuntal990/AI-Skill-builder",
            "X-Title": "AI Skill Scout",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return [query]

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Extract JSON array from response (LLM may wrap in markdown fences)
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if not match:
        return [query]
    try:
        variants = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [query]
    if not isinstance(variants, list):
        return [query]
    # Normalize: strings only, dedup case-insensitive, ensure original is first
    seen = set()
    result = []
    for v in [query] + [str(x) for x in variants if isinstance(x, str)]:
        low = v.lower().strip()
        if low and low not in seen:
            seen.add(low)
            result.append(v.strip())
    return result[:6]


# ── Step 2: SEARCH ─────────────────────────────────────────────────────


def _gh_search_code(query: str, limit: int = 20) -> list[dict]:
    """Search GitHub for SKILL.md files matching query via gh search code."""
    try:
        result = subprocess.run(
            ["gh", "search", "code", "--filename", "SKILL.md", query,
             "--limit", str(limit),
             "--json", "repository,path,sha,url,textMatches"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        return []
    return json.loads(result.stdout) if result.stdout.strip() else []


def _gh_search_repos(query: str, limit: int = 10) -> list[dict]:
    """Search GitHub repos containing SKILL.md via gh search repos."""
    try:
        result = subprocess.run(
            ["gh", "search", "repos", f"SKILL.md {query}",
             "--limit", str(limit),
             "--json", "fullName,stargazersCount,description,updatedAt"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        return []
    return json.loads(result.stdout) if result.stdout.strip() else []


def search_github(query: str) -> list[dict]:
    """Search with 1hr cache. Uses both gh search code and gh search repos.
    On timeout/failure, falls back to stale cache with notice."""
    cache_path = DATA_DIR / "search-cache.json"
    cache = _load_json(cache_path)

    cache_key = query.lower().strip()
    if cache_key in cache:
        entry = cache[cache_key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["results"]

    # Primary: gh search code
    results = _gh_search_code(query)

    # Secondary: gh search repos (adds repos that may have SKILL.md but weren't
    # matched by content search -- we synthesize a code-search-like entry)
    repo_results = _gh_search_repos(query)
    code_repos = {r["repository"]["nameWithOwner"] for r in results}
    for repo in repo_results:
        if repo["fullName"] not in code_repos:
            results.append({
                "repository": {"nameWithOwner": repo["fullName"]},
                "path": "SKILL.md",
                "sha": "",
                "url": f"https://github.com/{repo['fullName']}",
                "textMatches": [{"fragment": repo.get("description", "") or ""}],
            })

    # Graceful degradation: fall back to stale cache if search failed
    if not results and cache_key in cache:
        stale = cache[cache_key]
        stale["stale"] = True
        return stale["results"]

    # Deduplicate by repo+path
    seen = set()
    unique = []
    for r in results:
        key = f"{r['repository']['nameWithOwner']}:{r['path']}"
        if key not in seen:
            seen.add(key)
            unique.append(r)

    cache[cache_key] = {"timestamp": time.time(), "results": unique}
    _save_json(cache_path, cache)
    return unique


# ── Step 3: EVALUATE ───────────────────────────────────────────────────


def _get_repo_info(repo_full_name: str) -> dict:
    """Fetch repo metadata via gh api."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo_full_name}",
             "--jq", "{stars: .stargazers_count, owner: .owner.login, pushed: .pushed_at, description: .description}"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return {"stars": 0, "owner": "", "pushed": "", "description": ""}
    if result.returncode != 0:
        return {"stars": 0, "owner": "", "pushed": "", "description": ""}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stars": 0, "owner": "", "pushed": "", "description": ""}


def _trust_level(repo_info: dict, repo_full_name: str) -> str:
    owner = repo_full_name.split("/")[0]
    if owner in KNOWN_ORGS:
        return "HIGH"
    if repo_info.get("stars", 0) >= STAR_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _fetch_skill_content(repo: str, path: str) -> str:
    """Fetch raw SKILL.md content from GitHub. Returns empty string on failure."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/contents/{path}",
             "--jq", ".content"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return ""
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    # GitHub returns base64-encoded content
    try:
        return base64.b64decode(result.stdout.strip()).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _completeness_score(content: str) -> int:
    """Score 0-10 based on full SKILL.md content signals.

    Checks for the signals from PLAN.md: frontmatter, triggers,
    install steps, CLI commands, and proper skill format."""
    lower = content.lower()
    score = 0
    # Frontmatter (YAML header)
    if "---" in content and "name:" in lower:
        score += 1
    if "description:" in lower:
        score += 1
    # Trigger conditions
    if "trigger" in lower or "use when" in lower or "use this" in lower:
        score += 1
    # Install steps
    if "install" in lower:
        score += 1
    # CLI commands (code blocks)
    if "```" in content:
        score += 1
    # OpenClaw metadata
    if "metadata:" in lower and "openclaw" in lower:
        score += 1
    # Has requires/dependencies section
    if "requires" in lower or "bins:" in lower:
        score += 1
    # Has usage/examples section
    if "## usage" in lower or "## example" in lower:
        score += 1
    # Error handling / NOT-for guidance
    if "not for" in lower or "do not" in lower or "avoid" in lower:
        score += 1
    # Has references or supporting files
    if "references/" in lower or "scripts/" in lower:
        score += 1
    return score


def _classify(trust: str, completeness: int) -> str:
    """Classify candidate as ADOPT, EXTEND, or GAP per PLAN.md decision framework.
    ADOPT: high completeness + trusted. EXTEND: partial match. GAP: poor fit.
    Completeness scale is 0-10."""
    if completeness >= 5 and trust in ("HIGH", "MEDIUM"):
        return "ADOPT"
    if completeness >= 2:
        return "EXTEND"
    return "GAP"


def evaluate_results(results: list[dict], installed: list[str]) -> list[dict]:
    """Enrich with trust + completeness + classification, filter installed, rank, return top 5.
    Fetches full SKILL.md content from GitHub for accurate completeness scoring."""
    evaluated = []
    for r in results:
        repo_name = r["repository"]["nameWithOwner"]
        repo_info = _get_repo_info(repo_name)
        trust = _trust_level(repo_info, repo_name)
        skill_name = Path(r["path"]).parent.name

        # Dedup against already-installed skills
        if skill_name in installed:
            continue

        # Fetch full SKILL.md for accurate completeness scoring
        full_content = _fetch_skill_content(repo_name, r["path"])
        completeness = _completeness_score(full_content)

        classification = _classify(trust, completeness)

        evaluated.append({
            "repo": repo_name,
            "path": r["path"],
            "skill_name": skill_name,
            "url": r.get("url", ""),
            "trust": trust,
            "stars": repo_info.get("stars", 0),
            "completeness": completeness,
            "classification": classification,
            "pushed": repo_info.get("pushed", ""),
            "description": repo_info.get("description", ""),
        })

    # Ranking weights per PLAN.md section 3.2:
    #   Highest: repo trust (known org > 500+ stars > community)
    #   High:    skill completeness (frontmatter, triggers, CLI commands)
    #   Medium:  repo health (stars, recency)
    trust_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    class_order = {"ADOPT": 0, "EXTEND": 1, "GAP": 2}
    evaluated.sort(key=lambda x: (
        trust_order[x["trust"]],           # Highest weight: trust
        class_order[x["classification"]],  # Then classification
        -x["completeness"],                # Then completeness
        -x["stars"],                       # Then stars (repo health)
    ))
    return evaluated[:5]


# ── Step 5: SECURITY SCAN ─────────────────────────────────────────────


def security_scan(directory: Path) -> dict:
    """Scan all files for security threats."""
    blocks = []
    cautions = []

    for fpath in directory.rglob("*"):
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue

        rel = str(fpath.relative_to(directory))

        for pattern, label in BLOCK_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                blocks.append({"file": rel, "pattern": label})

        for pattern, label in CAUTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                cautions.append({"file": rel, "pattern": label})

    return {"clean": len(blocks) == 0, "blocks": blocks, "cautions": cautions}


# ── Download Helper ────────────────────────────────────────────────────


def _download_skill_dir(repo: str, path: str, dest: Path) -> bool:
    """Download a skill directory from GitHub to dest."""
    dest.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/contents/{path}",
             "--jq", '.[] | "\\(.type) \\(.name) \\(.download_url // "")"'],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False

    if result.returncode != 0:
        # Might be a single file, not a directory
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{repo}/contents/{path}",
                 "--jq", '"\\(.type) \\(.name) \\(.download_url // "")"'],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            return False
        if result.returncode != 0:
            return False

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) < 2:
            continue
        item_type, item_name = parts[0], parts[1]

        if item_type == "file":
            download_url = parts[2] if len(parts) > 2 else ""
            if download_url:
                try:
                    subprocess.run(
                        ["curl", "-sL", "-o", str(dest / item_name), download_url],
                        capture_output=True, timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    return False
        elif item_type == "dir":
            if not _download_skill_dir(repo, f"{path}/{item_name}", dest / item_name):
                return False

    return True


# ── Commands ───────────────────────────────────────────────────────────


def cmd_search(query: str, expand: bool = True) -> None:
    """Steps 1-4: check, expand (LLM), search, evaluate, present."""
    # Validate query
    stripped = query.strip()
    if not stripped or len(stripped) < 2:
        print(json.dumps({"error": "Query too short. Provide at least 2 characters."}))
        return
    if len(stripped) > 256:
        stripped = stripped[:256]

    # Step: LLM query expansion (uses OpenRouter via the user's API key)
    variants = expand_query(stripped) if expand else [stripped]
    _append_log(f'EXPAND "{stripped}" -> {variants}')

    installed = check_installed(stripped)

    # Search for each variant, merge results (deduplicated by repo+path)
    all_results: list[dict] = []
    seen_keys: set[str] = set()
    for v in variants:
        for r in search_github(v):
            key = f"{r['repository']['nameWithOwner']}:{r['path']}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_results.append(r)

    if not all_results:
        gaps = _load_json(DATA_DIR / "gaps.json")
        gaps[query] = {
            "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "expanded_to": variants,
        }
        _save_json(DATA_DIR / "gaps.json", gaps)
        _append_log(f'SEARCH "{query}" variants={len(variants)} results=0 gap=true')
        print(json.dumps({
            "query": stripped,
            "expanded_queries": variants,
            "already_installed": installed,
            "candidates": [],
            "gap": True,
        }))
        return

    candidates = evaluate_results(all_results, installed)
    _append_log(f'SEARCH "{query}" variants={len(variants)} results={len(all_results)} candidates={len(candidates)}')
    print(json.dumps({
        "query": stripped,
        "expanded_queries": variants,
        "already_installed": installed,
        "candidates": candidates,
        "gap": len(candidates) == 0,
    }, indent=2))


def cmd_install(repo: str, path: str) -> None:
    """Steps 5-7: quarantine, scan, install, log."""
    # Validate inputs
    if "/" not in repo or len(repo.split("/")) != 2:
        print(json.dumps({"status": "error", "message": f"Invalid repo format: '{repo}'. Expected 'owner/repo'."}))
        return
    if ".." in path:
        print(json.dumps({"status": "error", "message": "Path traversal not allowed."}))
        return

    skill_name = Path(path).name
    if not skill_name or skill_name in (".", ".."):
        print(json.dumps({"status": "error", "message": f"Invalid skill path: '{path}'."}))
        return

    quarantine_path = QUARANTINE_DIR / skill_name

    if quarantine_path.exists():
        shutil.rmtree(quarantine_path)

    # Download to quarantine
    print(json.dumps({"status": "downloading", "skill": skill_name, "repo": repo}),
          file=sys.stderr)
    if not _download_skill_dir(repo, path, quarantine_path):
        print(json.dumps({"status": "error", "message": "Failed to download skill"}))
        return

    if not (quarantine_path / "SKILL.md").exists():
        print(json.dumps({"status": "error", "message": "No SKILL.md found in downloaded content"}))
        shutil.rmtree(quarantine_path)
        return

    # Security scan
    scan = security_scan(quarantine_path)
    content = (quarantine_path / "SKILL.md").read_text()
    content_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    repo_info = _get_repo_info(repo)
    trust = _trust_level(repo_info, repo)

    # Dangerous patterns always block regardless of trust
    if not scan["clean"]:
        print(json.dumps({
            "status": "blocked",
            "skill": skill_name,
            "trust": trust,
            "scan": scan,
            "message": "Security scan found dangerous patterns. Install blocked.",
        }, indent=2))
        shutil.rmtree(quarantine_path)
        _append_log(f"BLOCKED {skill_name} source=github:{repo}/{path} trust={trust} blocks={len(scan['blocks'])}")
        return

    # Trust-level caution policy (PLAN.md section 3.3):
    #   HIGH:   allow caution with notice
    #   MEDIUM: report caution, ask user (agent handles interactive ask)
    #   LOW:    block if caution patterns found
    if scan["cautions"] and trust == "LOW":
        print(json.dumps({
            "status": "blocked",
            "skill": skill_name,
            "trust": trust,
            "scan": scan,
            "message": "LOW trust repo with caution patterns. Install blocked.",
        }, indent=2))
        shutil.rmtree(quarantine_path)
        _append_log(f"BLOCKED {skill_name} source=github:{repo}/{path} trust={trust} reason=low-trust-caution cautions={len(scan['cautions'])}")
        return

    # Install
    install_dest = SKILLS_DIR / skill_name
    install_dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(quarantine_path, install_dest, dirs_exist_ok=True)
    shutil.rmtree(quarantine_path)

    # Post-install verification via openclaw skills check
    verify_ok = True
    try:
        verify = subprocess.run(
            ["openclaw", "skills", "check"],
            capture_output=True, text=True, timeout=15,
        )
        verify_ok = verify.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        verify_ok = False

    # Update lockfile
    lockfile = _load_json(DATA_DIR / "installed-skills.json")
    lockfile[skill_name] = {
        "source": f"github:{repo}/{path}",
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trust_level": trust,
        "content_hash": content_hash,
        "scan_result": "clean",
        "cautions": len(scan["cautions"]),
    }
    _save_json(DATA_DIR / "installed-skills.json", lockfile)

    _append_log(f"INSTALL {skill_name} source=github:{repo}/{path} trust={trust} scan=clean verified={verify_ok} cautions={len(scan['cautions'])}")

    # Remove matching gaps
    gaps = _load_json(DATA_DIR / "gaps.json")
    to_remove = [k for k in gaps if k.lower() in skill_name.lower() or skill_name.lower() in k.lower()]
    for k in to_remove:
        del gaps[k]
    if to_remove:
        _save_json(DATA_DIR / "gaps.json", gaps)

    # For MEDIUM trust with cautions, flag for agent to confirm with user
    caution_notice = None
    if scan["cautions"] and trust == "MEDIUM":
        caution_notice = "MEDIUM trust repo with caution patterns -- confirm with user."
    elif scan["cautions"] and trust == "HIGH":
        caution_notice = "HIGH trust repo; caution patterns noted."

    print(json.dumps({
        "status": "installed",
        "skill": skill_name,
        "trust": trust,
        "scan": scan,
        "installed_to": str(install_dest),
        "content_hash": content_hash,
        "verified": verify_ok,
        "caution_notice": caution_notice,
    }, indent=2))


def cmd_installed() -> None:
    print(json.dumps(_load_json(DATA_DIR / "installed-skills.json"), indent=2))


def cmd_gaps() -> None:
    print(json.dumps(_load_json(DATA_DIR / "gaps.json"), indent=2))


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="AI Skill Scout -- search and install OpenClaw skills from GitHub."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Search GitHub for skills")
    p_search.add_argument("query", help="Search query (e.g., 'fine-tuning')")
    p_search.add_argument("--no-expand", action="store_true",
                          help="Skip LLM query expansion; search the literal query only")

    p_install = sub.add_parser("install", help="Install a skill from GitHub")
    p_install.add_argument("repo", help="GitHub repo (e.g., oracle/accelerated-data-science)")
    p_install.add_argument("path", help="Path to skill dir (e.g., skills/aqua-cli)")

    sub.add_parser("installed", help="List installed skills")
    sub.add_parser("gaps", help="Show unresolved skill gaps")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args.query, expand=not args.no_expand)
    elif args.command == "install":
        cmd_install(args.repo, args.path)
    elif args.command == "installed":
        cmd_installed()
    elif args.command == "gaps":
        cmd_gaps()


if __name__ == "__main__":
    main()
