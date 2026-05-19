# HLD: AI-Skill Scout — OpenClaw Agent

**Status:** Phase 1 Complete (2026-04-17)

## Goal

An OpenClaw agent that searches GitHub for high-quality OpenClaw skills (SKILL.md files) relevant to developing AI models, evaluates them, and installs them safely. Invoked on demand by the user or other tools.

## How It Works

1. User asks: *"I need a fine-tuning skill"* (via CLI or OpenClaw dashboard)
2. Agent invokes `skill_scout.py search "fine-tuning"` via its `exec` tool
3. Script checks if a matching skill is already installed
4. Script expands query via LLM to synonyms (`fine-tuning` → `lora, qlora, sft, peft, instruction-tuning`)
5. Script searches GitHub (`gh search code` + `gh search repos`) for each variant, merges deduplicated results
6. Script fetches each candidate's full SKILL.md and scores completeness (0-10)
7. Script ranks by trust → classification → completeness → stars, returns top 5 as JSON
8. Agent presents candidates conversationally (markdown table with trust signals, classifications, recommendations)
9. User approves one; agent invokes `skill_scout.py install <repo> <path>`
10. Script quarantines to `/tmp`, runs 60-pattern security scan, installs to workspace
11. Script runs `openclaw skills check` to verify, writes to lockfile + audit log
12. Agent reports result; gaps tracked for future re-scans

## Pipeline

```text
SEARCH → EXPAND (LLM) → EVALUATE → PRESENT → QUARANTINE → SCAN → INSTALL → VERIFY → LOG
```

| Stage | What happens |
|-------|-------------|
| Search | `gh search code --filename SKILL.md` + `gh search repos`, cached (1hr TTL), stale-cache fallback on failure |
| Expand | Claude Opus 4.6 via OpenRouter returns 3-6 related terms; each is searched and results merged |
| Evaluate | Fetch full SKILL.md content via `gh api`, score completeness (0-10), compute trust (HIGH/MEDIUM/LOW), dedup vs installed, classify ADOPT/EXTEND/GAP |
| Present | Top 5 as JSON with fragments; agent formats as markdown for the user |
| Quarantine | Recursive download to `/tmp/skill-quarantine/<name>/` — never directly to workspace |
| Scan | Regex scan: 52 BLOCK patterns + 8 CAUTION patterns across 7 threat categories |
| Install | Copy to `~/.openclaw/workspace/skills/<name>/` (preserves subdirs like `references/`, `scripts/`) |
| Verify | Run `openclaw skills check`; record success/failure in lockfile |
| Log | Append to `skill-audit.log`; update `installed-skills.json`; remove resolved gaps |

## Search

Source is **GitHub only**. Two methods, merged:
- `gh search code --filename SKILL.md "<query>"` — content-matched SKILL.md files
- `gh search repos "SKILL.md <query>"` — repos mentioning SKILL.md

LLM query expansion runs inside the script (not delegated to the agent), so expansion works whether the script is called via the agent or directly from the CLI.

## Quality Evaluation

- **Trust level:** Known orgs (huggingface, anthropics, oracle, openai, ...) = **HIGH**; 500+ stars = **MEDIUM**; everything else = **LOW**
- **Completeness (0-10):** Scores frontmatter, triggers, install steps, code blocks, OpenClaw metadata, requires/bins, usage section, NOT-for guidance, supporting files (references/scripts subdirs)
- **Classification:** **ADOPT** = completeness≥5 AND trust≥MEDIUM; **EXTEND** = completeness≥2; else **GAP**
- **Deduplication:** Skills already in `~/.openclaw/workspace/skills/` or the lockfile are filtered out of candidates

## Ranking (PLAN §3.2 weights)

Sort key: `(trust, classification, -completeness, -stars)` — trust is the highest weight, so a HIGH-trust EXTEND outranks a LOW-trust ADOPT.

## Security

**60 regex patterns** in 7 categories:
- Exfiltration (10): `curl|`, `wget --post`, `scp/rsync/sftp @host`, `socket.connect`, `requests.post`
- Injection (8): `eval(`, `exec(`, `os.system`, `subprocess shell=True`, `__import__`, `compile+exec`, `os.popen`
- Destructive (8): `rm -rf /`, `DROP TABLE`, `mkfs`, `dd of=/dev/`, `chmod 777 /`, `chown -R /`, `truncate -s 0`
- Obfuscation (8): `base64 -d | sh`, hex/unicode escapes, `chr() + chr()`, `codecs.decode rot`, `bytes.fromhex`, `zlib.decompress + base64`
- Credential access (10): `cat ~/.ssh`, `AWS_SECRET`, `.env`, `/etc/shadow`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `.netrc`, `.npmrc`, `cat *token*`
- Privilege escalation (4): `sudo`, `setuid()`, `chmod u+s`
- Reverse shell (4): `bash -i >& /dev/tcp`, `python socket dup2`, `php fsockopen`, `mkfifo | nc`
- Caution (8): external URLs, `pip install`, `npm install`, `curl`, `wget`, `git clone`, `docker run`, `chmod +x`

## Trust-Level Install Policy

| Trust | Safe | Caution patterns | Dangerous (BLOCK) patterns |
|-------|------|------------------|---------------------------|
| HIGH (known orgs) | Install | Install with notice | Block always |
| MEDIUM (500+ stars) | Install | Install with flagged notice for user confirmation | Block always |
| LOW (community) | Install | **Block** | Block always |

Dangerous patterns always block regardless of trust. Caution patterns are enforced for LOW trust and reported (agent judgment) for HIGH/MEDIUM.

## Decision Framework

Candidates are classified and surfaced to the agent:
- **ADOPT**: Exact match, trusted, passes scan → recommended for install
- **EXTEND**: Partial match, some signal → install with note to user
- **GAP**: Nothing suitable → log to `gaps.json`, tell the user

## Constraints

- **Caching:** `search-cache.json` with 1hr TTL; on GitHub timeout/failure, falls back to stale cache
- **Graceful degradation:** LLM expansion failure → searches the literal query only
- **Input validation:** query length 2–256 chars; repo must be `owner/repo` format; path traversal (`..`) rejected
- **Never auto-install:** agent always asks for user approval before running `install`
- **Install tracking:** `installed-skills.json` lockfile + `skill-audit.log` append-only log
- **Gap tracking:** `gaps.json` stores queries with no good results; resolved entries are cleared on successful install

## Invocation

Three ways to use it, all hitting the same pipeline:

```bash
# 1. Agent via CLI (conversational, Claude Opus 4.6 via OpenRouter)
openclaw agent --agent ai-skill-scout-opus -m "I need a fine-tuning skill"

# 2. Agent via dashboard
openclaw dashboard    # select ai-skill-scout-opus, chat in browser

# 3. Direct script (bypasses the LLM wrapper, still expands queries)
python3 agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py search "fine-tuning"
```

---

*Detailed design: [plan.md](plan.md)*
