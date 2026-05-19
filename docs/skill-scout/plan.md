# PLAN: AI-Skill Scout — OpenClaw Agent

> For the 1-page overview, see **[hld.md](hld.md)**.

**Status:** Phase 1 Complete
**Date:** 2026-04-17
**Scope:** An OpenClaw agent that searches GitHub for high-quality OpenClaw skills related to developing AI models, then installs them safely.

---

## 1. Goal

An **OpenClaw agent** (`ai-skill-scout` and `ai-skill-scout-opus` variants) that, on demand:

1. Searches GitHub for existing OpenClaw skills (SKILL.md files) relevant to AI model development
2. Evaluates quality using free signals (repo trust, skill completeness, security scan)
3. Installs the best match safely (quarantine → scan → user approval → install → verify)
4. Reports what was installed and what gaps remain

The search is **open-ended** — not a fixed list. An LLM (Claude Opus via OpenRouter, called from inside the script) expands queries with synonyms. The agent itself presents results conversationally.

### Current Landscape

78+ ML skills exist on GitHub but are scattered. The tools practitioners use most (Unsloth, Axolotl, TRL, lm-eval-harness, W&B, DVC) have zero OpenClaw skills in the default catalog. No agent currently finds and installs the right skill on demand.

---

## 2. Architecture — 9-Step Pipeline

```
User: "I need a fine-tuning skill"
  |
  v
[1. CHECK]   -- Already installed?
  |  Scan ~/.openclaw/workspace/skills/ for matching SKILL.md content
  |  Check installed-skills.json lockfile
  |
  v
[2. EXPAND]  -- LLM query expansion via OpenRouter
  |  Call Claude Opus 4.6 via OpenRouter /api/v1/chat/completions
  |  "fine-tuning" -> ["fine-tuning", "lora", "qlora", "sft", "peft", "instruction-tuning"]
  |  On failure -> fall back to [query] only
  |
  v
[3. SEARCH]  -- GitHub for each variant (merged, deduplicated)
  |  Check search-cache.json (1hr TTL)
  |  gh search code --filename SKILL.md "<variant>"
  |  gh search repos "SKILL.md <variant>"
  |  On timeout/429 -> fall back to stale cache
  |
  v
[4. EVALUATE] -- Rank by trust + completeness
  |  For each candidate, fetch full SKILL.md via gh api /contents
  |  Trust: known org (HIGH) > 500+ stars (MEDIUM) > community (LOW)
  |  Completeness (0-10): frontmatter, triggers, install, code blocks,
  |      openclaw metadata, requires, usage, avoid-guidance, subdirs
  |  Dedup vs installed skills
  |  Classify: ADOPT / EXTEND / GAP
  |  Sort: (trust, classification, -completeness, -stars)
  |
  v
[5. PRESENT]  -- Top 5 as JSON; agent renders markdown table
  |  User approves which to install (never auto-install)
  |
  v
[6. QUARANTINE] -- Recursive download to /tmp
  |  gh api /contents to enumerate files
  |  curl -sL to fetch each, preserving subdirs (references/, scripts/)
  |  Dest: /tmp/skill-quarantine/<name>/
  |
  v
[7. SCAN]    -- 60-pattern regex scan
  |  52 BLOCK + 8 CAUTION patterns across 7 threat categories
  |  Trust-level policy:
  |     HIGH/MEDIUM + caution -> install with notice
  |     LOW + caution -> BLOCK
  |     ANY + dangerous (BLOCK pattern) -> BLOCK
  |
  v
[8. INSTALL] -- Copy to workspace + verify
  |  shutil.copytree to ~/.openclaw/workspace/skills/<name>/
  |  Run openclaw skills check, record verified=true/false
  |  Write to installed-skills.json (lockfile)
  |
  v
[9. REPORT]  -- Confirm install + log + clean up gaps
  |  Append to skill-audit.log
  |  Remove resolved gaps from gaps.json
  |  Return JSON status to agent
  |  Agent tells user: "Installed trl-training (HIGH trust). 1 caution noted."
```

---

## 3. Detailed Design

### 3.1 Query Expansion (NEW in Phase 1)

LLM expansion happens **inside** `skill_scout.py` (not delegated to the agent), so it works uniformly whether invoked via `openclaw agent` or directly via CLI.

```python
def expand_query(query: str, model: str = "anthropic/claude-opus-4.6") -> list[str]:
    # Load OpenRouter key from OPENROUTER_API_KEY env or ~/.openclaw auth store
    # POST to https://openrouter.ai/api/v1/chat/completions
    # Ask for a JSON array of 3-6 related terms
    # Parse, deduplicate case-insensitively, ensure original is first
```

Behavior:
- Returns `[query]` if key is missing, network fails, or response malformed
- `--no-expand` CLI flag disables expansion for testing
- Cached results are keyed by the individual variant, so all 6 searches benefit from cache hits across sessions

### 3.2 Search

Two methods, both through `gh` CLI:
```bash
gh search code --filename SKILL.md "<variant>" --limit 20 --json repository,path,sha,url,textMatches
gh search repos "SKILL.md <variant>" --limit 10 --json fullName,stargazersCount,description,updatedAt
```

Repo-search results are synthesized as pseudo-code-search entries so they join the same pipeline. Results deduplicated by `repo:path`.

**Caching:** `data/search-cache.json` with 1hr TTL. On failure (both methods return empty), falls back to stale cache if one exists, tagged `stale: true`.

### 3.3 Evaluation

| Signal | Weight | Source | Notes |
|--------|--------|--------|-------|
| Repo trust | Highest | GitHub API (`gh api /repos/{owner}/{repo}`) | KNOWN_ORGS set + 500-star threshold |
| Skill completeness (0-10) | High | Full SKILL.md via `gh api /contents` (base64-decoded) | 10 signal checks |
| Stars | Medium | Same API call as trust | Tiebreaker |
| Deduplication | Pass/fail | Local scan of installed skills | Filters, not ranks |

**Completeness signals (1 point each):**
1. Has YAML frontmatter with `name:`
2. Has `description:`
3. Has trigger guidance (`trigger`, `use when`, `use this`)
4. Has install steps (`install` keyword)
5. Has code blocks (triple backticks)
6. Has OpenClaw metadata (`metadata:` + `openclaw`)
7. Declares dependencies (`requires`, `bins:`)
8. Has usage/examples section (`## usage`, `## example`)
9. Has NOT-for guidance (`not for`, `do not`, `avoid`)
10. Has supporting files referenced (`references/`, `scripts/`)

**Classification:**
- ADOPT: `completeness >= 5 AND trust in (HIGH, MEDIUM)`
- EXTEND: `completeness >= 2`
- GAP: otherwise

**Ranking:** Sort by `(trust_order, class_order, -completeness, -stars)`. Trust is the primary axis — a HIGH-trust EXTEND outranks a LOW-trust ADOPT.

### 3.4 Security (Quarantine + Scan)

After user approves a candidate:

1. **Quarantine:** Recursive download to `/tmp/skill-quarantine/<name>/`
   - `gh api /repos/{repo}/contents/{path}` to list files and get `download_url`s
   - `curl -sL` to fetch each file, preserving directory structure (`references/`, `scripts/`, etc.)
   - Requires `SKILL.md` at the root of the downloaded directory or install fails

2. **Scan:** 60 regex patterns (52 BLOCK + 8 CAUTION) across 7 categories. See HLD §Security for the full breakdown.

3. **Trust-level policy enforcement:**

| Trust | Caution patterns | Dangerous (BLOCK) patterns |
|-------|------------------|---------------------------|
| HIGH | Install, notice reported | Block always |
| MEDIUM | Install, caution_notice flagged for agent to confirm with user | Block always |
| LOW | **BLOCK** install | Block always |

### 3.5 Install + Tracking

```python
shutil.copytree(quarantine_path, ~/.openclaw/workspace/skills/<name>/, dirs_exist_ok=True)
subprocess.run(["openclaw", "skills", "check"], capture_output=True, timeout=15)
# verified = (returncode == 0)
```

**Lockfile** (`data/installed-skills.json`):
```json
{
  "trl-training": {
    "source": "github:huggingface/trl/trl/skills/trl-training",
    "installed_at": "2026-04-17T22:26:11Z",
    "trust_level": "HIGH",
    "content_hash": "sha256:cfdec413fcf030f2",
    "scan_result": "clean",
    "cautions": 1
  }
}
```

**Audit log** (`data/skill-audit.log`, append-only):

```text
2026-04-17T22:25:30Z EXPAND "fine-tuning" -> ['fine-tuning', 'lora', 'qlora', 'sft', 'peft', 'instruction-tuning']
2026-04-17T22:25:34Z SEARCH "fine-tuning" variants=6 results=27 candidates=5
2026-04-17T22:26:11Z INSTALL trl-training source=github:huggingface/trl/trl/skills/trl-training trust=HIGH scan=clean verified=True cautions=1
2026-04-17T22:27:00Z BLOCKED suspicious-skill source=github:bad/repo/skills/x trust=LOW reason=low-trust-caution cautions=3
```

**Gap tracking** (`data/gaps.json`): Queries with no results logged with timestamp + expanded variants. Cleared on successful install of a matching skill.

### 3.6 Input Validation

Hardening against random LLM input:
- Query length: 2–256 chars (truncated if longer, rejected if shorter)
- Install `repo` must be `owner/repo` format (exactly one `/`)
- Install `path` must not contain `..` (path traversal blocked)
- Skill name derived from `Path(path).name` must not be `.` or `..`

---

## 4. What the Agent Searches For

Open-ended, not a fixed list. Core areas (as seeded in AGENTS.md):

| Area | Example Tools | Existing Skills? |
|------|--------------|-----------------|
| Fine-tuning (LoRA, QLoRA, SFT) | Unsloth, Axolotl, TRL, LLaMA-Factory, torchtune | Found: `trl-training` (HF), `trl-fine-tuning` (Hermes) |
| Alignment (DPO, GRPO, RLHF) | TRL, OpenRLHF | Partial via HF trainer |
| Evaluation | lm-eval-harness, DeepEval, Promptfoo | `promptfoo` exists |
| Serving & inference | Ollama, vLLM, SGLang, llama.cpp | `ollama-local`, `vllm` exist |
| Quantization (GGUF, AWQ) | llama.cpp, AutoAWQ | GAP |
| Data preparation | distilabel, Argilla, Label Studio | GAP |
| Experiment tracking | W&B, MLflow | `mlflow` basic exists; W&B = GAP |
| Data versioning | DVC | GAP |
| Hyperparameter tuning | Optuna | GAP |
| MLOps | dstack, BentoML | `dstack` exists |

---

## 5. Implementation Status

### Phase 1: Discover & Install — **COMPLETE** ✓

- [x] Created two agent variants: `ai-skill-scout` (openai-codex/gpt-5.4) and `ai-skill-scout-opus` (openrouter/anthropic/claude-opus-4.6)
- [x] Configured `AGENTS.md`, `SOUL.md`, `IDENTITY.md`, `TOOLS.md` for agent identity and behavior
- [x] Registered tool via `SKILL.md` with 4 subcommands: `search`, `install`, `installed`, `gaps`
- [x] Implemented GitHub search (`gh search code` + `gh search repos`) with LLM query expansion (OpenRouter Claude Opus)
- [x] Implemented evaluation: full SKILL.md content fetch, 10-point completeness, trust levels, ADOPT/EXTEND/GAP classification
- [x] Implemented quarantine + 60-pattern security scan with trust-level caution policy
- [x] Implemented install with lockfile, audit log, and `openclaw skills check` verification
- [x] Implemented search cache (1hr TTL) with stale-cache fallback
- [x] Implemented gap tracking + cleanup on successful install
- [x] Input validation (query length, repo format, path traversal)
- [x] End-to-end verified: agent invokes script, expands query, presents markdown table, recommends top candidate, installs on approval

### Phase 1b: Continuous Discovery — Deferred

- [ ] Weekly cron via `openclaw cron add` that runs search against `gaps.json`
- [ ] Alert user to new high-quality skills found since last run

---

## 6. Stack

| Component | Technology | Location |
|-----------|-----------|----------|
| Runtime | OpenClaw 2026.4.14 | `/opt/homebrew/bin/openclaw` |
| LLM (query expansion) | Claude Opus 4.6 via OpenRouter | `https://openrouter.ai/api/v1/chat/completions` |
| LLM (agent reasoning) | `anthropic/claude-opus-4.6` or `openai-codex/gpt-5.4` | Per-agent `model` field in `~/.openclaw/openclaw.json` |
| Search | `gh search code` + `gh search repos` | authenticated `gh` CLI |
| Repo metadata | `gh api /repos/{owner}/{repo}` | Stars, pushed_at, description |
| Content fetch | `gh api /repos/{owner}/{repo}/contents/{path}` | Base64-decoded SKILL.md body |
| Install download | `gh api /contents` for listing + `curl -sL` for each file | Preserves subdirs |
| Verification | `openclaw skills check` | Confirms workspace load |
| Storage | JSON files in `agents/ai-skill-scout/data/` | Cache, lockfile, gaps, audit log |

No Python package dependencies — pure stdlib only (`argparse`, `urllib.request`, `subprocess`, `pathlib`, `json`, `re`, `hashlib`, `shutil`, `tempfile`, `base64`).

---

## 7. File Layout

```
agents/ai-skill-scout/
├── AGENTS.md                          # Operational playbook (gets injected into agent system prompt)
├── SOUL.md                            # Personality
├── IDENTITY.md                        # Name (Scout), emoji (🔭)
├── TOOLS.md                           # Local env notes
├── .gitignore                         # Ignores data/
├── skills/
│   └── find-ai-skill/
│       ├── SKILL.md                   # Tool definition + YAML frontmatter + metadata.openclaw
│       └── scripts/
│           └── skill_scout.py         # ~600 lines; search, evaluate, install, scan
└── data/                              # Runtime state (gitignored)
    ├── installed-skills.json          # Lockfile
    ├── search-cache.json              # 1hr TTL cache
    ├── gaps.json                      # Unresolved queries
    └── skill-audit.log                # Append-only
```

OpenClaw state (outside repo):
```
~/.openclaw/
├── openclaw.json                                # Global config, agent registry
├── agents/
│   ├── ai-skill-scout/agent/
│   │   ├── auth-profiles.json                  # Per-agent auth (openai-codex OAuth + openrouter api_key)
│   │   └── models.json                         # Per-agent model/provider config
│   └── ai-skill-scout-opus/agent/              # Same structure for opus variant
└── workspace/skills/<name>/                     # Installed skills land here
```

---

## 8. Invocation

```bash
# Via OpenClaw agent (conversational, Claude Opus)
openclaw agent --agent ai-skill-scout-opus -m "I need a fine-tuning skill"

# Via OpenClaw agent (ChatGPT Codex)
openclaw agent --agent ai-skill-scout -m "I need a fine-tuning skill"

# Via dashboard
openclaw dashboard   # select ai-skill-scout-opus

# Direct script
python3 agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py search "fine-tuning"
python3 agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py search "fine-tuning" --no-expand
python3 agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py install huggingface/trl trl/skills/trl-training
python3 agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py installed
python3 agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py gaps
```

---

## 9. Known Issues & Fixes Applied

| Issue | Root cause | Fix |
|-------|-----------|-----|
| OpenClaw `agent` returned HTML error page via OpenRouter, mislabeled as "DNS lookup failed" | Per-agent `models.json` had wrong baseUrl `https://openrouter.ai/v1` — missing `/api/` prefix. Wrong URL hits OpenRouter homepage which returns HTML | Patched `~/.openclaw/agents/*/agent/models.json` to `https://openrouter.ai/api/v1` |
| Expired ChatGPT Codex OAuth token kept re-syncing from `~/.codex/auth.json` | Codex CLI manages that profile; OpenClaw inherits the expired token | Refreshed via OAuth refresh_token flow to `https://auth.openai.com/oauth/token` |
| Original `auth-profiles.json` format rejected with `invalid_type` | Used `"type": "apiKey"`; OpenClaw expects `"type": "api_key"` and field `"key"` (not `apiKey`) | Fixed profile format |
| ChatGPT Pro quota exhaustion under repeated agent runs | Each agent turn is ~20K char system prompt; Codex subscription has strict per-request quota | Added OpenRouter variant (`ai-skill-scout-opus`) as a fallback model |

---

## 10. Open Questions

| # | Question | Resolution |
|---|----------|-----------|
| 1 | Where exactly should workspace skills go? | Confirmed: `~/.openclaw/workspace/skills/<name>/` |
| 2 | How to handle skills with supporting files (not just SKILL.md)? | Resolved: recursive download via `gh api /contents` preserves subdirs |
| 3 | Should gaps.json be shared across sessions? | Resolved: stored in agent workspace at `agents/ai-skill-scout/data/gaps.json` |

---

## 11. References

### Research (informing design patterns)

| Pattern | Source | What We Borrow |
|---------|--------|---------------|
| Skill search + install pipeline | Hermes Agent (86K stars) | Quarantine-scan-confirm-install, trust levels, audit log |
| Progressive disclosure | Agent Skills Survey (arXiv:2602.12430) | L1 fragment → L2 full SKILL.md fetch for scoring |
| Security scanning | Hermes skills_guard.py | 60 regex threat patterns, trust-level policy matrix |
| Query expansion | Gorilla LLM (12K stars) | LLM maps user intent to a small set of probe terms |
| Decision framework | claude-skill-search-first | ADOPT / EXTEND / GAP classification |
| Skill format | OpenClaw SKILL.md spec | YAML frontmatter + `metadata.openclaw` + markdown body |
| Community skill safety | "Six Million Fake Stars" (ICSE 2026) | 26.1% vulnerability rate; never auto-install |

### Key Repos

| Project | Stars | Relevance |
|---------|-------|-----------|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | 86K | Reference implementation for skill search/install pipeline |
| [OpenClaw](https://github.com/openclaw/openclaw) | 357K | Platform, SKILL.md format, Plugin SDK |
| [awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) | 46K | Curated skill catalog, quality filtering |
| [AI-Skill-builder](https://github.com/Kkuntal990/AI-Skill-builder) | — | This project |

### OpenClaw Docs

- Skills: https://docs.openclaw.ai/tools/skills
- Creating Skills: https://docs.openclaw.ai/tools/creating-skills
- Plugin SDK: https://docs.openclaw.ai/plugins/building-plugins
