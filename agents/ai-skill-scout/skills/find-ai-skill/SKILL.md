---
name: find-ai-skill
description: "Search GitHub for OpenClaw skills (SKILL.md files), evaluate by trust and quality, and install safely. Use when: user needs a new AI/ML skill, asks 'find me a skill for X', wants to discover what skills exist for a topic. NOT for: creating new skills (use skill-creator), managing installed skills (use openclaw skills list)."
metadata:
  {
    "openclaw":
      {
        "emoji": "🔭",
        "requires": { "bins": ["gh", "python3"] },
        "install":
          [
            {
              "id": "brew-gh",
              "kind": "brew",
              "formula": "gh",
              "bins": ["gh"],
              "label": "Install GitHub CLI (brew)",
            },
          ],
      },
  }
---

# Find AI Skill

Search GitHub for existing OpenClaw skills and install them safely.

## Usage

All operations go through the pipeline script:

```bash
# Search for skills matching a query
python3 scripts/skill_scout.py search "fine-tuning"

# Install a specific skill (after reviewing search results)
python3 scripts/skill_scout.py install oracle/accelerated-data-science skills/aqua-cli

# View currently installed skills from this agent
python3 scripts/skill_scout.py installed

# View gaps (queries with no good results)
python3 scripts/skill_scout.py gaps
```

## Pipeline

```
search: CHECK installed -> EXPAND (LLM) -> SEARCH GitHub -> EVALUATE trust+completeness -> PRESENT top candidates
install: QUARANTINE to /tmp -> SCAN for threats -> INSTALL to workspace -> LOG
```

## Query Expansion

The LLM (Claude Opus via OpenRouter) automatically expands single-word queries into
3-6 search variants before hitting GitHub. API key is read from `OPENROUTER_API_KEY`
env var, falling back to OpenClaw's auth store. On failure, searches the literal
query only. Pass `--no-expand` to disable.

## Trust Levels

- **HIGH**: Known orgs (huggingface, anthropics, oracle, google, microsoft, meta-llama, nvidia, aws, deepmind, openai, NousResearch, mistralai, databricks)
- **MEDIUM**: Repos with 500+ stars
- **LOW**: Everything else

## Security Scan

The scanner checks for exfiltration, injection, destructive commands, obfuscation, and credential access patterns. Dangerous patterns always block. Caution patterns are reported for agent judgment.

## Output

The script outputs JSON to stdout. Read it and present results conversationally to the user.
