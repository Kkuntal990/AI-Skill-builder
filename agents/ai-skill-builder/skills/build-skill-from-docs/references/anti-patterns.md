# Anti-Patterns — What the Validator Rejects

Adapted from `openclaw/skills/ivangdavila/skill-builder`.

| Trap | Why it fails | What to do |
|---|---|---|
| SKILL.md > 500 lines | Violates progressive disclosure; bloats context | Move bulk to `references/*.md` |
| Explaining what the package is | Models already know; wastes context | Explain *when* and *how* to use it |
| "Use when user needs X" in description | Passive, undertriggering | Action verbs + "Invoke when..." clause |
| Keyword-spam description ("lora, qlora, sft, peft, rlhf") | Looks spammy, confuses triggering | One clean sentence with concrete scenarios |
| Inline code templates > 20 lines | Bloats SKILL.md | Move to `references/` or `assets/` |
| Vague instructions ("analyze the data") | Not actionable | Be specific: "Run `df.describe()`; report mean/std for numeric columns" |
| Undeclared file creation | Security flag | Declare in a `## Data Storage` section |
| Duplicated content between SKILL.md and references/ | Violates single-source-of-truth | Reference from SKILL.md, don't repeat |
| Hallucinated shell commands not in the source | Skill breaks when run | Validator checks every shell command against source text |
| Missing frontmatter `name` or `description` | Skill won't load | Enforced by YAML parse |
| `openclaw skills check` errors | Skill won't install | Enforced by validator |

## Security Patterns Rejected

Inherits Scout's 60-pattern scanner (7 categories):

- Exfiltration (10 patterns) — `curl|`, `wget --post`, `scp/rsync @host`, `requests.post`
- Injection (8) — `eval()`, `exec()`, `os.system()`, `subprocess shell=True`
- Destructive (8) — `rm -rf /`, `DROP TABLE`, `mkfs`, `dd of=/dev/`
- Obfuscation (8) — `base64 -d | sh`, hex/unicode escapes
- Credential access (10) — `cat ~/.ssh`, `.env` reads, API key references
- Privilege escalation (4) — `sudo`, `setuid`, setuid bit
- Reverse shell (4) — `bash /dev/tcp`, socket dup2
- Caution (8) — external URLs, `pip install`, `npm install`, `git clone`

BLOCK patterns always reject. CAUTION patterns emit warnings in the report.
