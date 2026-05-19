# OpenClaw Frontmatter Spec

YAML frontmatter at the top of SKILL.md, delimited by `---`.

## Required Fields

- `name` — kebab-case skill identifier. Must match the directory name. Example: `trl-training`
- `description` — one paragraph. Includes *what the skill does* AND *when to trigger it*. **Use imperative action verbs, not "use when..."**.

## Description Best Practices

Claude undertriggers skills by default. Descriptions must be opinionated.

| ❌ Bad | ✅ Good |
|---|---|
| "Use when user needs to fine-tune a model" | "Train and fine-tune transformer language models using TRL. Supports SFT, DPO, GRPO, KTO. Use when user mentions fine-tuning, RLHF, preference optimization, or LLM alignment." |
| "Helper for TRL" | "Train transformers with reinforcement learning using TRL — SFT for instruction tuning, DPO for preference alignment, PPO/GRPO for reward optimization. Invoke whenever the user wants to align, fine-tune, or post-train a language model." |
| "Builds dashboards" | "Build interactive dashboards from tabular data. Use this whenever the user mentions dashboards, visualization, metrics displays, or wants to explore data, even if they don't explicitly say 'dashboard.'" |

Rules the validator enforces:

- First sentence starts with an action verb
- Includes at least one "Use when..." or "Invoke when..." clause with concrete triggers
- No bare keyword spam ("fine-tuning, lora, qlora, peft, rlhf")

## OpenClaw Metadata

Under the top-level `metadata:` key, the `openclaw:` block controls install and environment:

```yaml
metadata:
  {
    "openclaw":
      {
        "emoji": "🤖",
        "requires": { "bins": ["python3", "gh"] },
        "install": [
          {
            "id": "pip-trl",
            "kind": "pip",
            "packages": ["trl", "transformers", "accelerate"],
            "label": "Install TRL and dependencies",
          }
        ],
      },
  }
```

Supported `install.kind` values: `pip`, `brew`, `npm`, `apt`, `cargo`. For commands not covered, use `kind: shell` with an explicit `command` string (must pass the security scan).

## What the LLM Does NOT Write

The script writes frontmatter deterministically from source data:

- `name` — derived from repo name or `--name` flag
- `metadata.openclaw.requires.bins` — detected from README install commands
- `metadata.openclaw.install[]` — parsed from README install section
- `metadata.openclaw.emoji` — default `🤖` unless overridden

The LLM only writes the `description`.
