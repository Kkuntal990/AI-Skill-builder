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

## MCP capability contract (`metadata.openclaw.mcps`)

If the skill's body instructs live-doc fallback via MCP, declare the contract here. It is a
*declaration*, not an install — the agent runtime decides whether to invoke it, and some
runtimes (e.g. the MLEvolve eval harness) have **no MCP client at all**, so the body must
always work from `SKILL.md` + `references/` alone and treat MCP as a bonus.

```yaml
"mcps": {
  "preferred": ["hf-mcp/doc_search", "hf-mcp/doc_fetch"],
  "fallback":  ["context7/resolve-library-id", "context7/query-docs"],
  "on_unavailable": "Answer from SKILL.md + references/ only; state the uncertainty rather than fabricate API names, flags, or version details"
}
```

- **Exact tool names only.** Each entry is `<server>/<exact-tool-name>`. A stale name breaks
  runtime tool lookup. context7's real tools are **`resolve-library-id`** and **`query-docs`**
  — **NOT** `get-library-docs` (that's Anthropic's first-party name; see
  [skill-shape-principles.md](../../../../docs/skill-builder/skill-shape-principles.md)).
  The build's **MCP-name gate** (`check_mcp_tool_names`) verifies every declared/used name
  against `mcporter list <server> --json` and **blocks** on a stale name (`--no-mcp-check`
  falls back to the static known-stale denylist for offline builds).
- **`on_unavailable`** states the failure behavior when no MCP client is reachable.
- Body tool references use OpenClaw's double-underscore form: `context7__resolve-library-id`,
  `context7__query-docs` (the gate checks both the `/` frontmatter form and the `__` body form).

## What the LLM Does NOT Write

The script writes frontmatter deterministically from source data:

- `name` — derived from repo name or `--name` flag
- `metadata.openclaw.requires.bins` — detected from README install commands
- `metadata.openclaw.install[]` — parsed from README install section
- `metadata.openclaw.emoji` — default `🤖` unless overridden

The LLM only writes the `description`.
