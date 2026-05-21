---
name: <skill-name>
description: <one-sentence summary of what this skill teaches>
version: 0.0.1
references:
  - title: <paper or doc title>
    url: <url>
---

# <Skill name>

## When to use this skill

<Specific situations where the agent should reach for this skill. Be
concrete — name the dataset/task patterns, not abstract concepts.>

## Quick checklist

- [ ] <one-line action>
- [ ] <one-line action>
- [ ] <one-line action>

## Detailed guidance

<2-5 paragraphs of the actual content the agent needs. Include code
snippets in fenced blocks; the agent's code-gen prompts will see them.>

```python
# Example showing the typical API call sequence.
```

## Common pitfalls

- **<pitfall>**: <why it happens, how to avoid>.
- **<pitfall>**: <why it happens, how to avoid>.

## References

See the frontmatter `references` list. The agent may request additional
files from `infra/skills/<skill-name>/references/` via the harness.
