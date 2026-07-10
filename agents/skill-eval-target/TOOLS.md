# Tools

Available binaries:

- `python3` — for any computation.
- `mcporter` — calls MCP servers from the shell. Used to reach **live docs** when a
  loaded skill instructs an MCP fallback, or when the honest answer needs
  version-specific facts you don't have.

Available skills (whitelisted):

- `mcporter` — how to invoke MCP servers via `mcporter call <server>.<tool> ...`.

Skills loaded ad-hoc via a prompt marker (`(For context: a skill is installed at
<path> ...)`) should be read from disk and used as their `SKILL.md` instructs.

## Reaching context7 (live docs)

`context7` is the version-specific docs MCP. It is a **two-step** flow — you must
resolve a library ID before you can query docs:

```
# 1. resolve the package name to a Context7 library ID (/org/project)
mcporter call context7.resolve-library-id library-name="peft" query="how do I load a LoRA adapter"

# 2. query docs using the resolved libraryId
mcporter call context7.query-docs --args '{"libraryId":"/huggingface/peft","topic":"load LoRA adapter"}'
```

Discover the exact tool names / argument schema at runtime with
`mcporter list context7` — don't guess tool names. Only fall back to context7 when
a skill tells you to or when the static answer would be incomplete; don't call it
reflexively.
