# Tools

Available binaries:
- `python3` — for any computation
- `mcporter` — calls MCP servers (e.g. `mcporter call context7.query-docs '{"libraryName": "peft"}'`). Use this when a loaded skill says to fall back to live docs via an MCP.

Available skills (whitelisted):
- `mcporter` — knows how to invoke MCP servers via `mcporter call <server>.<tool> <json-args>`.

Skills loaded ad-hoc via prompt marker like `(For context: a skill is installed at <path>...)` should be read from disk and used as instructed by their SKILL.md.
