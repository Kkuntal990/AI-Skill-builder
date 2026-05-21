# `infra/skills/` — Skills under evaluation

A skill is a markdown document (the OpenClaw `SKILL.md` format) that the
A/B framework splices into the agent's task description in the `with_skill`
cell. The `without_skill` cell is identical except `MLEVAL_SKILL_PATH` is
unset, so no skill content reaches the agent.

```
infra/skills/<skill-name>/
├── SKILL.md          # required — the skill content (progressive disclosure)
├── references/       # optional — files the skill cites that the agent may pull
└── README.md         # optional — version notes, source paper, etc.
```

## How skills get into the agent

1. The orchestrator passes `--skill-path /results/skills/<name>/SKILL.md`.
2. The Job's `MLEVAL_SKILL_PATH` env is set to that path (or empty for `without_skill`).
3. `aide_sidecar.skill_inject` monkey-patches `aide.utils.config.load_task_desc`
   to splice the SKILL.md content into AIDE's task description before any prompt is built.
4. AIDE's code-gen and judge prompts both see the skill.

## Staging skills onto the PVC

Skills are tiny (~KB) compared to task data. Stage them once per sweep,
co-located with task data:

```bash
kubectl -n $K8S_NAMESPACE cp infra/skills/<name>/SKILL.md pvc-shell:/results/skills/<name>/SKILL.md
```

The default orchestrator expects `MLEVAL_SKILL_PATH=/results/skills/<name>/SKILL.md`
in the with-skill cell. Override via `--skill-path` if you put it elsewhere.

## Starting a new skill

Copy `_template/`:

```bash
cp -r infra/skills/_template infra/skills/<your-skill>
$EDITOR infra/skills/<your-skill>/SKILL.md
```

The `_template/SKILL.md` is a stub; real skills come from
`agents/ai-skill-builder/` or are hand-authored.
