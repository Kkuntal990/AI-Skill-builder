"""Orchestrate paired A/B Jobs across (task × cell × seed).

Reads `.env` for cluster config (registry, namespace, GPU, secrets), reads
CLI args for the sweep design, renders `infra/agents/mlevolve/job.yaml.tmpl`
per trajectory via envsubst-style substitution, kubectl-applies each Job,
waits for completion, and optionally pulls results off the PVC.

Single-agent on this branch: MLEvolve. AIDE was removed during the
mlevolve-smoke spike (see docs/eval/stage2.md for the pivot rationale).

Two-phase use:

    Phase A (preview, no cluster touch):
        python -m infra.orchestrator.run_ab --task <name> --seeds 0 1 --plan-only

    Phase B (live, applies Jobs):
        python -m infra.orchestrator.run_ab --task <name> --seeds 0 1 --apply

Idempotent: a trajectory whose Job already exists in the namespace is
treated as "already running" and only watched. A trajectory whose
``manifest.json`` already sits on the PVC is skipped entirely.

The orchestrator never re-runs a completed trajectory; bump `MLEVAL_RUN_ID`
in .env to start a fresh sweep.

This file is ~250 LOC, no external deps beyond kubectl + Python stdlib.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Spike: only the GPU profile exists for MLEvolve. CPU profile re-introduced
# if/when a tabular task lands on this branch.
JOB_TEMPLATES = {
    "gpu": REPO_ROOT / "infra/agents/mlevolve/job.yaml.tmpl",
}


# ---- env loading ----------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def _load_dotenv(path: Path) -> dict[str, str]:
    """Tiny .env parser — KEY=VALUE lines, strips matched surrounding quotes."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        value = m.group(2)
        # Strip a matched pair of surrounding quotes. Without this,
        # MLEVAL_LLM_MODEL="deepseek/..." renders as the literal quoted
        # string into the Job manifest and the OpenAI client rejects it.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        out[m.group(1)] = value
    return out


# ---- envsubst-style template rendering -----------------------------------

_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def render(template: str, env: dict[str, str]) -> str:
    def replace(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in env:
            raise KeyError(f"job template references ${{{name}}} but env has no value")
        return env[name]

    return _VAR_PATTERN.sub(replace, template)


# ---- trajectory plan -----------------------------------------------------


@dataclass
class Trajectory:
    task: str
    cell: str  # "with_skill" | "without_skill"
    seed: int
    skill_path: str
    skill_library: str
    run_id: str
    llm_model: str
    time_limit_sec: int
    step_limit: int
    llm_timeout_sec: int = 120
    # Per-exec subprocess cap. Decoupled from time_limit_sec so a single
    # training pass can't consume the entire trajectory budget. Default
    # is computed by entrypoint.sh as time_limit_sec / 2 when unset; pass
    # this explicitly when you need a specific per-exec window (e.g.
    # SAMSum needs ~40 min to finish 1 epoch of QLoRA + eval on A6000).
    exec_timeout_sec: int = 0  # 0 → entrypoint default (time_limit_sec/2)

    @property
    def trajectory_id(self) -> str:
        # k8s Job names must be DNS-1123 labels: lowercase alphanumeric + '-'
        # only (no '_'), <= 63 chars.
        safe_task = re.sub(r"[^a-z0-9-]", "-", self.task.lower())[:30]
        safe_cell = self.cell.replace("_", "-")
        return f"{self.run_id}-{safe_task}-{safe_cell}-s{self.seed}".lower()

    @property
    def task_reqs_path(self) -> str:
        # Mirrors task-data staging on PVC: /results/data/<task>/requirements.txt
        return f"/results/data/{self.task}/requirements.txt"

    @property
    def skill_reqs_path(self) -> str:
        # Sibling of SKILL.md. Empty if without_skill or no skill_path declared.
        if self.cell != "with_skill" or not self.skill_path:
            return ""
        skill_dir = self.skill_path.rsplit("/", 1)[0]
        return f"{skill_dir}/requirements.txt"

    def env_overrides(self, base_env: dict[str, str]) -> dict[str, str]:
        out = dict(base_env)
        out.update(
            {
                "MLEVAL_RUN_ID": self.run_id,
                "MLEVAL_TRAJECTORY_ID": self.trajectory_id,
                "TASK": self.task,
                "CELL": self.cell,
                "SEED": str(self.seed),
                "MLEVAL_LLM_MODEL": self.llm_model,
                "TIME_LIMIT_SECONDS": str(self.time_limit_sec),
                "STEP_LIMIT": str(self.step_limit),
                # +1200s buffer = ~10 min image pull + ~3 min first-run pip
                # install + ~5 min for analyzers / cleanup. Tighter buffer
                # killed mvp-002 mid-AIDE before the 1800s soft cap fired.
                "ACTIVE_DEADLINE_SECONDS": str(self.time_limit_sec + 1200),
                "MLEVAL_SKILL_PATH": self.skill_path if self.cell == "with_skill" else "",
                # Library dir (preferred): all skills available, model selector
                # routes. Empty for without_skill → zero skills → baseline.
                "MLEVAL_SKILL_LIBRARY": self.skill_library if self.cell == "with_skill" else "",
                "MLEVAL_TASK_REQS_PATH": self.task_reqs_path,
                "MLEVAL_SKILL_REQS_PATH": self.skill_reqs_path,
                "MLEVAL_LLM_TIMEOUT_SEC": str(self.llm_timeout_sec),
                # Empty string → entrypoint computes time_limit_sec / 2
                "MLEVAL_EXEC_TIMEOUT_SEC": (
                    str(self.exec_timeout_sec) if self.exec_timeout_sec > 0 else ""
                ),
            }
        )
        return out


@dataclass
class Plan:
    run_id: str
    task: str
    seeds: list[int]
    cells: list[str]
    namespace: str
    base_env: dict[str, str]
    skill_path: str = ""
    skill_library: str = ""
    trajectories: list[Trajectory] = field(default_factory=list)

    def build(
        self,
        llm_model: str,
        time_limit_sec: int,
        step_limit: int,
        llm_timeout_sec: int = 120,
        exec_timeout_sec: int = 0,
    ) -> None:
        for cell in self.cells:
            for seed in self.seeds:
                self.trajectories.append(
                    Trajectory(
                        task=self.task,
                        cell=cell,
                        seed=seed,
                        skill_path=self.skill_path,
                        skill_library=self.skill_library,
                        run_id=self.run_id,
                        llm_model=llm_model,
                        time_limit_sec=time_limit_sec,
                        step_limit=step_limit,
                        llm_timeout_sec=llm_timeout_sec,
                        exec_timeout_sec=exec_timeout_sec,
                    )
                )


# ---- kubectl wrappers ----------------------------------------------------


def _kubectl(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["kubectl", *args]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def job_exists(namespace: str, name: str) -> bool:
    cp = _kubectl(["-n", namespace, "get", "job", name, "-o", "name"], check=False)
    return cp.returncode == 0


def apply_job(namespace: str, rendered_yaml: str) -> None:
    cp = subprocess.run(
        ["kubectl", "-n", namespace, "apply", "-f", "-"],
        input=rendered_yaml,
        text=True,
        capture_output=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"kubectl apply failed:\n{cp.stderr}")
    print(f"  applied: {cp.stdout.strip()}")


def wait_for_job(namespace: str, name: str, timeout_sec: int) -> str:
    """Returns 'complete' | 'failed' | 'timeout'."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        cp = _kubectl(
            ["-n", namespace, "get", "job", name, "-o", "jsonpath={.status.conditions[*].type}"],
            check=False,
        )
        conds = (cp.stdout or "").split()
        if "Complete" in conds:
            return "complete"
        if "Failed" in conds:
            return "failed"
        time.sleep(10)
    return "timeout"


# ---- main flow -----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Orchestrate a paired A/B sweep")
    p.add_argument("--task", required=True, help="Task name (must match infra/tasks/<task>/)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="Seed list")
    p.add_argument(
        "--cells",
        nargs="+",
        default=["with_skill", "without_skill"],
        choices=["with_skill", "without_skill"],
    )
    p.add_argument("--skill-path", default="", help="In-pod path to a single SKILL.md (back-compat; only used when cell=with_skill)")
    p.add_argument("--skill-library", default="", help="In-pod path to a skill LIBRARY dir (e.g. /results/skills); all skills available, model selector routes (only used when cell=with_skill)")
    p.add_argument("--time-limit-sec", type=int, default=3600, help="Per-trajectory wall-clock cap (graceful — entrypoint watchdog kills agent PGID then runs analyzer/manifest).")
    p.add_argument("--step-limit", type=int, default=5, help="Agent max steps per trajectory (MLEvolve agent.steps; the only LOOP exit since agent.time_limit is soft).")
    p.add_argument("--llm-timeout-sec", type=int, default=120, help="Per-LLM-request HTTP timeout (read).")
    p.add_argument("--exec-timeout-sec", type=int, default=0, help="Per-exec subprocess kill (MLEvolve exec.timeout). 0 → entrypoint default = time_limit_sec/2.")
    p.add_argument(
        "--profile",
        choices=["gpu"],
        default="gpu",
        help="Job profile. Only 'gpu' available on the mlevolve-smoke branch.",
    )
    p.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    p.add_argument("--namespace", default=None, help="Override K8S_NAMESPACE from .env")
    p.add_argument("--apply", action="store_true", help="Actually kubectl-apply Jobs (default: dry-run)")
    p.add_argument("--wait", action="store_true", help="Block until all Jobs reach a terminal state")
    args = p.parse_args(argv)

    env = _load_dotenv(args.env_file)
    namespace = args.namespace or env.get("K8S_NAMESPACE", "").strip()
    if not namespace or namespace == "REPLACE_ME":
        print("ERROR: K8S_NAMESPACE missing/unset in .env", file=sys.stderr)
        return 1

    run_id = env.get("MLEVAL_RUN_ID", "mvp-001")
    llm_model = env.get("MLEVAL_LLM_MODEL", "deepseek/deepseek-v4-flash")

    plan = Plan(
        run_id=run_id,
        task=args.task,
        seeds=list(args.seeds),
        cells=list(args.cells),
        namespace=namespace,
        base_env=env,
        skill_path=args.skill_path,
        skill_library=args.skill_library,
    )
    plan.build(
        llm_model=llm_model,
        time_limit_sec=args.time_limit_sec,
        step_limit=args.step_limit,
        llm_timeout_sec=args.llm_timeout_sec,
        exec_timeout_sec=args.exec_timeout_sec,
    )

    template_path = JOB_TEMPLATES[args.profile]
    print(f"=== A/B sweep plan: {len(plan.trajectories)} trajectories ===")
    print(f"  run_id={run_id}  namespace={namespace}  llm={llm_model}  profile={args.profile}")
    print(f"  template={template_path.relative_to(REPO_ROOT)}")
    for t in plan.trajectories:
        skill = (t.skill_library or t.skill_path or "(none)") if t.cell == "with_skill" else "(none)"
        print(f"  - {t.trajectory_id}   cell={t.cell:14s}  seed={t.seed}  skill={skill}")

    if not args.apply:
        print("\n[plan-only mode] not applying. Re-run with --apply to live-deploy.")
        return 0

    print()
    print("=== applying Jobs ===")
    template = template_path.read_text()
    for t in plan.trajectories:
        env_for_render = t.env_overrides(env)
        try:
            rendered = render(template, env_for_render)
        except KeyError as e:
            print(f"  render error for {t.trajectory_id}: {e}", file=sys.stderr)
            continue
        if job_exists(namespace, t.trajectory_id):
            print(f"  skip (exists): {t.trajectory_id}")
            continue
        apply_job(namespace, rendered)

    if args.wait:
        print()
        print("=== waiting for completion ===")
        for t in plan.trajectories:
            status = wait_for_job(namespace, t.trajectory_id, timeout_sec=t.time_limit_sec + 1200)
            print(f"  {t.trajectory_id}: {status}")

    print()
    print(f"done. Results land on PVC under /results/{run_id}/<trajectory_id>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
