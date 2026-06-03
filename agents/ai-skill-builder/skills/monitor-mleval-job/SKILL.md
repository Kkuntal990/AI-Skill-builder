---
name: monitor-mleval-job
description: Background-watch one or more live mleval A/B trajectory Jobs on Nautilus with an adaptive cadence, staying SILENT while healthy and only surfacing when something is out of the ordinary — pod crash / OOMKill / CrashLoopBackOff / image-pull failure / restart / stuck-Pending, the agent drifting off the task, a stall, or normal completion. Use right after launching an A/B sweep (make ab-apply) to keep an eye on long (multi-hour) trajectories without manual polling.
---

# Monitor a live mleval A/B Job

Long trajectories (PEFT fine-tunes run hours; a single training node can take ~2 h)
shouldn't need a human babysitting `kubectl get pods`. This skill runs a
**background watcher** that polls the Job/pod health on an adaptive interval and
**only interrupts you when there's something to act on**.

## Design — quiet unless anomaly

The watcher (`scripts/monitor_job.sh`) is launched with `run_in_background: true`.
A backgrounded command keeps running across turns and **re-invokes the agent only
when it exits**. The script is written to *loop silently while healthy* and
*exit (→ notify) only* on:

- **Crash / failure** — Job `.status.failed ≥ 1`, pod phase `Failed`, or a non-zero
  terminal exit that isn't a clean wall-cap.
- **OOMKill** — pod `lastState.terminated.reason == OOMKilled` (the thing we watch
  most closely after right-sizing memory to 8 GiB).
- **Container-create trouble** — `CrashLoopBackOff`, `ImagePullBackOff`,
  `ErrImagePull`, `CreateContainerError`.
- **Unexpected restart** — `restartCount` increments.
- **Stuck Pending** — pod `Pending` beyond `PENDING_GRACE` (default 900 s; cold image
  pull is ~10 min, so the grace is generous).
- **Off-task / stall (handoff)** — after the run settles, the watcher can exit once
  with `SETTLED` so the operator (you) reads the agent's recent stdout and confirms
  it's actually doing the task in `instruction.md` (loading the right dataset,
  training, computing the right metric) and not thrashing or doing something
  unrelated. Then relaunch in anomaly-only mode.
- **Completion** — all Jobs `.status.succeeded == 1` (or wall-capped-with-result).

While none of these hold, the script sleeps and says nothing.

## Adaptive cadence

Start tight, widen once steady. The script polls every `INTERVAL` seconds (default
300 = 5 min) and, after `WIDEN_AFTER` consecutive healthy ticks (default 3), widens
to `WIDE_INTERVAL` (default 1200 = 20 min). Rationale: the failure-prone window is
startup (image pull, first node, OOM-on-load); once a training node is steadily
running, 20 min is plenty. Near a known wall-cap you can relaunch with a tighter
`INTERVAL` to catch the finalize/analyzer step.

## Usage

```bash
# Phase 1 — startup watch: tight cadence, hand back once settled for an on-task read
NS=ecepxie INTERVAL=300 SETTLE_EXIT=2 \
  bash scripts/monitor_job.sh <job-name-1> [<job-name-2> ...]
# (launch with run_in_background: true)
```

When it exits `SETTLED`, read the agent stdout and confirm on-task:

```bash
kubectl -n ecepxie logs <pod> --tail=120        # look for: right dataset, training, metric
kubectl -n ecepxie exec <helper-or-pod> -- tail /results/<run_id>/<traj>/agent_logs/mlevolve_stdout.log
```

Then relaunch in **anomaly-only** mode for the long haul:

```bash
NS=ecepxie INTERVAL=1200 SETTLE_EXIT=0 PENDING_GRACE=900 \
  bash scripts/monitor_job.sh <job-name-1> [<job-name-2> ...]
# (run_in_background: true) — now it only comes back on anomaly or completion.
```

## What the operator does on each re-invocation

| Exit reason | Action |
|---|---|
| `ANOMALY: OOMKilled` | Bump pod memory in `job.yaml.tmpl`, delete + relaunch the cell. |
| `ANOMALY: ImagePullBackOff/ErrImagePull` | Check `ghcr-pull` secret; confirm image digest exists. |
| `ANOMALY: CrashLoopBackOff / restart` | `kubectl logs --previous`; inspect entrypoint/sidecar import error. |
| `ANOMALY: Pending>grace` | `kubectl describe pod` events — GPU quota / taint / node pressure. |
| `SETTLED` | Read stdout; confirm on-task; relaunch anomaly-only (wide interval). |
| `DONE` | Pull results, run `scripts/l1_l2_compare.py`, update `docs/eval/peft-skill-eval-report.md`. |

## Notes

- The watcher is **read-only** — it never deletes/relaunches Jobs. Remediation is the
  operator's call (some "anomalies" like a clean wall-cap are expected — exit 143 with
  a metric already in hand is a success, not a crash; see `docs/eval/peft-skill-eval-report.md`).
- Pair with **build-mleval-image** (image changes) and **refresh-mleval-pvc** (data changes).
- Tune `INTERVAL`/`WIDE_INTERVAL`/`PENDING_GRACE` per job length; the defaults suit a
  multi-hour PEFT A/B.
