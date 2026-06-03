#!/usr/bin/env bash
# Adaptive, quiet-unless-anomaly watcher for mleval A/B Jobs on Nautilus.
#
# Launch with run_in_background: true. The script loops silently while every
# watched Job is healthy and EXITS (which re-invokes the agent) only on an
# anomaly, a SETTLED handoff, or completion. Read-only: never mutates Jobs.
#
# Usage:
#   NS=ecepxie INTERVAL=300 WIDE_INTERVAL=1200 WIDEN_AFTER=3 \
#   PENDING_GRACE=900 SETTLE_EXIT=0 \
#     bash monitor_job.sh <job-1> [<job-2> ...]
#
# Env knobs:
#   NS            namespace (default ecepxie)
#   INTERVAL      base poll seconds (default 300 = 5 min)
#   WIDE_INTERVAL poll seconds after settling (default 1200 = 20 min)
#   WIDEN_AFTER   healthy ticks before widening (default 3)
#   PENDING_GRACE max seconds a pod may stay Pending before it's an anomaly (default 900)
#   SETTLE_EXIT   if >0, exit "SETTLED" after this many healthy ticks (handoff for on-task review)
set -uo pipefail

NS="${NS:-ecepxie}"
INTERVAL="${INTERVAL:-300}"
WIDE_INTERVAL="${WIDE_INTERVAL:-1200}"
WIDEN_AFTER="${WIDEN_AFTER:-3}"
PENDING_GRACE="${PENDING_GRACE:-900}"
SETTLE_EXIT="${SETTLE_EXIT:-0}"
JOBS=("$@")

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "usage: monitor_job.sh <job-name> [<job-name> ...]" >&2
  exit 2
fi

k() { kubectl -n "$NS" "$@" 2>/dev/null; }

# macOS ships bash 3.2 (no associative arrays); keep this POSIX-simple so the
# watcher runs from the operator's Mac without a newer bash.
healthy_ticks=0

echo "[monitor] watching ${JOBS[*]} in ns=$NS (interval=${INTERVAL}s, settle_exit=$SETTLE_EXIT)"

while true; do
  all_done=1
  for job in "${JOBS[@]}"; do
    failed=$(k get job "$job" -o jsonpath='{.status.failed}')
    succeeded=$(k get job "$job" -o jsonpath='{.status.succeeded}')
    [ "${succeeded:-0}" = "1" ] || all_done=0

    if [ "${failed:-0}" != "" ] && [ "${failed:-0}" -ge 1 ] 2>/dev/null; then
      echo "[monitor] ANOMALY: Job $job .status.failed=$failed"
      k get job "$job" -o wide
      k describe job "$job" | tail -20
      exit 1
    fi

    # Pod-level checks (newest pod for this job).
    pod=$(k get pods -l job-name="$job" --sort-by=.metadata.creationTimestamp \
            -o jsonpath='{.items[-1:].metadata.name}')
    [ -z "$pod" ] && continue

    phase=$(k get pod "$pod" -o jsonpath='{.status.phase}')
    waiting=$(k get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}')
    term_reason=$(k get pod "$pod" -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}')
    restarts=$(k get pod "$pod" -o jsonpath='{.status.containerStatuses[0].restartCount}')
    restarts="${restarts:-0}"

    case "$waiting" in
      CrashLoopBackOff|ImagePullBackOff|ErrImagePull|CreateContainerError|InvalidImageName)
        echo "[monitor] ANOMALY: pod $pod waiting reason=$waiting"
        k describe pod "$pod" | tail -25
        exit 1 ;;
    esac

    if [ "$term_reason" = "OOMKilled" ]; then
      echo "[monitor] ANOMALY: pod $pod OOMKilled (bump job.yaml.tmpl memory)"
      k get pod "$pod" -o wide
      exit 1
    fi

    if [ "${restarts:-0}" -ge 1 ] 2>/dev/null; then
      echo "[monitor] ANOMALY: pod $pod restartCount=$restarts (container restarted — trajectory pods should never restart)"
      k logs "$pod" --previous --tail=40
      exit 1
    fi

    if [ "$phase" = "Pending" ]; then
      start=$(k get pod "$pod" -o jsonpath='{.metadata.creationTimestamp}')
      if [ -n "$start" ]; then
        age=$(( $(date +%s) - $(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$start" +%s 2>/dev/null || echo "$(date +%s)") ))
        if [ "$age" -gt "$PENDING_GRACE" ] 2>/dev/null; then
          echo "[monitor] ANOMALY: pod $pod Pending for ${age}s (> ${PENDING_GRACE}s grace)"
          k describe pod "$pod" | sed -n '/Events:/,$p' | tail -20
          exit 1
        fi
      fi
      all_done=0
    fi
    if [ "$phase" = "Failed" ]; then
      echo "[monitor] ANOMALY: pod $pod phase=Failed"
      k describe pod "$pod" | tail -25
      exit 1
    fi
    [ "$phase" = "Succeeded" ] || all_done=0
  done

  if [ "$all_done" = "1" ]; then
    echo "[monitor] DONE: all jobs succeeded — pull results + run l1_l2_compare.py + update report."
    for job in "${JOBS[@]}"; do k get job "$job" -o wide; done
    exit 0
  fi

  healthy_ticks=$((healthy_ticks + 1))
  if [ "$SETTLE_EXIT" -gt 0 ] && [ "$healthy_ticks" -ge "$SETTLE_EXIT" ]; then
    echo "[monitor] SETTLED: $healthy_ticks healthy ticks — read agent stdout, confirm on-task, then relaunch anomaly-only."
    for job in "${JOBS[@]}"; do
      pod=$(k get pods -l job-name="$job" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
      echo "--- $pod ---"; k get pod "$pod" -o wide
    done
    exit 0
  fi

  if [ "$healthy_ticks" -ge "$WIDEN_AFTER" ]; then
    sleep_s="$WIDE_INTERVAL"
  else
    sleep_s="$INTERVAL"
  fi
  sleep "$sleep_s"
done
