#!/usr/bin/env bash
# =============================================================================
# watch_pod_failures.sh — capture WHY an A/B pod died, before k8s GCs the proof
# =============================================================================
# Failed pod objects are garbage-collected within minutes and their events
# expire ~1h, so a post-hoc "why did it restart?" is usually unrecoverable
# (mvp-029 with-skill: we could only INFER eviction). This watcher polls every
# POLL_SEC and, the moment a pod for the run goes Failed/Evicted/OOMKilled or
# the Job swaps in a new pod, it snapshots the forensics to a persistent log
# and EXITS — so the background runner re-invokes the caller to report.
#
# Usage:  scripts/watch_pod_failures.sh [RUN_ID] [TASK] [POLL_SEC] [SEED]
#   RUN_ID    defaults to MLEVAL_RUN_ID from .env
#   TASK      defaults to gsm8k
#   POLL_SEC  defaults to 30
#   SEED      defaults to 0 (run one watcher per seed)
# Run it in the BACKGROUND so it keeps watching across turns:
#   (the Bash tool's run_in_background re-invokes on exit = on incident/done)
#
# Exits when: (a) a NEW pod failure/restart is captured, or (b) BOTH cells'
# Jobs reach a terminal state (run finished, no incident). Log persists at
# $PODWATCH_LOG (default scratchpad/pod_failures_<run>.log).
# bash 3.2 compatible (no associative arrays) — runs on macOS.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || true
[ -f .env ] && set -a && . ./.env && set +a
RUN="${1:-${MLEVAL_RUN_ID:-mvp-029}}"
TASK="${2:-gsm8k}"
POLL="${3:-30}"
SEED="${4:-0}"
NS="${K8S_NAMESPACE:-ecepxie}"
LOG="${PODWATCH_LOG:-$(dirname "$0")/../.podwatch_${RUN}_s${SEED}.log}"

JOBS="${RUN}-${TASK}-with-skill-s${SEED} ${RUN}-${TASK}-without-skill-s${SEED}"
SEEN=""          # pod names already captured (space-delimited)
PREV_ACTIVE=""   # "job=podname" pairs from last poll

note(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

capture(){ # $1=pod  $2=job  $3=context
  local p="$1" job="$2" ctx="$3"
  note "════════ INCIDENT: pod failure/restart ($ctx) ════════"
  note "job=$job  pod=$p"
  # Live pod status (if not yet GC'd) — the authoritative terminated reason.
  local js
  js=$(kubectl -n "$NS" get pod "$p" -o jsonpath='phase={.status.phase} reason={.status.reason} msg={.status.message} node={.spec.nodeName} term={.status.containerStatuses[0].lastState.terminated.reason} exit={.status.containerStatuses[0].lastState.terminated.exitCode} sig={.status.containerStatuses[0].lastState.terminated.signal} restarts={.status.containerStatuses[0].restartCount}' 2>/dev/null)
  if [ -n "$js" ]; then note "POD STATUS: $js"; else note "POD STATUS: (already GC'd — relying on events)"; fi
  # Events for this pod by name (persist ~1h even after the pod is GC'd).
  note "POD EVENTS:"
  kubectl -n "$NS" get events --field-selector involvedObject.name="$p" \
    -o custom-columns='T:.lastTimestamp,TYPE:.type,REASON:.reason,MSG:.message' --no-headers 2>/dev/null \
    | tee -a "$LOG" | sed 's/^/  /' || true
  # Node-level events (eviction/NotReady/OOM) for the node it ran on.
  local node; node=$(echo "$js" | sed -n 's/.*node=\([^ ]*\).*/\1/p')
  if [ -n "$node" ]; then
    note "NODE $node events (kill/evict/NotReady/Oom):"
    kubectl -n "$NS" get events --field-selector involvedObject.name="$node" \
      -o custom-columns='T:.lastTimestamp,REASON:.reason,MSG:.message' --no-headers 2>/dev/null \
      | grep -iE 'evict|notready|oom|kill|preempt|taint|pressure' | tail -5 | sed 's/^/  /' || true
  fi
  note "─────────────────────────────────────────────────────"
}

job_terminal(){ # echoes "done" if job reached Complete/Failed — or is ABSENT
  # An absent job (e.g. single-cell run where without-skill was never launched)
  # is treated as non-blocking so "all jobs terminal" can still fire.
  kubectl -n "$NS" get job "$1" >/dev/null 2>&1 || { echo done; return; }
  local s; s=$(kubectl -n "$NS" get job "$1" -o jsonpath='{.status.succeeded}-{.status.failed}-{range .status.conditions[*]}{.type}={.status};{end}' 2>/dev/null)
  case "$s" in *"Complete=True"*|*"Failed=True"*) echo done;; *) echo "";; esac
}

note "watch start: run=$RUN task=$TASK poll=${POLL}s ns=$NS  (log: $LOG)"
while true; do
  incident=0
  active_now=""
  alldone=1
  for job in $JOBS; do
    [ -n "$(job_terminal "$job")" ] || alldone=0
    pods=$(kubectl -n "$NS" get pods -l job-name="$job" --no-headers 2>/dev/null | awk '{print $1"|"$3}')
    cur_active=""
    for entry in $pods; do
      p="${entry%%|*}"; phase="${entry##*|}"
      case "$phase" in
        Running|Pending|ContainerCreating) cur_active="$p" ;;
        Failed|Error|Evicted|OOMKilled|CrashLoopBackOff)
          case " $SEEN " in *" $p "*) : ;; *) capture "$p" "$job" "phase=$phase"; SEEN="$SEEN $p"; incident=1 ;; esac ;;
      esac
    done
    [ -n "$cur_active" ] && active_now="$active_now $job=$cur_active"
    # restart detection: this job's active pod name changed since last poll
    prev=$(echo "$PREV_ACTIVE" | tr ' ' '\n' | sed -n "s/^$job=//p")
    if [ -n "$prev" ] && [ -n "$cur_active" ] && [ "$prev" != "$cur_active" ]; then
      case " $SEEN " in *" $prev "*) : ;; *) capture "$prev" "$job" "RESTART: $prev -> $cur_active"; SEEN="$SEEN $prev"; incident=1 ;; esac
    fi
  done
  PREV_ACTIVE="$active_now"
  if [ "$incident" = "1" ]; then note "EXIT: incident captured (see above)"; exit 0; fi
  if [ "$alldone" = "1" ]; then note "EXIT: both jobs terminal (run finished, no new incident)"; exit 0; fi
  sleep "$POLL"
done
