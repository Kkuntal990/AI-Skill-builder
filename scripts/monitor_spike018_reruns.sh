#!/usr/bin/env bash
# Monitor the two live spike-018 reruns until both reach a terminal state.
# Reports periodic status + flags anomalies; on completion reads held-out score.
# macOS bash 3.2 safe: no associative arrays.
set -u
NS=ecepxie
JOBS="spike-018-samsum-with-skill-s0 spike-018-samsum-without-skill-s1"
INTERVAL=300   # 5 min between polls
TICK=0

job_phase() { # -> Complete | Failed | Running | Unknown
  local j="$1" c
  c=$(kubectl -n "$NS" get job "$j" -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
  case "$c" in
    *Complete*) echo Complete ;;
    *Failed*)   echo Failed ;;
    *)          [ -n "$(kubectl -n "$NS" get job "$j" -o jsonpath='{.status.active}' 2>/dev/null)" ] && echo Running || echo Unknown ;;
  esac
}

probe() { # tail the live run log + flag anomalies for job $1
  local j="$1" P L
  P=$(kubectl -n "$NS" get pods -l job-name="$j" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -z "$P" ] && { echo "   (no pod)"; return; }
  L=$(kubectl -n "$NS" exec "$P" -- bash -lc "ls -t /results/spike-018/$j/mlevolve_runs/*/logs/MLEvolve.log 2>/dev/null | head -1" 2>/dev/null)
  [ -z "$L" ] && { echo "   (no log yet)"; return; }
  kubectl -n "$NS" exec "$P" -- bash -lc "
    echo -n '   nodes_done='; grep -cE 'Execution completed|passed code review without changes' '$L' 2>/dev/null
    tail -2 '$L' 2>/dev/null | sed 's/^/   | /'
    grep -qiE 'segmentation fault|exit code 139|SIGSEGV' '$L' 2>/dev/null && echo '   !! SEGFAULT signature'
    grep -qiE 'imdb|sst2|sentiment|sequence-classification|distilbert' '$L' 2>/dev/null && echo '   !! DRIFT signature (wrong task/model)'
  " 2>/dev/null
}

held_out() { # print held-out score for a finished job
  local j="$1" P
  P=$(kubectl -n "$NS" get pods -l job-name="$j" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -z "$P" ] && return
  kubectl -n "$NS" exec "$P" -- bash -lc "cat /results/spike-018/$j/held_out_score.json 2>/dev/null" 2>/dev/null \
    | python3 -c "import sys,json;j=json.load(sys.stdin);print('   held_out: valid=%s score=%s err=%s'%(j.get('valid'),j.get('score'),j.get('error') or j.get('errors')))" 2>/dev/null \
    || echo "   held_out: (not written yet)"
}

while :; do
  TICK=$((TICK+1))
  done_count=0
  echo "=== tick $TICK | $(date '+%H:%M:%S') ==="
  for j in $JOBS; do
    ph=$(job_phase "$j")
    echo "[$j] $ph"
    case "$ph" in
      Complete|Failed) done_count=$((done_count+1)); held_out "$j" ;;
      *)               probe "$j" ;;
    esac
  done
  [ "$done_count" -ge 2 ] && { echo "=== both terminal; monitor exiting ==="; break; }
  sleep "$INTERVAL"
done
