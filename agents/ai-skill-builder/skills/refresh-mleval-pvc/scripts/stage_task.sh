#!/usr/bin/env bash
# Stage one task directory's files to the mleval-results PVC.
#
# Usage:
#   ./stage_task.sh <task_name>
#
# Example:
#   ./stage_task.sh samsum
#
# Pushes:
#   infra/tasks/<task>/*.md  -> /results/data/<task>/
#   infra/tasks/<task>/*.yaml -> /results/data/<task>/
#
# Removes from PVC any file in /results/data/<task>/ that no longer has
# a local counterpart with the same basename.
#
# Idempotent — safe to re-run.

set -euo pipefail

TASK="${1:?usage: $0 <task_name>}"
NAMESPACE="${KUBECTL_NS:-ecepxie}"
HELPER_POD="${HELPER_POD:-mleval-jupyter-1gpu}"

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/infra/tasks/$TASK"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "[stage_task] ERROR: $LOCAL_DIR does not exist" >&2
    exit 1
fi

if ! kubectl -n "$NAMESPACE" get pod "$HELPER_POD" >/dev/null 2>&1; then
    echo "[stage_task] ERROR: helper pod $HELPER_POD not found in $NAMESPACE" >&2
    echo "[stage_task]   redeploy with: envsubst < deploy/k8s/helper-jupyter-1gpu.yaml | kubectl -n $NAMESPACE apply -f -" >&2
    exit 1
fi

echo "[stage_task] task=$TASK  local=$LOCAL_DIR  pvc=/results/data/$TASK"

# Ensure PVC target dir exists
kubectl -n "$NAMESPACE" exec "$HELPER_POD" -- mkdir -p "/results/data/$TASK"

# Push local files (md + yaml only — data/* is handled separately, usually large)
PUSHED=0
for f in "$LOCAL_DIR"/*.md "$LOCAL_DIR"/*.yaml; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    kubectl -n "$NAMESPACE" cp "$f" "$HELPER_POD:/results/data/$TASK/$base"
    PUSHED=$((PUSHED + 1))
    echo "  pushed: $base"
done

# Detect & report stale (don't auto-delete — confirm interactively)
STALE=$(kubectl -n "$NAMESPACE" exec "$HELPER_POD" -- bash -c "
    for pf in /results/data/$TASK/*.md /results/data/$TASK/*.yaml; do
        [ -f \"\$pf\" ] || continue
        echo \"\$(basename \"\$pf\")\"
    done
" | while read base; do
    [ -z "$base" ] && continue
    if [ ! -f "$LOCAL_DIR/$base" ]; then
        echo "$base"
    fi
done)

if [ -n "$STALE" ]; then
    echo "[stage_task] STALE files on PVC (no local counterpart):"
    echo "$STALE" | sed 's/^/    /'
    echo "[stage_task] To delete, run:"
    echo "  kubectl -n $NAMESPACE exec $HELPER_POD -- rm -fv $(echo "$STALE" | sed "s|^|/results/data/$TASK/|" | tr '\n' ' ')"
fi

echo "[stage_task] done — pushed $PUSHED file(s)"
