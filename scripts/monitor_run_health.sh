#!/usr/bin/env bash
# =============================================================================
# monitor_run_health.sh — early KILL-or-KEEP health monitor for an MLEvolve A/B
# =============================================================================
# Usage:
#   scripts/monitor_run_health.sh [RUN_ID] [TASK] [EXPECT_DIGEST]
#     RUN_ID        defaults to MLEVAL_RUN_ID from .env
#     TASK          defaults to gsm8k
#     EXPECT_DIGEST optional sha256 the pods SHOULD be running (catch stale :dev)
#
# Run it ~3-5 min after launch (Tier 0-1), then again at ~15-40 min (Tier 2-4).
# It reads the shared PVC by exec-ing into a RUNNING pod of the run — no extra
# pods (NRP blocks bare inspection pods on low utilization).
#
# Verdicts:  [OK]  fine    [WARN] watch / investigate    [KILL] abort early
#
# -----------------------------------------------------------------------------
# THE CHECKLIST (each check cites the failure trace that motivated it)
# -----------------------------------------------------------------------------
# TIER 0 — deployment (immediate):
#   0.1 Pod phase Running; not ImagePullBackOff / CrashLoopBackOff / Pending>2m
#       KILL on ImagePullBackOff      [trace: mvp-028 expired GHCR token, 403]
#   0.2 Restarts == 0                 [WARN if >0]
#   0.3 Image digest == EXPECT_DIGEST [WARN: stale cached :dev despite Always]
#   0.4 Entrypoint banner present + cell/seed/caps + "metric direction pinned"
#       KILL if no banner after ~2m   [entrypoint crashed before MLEvolve]
#
# TIER 1 — our inputs reached the agent INTACT (~3-5 min):
#   1.1 Instruction integrity: clean_task_desc output is the REAL instruction
#       (non-trivial, has task markers), not empty / "REQUIRED SUBMISSION FORMAT"
#       only.  KILL if gutted.        [trace: mvp-029 clean_task_desc -> 2 chars
#                                      -> tabular regressor; spike-025 garbage]
#   1.2 (skill cell) skill library injected: banner "skill library:" + selector
#       fired with NON-EMPTY selections + skill bodies in prompts.
#       KILL/WARN if skill cell but selections empty
#                                      [trace: spike-023 selections=[] x6 ->
#                                       treatment silently empty]
#
# TIER 2 — agent started the RIGHT task (~10-20 min, first draft):
#   2.1 First draft uses AutoModelForCausalLM + LoRA + generate (generative
#       framing). KILL if tabular-regression signature (Tfidf/RandomForest/
#       XGB/LGBM/MLP-regressor / float() on answer strings) and NO causal LM.
#                                      [trace: mvp-029 gutted -> TF-IDF/MLP regr]
#
# TIER 3 — early node health / yield (~20-40 min, first 3-6 nodes):
#   3.1 No fork-after-CUDA segfault (exit 139 / "Segmentation fault"). KILL.
#                                      [trace: spike-018 killed both reruns x3]
#   3.2 No diff-applier corruption (=======, <<<<<<<, SyntaxError-from-patch).
#                                      [trace: diff-applier ======= defect]
#   3.3 No systemic recurring crash: same exc across >=3 early nodes incl a
#       debug child -> won't converge. WARN->KILL.
#                                      [trace: mvp-029 grad-path x3; trl import;
#                                       lightgbm ModuleNotFound]
#   3.4 Not timeout-dominated: most early nodes hitting per-exec TimeoutError ->
#       approach too heavy for the budget. WARN.
#                                      [trace: mvp-029 without-skill 6 timeouts,
#                                       best=None to the 16h wall]
#   3.5 Yield: >=1 non-buggy node / finite metric by ~step 6, else WARN.
#                                      [trace: mvp-028/029 best=None]
#
# TIER 4 — silent-correctness warnings (don't crash, corrupt results):
#   4.1 Right-padding on decoder-only generation ("right-padding was detected")
#       -> WARN (garbage generations).  [trace: mvp-029 with-skill step18]
#   4.2 Agent requests absent deps: flash_attn / lightgbm / xgboost
#       ModuleNotFound -> WARN.         [trace: mvp-029]
#
# -----------------------------------------------------------------------------
# IF YOU KILL:  kubectl -n <ns> delete job <job-name>
#   To relaunch the SAME run_id you MUST first wipe /results/<run_id>/<traj>,
#   because the SIGTERM finalize writes manifest.json and the orchestrator then
#   SKIPS that trajectory ("manifest exists"). Or bump MLEVAL_RUN_ID.
# Logs live at: /results/<run_id>/<traj>/mlevolve_runs/*/logs/MLEvolve.log,
#               /results/<run_id>/<traj>/{prompts.jsonl,state.json,
#                                          mlevolve_runs/*/logs/journal.json}
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || true
[ -f .env ] && set -a && . ./.env && set +a

RUN_ID="${1:-${MLEVAL_RUN_ID:-mvp-001}}"
TASK="${2:-gsm8k}"
EXPECT_DIGEST="${3:-}"
SEEDS="${4:-0 1}"          # space-separated seed list (was hardcoded to s0)
NS="${K8S_NAMESPACE:-ecepxie}"

echo "================================================================"
echo " run-health monitor   run_id=$RUN_ID  task=$TASK  ns=$NS"
echo "================================================================"

# ---- TIER 0: deployment (host-side kubectl) --------------------------------
RUNNING_POD=""
for s in $SEEDS; do
for cell in with-skill without-skill; do
  traj="${RUN_ID}-${TASK}-${cell}-s${s}"
  pod=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | awk -v j="$traj" '$1 ~ j {print $1; exit}')
  echo
  echo "### CELL: $cell   ($traj)"
  if [ -z "$pod" ]; then echo "  [WARN] no pod found (not scheduled yet, or job missing)"; continue; fi
  phase=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
  waiting=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null)
  restarts=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  digest=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].imageID}' 2>/dev/null)
  echo "  pod=$pod  phase=${phase:-?}  restarts=${restarts:-0}  waiting=${waiting:-none}"
  case "$waiting" in
    ImagePullBackOff|ErrImagePull) echo "  [KILL] 0.1 image pull failing ($waiting) — check ghcr-pull secret / GHCR_READ_TOKEN";;
    CrashLoopBackOff)              echo "  [KILL] 0.1 CrashLoopBackOff — entrypoint failing";;
  esac
  [ "$phase" = "Running" ] && { echo "  [OK]   0.1 pod Running"; [ -z "$RUNNING_POD" ] && RUNNING_POD="$pod"; }
  [ "${restarts:-0}" != "0" ] && echo "  [WARN] 0.2 restarts=${restarts}"
  if [ -n "$EXPECT_DIGEST" ]; then
    case "$digest" in *"$EXPECT_DIGEST"*) echo "  [OK]   0.3 image digest matches";; *) echo "  [WARN] 0.3 image digest ${digest##*@} != expected $EXPECT_DIGEST (stale cache?)";; esac
  fi
  banner=$(kubectl -n "$NS" logs "$pod" --tail=40 2>/dev/null | grep -E '\[entrypoint\]')
  if echo "$banner" | grep -q "task=$TASK cell="; then
    echo "  [OK]   0.4 entrypoint banner present"
    echo "$banner" | grep -E 'wall_cap|metric direction|skill library' | sed 's/^/         /'
  else
    echo "  [WARN] 0.4 no entrypoint banner yet (too early, or entrypoint died)"
  fi
done
done

# ---- TIER 1-4: PVC-side (exec into a running pod; shared PVC sees both cells)
if [ -z "$RUNNING_POD" ]; then
  echo; echo "No Running pod to read the PVC from — Tier 1-4 skipped (re-run when a pod is Running)."
  exit 0
fi
echo; echo "----------------------------------------------------------------"
echo " Tier 1-4 (PVC via exec into $RUNNING_POD)"
echo "----------------------------------------------------------------"

PROBE="$(mktemp -t runhealth.XXXX.py)"
cat > "$PROBE" <<'PY'
import sys, os, glob, json, re, collections
RUN, TASK = sys.argv[1], sys.argv[2]
SEEDS = sys.argv[3].split() if len(sys.argv) > 3 else ["0"]
def log(v, c, m): print(f"  [{v}] {c} {m}")
for s in SEEDS:
 for cell in ("with-skill","without-skill"):
    traj=f"{RUN}-{TASK}-{cell}-s{s}"
    base=f"/results/{RUN}/{traj}"
    print(f"\n### CELL: {cell}-s{s}")
    if not os.path.isdir(base): print("  (no PVC dir yet)"); continue
    mlog = (glob.glob(f"{base}/mlevolve_runs/*/logs/MLEvolve.log") or [None])[0]
    jpath = (glob.glob(f"{base}/mlevolve_runs/*/logs/journal.json") or [None])[0]
    prompts = f"{base}/prompts.jsonl"
    logtxt = open(mlog,errors="ignore").read() if mlog and os.path.exists(mlog) else ""

    # 1.1 instruction integrity: judge the WHOLE cleaned task desc (printed after
    # "Generating Task desc:") by task markers + length — NOT the first line (the
    # instruction legitimately starts with a short '<!--' provenance comment).
    gi = logtxt.find("Generating Task desc:")
    if not logtxt:
        log("PEND","1.1","MLEvolve.log not present yet (pre-init)")
    elif gi == -1:
        log("PEND","1.1","task-desc not generated yet")
    else:
        after = logtxt[gi+len("Generating Task desc:"):]
        nxt = re.search(r"\n\[\d{4}-\d\d-\d\d ", after)       # next timestamped log line ends the desc
        desc = after[:nxt.start()] if nxt else after[:6000]
        markers = sum(k in desc for k in ("## Description","## Model","Qwen","chain-of-thought","#### "))
        L = len(desc.strip())
        if markers == 0 and L < 50:
            log("KILL","1.1",f"instruction GUTTED (~{L} chars, 0 task markers) -> agent will mis-frame [clean_task_desc bug]")
        elif markers == 0:
            log("WARN","1.1",f"no task markers in cleaned desc (~{L} chars) — verify instruction reached agent")
        else:
            log("OK","1.1",f"instruction intact ({markers} markers, ~{L} chars) -> determinism fix holding")

    # 1.2 skill injection (skill cell only)
    if cell == "with-skill":
        if not os.path.exists(prompts):
            log("PEND","1.2","prompts.jsonl not present yet")
        else:
            pt=open(prompts,errors="ignore").read()
            inj = sum(pt.count(k) for k in ("lora-methods.md","offline-inference","get_peft_model"))
            sel = re.findall(r"\[skill_injector\].*selected=\[(.*?)\]", logtxt)
            nonempty = [s for s in sel if s.strip()]
            if inj == 0 and not sel:
                log("PEND","1.2","no skill content in a prompt yet (pre-node)")
            elif inj == 0:
                log("KILL","1.2","skill cell but NO skill content in prompts (catalog/bodies absent)")
            elif sel and not nonempty:
                log("KILL","1.2",f"selector fired but ALL selections EMPTY (x{len(sel)}) -> treatment=baseline [spike-023]")
            else:
                log("OK","1.2",f"skills injected (body-marker hits={inj}; selector lines={len(sel)}, non-empty={len(nonempty)})")

    if not jpath or not os.path.exists(jpath):
        log("PEND","2-3","journal.json not present yet (no nodes)"); continue
    nodes=sorted(json.load(open(jpath)).get("nodes",[]), key=lambda x:x.get("step",-1))
    code=lambda n:(n.get("code") or "")
    term=lambda n:("".join(n.get("_term_out")) if isinstance(n.get("_term_out"),list) else (n.get("_term_out") or ""))
    drafts=[n for n in nodes if n.get("stage")=="draft"]

    # 2.1 first-draft framing
    if drafts:
        d=drafts[0]; c=code(d)
        causal = bool(re.search(r"AutoModelForCausalLM",c))
        gen = bool(re.search(r"\.generate\(|SamplingParams",c))
        tabular = bool(re.search(r"TfidfVectorizer|RandomForest|XGB|lightgbm|LGBM|GradientBoost|MSELoss|mean_squared_error",c))
        if tabular and not causal:
            log("KILL","2.1","first draft is TABULAR-REGRESSION (no causal LM) -> wrong task framing")
        elif causal and (gen or "LoraConfig" in c):
            log("OK","2.1","first draft uses causal LM + LoRA/generate (correct framing)")
        else:
            log("WARN","2.1","first draft framing unclear (no causal LM and no tabular signature)")
        # 2.2 PEFT discovery (informational; matters since the LoRA nudge was
        # removed from instruction.md — the agent should reach for it itself).
        lora = bool(re.search(r"LoraConfig|get_peft_model|prepare_model_for_kbit",c))
        quant = "4bit" if re.search(r"load_in_4bit\s*=\s*True",c) else ("8bit" if re.search(r"load_in_8bit\s*=\s*True",c) else "fp16/bf16")
        log("OK","2.2",f"PEFT choice (de-nudged, informational): lora={lora} quant={quant}")

    # 3.1-3.4 node health
    alltext="\n".join(term(n) for n in nodes)
    if re.search(r"Segmentation fault|exit code 139|SIGSEGV",alltext): log("KILL","3.1","fork-after-CUDA SEGFAULT detected")
    else: log("OK","3.1","no segfault")
    if re.search(r"^=======|<<<<<<<|>>>>>>>",alltext,re.M): log("KILL","3.2","diff-applier corruption markers in node output")
    else: log("OK","3.2","no diff corruption")
    excs=[(str(n.get("exc_type")),str((n.get("exc_info") or {}).get("message",""))[:60]) for n in nodes if str(n.get("exc_type"))!="None"]
    sig=collections.Counter(e for e in excs)
    worst=sig.most_common(1)
    if worst and worst[0][1]>=3: log("WARN","3.3",f"recurring crash x{worst[0][1]}: {worst[0][0][0]} {worst[0][0][1]!r} -> may not converge")
    else: log("OK","3.3","no single error dominates early nodes")
    nto=sum(1 for n in nodes if str(n.get("exc_type"))=="TimeoutError")
    done=[n for n in nodes if n.get("step",0)>=1]
    if done and nto >= max(3, len(done)*0.6): log("WARN","3.4",f"timeout-dominated ({nto}/{len(done)}) -> approach too heavy for per-exec cap")
    else: log("OK","3.4",f"timeouts {nto}/{len(done)}")
    nonbuggy=[n for n in nodes if n.get("is_buggy") is False]
    maxstep=max((n.get("step",0) for n in nodes), default=0)
    if maxstep>=6 and not nonbuggy: log("WARN","3.5",f"no non-buggy node by step {maxstep} (best likely None) -> low yield")
    else: log("OK","3.5",f"steps={maxstep}, non-buggy nodes={len(nonbuggy)}")

    # 4.1-4.2 silent correctness
    if re.search(r"right-padding was detected",alltext): log("WARN","4.1","RIGHT-padding on decoder-only generation -> garbage generations (need padding_side=left)")
    miss=set(re.findall(r"No module named '([^']+)'",alltext))
    if miss: log("WARN","4.2",f"agent requested absent deps: {sorted(miss)}")
PY
kubectl -n "$NS" exec -i "$RUNNING_POD" -- python3 - "$RUN_ID" "$TASK" "$SEEDS" < "$PROBE"
rm -f "$PROBE"
echo; echo "done."
