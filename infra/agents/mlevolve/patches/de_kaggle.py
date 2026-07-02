"""Build-time de-Kaggle patch for the vendored MLEvolve upstream.

WHY: MLEvolve is a Kaggle/MLE-bench agent. Its prompts hardcode a
"Kaggle Grandmaster competing to WIN on a leaderboard" persona AND a
"data lives in ./input / read sample_submission.csv" framing. On our
contract-only tasks this derailed the agent into generic Kaggle tabular code
(`pd.read_csv("./input/train.csv")`, classification heads) → off-task drift.
File-staged tasks (gsm8k jsonl) and HF-loaded tasks (samsum) both defer to the
task description for the loader — see the INPUT-side patch below.

This patch neutralizes the *competition persona* and the *./input read*
framing image-wide so the agent stays on the contract. Two notes on the
OUTPUT side, which changed under the held-out-grader design:

  - We DO want the agent to emit a per-example `submission.csv` now. We run
    `no_submission_mode: False` so MLEvolve's native machinery preserves the
    best node's predictions at `best_submission/submission.csv`, which our
    independent held-out grader (``mleval.grader``) scores against held-out
    references — the trustworthy A/B metric. So this patch only neutralizes
    the Kaggle *persona* + *input-reading* nudges; it does NOT suppress
    submission writing (the output contract is carried by instruction.md +
    MLEvolve's own impl_guideline).
  - `no_submission_mode: False` makes result_parse run a format-validation
    path whose `exp_name.split("_")[2]` assumes an mle-bench competition id
    and IndexErrors on our exp_name; one rule below makes that split
    index-safe (the mle-bench grading call itself is already skipped via
    `use_grading_server: False`).

It edits the image copy at /workspace/mlevolve (upstream submodule files
are reset by `git submodule update` on every build, so we cannot edit them
durably; we patch the copied tree at build time instead).

Each REQUIRED replacement asserts it applied (count >= min) so an upstream
refactor FAILS THE BUILD loudly rather than silently leaving the nudge in.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("MLEVOLVE_ROOT", "/workspace/mlevolve"))
AGENTS = ROOT / "agents"

NEUTRAL_PERSONA = "You are an expert ML engineer implementing exactly the task described below"

# GPU fork-after-CUDA safety preamble, prepended to EVERY executed node
# (engine/executor.py). Forces DataLoader(num_workers=0): num_workers>0 forks
# worker processes AFTER the model has initialized CUDA (device_map="auto" /
# .to("cuda")). fork-after-CUDA is not safe and SIGSEGVs (exit 139), which kills
# the whole trajectory — spike-018 lost both reruns this way, x3 attempts each.
# The subprocess boundary doesn't contain it on a shared single GPU and we have
# no container-in-container isolation on Nautilus, so we neutralize the trigger
# at execution time. Wrapped in try/except so non-torch tasks are unaffected.
# The \n are written literally into executor.py (they become real newlines in
# the executed node string). Applies to BOTH cells -> no A/B confound.
_GPU_SAFETY = (
    "try:\\n"
    "    import torch.utils.data as _mtud\\n"
    "    _mo = _mtud.DataLoader.__init__\\n"
    "    def _mw(s, *a, **k):\\n"
    "        k['num_workers'] = 0\\n"
    "        k.pop('persistent_workers', None); k.pop('prefetch_factor', None)\\n"
    "        return _mo(s, *a, **k)\\n"
    "    _mtud.DataLoader.__init__ = _mw\\n"
    "except Exception:\\n"
    "    pass\\n"
)

# Determinism fix: BYPASS the LLM task-desc rewrite (utils/data_preview.py
# clean_task_desc). It piped instruction.md through an LLM ONCE per run, cached
# as self.task_desc for every node, with NO empty-output guard (only catches
# exceptions). On OpenRouter/DeepSeek temp=0 is not reproducible, so it
# occasionally returned "" and silently GUTTED the instruction (mvp-029
# with-skill: cleaned to 2 chars -> agent saw only the id,prediction stub ->
# tabular regressor, all nodes invalid; spike-025: hallucinated "Unihandecode
# Ecosphere"). Our instruction.md is authored clean (docs/eval/task-authoring.md),
# so the rewrite buys nothing and only adds nondeterminism. Replace ONLY the
# query() block with `cleaned_desc = task_desc`; the submission_format append
# that follows is untouched -> task_desc is deterministic & identical across
# cells/runs. (data_preview.py is NOT under agents/, hence an explicit path.)
_CLEAN_TASK_DESC_OLD = '''    try:
        cleaned_desc = query(
            system_message=prompt,
            user_message=None,
            model=acfg.code.model,
            temperature=0.0,
            cfg=cfg
        )
        logger.info(f"Task description cleaned for code review")
        cleaned_desc = cleaned_desc.strip()
    except Exception as e:
        logger.warning(f"Failed to clean task_desc with LLM: {e}. Using original.")
        cleaned_desc = task_desc'''

_CLEAN_TASK_DESC_NEW = '''    # de_kaggle: BYPASS the nondeterministic LLM task-desc rewrite (see de_kaggle.py
    # note). Use the already-clean instruction.md verbatim; submission_format is
    # still appended below.
    cleaned_desc = task_desc'''

# (path-relative-to-ROOT, old, new, required, min_count)
RULES = [
    # --- recurring competition persona across draft/improve/evolution/fusion/
    #     aggregation/planner/result_parse/stepwise (global substring) ---
    ("__GLOB_AGENTS__", "You are a Kaggle grandmaster attending a competition",
     NEUTRAL_PERSONA, True, 5),
    # --- draft_agent.py aggressive competition block (the code-gen path) ---
    ("agents/draft_agent.py",
     "\U0001f3c6 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.",
     "You are an expert ML engineer. Implement exactly the task described below.", True, 1),
    ("agents/draft_agent.py", "Compete for TOP performance, not trivial baselines",
     "Aim for strong performance on the task's stated metric, not trivial baselines", True, 1),
    ("agents/draft_agent.py",
     "Your solution will be evaluated on a real leaderboard. Treat this with professionalism.",
     "Your solution will be evaluated by the task's stated metric. Treat this with professionalism.", True, 1),
    ("agents/draft_agent.py", "Now, let's begin the competition.", "Now, let's begin.", True, 1),
    ("agents/draft_agent.py", "with the quality expected of a Kaggle Grandmaster",
     "with the quality expected of an expert ML engineer", True, 1),
    # --- INPUT-side data-location nudge ---
    # Task-description-driven: gsm8k reads staged jsonl under ./input/; samsum
    # uses datasets.load_dataset per instruction. Do NOT steer all tasks toward HF
    # or away from ./input/ — that contradicted file-staged tasks (spike-028).
    ("agents/draft_agent.py",
     "- The data is already prepared in `./input` directory. No need to unzip files.",
     "- Follow the dataset loader and file paths named in the task description exactly. "
     "When the task lists local files under `./input/` (e.g. `.jsonl`, `.csv`), read those files. "
     "When the task names a HuggingFace dataset slug, use `datasets.load_dataset(...)` as specified. "
     "Do not assume MLE-Bench CSV layouts (`train.csv`, `test.csv`) unless the task describes them.", True, 1),
    # --- OUTPUT-side submission-format warning (injected into EVERY prompt
    #     via update_data_preview) ---
    ("engine/agent_search.py", "self.data_preview = base_preview + submission_format_warning",
     "self.data_preview = base_preview  # de_kaggle: dropped submission_format_warning", True, 1),
    # --- no_submission_mode:False compatibility ---
    # We run with no_submission_mode:False so MLEvolve natively preserves each
    # node's predictions (submission/submission_<id>.csv) and the best node's
    # (best_submission/submission.csv) for our held-out grader. That path runs
    # _validate_format_with_retry / _validate_format_simple, which start with
    #   exp_id = agent.cfg.exp_name.split("_")[2]
    # assuming an mle-bench competition id. Our exp_name has no such underscore
    # structure, so [2] IndexErrors BEFORE the use_grading_server=False skip
    # (quality_check.py:219). exp_id is unused once validation is skipped, so
    # make the split index-safe. Two call sites (lines ~223 in the live
    # _validate_format_with_retry, and ~260 in _validate_format_simple which
    # is currently dead code but patched too so it can never regress).
    ("agents/result_parse_agent.py",
     'exp_id = agent.cfg.exp_name.split("_")[2]',
     'exp_id = (agent.cfg.exp_name.split("_") + ["", "", ""])[2]', True, 2),
    # --- time-budget signal accuracy ---
    # Upstream hardcodes a "9 hours" execution budget in TWO agent-facing
    # prompts, which contradicts the real per-exec cap (config.exec.timeout via
    # MLEVAL_EXEC_TIMEOUT_SEC) and the accurate dynamic line in impl_guideline
    # ("Max execution time per run = {naturaldelta(exec_timeout)}"). The agent
    # believes it has 9h and plans a full-epoch run, then gets SIGKILLed at the
    # real cap (spike-014: every node TimedOut). MLE-bench's own scaffold
    # (its agent additional-notes file) parameterizes this as ${TIME_LIMIT} and
    # frames it as "program runtime counts toward this limit" — never a hardcoded
    # number. Rewrite both to reference the real per-run limit instead of "9h".
    ("agents/prompts/impl_guideline.py", "9 hours (hard limit)",
     "the per-run execution time limit shown above (both training AND the final "
     "evaluation must finish within it; program runtime counts toward it)", True, 1),
    ("agents/prompts/validation_template_prompts.py", "9 hours available",
     "bounded by the per-run execution time limit (training + evaluation must finish within it)", True, 1),
    # --- fork-after-CUDA segfault, prompt side ---
    # Upstream impl_guideline tells the agent "Use DataLoader with num_workers>=2
    # for speed" — inherited from the prior agent's CPU-tabular origin. On our single-GPU
    # task that induces the fork-after-CUDA SIGSEGV (exit 139) that killed
    # spike-018's reruns. Steer to the safe pattern (the harness also enforces
    # num_workers=0 via _GPU_SAFETY below, but don't actively recommend the crash).
    ("agents/prompts/impl_guideline.py", "Use DataLoader with num_workers>=2 for speed",
     "Use DataLoader(num_workers=0) when the model is on GPU (num_workers>0 forks "
     "after CUDA init and crashes the run; the harness enforces num_workers=0)", True, 1),
    # --- fork-after-CUDA segfault, execution side (the real guard) ---
    # Prepend _GPU_SAFETY to every executed node so DataLoader cannot fork
    # workers after CUDA init regardless of what the agent wrote. See the
    # _GPU_SAFETY comment above. One call site: code = pre_code + code.
    ("engine/executor.py", "code = pre_code + code",
     'code = pre_code + "' + _GPU_SAFETY + '" + code', True, 1),
    # --- optional leftovers (best-effort; don't fail build if upstream shifts) ---
    ("agents/coder/stepwise_coder.py", "competition-winning Python code", "high-quality Python code", False, 0),
    ("agents/improve_agent.py", "As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance",
     "As an expert ML engineer, make MEANINGFUL improvements that boost the task's stated metric", False, 0),
    ("agents/improve_agent.py", "distilled from the kaggle award-winning solutions",
     "distilled from strong ML solutions", False, 0),
    # --- DETERMINISM: bypass the clean_task_desc LLM rewrite (see note above).
    #     Required: if upstream refactors this block, FAIL the build loudly so the
    #     nondeterministic task-desc rewrite can never silently come back. ---
    ("utils/data_preview.py", _CLEAN_TASK_DESC_OLD, _CLEAN_TASK_DESC_NEW, True, 1),
]


def _apply(path: pathlib.Path, old: str, new: str) -> int:
    text = path.read_text()
    n = text.count(old)
    if n:
        path.write_text(text.replace(old, new))
    return n


def main() -> int:
    failures = []
    for rel, old, new, required, min_count in RULES:
        if rel == "__GLOB_AGENTS__":
            total = sum(_apply(p, old, new) for p in AGENTS.rglob("*.py"))
            label = "agents/**/*.py"
        else:
            p = ROOT / rel
            total = _apply(p, old, new) if p.exists() else 0
            label = rel
        status = "OK" if total >= min_count else ("MISSING" if required else "skip")
        print(f"[de_kaggle] {status:7} {total:>2}x  {label}  <- {old[:48]!r}")
        if required and total < min_count:
            failures.append(f"{label}: expected >= {min_count}, got {total} for {old[:60]!r}")
    if failures:
        print("\n[de_kaggle] FAILED — upstream strings drifted; update patches/de_kaggle.py:")
        for f in failures:
            print("  - " + f)
        return 1
    print("[de_kaggle] all required Kaggle-framing nudges neutralized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
