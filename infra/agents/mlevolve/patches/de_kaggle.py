"""Build-time de-Kaggle patch for the vendored MLEvolve upstream.

WHY: MLEvolve is a Kaggle/MLE-bench agent. Its prompts hardcode a
"Kaggle Grandmaster competing to WIN on a leaderboard" persona AND a
"data lives in ./input" + "use sample_submission.csv" framing. On our
contract-only tasks (SAMSum etc., HF-loaded, no submission file) this
derails the agent into writing generic Kaggle tabular code
(`pd.read_csv("./input/train.csv")`, `submission.csv`, generic NN) →
FileNotFoundError, never reaching the actual task. `no_submission_mode`
only gates the OUTPUT-side validation, not these prompt-level nudges.

This patch neutralizes the competition framing image-wide. We run ONLY
no-submission tasks, so it's safe to apply unconditionally. It edits the
image copy at /workspace/mlevolve (upstream submodule files are reset by
`git submodule update` on every build, so we cannot edit them durably;
we patch the copied tree at build time instead).

Each REQUIRED replacement asserts it applied (count >= min) so an upstream
refactor FAILS THE BUILD loudly rather than silently leaving the nudge in.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("MLEVOLVE_ROOT", "/workspace/mlevolve"))
AGENTS = ROOT / "agents"

NEUTRAL_PERSONA = "You are an expert ML engineer implementing exactly the task described below"

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
    ("agents/draft_agent.py",
     "- The data is already prepared in `./input` directory. No need to unzip files.",
     "- Load the dataset using the loader named in the task description (e.g. a HuggingFace `datasets.load_dataset(...)` call). The `./input` directory may be EMPTY — do NOT assume local train.csv/test.csv/sample_submission.csv exist; do NOT read CSVs from `./input`.", True, 1),
    # --- OUTPUT-side submission-format warning (injected into EVERY prompt
    #     via update_data_preview) ---
    ("engine/agent_search.py", "self.data_preview = base_preview + submission_format_warning",
     "self.data_preview = base_preview  # de_kaggle: dropped submission_format_warning", True, 1),
    # --- optional leftovers (best-effort; don't fail build if upstream shifts) ---
    ("agents/coder/stepwise_coder.py", "competition-winning Python code", "high-quality Python code", False, 0),
    ("agents/improve_agent.py", "As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance",
     "As an expert ML engineer, make MEANINGFUL improvements that boost the task's stated metric", False, 0),
    ("agents/improve_agent.py", "distilled from the kaggle award-winning solutions",
     "distilled from strong ML solutions", False, 0),
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
