"""Prepare GSM8K for an MLE-Bench-style held-out run.

Run ONCE on a pod that has `datasets` + network (e.g. the helper pod):

    python3 make_grading_data.py --out-root /results/data/gsm8k

Produces the MLE-Bench split shape (docs/eval/task-authoring.md §C3):

  AGENT-FACING (mounted into the trajectory's input dir):
    <out-root>/data/train.jsonl          question + answer   (7,473 — train, labels OK)
    <out-root>/data/test.jsonl           id + question ONLY  (1,319 — targets WITHHELD)
    <out-root>/data/sample_submission.csv  id,prediction     (format + id-set, empty preds)

  PRIVATE (graded against post-run; NEVER in the agent's input dir):
    <out-root>/refs/test_refs.csv        id,reference_answer (held-out gold)

The agent never sees the test targets: `test.jsonl` carries only `id`+`question`,
and the gold numbers live in `refs/` (entrypoint sets MLEVAL_TASK_REFS_PATH to
<out-root>/refs/test_refs.csv). The search signal must come from a validation
slice the agent carves out of `train.jsonl` (see instruction.md), not from test.

GSM8K (`openai/gsm8k`, config `main`) has NO native id field. The submission
`id` is the **0-based row index in the test split** (enumerate order), which is
fixed/deterministic — so this generator and the agent (iterating `test.jsonl` in
file order) agree. The gold number is the token after the final `#### ` in
`answer`, with thousands-commas stripped.

Deterministic: rows are written in split order (id = 0..N-1), so files are
byte-stable across runs. Keep in lockstep with mleval.grader._TASKS
("gsm8k": exact_match on id/prediction vs id/reference_answer).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASET = "openai/gsm8k"
CONFIG = "main"


def _gold_number(answer: str) -> str:
    """Final number after the last '#### ', commas stripped (else '')."""
    if answer is None or "####" not in answer:
        return ""
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/results/data/gsm8k"))
    args = ap.parse_args()

    from datasets import load_dataset  # lazy; only needed at gen time

    ds = load_dataset(DATASET, CONFIG)
    train, test = ds["train"], ds["test"]

    data_dir = args.out_root / "data"
    refs_dir = args.out_root / "refs"
    data_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    # --- agent-facing: train (with answers) ---------------------------------
    train_path = data_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps({"question": ex["question"], "answer": ex["answer"]}) + "\n")

    # --- agent-facing: test (id + question ONLY; targets withheld) ----------
    test_path = data_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            f.write(json.dumps({"id": str(i), "question": ex["question"]}) + "\n")

    # --- agent-facing: sample_submission (format + id-set) ------------------
    sample_path = data_dir / "sample_submission.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        w.writerows((str(i), "") for i in range(len(test)))

    # --- PRIVATE: held-out references (graded against post-run) -------------
    refs_path = refs_dir / "test_refs.csv"
    rows = [(str(i), _gold_number(ex["answer"])) for i, ex in enumerate(test)]
    with refs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference_answer"])
        w.writerows(rows)

    blank = sum(1 for _, g in rows if g == "")
    print(f"[make_grading_data] train={len(train)}  test={len(test)} ({blank} unparseable gold)")
    print("  agent-facing:")
    print(f"    {train_path}")
    print(f"    {test_path}        (no answers)")
    print(f"    {sample_path}")
    print("  PRIVATE (held out from agent):")
    print(f"    {refs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
