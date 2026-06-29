"""Prepare BoolQ for an MLE-Bench-style held-out run.

Run ONCE on a pod that has `datasets` + network (e.g. the helper pod):

    python3 make_grading_data.py --out-root /results/data/boolq

Produces the MLE-Bench split shape (docs/eval/task-authoring.md §C3):

  AGENT-FACING (mounted into the trajectory's input dir):
    <out-root>/data/train.jsonl          question + passage + answer  (9,427 — train, labels OK)
    <out-root>/data/test.jsonl           id + question + passage ONLY (3,270 — targets WITHHELD)
    <out-root>/data/sample_submission.csv  id,prediction              (format + id-set, empty preds)

  PRIVATE (graded post-run; NEVER in the agent's input dir):
    <out-root>/refs/test_refs.csv        id,reference_answer ("true"/"false")

Why `validation` becomes the test: BoolQ ships NO labelled `test` split, only
`train` (9,427) and `validation` (3,270). We use `validation` as the withheld
test — its labels are stripped from the agent-facing `test.jsonl` and live only
in `refs/`. (BoolQ's validation labels ARE public on HF, so this is not true
secrecy; the binding control is the harness "do not download from the internet,
use only the provided files" rule + the staged-file design — same posture as
GSM8K. It removes the casual re-label leak and gives a deterministic id.)

BoolQ has NO native id field — the submission `id` is the **0-based row index in
the validation split** (enumerate order), fixed/deterministic, so this generator
and the agent (iterating `test.jsonl` in file order) agree. The gold `answer`
is a Python bool; we write it canonically as "true"/"false".

Deterministic: rows are written in split order (id = 0..N-1), so files are
byte-stable across runs. Keep in lockstep with mleval.grader._TASKS
("boolq": accuracy on id/prediction vs id/reference_answer).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASET = "google/boolq"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/results/data/boolq"))
    args = ap.parse_args()

    from datasets import load_dataset  # lazy; only needed at gen time

    ds = load_dataset(DATASET)
    train, test = ds["train"], ds["validation"]  # validation = withheld test

    data_dir = args.out_root / "data"
    refs_dir = args.out_root / "refs"
    data_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    # --- agent-facing: train (with answers) ---------------------------------
    train_path = data_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps({
                "question": ex["question"],
                "passage": ex["passage"],
                "answer": bool(ex["answer"]),
            }) + "\n")

    # --- agent-facing: test (id + question + passage ONLY; answer withheld) --
    test_path = data_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            f.write(json.dumps({
                "id": str(i),
                "question": ex["question"],
                "passage": ex["passage"],
            }) + "\n")

    # --- agent-facing: sample_submission (format + id-set) ------------------
    sample_path = data_dir / "sample_submission.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        w.writerows((str(i), "") for i in range(len(test)))

    # --- PRIVATE: held-out references (graded post-run) ---------------------
    refs_path = refs_dir / "test_refs.csv"
    rows = [(str(i), "true" if bool(ex["answer"]) else "false") for i, ex in enumerate(test)]
    with refs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference_answer"])
        w.writerows(rows)

    n_true = sum(1 for _, g in rows if g == "true")
    print(f"[make_grading_data] train={len(train)}  test={len(test)} ({n_true} true / {len(test) - n_true} false)")
    print("  agent-facing:")
    print(f"    {train_path}")
    print(f"    {test_path}        (no answers)")
    print(f"    {sample_path}")
    print("  PRIVATE (held out from agent):")
    print(f"    {refs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
