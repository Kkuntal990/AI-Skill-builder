"""Generate GSM8K grading data: held-out references + a sample submission.

Run ONCE on a pod that has `datasets` + network (e.g. the helper pod), writing
into the PVC task tree:

    python3 make_grading_data.py --out-root /results/data/gsm8k

Produces (BOTH under refs/, intentionally OUTSIDE data/):
    <out-root>/refs/test_refs.csv          id,reference_answer   (held-out; graded against)
    <out-root>/refs/sample_submission.csv  id,prediction         (schema reference only)

GSM8K (`openai/gsm8k`, config `main`) has NO native id field — only
`question` + `answer`. The submission `id` is therefore the **0-based row index
of the example in the `test` split** (enumerate order). load_dataset returns the
split in a fixed order, so this index is deterministic and reproducible by both
the agent (iterating the same split) and this generator. The gold number is the
token after the final `#### ` in `answer`, with thousands-commas stripped.

Deterministic: rows are written in test-split order (id = 0..N-1), so the file
is byte-stable across runs. Keep this in lockstep with mleval.grader._TASKS
("gsm8k": exact_match on id/prediction vs id/reference_answer).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATASET = "openai/gsm8k"
CONFIG = "main"
SPLIT = "test"


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

    ds = load_dataset(DATASET, CONFIG)[SPLIT]
    # id = 0-based position in the test split (GSM8K has no native id).
    rows = [(str(i), _gold_number(ex["answer"])) for i, ex in enumerate(ds)]

    refs_dir = args.out_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    refs_path = refs_dir / "test_refs.csv"
    with refs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference_answer"])
        w.writerows(rows)

    sample_path = refs_dir / "sample_submission.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        w.writerows((rid, "") for rid, _ in rows)

    blank = sum(1 for _, g in rows if g == "")
    print(f"[make_grading_data] wrote {len(rows)} rows ({blank} with no parseable gold)")
    print(f"  refs:   {refs_path}")
    print(f"  sample: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
