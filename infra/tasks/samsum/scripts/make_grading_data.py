"""Generate SAMSum grading data: held-out references + a sample submission.

Run ONCE on a pod that has `datasets` + network (e.g. the helper pod),
writing into the PVC task tree:

    python3 make_grading_data.py --out-root /results/data/samsum

Produces (BOTH under refs/, intentionally OUTSIDE data/):
    <out-root>/refs/test_refs.csv          id,reference_summary   (held-out; graded against)
    <out-root>/refs/sample_submission.csv  id,generated_summary   (schema reference only)

Nothing is written under `data/`, so the entrypoint (which symlinks
`data/*` into the agent's `./input`) keeps `./input` empty — consistent with
the de_kaggle patch that tells the agent "`./input` may be EMPTY; do NOT read
CSVs from `./input`". The agent gets the submission schema from
`instruction.md` and derives the test id set from the HF `test` split itself;
it never needs these files. They exist for OUR grader (test_refs.csv) and as a
human schema reference (sample_submission.csv). The references are public on
HF anyway — the protection is that WE recompute the metric, not secrecy.

Deterministic: rows sorted by id, so the file is byte-stable across runs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATASET = "knkarthick/samsum"
SPLIT = "test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/results/data/samsum"))
    args = ap.parse_args()

    from datasets import load_dataset  # imported lazily; only needed at gen time

    ds = load_dataset(DATASET)[SPLIT]
    rows = sorted(
        ((str(ex["id"]), (ex["summary"] or "").replace("\n", " ").strip()) for ex in ds),
        key=lambda t: t[0],
    )

    refs_dir = args.out_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    refs_path = refs_dir / "test_refs.csv"
    with refs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference_summary"])
        w.writerows(rows)

    sample_path = refs_dir / "sample_submission.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "generated_summary"])
        # placeholder summaries (empty) — schema + id list only
        w.writerows((rid, "") for rid, _ in rows)

    print(f"[make_grading_data] wrote {len(rows)} rows")
    print(f"  refs:   {refs_path}")
    print(f"  sample: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
