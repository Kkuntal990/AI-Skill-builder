"""Tests for the held-out grader (mleval.grader).

Covers three layers:
  1. ROUGE-L math — hand-computed values, LCS order-sensitivity, edge cases.
  2. Validation gates — id-set mismatch, duplicates, missing columns, empties.
  3. CLI — artifact auto-location, held_out_score.json contents, the
     never-raise / always-exit-0 contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mleval.grader import grade_predictions, rouge_l_f, tokenize
from mleval.grader.__main__ import main, run
from mleval.grader.rouge import _lcs_length, mean_rouge_l_f


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)]
    for r in rows:
        # quote every field to keep commas inside summaries intact
        lines.append(",".join('"' + c.replace('"', '""') + '"' for c in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run_dir(tmp_path: Path, pred_rows: list[list[str]]) -> Path:
    """Build OUT_DIR/mlevolve_runs/<ts>/workspace/best_submission/submission.csv."""
    out_dir = tmp_path / "out"
    sub = out_dir / "mlevolve_runs" / "20260603_120000_exp" / "workspace" / "best_submission"
    sub.mkdir(parents=True)
    _write_csv(sub / "submission.csv", ["id", "generated_summary"], pred_rows)
    return out_dir


# --------------------------------------------------------------------------
# 1. ROUGE-L math
# --------------------------------------------------------------------------
def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, WORLD! 42") == ["hello", "world", "42"]


def test_tokenize_non_string_is_empty():
    assert tokenize(None) == []  # type: ignore[arg-type]
    assert tokenize(3.14) == []  # type: ignore[arg-type]


def test_rouge_identical_is_one():
    assert rouge_l_f("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_rouge_disjoint_is_zero():
    assert rouge_l_f("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_empty_is_zero():
    assert rouge_l_f("", "anything here") == 0.0
    assert rouge_l_f("anything here", "") == 0.0
    assert rouge_l_f("", "") == 0.0


def test_rouge_partial_known_value():
    # cand=[the,cat,sat] (3), ref=[the,cat,sat,on,the,mat] (6), LCS=3
    # P=3/3=1.0, R=3/6=0.5, F = 2*1*0.5/(1.5) = 0.6667
    assert rouge_l_f("the cat sat", "the cat sat on the mat") == pytest.approx(2 / 3)


def test_rouge_is_subsequence_not_substring():
    # cand=[cat,the,sat], ref=[the,cat,sat]; LCS=2 ("cat sat" or "the sat")
    # P=2/3, R=2/3, F=2/3
    assert rouge_l_f("cat the sat", "the cat sat") == pytest.approx(2 / 3)


def test_lcs_length_basic():
    assert _lcs_length(["a", "b", "c", "d"], ["b", "d"]) == 2
    assert _lcs_length(["a", "b", "c"], []) == 0
    assert _lcs_length(["x"], ["x"]) == 1


def test_mean_rouge():
    assert mean_rouge_l_f([]) == 0.0
    assert mean_rouge_l_f([("a b", "a b"), ("c", "z")]) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 2. grade_predictions validation gates
# --------------------------------------------------------------------------
def test_grade_perfect(tmp_path: Path):
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "the cat sat"], ["2", "a dog ran"]])
    _write_csv(preds, ["id", "generated_summary"], [["1", "the cat sat"], ["2", "a dog ran"]])
    res = grade_predictions(preds, refs)
    assert res.valid is True
    assert res.score == pytest.approx(1.0)
    assert res.n_scored == 2 and res.n_expected == 2


def test_grade_partial(tmp_path: Path):
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "the cat sat on the mat"]])
    _write_csv(preds, ["id", "generated_summary"], [["1", "the cat sat"]])
    res = grade_predictions(preds, refs)
    assert res.valid is True
    assert res.score == pytest.approx(2 / 3)


def test_grade_missing_id_is_invalid(tmp_path: Path):
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"], ["2", "y"]])
    _write_csv(preds, ["id", "generated_summary"], [["1", "x"]])  # missing id 2
    res = grade_predictions(preds, refs)
    assert res.valid is False and res.score is None
    assert any("missing" in e for e in res.errors)


def test_grade_extra_id_is_invalid(tmp_path: Path):
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"]])
    _write_csv(preds, ["id", "generated_summary"], [["1", "x"], ["99", "z"]])
    res = grade_predictions(preds, refs)
    assert res.valid is False and res.score is None
    assert any("unexpected" in e for e in res.errors)


def test_grade_duplicate_id_is_invalid(tmp_path: Path):
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"]])
    _write_csv(preds, ["id", "generated_summary"], [["1", "x"], ["1", "x again"]])
    res = grade_predictions(preds, refs)
    assert res.valid is False
    assert any("duplicate" in e for e in res.errors)


def test_grade_missing_column_is_invalid(tmp_path: Path):
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"]])
    _write_csv(preds, ["id", "wrong_col"], [["1", "x"]])
    res = grade_predictions(preds, refs)
    assert res.valid is False
    assert any("missing required columns" in e for e in res.errors)


def test_grade_empty_summary_present_scores_zero_but_valid(tmp_path: Path):
    # An empty prediction for a present id is NOT a missing row: it scores 0
    # ROUGE-L, the submission stays valid.
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "preds.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "the cat sat"], ["2", "a dog ran"]])
    _write_csv(preds, ["id", "generated_summary"], [["1", "the cat sat"], ["2", ""]])
    res = grade_predictions(preds, refs)
    assert res.valid is True
    assert res.score == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_grade_references_unreadable_is_invalid(tmp_path: Path):
    preds = tmp_path / "preds.csv"
    _write_csv(preds, ["id", "generated_summary"], [["1", "x"]])
    res = grade_predictions(preds, tmp_path / "does_not_exist.csv")
    assert res.valid is False and res.score is None


# --------------------------------------------------------------------------
# 3. CLI
# --------------------------------------------------------------------------
def test_cli_auto_locates_and_writes_json(tmp_path: Path):
    out_dir = _make_run_dir(tmp_path, [["1", "the cat sat"], ["2", "a dog ran"]])
    refs = tmp_path / "refs.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "the cat sat"], ["2", "a dog ran"]])
    out_json = out_dir / "held_out_score.json"

    payload = run(out_dir, "samsum", refs, out_json)
    assert payload["valid"] is True
    assert payload["score"] == pytest.approx(1.0)
    assert payload["source"] == "held_out_grader"
    assert payload["metric"] == "rougeL_f"
    # written to disk identically
    on_disk = json.loads(out_json.read_text())
    assert on_disk["score"] == pytest.approx(1.0)
    assert on_disk["predictions_path"].endswith("best_submission/submission.csv")


def test_cli_no_artifact_is_invalid_but_exit_zero(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    refs = tmp_path / "refs.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"]])
    rc = main([str(out_dir), "--task", "samsum", "--refs", str(refs)])
    assert rc == 0  # never breaks the entrypoint
    payload = json.loads((out_dir / "held_out_score.json").read_text())
    assert payload["valid"] is False and payload["score"] is None
    assert "no best_submission" in payload["error"]


def test_cli_pred_override_and_drift_scores_invalid(tmp_path: Path):
    # Simulate an off-task (IMDB) submission: ids don't match the SAMSum
    # test set → the held-out grader rejects it (the whole point).
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    refs = tmp_path / "refs.csv"
    preds = tmp_path / "imdb_like.csv"
    _write_csv(refs, ["id", "reference_summary"], [["s1", "the cat sat"], ["s2", "a dog ran"]])
    _write_csv(preds, ["id", "generated_summary"], [["0", "positive"], ["1", "negative"]])
    rc = main([str(out_dir), "--task", "samsum", "--refs", str(refs), "--pred", str(preds)])
    assert rc == 0
    payload = json.loads((out_dir / "held_out_score.json").read_text())
    assert payload["valid"] is False and payload["score"] is None


def test_cli_never_raises_on_unwritable_out_path(tmp_path: Path):
    # A grading hiccup (here: --out under a regular file → NotADirectoryError on
    # mkdir) must NOT propagate; the entrypoint relies on exit 0.
    out_dir = _make_run_dir(tmp_path, [["1", "x"]])
    refs = tmp_path / "refs.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"]])
    a_file = tmp_path / "afile"
    a_file.write_text("not a dir")
    bad_out = a_file / "sub" / "score.json"
    rc = main([str(out_dir), "--task", "samsum", "--refs", str(refs), "--out", str(bad_out)])
    assert rc == 0


def test_cli_unknown_task_is_invalid(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    refs = tmp_path / "refs.csv"
    _write_csv(refs, ["id", "reference_summary"], [["1", "x"]])
    rc = main([str(out_dir), "--task", "nope", "--refs", str(refs)])
    assert rc == 0
    payload = json.loads((out_dir / "held_out_score.json").read_text())
    assert payload["valid"] is False
    assert "unknown task" in payload["error"]
