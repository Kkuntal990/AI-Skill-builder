"""Tests for the format-only submission validator (MLE-Bench validate affordance).

Mirrors the two real spike-018 failure modes — a wrong column (``label``) and a
wrong id-set (hashed ids) — plus the happy path. Asserts the validator never
emits a score, only validity + structural reasons.
"""
from __future__ import annotations

from pathlib import Path

from mleval.grader.validate import main, validate_format

_REFS = "id,reference_summary\n13862856,a\n13611370,b\n13611413,c\n"


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_valid_submission(tmp_path):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv",
                 "id,generated_summary\n13862856,x\n13611370,y\n13611413,z\n")
    valid, errors = validate_format(sub, refs, id_col="id", pred_col="generated_summary")
    assert valid is True
    assert errors == []


def test_wrong_column_is_invalid(tmp_path):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv", "id,label\n13862856,0\n13611370,1\n13611413,2\n")
    valid, errors = validate_format(sub, refs, id_col="id", pred_col="generated_summary")
    assert valid is False
    assert any("missing required columns" in e for e in errors)


def test_hashed_ids_are_invalid(tmp_path):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv",
                 "id,generated_summary\nca8de9c0,x\ndeadbeef,y\nfeedface,z\n")
    valid, errors = validate_format(sub, refs, id_col="id", pred_col="generated_summary")
    assert valid is False
    assert any("missing" in e for e in errors)  # all real ids missing
    assert any("unexpected" in e for e in errors)  # hashed ids extra


def test_duplicate_id_is_invalid(tmp_path):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv",
                 "id,generated_summary\n13862856,x\n13862856,x2\n13611370,y\n13611413,z\n")
    valid, errors = validate_format(sub, refs, id_col="id", pred_col="generated_summary")
    assert valid is False
    assert any("duplicate" in e for e in errors)


def test_missing_subset_is_invalid(tmp_path):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv", "id,generated_summary\n13862856,x\n")
    valid, errors = validate_format(sub, refs, id_col="id", pred_col="generated_summary")
    assert valid is False
    assert any("missing 2 required ids" in e for e in errors)


def test_cli_prints_valid_no_score(tmp_path, capsys):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv",
                 "id,generated_summary\n13862856,x\n13611370,y\n13611413,z\n")
    rc = main([str(sub), "--task", "samsum", "--refs", str(refs)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("VALID")
    assert "does NOT report your score" in out
    # The validator must never leak a numeric score.
    assert "rouge" not in out.lower()


def test_cli_prints_invalid_with_reasons(tmp_path, capsys):
    refs = _write(tmp_path / "refs.csv", _REFS)
    sub = _write(tmp_path / "sub.csv", "id,label\n13862856,0\n")
    rc = main([str(sub), "--task", "samsum", "--refs", str(refs)])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.startswith("INVALID")
    assert "missing required columns" in out


def test_cli_unknown_task(tmp_path, capsys):
    sub = _write(tmp_path / "sub.csv", "id,generated_summary\n1,x\n")
    rc = main([str(sub), "--task", "nope", "--refs", str(tmp_path / "refs.csv")])
    assert rc == 1
    assert "unknown task" in capsys.readouterr().out
