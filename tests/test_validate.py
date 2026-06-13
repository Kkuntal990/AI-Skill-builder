"""Tests for the format-only submission validator (MLE-Bench validate affordance).

Mirrors the two real spike-018 failure modes — a wrong column (``label``) and a
wrong id-set (hashed ids) — plus the happy path. Asserts the validator never
emits a score, only validity + structural reasons.
"""
from __future__ import annotations

from pathlib import Path

from mleval.grader.validate import _default_idset, main, validate_format

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


def test_idset_path_preferred_over_refs(tmp_path, monkeypatch):
    """C3 held-out: the validator must read the PUBLIC id-set, not the gold refs.

    When MLEVAL_TASK_IDSET_PATH is set (the agent-facing sample_submission.csv),
    it wins over MLEVAL_TASK_REFS_PATH (gold) so the agent never gets the gold
    path. The gold-only fallback remains for tasks staged before the split.
    """
    idset = _write(tmp_path / "sample_submission.csv", "id,prediction\n0,\n1,\n")
    gold = _write(tmp_path / "test_refs.csv", "id,reference_answer\n0,18\n1,3\n")
    monkeypatch.setenv("MLEVAL_TASK_IDSET_PATH", str(idset))
    monkeypatch.setenv("MLEVAL_TASK_REFS_PATH", str(gold))
    assert _default_idset() == idset  # public id-set wins
    monkeypatch.delenv("MLEVAL_TASK_IDSET_PATH")
    assert _default_idset() == gold   # falls back to refs (back-compat)


def test_cli_validates_against_public_idset(tmp_path, capsys, monkeypatch):
    """A gsm8k submission validates against sample_submission.csv (no gold needed)."""
    idset = _write(tmp_path / "sample_submission.csv", "id,prediction\n0,\n1,\n2,\n")
    monkeypatch.setenv("MLEVAL_TASK_IDSET_PATH", str(idset))
    monkeypatch.delenv("MLEVAL_TASK_REFS_PATH", raising=False)
    sub = _write(tmp_path / "sub.csv", "id,prediction\n0,18\n1,3\n2,42\n")
    rc = main([str(sub), "--task", "gsm8k"])  # no --refs: resolved from idset env
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("VALID")
