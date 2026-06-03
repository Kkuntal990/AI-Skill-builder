"""Per-sub-stage metrics over a universal ``trajectory.jsonl``.

Three metrics, all derived from telemetry we already capture (no new
instrumentation, no working-dir snapshots). All three operate only on a
node's multi-label ``stage_classifier.all_sub_stages`` set plus its
``is_buggy`` / ``exc_type`` / ``parent_id`` fields, so they are immune to
the *co-location* confound that blocks per-stage time/token attribution
(one node's script spans many stages, so its wall-time can't be split,
but its label-set and pass/fail status attribute cleanly).

For each sub-stage ``s``:

1. **clean_reach** — of the nodes that touch ``s``, the fraction that ran
   without a bug. "Where does the agent get this stage *right*."
   ``clean(s) / touches(s)``.
2. **rework** — how many times ``s`` was re-attempted beyond the first.
   ``max(0, touches(s) - 1)``. "Where does the agent *thrash*." (Process
   mining's rework/self-loop count; on our near-linear search trees the
   touches-1 form and the parent-edge self-loop form coincide — we also
   expose ``rework_edges`` for the stricter parent→child definition.)
3. **fail_modes** — distribution of ``exc_type`` over the *buggy* nodes
   that touch ``s``. "Which errors live at this stage." Categorical;
   attributed to every stage a buggy node touches (we can't know which of
   the co-located stages actually raised).

CLI::

    python -m mleval.analyzer.stage_metrics <traj_dir>              # one cell
    python -m mleval.analyzer.stage_metrics <with_dir> <without_dir>  # paired A/B table

Reads ``<dir>/trajectory.jsonl`` (already classified). Writes nothing by
default; prints a markdown table. The single-dir form returns the raw dict.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Canonical sub-stage -> human label, sourced from the classifier's own rules
# so the two never drift. Falls back to "?" for any sub-stage without a rule.
try:  # pragma: no cover - import shim for both `python -m` and direct runs
    from mleval.analyzer.stage_classifier import _RULES
except ImportError:  # running from a checkout without the package installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mleval.analyzer.stage_classifier import _RULES

_LABELS: dict[str, str] = {r.sub_stage: r.label for r in _RULES}
# Stable display order across the 6 top-levels.
_ORDER = ["1a", "1b", "2a", "2b", "2c", "3a", "3b", "3c",
          "4a", "4b", "4c", "5a", "6a", "6b", "6c"]


def _read_traj(traj_dir: Path) -> list[dict[str, Any]]:
    path = traj_dir / "trajectory.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run adapter + classifier first")
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _subs(rec: dict[str, Any]) -> list[str]:
    return (rec.get("stage_classifier") or {}).get("all_sub_stages") or []


def stage_metrics(traj_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{sub_stage: {touches, clean, clean_reach, rework, rework_edges,
    fail_modes}}`` for one trajectory."""
    recs = _read_traj(traj_dir)
    by_id = {r.get("node_id"): r for r in recs}

    out: dict[str, dict[str, Any]] = {}
    for rec in recs:
        for s in _subs(rec):
            a = out.setdefault(s, {
                "touches": 0, "clean": 0, "rework_edges": 0,
                "fail_modes": Counter(),
            })
            a["touches"] += 1
            # is_buggy is False for a clean run, True for buggy, None for root.
            if rec.get("is_buggy") is False:
                a["clean"] += 1
            elif rec.get("is_buggy") is True:
                exc = rec.get("exc_type") or "UnknownError"
                a["fail_modes"][exc] += 1
            # Parent-edge self-loop: this stage persisted from parent to child.
            parent = by_id.get(rec.get("parent_id"))
            if parent is not None and s in _subs(parent):
                a["rework_edges"] += 1

    for s, a in out.items():
        a["clean_reach"] = (a["clean"] / a["touches"]) if a["touches"] else None
        a["rework"] = max(0, a["touches"] - 1)
        a["fail_modes"] = dict(a["fail_modes"])
    return out


def _fmt_reach(a: dict[str, Any] | None) -> str:
    if not a or not a.get("touches"):
        return "—"
    return f"{a['clean_reach']:.2f} ({a['clean']}/{a['touches']})"


def _fmt_modes(a: dict[str, Any] | None) -> str:
    if not a or not a.get("fail_modes"):
        return "—"
    return ", ".join(f"{k}×{v}" for k, v in sorted(a["fail_modes"].items()))


def _all_substages(*metric_dicts: dict[str, Any]) -> list[str]:
    seen = set()
    for d in metric_dicts:
        seen |= set(d)
    ordered = [s for s in _ORDER if s in seen]
    ordered += sorted(s for s in seen if s not in _ORDER)
    return ordered


def print_single(traj_dir: Path) -> None:
    m = stage_metrics(traj_dir)
    print(f"# Per-sub-stage metrics — {traj_dir.name}\n")
    print("| Sub-stage | Label | clean-reach | rework | failure modes |")
    print("|---|---|---|---|---|")
    for s in _all_substages(m):
        a = m.get(s)
        print(f"| {s} | {_LABELS.get(s, '?')} | {_fmt_reach(a)} "
              f"| {a['rework'] if a else '—'} | {_fmt_modes(a)} |")


def print_paired(with_dir: Path, without_dir: Path) -> None:
    mw = stage_metrics(with_dir)
    mn = stage_metrics(without_dir)
    subs = _all_substages(mw, mn)

    print("# Per-sub-stage A/B — clean-reach · rework · failure-modes\n")
    print("## 1. clean-reach rate (clean nodes / nodes touching stage)\n")
    print("| Sub-stage | Label | with_skill | without_skill |")
    print("|---|---|---|---|")
    for s in subs:
        print(f"| {s} | {_LABELS.get(s, '?')} | {_fmt_reach(mw.get(s))} | {_fmt_reach(mn.get(s))} |")

    print("\n## 2. rework (re-attempts beyond first = touches − 1)\n")
    print("| Sub-stage | Label | with_skill | without_skill |")
    print("|---|---|---|---|")
    for s in subs:
        rw = mw.get(s, {}).get("rework", "—")
        rn = mn.get(s, {}).get("rework", "—")
        print(f"| {s} | {_LABELS.get(s, '?')} | {rw} | {rn} |")

    print("\n## 3. failure modes (exc_type over buggy nodes touching stage)\n")
    print("| Sub-stage | Label | with_skill | without_skill |")
    print("|---|---|---|---|")
    for s in subs:
        print(f"| {s} | {_LABELS.get(s, '?')} | {_fmt_modes(mw.get(s))} | {_fmt_modes(mn.get(s))} |")

    # parse_error nodes carry no sub-stage labels, so per-stage rework/fail-modes
    # cannot see them. Surface that gap so the table isn't misread as "no thrash".
    def _parse_err(d: Path) -> int:
        return sum(1 for r in _read_traj(d)
                   if (r.get("stage_classifier") or {}).get("parse_status") == "parse_error")
    print(f"\n> Note: parse_error nodes (unclassifiable, no sub-stages) — "
          f"with_skill={_parse_err(with_dir)}, without_skill={_parse_err(without_dir)}. "
          f"Their thrash is invisible to per-stage rework/fail-modes by construction.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-sub-stage metrics over trajectory.jsonl")
    p.add_argument("dirs", type=Path, nargs="+",
                   help="one trajectory dir, or two (with_skill without_skill) for a paired table")
    args = p.parse_args(argv)
    if len(args.dirs) == 1:
        print_single(args.dirs[0])
    elif len(args.dirs) == 2:
        print_paired(args.dirs[0], args.dirs[1])
    else:
        print("error: pass 1 dir (single) or 2 dirs (paired)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
