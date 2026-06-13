# Task-instruction authoring guideline (MLE-Bench-aligned)

How to write the `instruction.md` + grading contract for a Stage-2 task. The
conventions here are ported from **MLE-Bench** (OpenAI's ML-engineering agent
benchmark) so our freeform PEFT tasks are scored with the same discipline as a
peer-reviewed benchmark. Apply this to every new task and when revising an
existing one.

> All MLE-Bench claims below were read from source (not training memory):
> - instructions template — `environment/instructions.txt`
> - description schema — `mlebench/competitions/spaceship-titanic/description.md`
> - task config — `mlebench/competitions/spaceship-titanic/config.yaml`
> - grading rubric — `mlebench/grade_helpers.py` (`CompetitionReport`)
> - re-split / held-out design — `README.md` + per-competition `prepare.py`
> - paper: *MLE-bench* (arXiv 2410.07095)

---

## 1. Why mirror MLE-Bench

MLE-Bench is the de-facto standard for "did the agent actually engineer a
working ML pipeline." Our Stage-2 A/B asks the same question with one extra
axis (skill / no-skill). Adopting its instruction + grading conventions means
(a) reviewers recognise the protocol, (b) the gameable failure modes it already
solved (label leakage, format gaming, self-reported scores) don't reappear in
our results, and (c) every task we add is authored the same way.

Our grader already states the lineage — `src/mleval/grader/grade.py` docstring:
*"The validation gates mirror mle-bench's grade functions (id-set equality, no
duplicates, real per-example predictions)."* This doc extends that lineage from
the **grader** to the **instructions** and the **rubric**.

---

## 2. MLE-Bench's design (verified)

Six load-bearing decisions:

| # | Decision | Source |
|---|---|---|
| D1 | **Two instruction layers**: a fixed benchmark-wide `instructions.txt` + a per-competition `description.md`. Benchmark rules **override** competition text on conflict. | `environment/instructions.txt` |
| D2 | **`description.md` has a fixed schema**: `Description` · `Evaluation` · `Submission Format` · `Dataset Description` (Files, Columns). Terse — no defensive nagging. | `spaceship-titanic/description.md` |
| D3 | **Test labels are physically held out via `prepare.py`**: the public train is re-split into a *new* train+test; the new test answers go to `prepared/private/test.csv` (the grader's `dataset.answers`); only `prepared/public/` (train + test-inputs + `sample_submission.csv`) is shipped to the agent. The agent **cannot see test labels** — it's not a prose request. | `config.yaml`, `prepare.py`, README |
| D4 | **One fixed submission path**, and only that path is graded: `/home/submission/submission.csv`. `sample_submission.csv` is the canonical format spec ("same format as the sample submission"). | `environment/instructions.txt` |
| D5 | **A validate-but-don't-score checker** (`localhost:5000/validate` / `validate_submission.sh`) tells the agent valid/invalid before it finishes — never the score. | `environment/instructions.txt` |
| D6 | **Anti-gaming rules**: no hand-labeling the test set, no plagiarising solutions. Grading = the competition metric on the private test, then positioned against the human leaderboard into a **threshold ladder** (gold/silver/bronze/median). | `instructions.txt`, `grade_helpers.py` |

The grading **rubric** is not a bare number. `CompetitionReport` fields:

```
score · gold_threshold · silver_threshold · bronze_threshold · median_threshold
any_medal · gold_medal · silver_medal · bronze_medal · above_median
submission_exists · valid_submission · is_lower_better
```

A result is `(valid?, score, which tier)` — interpretable against a reference
distribution, not an absolute float in a vacuum.

---

## 3. Where we stand today (gap analysis)

| MLE-Bench decision | Our current state (gsm8k / samsum) | Gap |
|---|---|---|
| D1 two-layer instructions | One monolithic `instruction.md` mixing harness rules + task contract + heavy `⚠️` warnings | **Gap** — no shared rules layer; rules re-stated (and drift) per task |
| D2 description schema | Ad-hoc sections (`Task`, `Data`, `Model`, `Evaluation`, `Output contract`) | Partial — close, but not the canonical 4 + bloated with warnings |
| D3 held-out test | Agent loads the **public** `test` split **with labels** (`ex["answer"]` / `summary`). Our grader recomputes from a private `refs/` copy. | **Biggest gap** — "held-out" = independent *recompute* (integrity), **not** label privacy. Search self-scores on test → model-selection-against-test bias. |
| D4 fixed path + sample | `./submission/submission.csv` + gsm8k writes `sample_submission.csv` | **Met** ✓ (keep) |
| D5 validate-don't-score | `python -m mleval.grader.validate` | **Met** ✓ (keep) |
| D6 anti-gaming + rubric | "use exactly this model" prose; grader returns `valid/score` but **no threshold ladder** | Partial — rules informal; **no rubric tiers** |

The two real gaps are **D3 (held-out)** and **D6 (rubric tiers)**; **D1/D2**
are a cleanup. D4/D5 are already MLE-Bench-shaped.

---

## 4. The conventions we adopt

### C1 — Split instructions into two layers (D1)

- **`infra/tasks/_harness_rules.md`** (NEW, shared, prepended to every task):
  the **task-agnostic** rules only — provided-data-only / no external download,
  *build a model from the data, don't fabricate or copy answers*, *no
  training/selecting on the held-out test*, fixed submission path + match
  `sample_submission.csv`, the `validate` tool, *submission-is-the-score /
  printed-number-is-only-the-search-signal*, the resource budget, and the
  conflict-priority clause ("these rules override anything in the task below").
  It must stay generic: the concrete **backbone**, dataset, metric, and exact
  columns are per-task (a tabular task has no LLM backbone), so they live in the
  task `instruction.md`, never here — exactly as MLE-Bench keeps model specifics
  in `description.md`, not `instructions.txt`.
- **`infra/tasks/<task>/instruction.md`**: only the task-specific contract,
  in the C2 schema. The sidecar concatenates `_harness_rules.md` + the task file
  (mirrors MLE-Bench's `instructions.txt` + `description.md` assembly).

This stops the `⚠️` id/backbone warnings from being copy-pasted (and silently
diverging) across tasks — they live once in the shared layer.

### C2 — Canonical `description.md` schema (D2)

Every task `instruction.md`, in this order:

1. **Description** — task family + what to produce, 2–4 sentences. State the
   contract is FIXED, the recipe (method/library/schedule/inference) is OPEN.
2. **Dataset Description** — slug/loader, splits **with the held-out caveat**
   (see C3), and a **Fields** list naming *every* field the output references
   (especially the id key). Pin the backbone here or in its own line.
3. **Evaluation** — the exact metric, computed over which split, and the one-line
   "the submission file is the official score; your printed number is only the
   search signal."
4. **Submission Format** — exact columns + header + an example row; point at
   `sample_submission.csv` as canonical; the fixed path; the `validate` call.

Keep it terse. The mechanical guards (held-out split + `validate`) replace prose
nagging — that is *why* MLE-Bench descriptions are short.

### C3 — Make held-out REAL: a `prepare.py`-style re-split (D3) ⭐

This is the centerpiece and it resolves the methodology problem we hit on
samsum (search self-scoring on test → optimistic, selection-biased). Mirror
MLE-Bench exactly:

- **The agent never receives test labels.** Ship it: `train` (with labels) +
  `test` **inputs only** (target/answer field stripped) + a `sample_submission.csv`
  (all test ids, empty predictions). Keep the answers in a **private** `refs/`
  dir that is mounted only for *our* post-exit grader, never in the agent's data
  dir. (gsm8k already generates `refs/test_refs.csv` privately — extend the
  pattern to *withhold* the public test answers from the agent's copy.)
- **The search signal is a held-out-from-grade set the agent self-scores on** —
  a `validation` split (samsum has 818) or, when none exists (gsm8k), a slice the
  agent carves from `train`. **Never test.** This is strictly better than "score
  on a subset of test": a train/val holdout leaks nothing into the graded test.
- **We grade the full `submission.csv` on the private test**, once, post-exit.

Net effect: `test` becomes genuinely held out (label privacy, like MLE-Bench),
the search can't select against test, and search nodes get *cheaper* (eval on a
200-row val, not the full test) — which directly attacks the gsm8k timeout wall.

**Implementation note — what "private" means in our single-pod runtime.** Our
held-out grader runs **in-pod** (`entrypoint.sh`, for watchdog-harvest safety),
so the gold `refs/test_refs.csv` is physically on the mounted PVC during the
agent's run. We harden the two avoidable leaks:
1. The agent's **normal data path** (`train.*` / `test.*`) carries no test
   targets — the by-default access is gone.
2. The entrypoint exports only the **public** id-set
   (`MLEVAL_TASK_IDSET_PATH` → `sample_submission.csv`) to the agent's
   subprocesses for self-validation; the **gold** path is a parent-only shell
   var, never in the agent's environment. (Pre-C3 we exported the gold refs
   path itself — under label-withholding that would hand over the answers.)

**Residual (document honestly):** two deliberate-cheat vectors remain — (a) an
agent could re-download the public benchmark from HF, and (b) the gold file is
still readable by *path* on the shared PVC (`/results/data/<task>/refs/`),
though it is never named to the agent. Both require the agent to go out of its
way; both are **symmetric across cells** (can't alone explain a Lift); both are
caught by the `data_leakage` helper + state predicates. MLE-Bench avoids (b)
entirely by grading in a separate process the agent can't read — **true physical
isolation in our runtime would need a grader sidecar with a non-shared volume,
or out-of-pod grading** (loses the in-pod watchdog-harvest). Flagged as future
work; not required for a valid, leak-detected A/B. Belongs in threats-to-validity.

### C4 — Keep the fixed path + sample + validate (D4, D5)

Already met. Don't regress: one graded path (`./submission/submission.csv`),
`sample_submission.csv` as the format truth, `validate` returns valid/invalid
only.

### C5 — Anti-gaming rules block (D6)

**Shared (`_harness_rules.md`, generic):** predictions must come from a model
trained on the provided data — no hand-authored or copied answers; no training
or selecting on the held-out test. PEFT analogue of MLE-Bench's "no hand-labeling
/ no plagiarism." **Per-task (`instruction.md`):** the concrete **pinned
backbone** and "use exactly this model" (the grader does *not* check the model,
so a wrong/smaller backbone surfaces only as a bad/invalid submission) — this is
task-specific and a tabular task won't have it.

### C6 — Adopt a threshold-ladder rubric (D6)

Extend `GradeResult` (currently `valid · score · metric · n_scored ·
n_expected · errors`) with reference-anchored tiers, mirroring
`CompetitionReport`:

- Per task, define reference scores → `bronze` (a competent baseline fine-tune),
  `silver`, `gold` (a strong reference), plus a `pass`/`above_baseline` gate.
- Grader emits `score` **and** the tier reached + `is_lower_better`.
- A/B reporting then reads "with_skill reached silver, without_skill bronze"
  instead of "0.41 vs 0.29" — interpretable, and robust to a task whose raw
  scale is unintuitive.

Thresholds come from a small reference sweep (a no-skill baseline fine-tune at a
few budgets), recorded in the task's `config` next to the refs. They do not need
a human leaderboard — they need a *documented reference*.

---

## 5. Canonical task layout

```
infra/tasks/
├── _harness_rules.md          NEW — shared benchmark rules (C1, C5); prepended to every task
├── _template/
│   ├── instruction.md         the C2 4-section schema with <placeholders>
│   ├── predicates.py
│   └── README.md
└── <task>/
    ├── instruction.md         task-specific contract only (C2)
    ├── scripts/
    │   └── make_grading_data.py   builds refs/ (private answers) + sample_submission + (NEW) the agent-facing label-stripped test
    ├── refs/                   PRIVATE — test answers + thresholds; mounted only for our grader, never in the agent data dir
    │   ├── test_refs.csv
    │   ├── sample_submission.csv
    │   └── thresholds.json     (NEW, C6) {bronze, silver, gold, is_lower_better}
    └── data/                   agent-facing: train (labels) + test inputs (NO labels)
```

---

## 6. Authoring checklist

Supersedes the `project_task_instruction_authoring_checklist` memory. Tick all
before staging a task to the PVC:

- [ ] **Shared rules prepended** — `_harness_rules.md` carries submission path,
      validate tool, backbone-must-match, no-hand-labels, no-train-on-test,
      conflict-priority. Task file does **not** restate them.
- [ ] **C2 schema** — Description · Dataset Description (+ full Fields list incl.
      id key) · Evaluation · Submission Format, in that order, terse.
- [ ] **Backbone pinned** verbatim; note the grader doesn't check it.
- [ ] **Held-out is real (C3)** — agent's `data/` has train-with-labels + test
      **inputs only**; answers live in private `refs/`; search signal is
      val/train-holdout, never test; this is stated in Dataset Description.
- [ ] **Exact output contract** — verbatim id key (no hashing/renumber unless the
      task *defines* index-as-id, like gsm8k), exact columns + header, example row,
      fixed `./submission/submission.csv` path, `validate` call shown.
- [ ] **`sample_submission.csv`** generated with the exact id-set + empty preds.
- [ ] **Rubric thresholds (C6)** — `refs/thresholds.json` from a reference sweep.
- [ ] **Resource note** — per-exec wall cap stated; "submission is the score, the
      printed number is only the search signal"; batch/decode hints if eval-heavy.
- [ ] **Synced to PVC** — `/results/data/<task>/{instruction.md,data/}` updated;
      private `refs/` synced to its grader-only mount. Pods read the PVC, not the
      repo/image.

---

## 7. Worked example — gsm8k (the changes this prescribes)

Current `infra/tasks/gsm8k/instruction.md` violates C1 (monolithic), C3 (ships
`test` with `answer`), and C6 (no rubric). Concrete diffs:

1. **Strip test labels (C3).** `make_grading_data.py` already writes the private
   `refs/test_refs.csv`. Add a step that writes the **agent-facing** test as
   *questions only* (drop `answer`), plus the existing `sample_submission.csv`
   (ids `"0".."1318"`, empty `prediction`). The agent's `data/` no longer
   contains gold answers.
2. **Move the search signal off test (C3).** Dataset Description instructs: carve
   a validation slice from `train` (e.g. last 500) for the self-reported
   `Final Validation Score`; **do not** self-score on test. Full `submission.csv`
   still covers all 1319; we grade it privately. → search nodes evaluate ~500
   rows, not 1319 → **the 60-min timeout wall drops** (the spike-022 failure mode).
3. **Two-layer split (C1).** Lift the backbone / id / no-substitution `⚠️` blocks
   into `_harness_rules.md`; the gsm8k file keeps only the math-specific contract
   (`#### <number>` extraction, index-as-id, `main` config).
4. **Rubric (C6).** Run a no-skill reference sweep, write
   `refs/thresholds.json` (e.g. bronze=0.20, silver=0.40, gold=0.55 exact-match,
   `is_lower_better=false`); grader reports the tier.

Combined with the exec-cap bump, this is what gives a gsm8k node a real chance to
produce a *valid, cheaply-searched* result instead of timing out on a full-test
self-eval.

---

## 8. Threats to validity (carry into the report)

- **Public-benchmark label exposure** — see C3 residual; symmetric across cells,
  detected not prevented.
- **Self-reported search signal is gameable** — mitigated because the *graded*
  number is our private recompute; the printed scalar only steers the search.
- **Reference-anchored thresholds are ours, not a leaderboard** — document the
  reference sweep that produced them so tiers are reproducible.
