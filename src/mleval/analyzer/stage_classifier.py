"""AST-based pipeline-stage classifier (MVP — to be upgraded with PyCG).

Walks each record's code (from trajectory.jsonl + ``code/op_NNN.py``) and
labels it with one of the 16 sub-stages defined in ``docs/eval/stage2.md``:

    1a data-loading           4a optimizer/scheduler
    1b EDA                    4b training-loop
    2a cleaning/encoding      4c preference-opt
    2b split-validation       5a HPO
    2c feature-engineering    5b ablation
    3a architecture           6a held-out-eval
    3b loss                   6b inference-merge
    3c adapter-config         6c submission

Classification source = ``ast_choice_extractor`` with a confidence in [0, 1]
derived from the highest-priority rule that matched.

This is the MVP version: a flat priority-ordered rule table over
imports and call names. The Ramasamy-validation upgrade to full
PyCG-Extended (task #62, gated by #70) replaces ``_classify_one`` and
keeps the rest of the module intact.

CLI:
    python -m mleval.analyzer.stage_classifier $MLEVAL_OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    """A single classification rule. Higher priority wins ties."""

    sub_stage: str
    label: str
    top_level: str
    confidence: float
    priority: int
    # Match if ANY import name in this set is present.
    import_any: frozenset[str] = frozenset()
    # AND ANY call name in this set is present (if non-empty).
    call_any: frozenset[str] = frozenset()


# Rules sorted by priority (highest = most specific). Tweak as we learn more.
_RULES: list[Rule] = [
    # 3c — adapter config (PEFT — most specific)
    Rule(sub_stage="3c", label="adapter_config", top_level="3", confidence=0.95, priority=100,
         import_any=frozenset({"peft"}),
         call_any=frozenset({"LoraConfig", "get_peft_model", "PromptTuningConfig", "AdaLoraConfig"})),
    # 4c — preference optimization (TRL DPO/GRPO/SimPO)
    Rule(sub_stage="4c", label="preference_opt", top_level="4", confidence=0.92, priority=95,
         import_any=frozenset({"trl"}),
         call_any=frozenset({"GRPOTrainer", "DPOTrainer", "SimPOTrainer", "KTOTrainer", "SFTTrainer"})),
    # 5a — hyperparameter optimization (kept above 6b/6c but below adapter/preference rules)
    # 6b/6c are intentionally LOW-priority: agents frequently call from_pretrained or
    # to_csv as side-effects of a step whose primary purpose is training / architecture /
    # eval. They're meant as fallback labels when no other signal fires. Without this
    # demotion, every house-prices step landed in 6c (#104).
    # 6b — inference / merge (fallback, fires only if no training/HPO signal)
    Rule(sub_stage="6b", label="inference_merge", top_level="6", confidence=0.7, priority=12,
         import_any=frozenset({"peft", "transformers"}),
         call_any=frozenset({"merge_and_unload", "generate"})),
    # 6c — submission write (lowest fallback)
    Rule(sub_stage="6c", label="submission", top_level="6", confidence=0.7, priority=10,
         call_any=frozenset({"to_csv", "to_parquet", "to_json"})),
    # 5a — hyperparameter optimization
    Rule(sub_stage="5a", label="hpo", top_level="5", confidence=0.85, priority=75,
         import_any=frozenset({"optuna", "ray.tune", "hyperopt", "sklearn.model_selection"}),
         call_any=frozenset({"GridSearchCV", "RandomizedSearchCV", "BayesianOptimization", "create_study"})),
    # 4b — training loop (Trainer or model.fit)
    Rule(sub_stage="4b", label="training_loop", top_level="4", confidence=0.8, priority=70,
         call_any=frozenset({"Trainer", "fit", "train", "train_step"})),
    # 4a — optimizer/scheduler instantiation
    Rule(sub_stage="4a", label="optimizer", top_level="4", confidence=0.8, priority=65,
         import_any=frozenset({"torch.optim", "transformers.optimization"}),
         call_any=frozenset({"AdamW", "Adam", "SGD", "get_linear_schedule_with_warmup", "get_scheduler"})),
    # 3b — loss / objective
    Rule(sub_stage="3b", label="loss", top_level="3", confidence=0.75, priority=60,
         import_any=frozenset({"torch.nn"}),
         call_any=frozenset({"CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "compute_loss"})),
    # 3a — architecture
    Rule(sub_stage="3a", label="architecture", top_level="3", confidence=0.78, priority=55,
         import_any=frozenset({"transformers", "torchvision.models", "timm", "torch.nn"}),
         call_any=frozenset({"AutoModelForCausalLM", "AutoModel", "AutoModelForSequenceClassification", "Sequential", "Module"})),
    # 6a — held-out evaluation
    Rule(sub_stage="6a", label="held_out_eval", top_level="6", confidence=0.7, priority=50,
         call_any=frozenset({"accuracy_score", "roc_auc_score", "f1_score", "mean_absolute_error", "mean_squared_error", "evaluate", "compute"})),
    # 2b — split / validation
    Rule(sub_stage="2b", label="split_validation", top_level="2", confidence=0.85, priority=45,
         call_any=frozenset({"train_test_split", "KFold", "StratifiedKFold", "TimeSeriesSplit"})),
    # 2c — feature engineering
    Rule(sub_stage="2c", label="feature_engineering", top_level="2", confidence=0.6, priority=40,
         call_any=frozenset({"PolynomialFeatures", "TfidfVectorizer", "CountVectorizer", "OneHotEncoder", "get_dummies"})),
    # 2a — cleaning / encoding
    Rule(sub_stage="2a", label="cleaning", top_level="2", confidence=0.6, priority=35,
         call_any=frozenset({"StandardScaler", "MinMaxScaler", "LabelEncoder", "fillna", "dropna", "Imputer", "SimpleImputer"})),
    # 1b — EDA
    Rule(sub_stage="1b", label="eda", top_level="1", confidence=0.55, priority=20,
         call_any=frozenset({"describe", "info", "head", "value_counts", "corr", "hist", "boxplot"})),
    # 1a — data loading (fallback for almost-anything-pandas)
    Rule(sub_stage="1a", label="data_loading", top_level="1", confidence=0.6, priority=15,
         call_any=frozenset({"read_csv", "read_parquet", "read_json", "load_dataset", "ImageFolder"})),
]


def _extract(code: str) -> tuple[set[str], set[str], bool]:
    """Return (imports, calls, parse_ok).

    ``parse_ok=False`` distinguishes ``SyntaxError`` (e.g. diff-patch merge
    conflict markers in the source) from "valid code, no rule matched"
    (which the classify() caller maps to ``unknown``). spike-012's
    without_skill cell showed 4 nodes with diff-patch-corrupted code —
    flagging those as ``parse_error`` instead of ``unknown`` is the
    difference between "agent picked nothing classifiable" and "agent
    output unparsable garbage".
    """
    imports: set[str] = set()
    calls: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports, calls, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            for alias in node.names:
                # `from peft import LoraConfig` -> also count LoraConfig as a call hint
                imports.add(alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    return imports, calls, True


def _import_matches(rule: Rule, imports: set[str]) -> bool:
    if not rule.import_any:
        return True
    return any(any(imp == r or imp.startswith(f"{r}.") for r in rule.import_any) for imp in imports)


def classify(code: str) -> dict:
    """Multi-label classification: walk ALL rules and report every match.

    Output schema:
      top_level / sub_stage / label / classifier_confidence  — the HIGHEST-priority
        match (back-compat; consumers that want one canonical label use these).
      all_matches: list of {top_level, sub_stage, label, confidence}, priority-ordered.
      all_top_levels / all_sub_stages: convenience flat lists for aggregation
        (each unique top_level / sub_stage that fired, in priority order).
      parse_status: "ok" | "parse_error" — distinguishes broken code from
        valid-but-uninteresting code.
      imports_top: first 10 imports for debugging.
    """
    imports, calls, parse_ok = _extract(code)

    if not parse_ok:
        return {
            "top_level": "-1",
            "sub_stage": "parse_error",
            "label": "parse_error",
            "classifier_source": "ast_choice_extractor",
            "classifier_confidence": 0.0,
            "parse_status": "parse_error",
            "imports_top": [],
            "all_matches": [],
            "all_top_levels": [],
            "all_sub_stages": [],
        }

    matches: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for rule in sorted(_RULES, key=lambda r: -r.priority):
        if not _import_matches(rule, imports):
            continue
        if rule.call_any and not (rule.call_any & calls):
            continue
        pair = (rule.top_level, rule.sub_stage)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        matches.append({
            "top_level": rule.top_level,
            "sub_stage": rule.sub_stage,
            "label": rule.label,
            "confidence": rule.confidence,
        })

    if not matches:
        # Code parsed cleanly but no rule fired — agent wrote valid Python
        # that doesn't touch any classifier-known API (e.g. pure config
        # script, pure helper functions, or a test stub).
        return {
            "top_level": "0",
            "sub_stage": "unknown",
            "label": "unknown",
            "classifier_source": "ast_choice_extractor",
            "classifier_confidence": 0.0,
            "parse_status": "ok",
            "imports_top": sorted(imports)[:10],
            "all_matches": [],
            "all_top_levels": [],
            "all_sub_stages": [],
        }

    # Highest-priority match drives the canonical label fields.
    best = matches[0]
    return {
        "top_level": best["top_level"],
        "sub_stage": best["sub_stage"],
        "label": best["label"],
        "classifier_source": "ast_choice_extractor",
        "classifier_confidence": best["confidence"],
        "parse_status": "ok",
        "imports_top": sorted(imports)[:10],
        "all_matches": matches,
        "all_top_levels": list(dict.fromkeys(m["top_level"] for m in matches)),
        "all_sub_stages": list(dict.fromkeys(m["sub_stage"] for m in matches)),
    }


def classify_trajectory(output_dir: Path) -> Path:
    """Re-write trajectory.jsonl in place, replacing each record's ``stage`` and ``code.imports_top``."""
    path = output_dir / "trajectory.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run adapter first")

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for rec in records:
        # Schema evolution: adapter_mlevolve writes ``code`` as the raw source
        # string (inline). Earlier AIDE adapter wrote a dict ``{"emitted_path":
        # ...}``. Handle both so the classifier survives reruns on archived data.
        code_field = rec.get("code")
        if isinstance(code_field, dict):
            code_path = code_field.get("emitted_path")
            if not code_path:
                continue
            code = (output_dir / code_path).read_text()
        elif isinstance(code_field, str) and code_field.strip():
            code = code_field
        else:
            continue
        cls = classify(code)
        # Preserve top-level node fields and add a "stage_classifier" sub-dict
        # alongside the upstream ``stage`` label (a string from adapter_mlevolve).
        # This avoids the AIDE-era convention of overwriting ``stage`` with a
        # dict, which broke aggregate.py's str-comparison on the upstream label.
        rec["stage_classifier"] = {
            "top_level": cls["top_level"],
            "sub_stage": cls["sub_stage"],
            "label": cls["label"],
            "classifier_source": cls["classifier_source"],
            "classifier_confidence": cls["classifier_confidence"],
            "parse_status": cls.get("parse_status"),
            "all_top_levels": cls.get("all_top_levels", []),
            "all_sub_stages": cls.get("all_sub_stages", []),
            "all_matches": cls.get("all_matches", []),
        }
        if isinstance(rec.get("code"), dict):
            rec["code"]["imports_top"] = cls["imports_top"]
        else:
            rec["imports_top"] = cls["imports_top"]

    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label trajectory.jsonl with pipeline stages")
    parser.add_argument("output_dir", type=Path, help="$MLEVAL_OUTPUT_DIR")
    args = parser.parse_args(argv)
    out = classify_trajectory(args.output_dir)
    print(f"[stage_classifier] updated {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
