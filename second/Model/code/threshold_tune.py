"""
threshold_tune.py — Validation-tuned decision threshold for binary outcome models.

Why this script exists
----------------------
The default decision threshold τ=0.5 is statistically optimal only when
(a) class priors are 50/50 *and* (b) model probabilities are calibrated.
Neither holds here:
  * Test set is 14 pos / 9 neg  (61 % positive prevalence)
  * Mistral / BioMistral softmax over Yes/No tokens is not calibrated
    (Guo et al. 2017, "On Calibration of Modern Neural Networks")

The consequence on the τ=0.5 evaluation: high AUROC (model ranks correctly)
but poor Macro F1 / MCC / Specificity (model defaults to "Yes" for almost
everyone). To recover an honest Macro F1 we choose a single threshold τ*
on the *validation* set, then HOLD IT FIXED for the test report.

This script does NOT change AUROC or AUPRC (both are threshold-independent).

Method (standard ML, see e.g. scikit-learn "TunedThresholdClassifierCV"):
  1. For each cell tag, load predictions_valid.jsonl (re-infer if absent).
  2. Sweep τ ∈ {0.01, 0.02, ..., 0.99}.
  3. Pick τ* = argmax  Macro_F1(valid).
     Tie-breaker: smallest τ that achieves the max F1 (favours sensitivity).
  4. Apply τ* to predictions_test.jsonl, recompute the full metric set.
  5. Write results/<tag>/predictions_test.metrics_tuned.json AND
     append a row to results/aggregate_threshold_tuned.csv.

The original aggregate.csv (τ=0.5) is left untouched for transparency —
the milestone report will quote BOTH numbers so reviewers can verify that
threshold tuning is not silently inflating any score.

Usage
-----
    # Tune all cells that have predictions_valid.jsonl already
    python code/threshold_tune.py --all

    # Single cell
    python code/threshold_tune.py --tag ExpC_TxVLM__beep__medcpt__biomistral

    # Re-infer validation predictions first (GPU-intensive; one model load
    # per cell). Takes ~1-2 min/cell on Apple MPS.
    python code/threshold_tune.py --all --reinfer
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import evaluate as evaluate_mod

ROOT      = Path(__file__).resolve().parents[2]    # Second_Recur/
MODEL_DIR = ROOT / "Model"
RESULT_D  = MODEL_DIR / "results"

# Reasonable fine-grained sweep. Going below 0.01 or above 0.99 produces
# all-Yes / all-No predictors, which we exclude (Macro F1 ill-defined).
TAU_GRID = np.round(np.arange(0.01, 1.00, 0.01), 4)


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = [], []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            labels.append(int(r["label"]))
            scores.append(float(r["score"]))
    return np.asarray(labels), np.asarray(scores)


def _macro_f1_at(threshold: float, labels: np.ndarray, scores: np.ndarray) -> float:
    """Macro F1 at a given threshold (0 if predictions collapse to one class)."""
    from sklearn.metrics import f1_score
    preds = (scores >= threshold).astype(int)
    if len(np.unique(preds)) == 1:
        return 0.0
    return float(f1_score(labels, preds, average="macro"))


def select_threshold(valid_path: Path, criterion: str = "macro_f1") -> tuple[float, dict]:
    """Pick τ* on the validation set under a chosen criterion.

    Returns (tau_star, sweep_log) where sweep_log is the full {τ: F1} dict
    so the milestone report can plot the sweep curve and show the choice
    is not over-fit to a single bin.
    """
    labels, scores = _load_predictions(valid_path)
    if criterion != "macro_f1":
        raise NotImplementedError(criterion)

    sweep = {float(tau): _macro_f1_at(float(tau), labels, scores) for tau in TAU_GRID}
    best_f1 = max(sweep.values())
    # Tie-breaker: smallest τ that hits the max F1 (favours sensitivity,
    # which matches the clinical preference for not missing recurrences).
    tau_star = min(tau for tau, f1 in sweep.items() if f1 == best_f1)
    return tau_star, {
        "criterion":    criterion,
        "valid_n":      int(len(labels)),
        "valid_n_pos":  int(labels.sum()),
        "valid_best_f1": round(best_f1, 4),
        "tau_star":     round(tau_star, 4),
        "sweep":        {str(k): round(v, 4) for k, v in sweep.items()},
    }


def tune_one(tag: str, *, reinfer: bool = False) -> dict:
    """Tune τ on validation, apply to test, write metrics_tuned.json."""
    cell_dir = RESULT_D / tag
    valid_path = cell_dir / "predictions_valid.jsonl"
    test_path  = cell_dir / "predictions_test.jsonl"

    if not test_path.exists():
        raise FileNotFoundError(f"missing test predictions: {test_path}")

    if not valid_path.exists():
        if not reinfer:
            raise FileNotFoundError(
                f"missing validation predictions: {valid_path}\n"
                f"  → re-run with --reinfer, or run "
                f"`python code/infer.py --tag {tag} --split valid` first."
            )
        # Defer the heavy MLX import until we actually need it.
        import infer as infer_mod
        print(f"[tune_one] re-inferring validation split for {tag} ...")
        infer_mod.infer(tag, split="valid")

    # Pick τ* on validation, evaluate on test at that τ*.
    tau_star, sweep_log = select_threshold(valid_path)

    # Test metrics @ τ* and @ 0.5 (for transparent comparison).
    labels_t, scores_t = _load_predictions(test_path)
    metrics_tuned   = evaluate_mod.compute_metrics(labels_t, scores_t,
                                                    threshold=tau_star,
                                                    bootstrap=1000)
    metrics_default = evaluate_mod.compute_metrics(labels_t, scores_t,
                                                    threshold=0.5,
                                                    bootstrap=0)

    out = {
        "tag":               tag,
        "tau_star":          round(tau_star, 4),
        "valid_best_f1":     sweep_log["valid_best_f1"],
        "valid_n":           sweep_log["valid_n"],
        "valid_n_pos":       sweep_log["valid_n_pos"],
        "metrics_tuned":     metrics_tuned,
        "metrics_default":   metrics_default,
        "sweep":             sweep_log["sweep"],
    }
    out_path = cell_dir / "predictions_test.metrics_tuned.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  → {tag}: τ*={tau_star:.2f}  "
          f"F1: {metrics_default['Macro_F1']:.3f} → {metrics_tuned['Macro_F1']:.3f}  "
          f"MCC: {metrics_default['MCC']:.3f} → {metrics_tuned['MCC']:.3f}  "
          f"Spec: {metrics_default['Specificity']:.3f} → {metrics_tuned['Specificity']:.3f}")
    return out


def write_aggregate(rows: list[dict], out_csv: Path) -> None:
    """Write a single CSV summarising τ-tuned vs τ=0.5 metrics for all cells."""
    headers = [
        "tag", "tau_star", "valid_n", "valid_n_pos", "valid_best_f1",
        # τ=0.5 (default) test metrics
        "AUROC", "AUPRC",
        "F1_default", "Acc_default", "Sens_default", "Spec_default",
        "MCC_default", "Brier_default",
        # τ=τ* test metrics
        "F1_tuned", "Acc_tuned", "Sens_tuned", "Spec_tuned",
        "MCC_tuned", "Brier_tuned",
        # deltas (tuned − default)
        "dF1", "dMCC", "dSpec",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            md = r["metrics_default"]
            mt = r["metrics_tuned"]
            w.writerow([
                r["tag"], r["tau_star"], r["valid_n"], r["valid_n_pos"], r["valid_best_f1"],
                md["AUROC"], md["AUPRC"],
                md["Macro_F1"], md["Accuracy"], md["Sensitivity"], md["Specificity"],
                md["MCC"], md["Brier"],
                mt["Macro_F1"], mt["Accuracy"], mt["Sensitivity"], mt["Specificity"],
                mt["MCC"], mt["Brier"],
                round(mt["Macro_F1"] - md["Macro_F1"], 4),
                round(mt["MCC"]      - md["MCC"],      4),
                round(mt["Specificity"] - md["Specificity"], 4),
            ])
    print(f"\nAggregate threshold-tuned report → {out_csv}")


def list_tunable_tags() -> list[str]:
    """All cells that have a predictions_test.jsonl (i.e. completed cells)."""
    return sorted(
        d.name for d in RESULT_D.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "predictions_test.jsonl").exists()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="Single cell tag, e.g. "
                    "ExpC_TxVLM__beep__medcpt__biomistral")
    ap.add_argument("--all", action="store_true",
                    help="Tune every completed cell under results/")
    ap.add_argument("--reinfer", action="store_true",
                    help="If predictions_valid.jsonl is missing, re-run "
                         "infer.py --split valid (GPU-intensive).")
    ap.add_argument("--out-csv", default=str(RESULT_D / "aggregate_threshold_tuned.csv"))
    args = ap.parse_args()

    if not args.tag and not args.all:
        ap.error("supply --tag <tag> or --all")

    tags = [args.tag] if args.tag else list_tunable_tags()
    print(f"Threshold tuning {len(tags)} cells:")
    rows = []
    for t in tags:
        try:
            rows.append(tune_one(t, reinfer=args.reinfer))
        except FileNotFoundError as e:
            print(f"  [skip] {t}: {e}")
        except Exception as e:
            print(f"  [FAIL] {t}: {type(e).__name__}: {e}")
    if rows:
        write_aggregate(rows, Path(args.out_csv))


if __name__ == "__main__":
    main()
