"""
evaluate.py — Score one predictions JSONL with the metric set BEEP and
clinical-prediction reviewers expect.

Metrics:
  * AUROC, AUPRC          (rank quality)
  * Macro F1, Accuracy    (operating-point quality at threshold=0.5)
  * Sensitivity, Specificity
  * MCC                   (balanced summary insensitive to class skew)
  * Brier score           (calibration quality, lower=better)
  * ECE (10-bin)          (expected calibration error)
  * 1 000-resample bootstrap 95 % CIs for AUROC / AUPRC

Inputs are JSONL produced by `infer.py`:
    {"patient_id": "...", "label": 0|1, "score": float}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = [], []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            labels.append(int(r["label"]))
            scores.append(float(r["score"]))
    return np.asarray(labels), np.asarray(scores)


def _ece(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids  = np.digitize(scores, bins) - 1
    ids  = np.clip(ids, 0, n_bins - 1)
    ece = 0.0
    n   = len(labels)
    for b in range(n_bins):
        m = ids == b
        if not np.any(m):
            continue
        bin_acc  = labels[m].mean()
        bin_conf = scores[m].mean()
        ece += (m.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _bootstrap_ci(metric_fn, labels, scores, n_iter=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n   = len(labels)
    vals = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(metric_fn(labels[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return (lo, hi)


def compute_metrics(labels: np.ndarray, scores: np.ndarray,
                    threshold: float = 0.5,
                    bootstrap: int = 1000) -> dict:
    preds = (scores >= threshold).astype(int)
    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    auprc = float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    f1    = float(f1_score(labels, preds, average="macro"))
    acc   = float(accuracy_score(labels, preds))
    cm    = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens  = tp / (tp + fn) if (tp + fn) else 0.0
    spec  = tn / (tn + fp) if (tn + fp) else 0.0
    ppv   = tp / (tp + fp) if (tp + fp) else 0.0
    npv   = tn / (tn + fn) if (tn + fn) else 0.0
    mcc   = float(matthews_corrcoef(labels, preds)) if len(np.unique(preds)) > 1 else 0.0
    brier = float(np.mean((scores - labels) ** 2))
    ece   = _ece(labels, scores)

    auroc_ci = _bootstrap_ci(lambda y, s: roc_auc_score(y, s), labels, scores, n_iter=bootstrap)
    auprc_ci = _bootstrap_ci(lambda y, s: average_precision_score(y, s), labels, scores, n_iter=bootstrap)

    return {
        "n":             int(len(labels)),
        "n_pos":         int(labels.sum()),
        "n_neg":         int(len(labels) - labels.sum()),
        "AUROC":         round(auroc, 4),
        "AUROC_95CI":    [round(auroc_ci[0], 4), round(auroc_ci[1], 4)],
        "AUPRC":         round(auprc, 4),
        "AUPRC_95CI":    [round(auprc_ci[0], 4), round(auprc_ci[1], 4)],
        "Macro_F1":      round(f1, 4),
        "Accuracy":      round(acc, 4),
        "Sensitivity":   round(sens, 4),
        "Specificity":   round(spec, 4),
        "PPV":           round(ppv, 4),
        "NPV":           round(npv, 4),
        "MCC":           round(mcc, 4),
        "Brier":         round(brier, 4),
        "ECE":           round(ece, 4),
        "Threshold":     threshold,
    }


def evaluate(predictions_path: str | Path, threshold: float = 0.5,
             bootstrap: int = 1000) -> dict:
    p = Path(predictions_path)
    labels, scores = _load(p)
    m = compute_metrics(labels, scores, threshold=threshold, bootstrap=bootstrap)
    out = p.with_suffix(".metrics.json")
    out.write_text(json.dumps({"path": str(p), **m}, indent=2))
    print("\n" + "=" * 50)
    print(f"Eval  : {p}")
    print("=" * 50)
    for k, v in m.items():
        if isinstance(v, list):
            v = f"[{v[0]:.4f}, {v[1]:.4f}]"
        elif isinstance(v, float):
            v = f"{v:.4f}"
        print(f"  {k:<15s}: {v}")
    print(f"\n  metrics → {out}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--threshold",   type=float, default=0.5)
    ap.add_argument("--bootstrap",   type=int,   default=1000)
    args = ap.parse_args()
    evaluate(args.predictions, threshold=args.threshold, bootstrap=args.bootstrap)
