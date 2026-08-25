"""
External-validation evaluation on PPUM for ALL models, F1-first + AUPRC + AUROC + CI.

For each Exp and arm (no-RAG / RAG k3):
  * Platt calibrator is FIT on the MU validation split (Model/results/<mu_tag>/predictions_valid.jsonl)
    -> the in-distribution data the model was tuned on.
  * Applied to PPUM test scores
    (PPUM/generated/results/<mu_tag>__ppum/predictions_test.jsonl).
Also folds in the clinical-XGBoost external numbers for a single comparison table.

Run: python PPUM/eval_ppum.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, roc_auc_score)

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
MU_RES = SUBMISSION_ROOT / "first" / "Model" / "results"
PPUM_ROOT = Path(__file__).resolve().parent
PPUM_RES = PPUM_ROOT / "generated" / "results"
OUT = PPUM_ROOT / "generated" / "evaluation"; OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-6; RNG = np.random.default_rng(0)
EXPS = ["Exp1", "Exp2", "Exp3", "Exp4"]


def load_jsonl(p):
    if not p.exists(): return None, None
    y, s = [], []
    for ln in open(p):
        r = json.loads(ln); y.append(int(r["label"])); s.append(float(r["score"]))
    return np.array(y), np.clip(np.array(s), EPS, 1 - EPS)


def logit(p): return np.log(p / (1 - p))


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(MacroF1=f1_score(y, yh, average="macro"),
                Sens=tp/(tp+fn) if (tp+fn) else np.nan,
                Spec=tn/(tn+fp) if (tn+fp) else np.nan,
                AUPRC=average_precision_score(y, p) if len(set(y)) > 1 else np.nan,
                AUROC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan)


def boot_ci(y, p, thr=0.5, B=2000):
    n = len(y); v = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if len(set(y[idx])) < 2: continue
        v.append(f1_score(y[idx], (p[idx] >= thr).astype(int), average="macro"))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def eval_llm(mu_tag):
    yv, sv = load_jsonl(MU_RES / mu_tag / "predictions_valid.jsonl")     # MU valid (calibrator)
    yt, st = load_jsonl(PPUM_RES / f"{mu_tag}__ppum" / "predictions_test.jsonl")  # PPUM test
    if yt is None or yv is None: return None
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(sv).reshape(-1, 1), yv)
    p = lr.predict_proba(logit(st).reshape(-1, 1))[:, 1]
    lo, hi = boot_ci(yt, p)
    return {**metrics(yt, p), "F1_lo": lo, "F1_hi": hi}


rows = []
# clinical XGB external (from clinical_external_ppum.csv, if present)
clin_csv = OUT / "clinical_external_ppum.csv"
if clin_csv.exists():
    cdf = pd.read_csv(clin_csv)
    for exp in EXPS:
        r = cdf[(cdf.exp == exp) & (cdf.model == "xgboost")]
        if len(r):
            rows.append({"exp": exp, "system": "Clinical XGBoost",
                         "MacroF1": float(r["MacroF1"].iloc[0]),
                         "F1_lo": float(r["F1_lo"].iloc[0]), "F1_hi": float(r["F1_hi"].iloc[0]),
                         "Sens": float(r["Sens"].iloc[0]), "Spec": float(r["Spec"].iloc[0]),
                         "AUPRC": float(r["AUPRC"].iloc[0]), "AUROC": float(r["AUROC"].iloc[0])})

for exp in EXPS:
    for arm, label in [("baseline", "LLM no-RAG"), ("beep__beep", "LLM + RAG k3")]:
        m = eval_llm(f"{exp}__{arm}")
        if m is None:
            rows.append({"exp": exp, "system": label, "MacroF1": np.nan, "note": "MISSING"})
        else:
            rows.append({"exp": exp, "system": label, **m})

tab = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
cols = ["exp", "system", "MacroF1", "F1_lo", "F1_hi", "Sens", "Spec", "AUPRC", "AUROC"]
print("=== PPUM EXTERNAL VALIDATION (calibrated on MU validation, threshold 0.5) ===\n")
print(tab[[c for c in cols if c in tab.columns]].to_string(index=False))
tab.to_csv(OUT / "ppum_external_all_models.csv", index=False)

print("\n=== RAG vs no-RAG on PPUM (Macro-F1) ===")
piv = tab[tab.system.isin(["LLM no-RAG", "LLM + RAG k3"])].pivot_table(index="exp", columns="system", values="MacroF1")
if "LLM + RAG k3" in piv and "LLM no-RAG" in piv:
    piv["RAG-minus-noRAG"] = piv["LLM + RAG k3"] - piv["LLM no-RAG"]
print(piv.round(3).to_string())
print("\nsaved ->", OUT / "ppum_external_all_models.csv")
