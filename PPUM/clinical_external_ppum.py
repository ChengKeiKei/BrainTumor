"""
External validation of the CLINICAL-features-only models.
Train on the full MU-Glioma-Post cohort (203), test on the held-out PPUM cohort (31).
Exp1-4 feature groups. Reports Macro-F1@0.5, AUPRC, AUROC, Sens, Spec + bootstrap 95% CI.

Run: python PPUM/clinical_external_ppum.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, roc_auc_score)
from xgboost import XGBClassifier

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = SUBMISSION_ROOT / "dataset" / "first"
SPLIT_DIR = DATA_ROOT / "splits"
FEATURE_GROUPS_JSON = DATA_ROOT / "Processed" / "feature_groups.json"
PPUM_ROOT = Path(__file__).resolve().parent
PPUM = PPUM_ROOT / "generated" / "PPUM.csv"
OUT = PPUM_ROOT / "generated" / "evaluation"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
GROUPS = {"Exp1": "Exp1_metadata", "Exp2": "Exp2_metadata_molecular",
          "Exp3": "Exp3_metadata_treatment", "Exp4": "Exp4_metadata_molecular_treatment"}


def mu_pooled():
    frames = [pd.read_csv(SPLIT_DIR / f) for f in ["Train.csv", "Validation.csv", "Test.csv"]]
    return pd.concat(frames, ignore_index=True).drop_duplicates("Patient_ID")


def encode(train_df, test_df):
    num = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    cat = [c for c in train_df.columns if c not in num]
    tn = train_df[num].apply(pd.to_numeric, errors="coerce").fillna(-999.0)
    te = test_df[num].apply(pd.to_numeric, errors="coerce").fillna(-999.0)
    tc = pd.get_dummies(train_df[cat], dummy_na=True); tc = tc.loc[:, ~tc.columns.duplicated()]
    ec = pd.get_dummies(test_df[cat], dummy_na=True); ec = ec.loc[:, ~ec.columns.duplicated()]
    ec = ec.reindex(columns=tc.columns, fill_value=0.0)
    return (pd.concat([tn, tc], axis=1).to_numpy(np.float32),
            pd.concat([te, ec], axis=1).to_numpy(np.float32))


def model(kind, ytr):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    pos = int(ytr.sum()); spw = (len(ytr) - pos) / max(pos, 1)
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.85, min_child_weight=2, reg_lambda=2.0,
                         objective="binary:logistic", eval_metric="logloss",
                         scale_pos_weight=spw, random_state=0, n_jobs=4, device="cpu")


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
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


mu = mu_pooled(); ppum = pd.read_csv(PPUM)
groups = json.loads(FEATURE_GROUPS_JSON.read_text())
print(f"Train = MU pooled n={len(mu)} (pos={int(mu['y'].sum())}) | Test = PPUM n={len(ppum)} (pos={int(ppum['y'].sum())})\n")

rows = []
for exp, gname in GROUPS.items():
    cols = [c for c in groups[gname] if c in mu.columns and c in ppum.columns]
    ytr = mu["y"].astype(int).to_numpy(); yte = ppum["y"].astype(int).to_numpy()
    Xtr, Xte = encode(mu[cols], ppum[cols])
    for kind in ["logreg", "xgboost"]:
        m = model(kind, ytr); m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        met = metrics(yte, p); lo, hi = boot_ci(yte, p)
        rows.append({"exp": exp, "model": kind, **met, "F1_lo": lo, "F1_hi": hi})
        print(f"{exp} {kind:8s}: MacroF1={met['MacroF1']:.3f} [{lo:.3f},{hi:.3f}] "
              f"AUPRC={met['AUPRC']:.3f} AUROC={met['AUROC']:.3f} Sens={met['Sens']:.3f} Spec={met['Spec']:.3f}")

pd.DataFrame(rows).to_csv(OUT / "clinical_external_ppum.csv", index=False)
print("\nsaved ->", OUT / "clinical_external_ppum.csv")
