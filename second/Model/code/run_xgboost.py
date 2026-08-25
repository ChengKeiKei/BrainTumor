"""
run_xgboost.py — XGBoost baseline for First_Recur AND Second_Recur.

Mirrors EXACTLY the same experiment design as the LoRA pipeline so that the
results are directly comparable cell-by-cell.

First_Recur experiments (n=203 patients, no radiomic / no VLM available):
  Exp1 :  Demographics + Diagnosis
  Exp2 :  Demographics + Diagnosis + Molecular
  Exp3 :  Demographics + Diagnosis + Treatment
  Exp4 :  Demographics + Diagnosis + Molecular + Treatment

Second_Recur experiments (n=147 patients, with radiomic + VLM):
  ExpA_TxNoMol     :  Demographics + Diagnosis + InitialTx + SalvageTx
  ExpA_Tx          :  Demographics + Diagnosis + Molecular + InitialTx + SalvageTx
  ExpB_TxRadiomic  :  ExpA_Tx + Timepoints + Radiomic
  ExpC_TxVLM       :  ExpA_Tx + Timepoints + VLM-v3 (7 binary fields)
  ExpD_TxRadVLM    :  ExpA_Tx + Timepoints + Radiomic + VLM-v3

Usage:
    python Model/code/run_xgboost.py                          # both projects
    python Model/code/run_xgboost.py --project second_recur   # SR only
    python Model/code/run_xgboost.py --project first_recur    # FR only
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ===========================================================================
# Paths
# ===========================================================================
SR_ROOT   = Path(__file__).resolve().parents[2]      # Second_Recur/
DATA_ROOT = SR_ROOT.parent / "dataset"
SR_DATA   = DATA_ROOT / "second" / "Processed"
SR_SPLITS = DATA_ROOT / "second" / "splits"
RESULT_D  = SR_ROOT / "Model" / "results"
RESULT_D.mkdir(parents=True, exist_ok=True)
OUT_CSV   = RESULT_D / "aggregate_xgboost.csv"

FR_ROOT   = SR_ROOT.parent / "first"
FR_DATA   = DATA_ROOT / "first" / "Processed"
FR_SPLITS = DATA_ROOT / "first" / "splits"

# ===========================================================================
# Feature column groups — MATCH feature_render.py EXACTLY
# ===========================================================================
DEMOGRAPHIC_COLS = ["Sex at Birth", "Race", "Age at diagnosis"]

DIAGNOSIS_COLS = [
    "Primary Diagnosis",
    "Grade of Primary Brain Tumor",
    "Stereotactic Biopsy before Surgical Resection",
    "Previous Brain Tumor",
    "Type of previous brain tumor",
    "Year of previous surgery",
    "Grade of Previous brain tumor",
]

MOLECULAR_COLS = [
    "IDH1 mutation", "IDH2 mutation", "1p/19q", "ATRX mutation",
    "MGMT methylation", "BRAF V600E mutation", "TERT promoter mutation",
    "Chromosome 7 gain and Chromosome 10 loss", "H3-3A mutation",
    "EGFR amplification", "PTEN mutation", "CDKN2A/B deletion",
    "TP53 alteration",
]

# First_Recur Treatment block (initial therapy only)
FR_TREATMENT_COLS = [
    "Number of days from Diagnosis to First surgery or procedure ",
    "Initial Chemo Therapy",
    "Name of Initial Chemo Therapy",
    " Number of days from Diagnosis to Initial Chemo Therapy Start date",
    " Number of days from Diagnosis to Initial Chemo Therapy end date",
    "Radiation Therapy",
    "Number of days from Diagnosis to Radiation Therapy Start date",
    "Number of days from Diagnosis to Radiation Therapy end date",
    "Dose",
    "Number of Fractions",
]

# Second_Recur — InitialTx (same as First_Recur) + SalvageTx blocks
SR_INITIAL_TX_COLS = FR_TREATMENT_COLS

SR_SALVAGE_TX_COLS = [
    "Additional Therapy",
    "Cycle length of Additional Therapy (q days)",
    "Number of Days from Diagnosis to Starting Additional Therapy ",
    "Number of Days from Diagnosis to Complete Additional Therapy ",
    "Number of Cycles of Additional Therapy",
    "Immuno therapy",
    "Cycle length of Immunotherapy (q days)",
    "Number of Days from Diagnosis to Start Immunotherapy ",
    "Number of Days from Diagnosis to Complete Immunotherapy ",
    "Number of Cycles of Immunotherapy",
    "Brachy therapy",
    "Number of Days from Diagnosis to the day of Insertion of Brachytherapy ",
    "Other Types of Therapy (LITT, more chemo, proton therapy)",
    "Number of Days from Diagnosis to Start Other Additional Therapy ",
    "Number of Days from Diagnosis to Complete Other Additional Therapy ",
]

SR_TIMEPOINT_COLS = [
    "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ",
    "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ",
    "Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ",
    "Number of Days from Diagnosis to 4th MRI (Timepoint_4) ",
    "Number of Days from Diagnosis to 5th MRI (Timepoint_5) ",
    "Number of Days from Diagnosis to 6th MRI (Timepoint_6) ",
]

RADIOMIC_COLS = [
    't1c_mean_ET', 't1c_mean_NCR', 't1c_mean_RC', 't1c_mean_SNFH',
    't1n_mean_ET', 't1n_mean_NCR', 't1n_mean_RC', 't1n_mean_SNFH',
    't2f_mean_ET', 't2f_mean_NCR', 't2f_mean_RC', 't2f_mean_SNFH',
    't2w_mean_ET', 't2w_mean_NCR', 't2w_mean_RC', 't2w_mean_SNFH',
    'vol_ET', 'vol_NCR', 'vol_RC', 'vol_SNFH',
    'n_pre_landmark_scans', 'latest_scan_day',
]

VLM_V3_FIELDS = [
    "enhancement", "necrosis", "hemorrhage",
    "edema", "mass_effect", "multifocal", "larger_baseline",
]


# ===========================================================================
# Helpers
# ===========================================================================
def _cap_mri_days_at_landmark(df: pd.DataFrame) -> pd.DataFrame:
    """Null out raw MRI day cells where ``day >= Landmark_day``.

    Mirrors the cap applied in ``feature_render._block_timepoints()`` so
    the XGBoost SR baseline sees the same pre-landmark window the LLM
    does. Applied only to SR — FR XGBoost does not include MRI day
    columns. Post-landmark cells become NaN and are median-imputed
    downstream.
    """
    df = df.copy()
    if "Landmark_day" not in df.columns:
        return df
    L = pd.to_numeric(df["Landmark_day"], errors="coerce")
    for col in SR_TIMEPOINT_COLS:
        if col not in df.columns:
            continue
        v   = pd.to_numeric(df[col], errors="coerce")
        bad = v.notna() & L.notna() & (v >= L)
        if bad.any():
            df.loc[bad, col] = np.nan
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all object cols (except Patient_ID) to numeric."""
    df = df.copy()
    skip = {"Patient_ID"}
    if "Sex at Birth" in df.columns:
        df["Sex at Birth"] = (df["Sex at Birth"].astype(str).str.lower() == "male").astype(float)
    if "Race" in df.columns:
        race = df["Race"].astype(str).str.lower()
        df["Race"] = race.map({"white": 0, "black": 1, "asian": 2,
                               "hispanic": 3, "other": 4}).fillna(4).astype(float)
    if "Primary Diagnosis" in df.columns:
        df["Primary Diagnosis"] = (df["Primary Diagnosis"].astype(str).str.upper().str.contains("GBM")
                                   ).astype(float)
    for col in ["Initial Chemo Therapy", "Radiation Therapy",
                "Previous Brain Tumor", "Brachy therapy"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.lower() == "yes").astype(float)
    for col in ["Additional Therapy", "Name of Initial Chemo Therapy",
                "Type of previous brain tumor",
                "Other Types of Therapy (LITT, more chemo, proton therapy)",
                "Immuno therapy"]:
        if col in df.columns:
            v = df[col].astype(str).str.strip()
            df[col] = (~v.isin(["", "nan", "None", "NaN"])).astype(float)
    for col in df.columns:
        if col not in skip and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_v3_caption(caption: str) -> dict[str, float]:
    """v3_structured caption → 7 numeric features. yes=1, no=0, unclear=0.5."""
    mapping = {"yes": 1.0, "no": 0.0}
    result  = {f: 0.5 for f in VLM_V3_FIELDS}
    key_map = {
        "contrast enhancement":     "enhancement",
        "necrotic core":            "necrosis",
        "intratumoral hemorrhage":  "hemorrhage",
        "peritumoral edema":        "edema",
        "mass effect":              "mass_effect",
        "multifocal":               "multifocal",
        "larger than expected":     "larger_baseline",
    }
    cap_lc = caption.lower()
    for phrase, field in key_map.items():
        m = re.search(rf"{re.escape(phrase)}[^;.]*?:\s*(yes|no|unclear)", cap_lc)
        if m:
            result[field] = mapping.get(m.group(1), 0.5)
    return result


def _build_vlm_table(captions_path: Path,
                     landmark_map: dict[str, float] | None = None) -> pd.DataFrame:
    """Aggregate v3 captions per patient (latest *pre-landmark* scan).

    If ``landmark_map`` (Patient_ID -> Landmark_day) is provided, captions
    whose ``Day_from_diag >= Landmark_day`` are dropped before the per-patient
    "latest" aggregation. This keeps the XGBoost VLM features in lock-step
    with the LLM prompt path (see ``feature_render._filter_captions_pre_landmark``).
    """
    df3 = pd.read_csv(captions_path)
    df3_ok = df3[df3["error"].isna()].copy()
    if landmark_map is not None:
        df3_ok["_L"] = df3_ok["Patient_ID"].map(landmark_map)
        d = pd.to_numeric(df3_ok["Day_from_diag"], errors="coerce")
        L = pd.to_numeric(df3_ok["_L"], errors="coerce")
        before = len(df3_ok)
        df3_ok = df3_ok[L.isna() | (d.notna() & (d < L))].drop(columns=["_L"])
        dropped = before - len(df3_ok)
        if dropped:
            print(f"[_build_vlm_table] dropped {dropped} caption rows at/after Landmark_day")
    df3_ok = df3_ok.sort_values("Day_from_diag")
    latest = df3_ok.groupby("Patient_ID").last().reset_index()
    rows = []
    for _, row in latest.iterrows():
        feats = _parse_v3_caption(str(row["caption"]))
        feats["Patient_ID"] = row["Patient_ID"]
        rows.append(feats)
    return pd.DataFrame(rows)


def _auroc_bootstrap_ci(y_true, y_prob, n_resamples=1000, seed=42):
    """Percentile bootstrap 95% CI for AUROC.

    Resamples (y_true, y_prob) PAIRS so the (label, score) link is preserved.
    Skips resamples that contain only one class (AUROC is undefined there).
    Returns (lo, hi) or (nan, nan) on failure.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        try:
            aucs.append(roc_auc_score(yt, y_prob[idx]))
        except Exception:
            continue
    if len(aucs) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _metrics(y_true, y_prob, tau=0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= tau).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auroc_lo, auroc_hi = _auroc_bootstrap_ci(y_true, y_prob, n_resamples=1000, seed=42)
    return {
        "AUROC":       round(auroc, 4),
        "AUROC_lo":    round(auroc_lo, 4) if not np.isnan(auroc_lo) else float("nan"),
        "AUROC_hi":    round(auroc_hi, 4) if not np.isnan(auroc_hi) else float("nan"),
        "AUPRC":       round(average_precision_score(y_true, y_prob), 4),
        "Macro_F1":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1), 4),
        "Specificity": round(((y_pred == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1), 4),
    }


def _tune_tau(y_val, prob_val) -> float:
    best_tau, best_f1 = 0.5, 0.0
    for tau in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_val, (prob_val >= tau).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return round(float(best_tau), 2)


def _train_one_cell(tag, feat_cols, train_df, val_df, test_df, merged):
    """Train one XGBoost cell; return result row + model."""
    def _merge(split):
        return split[["Patient_ID", "y"]].merge(merged, on="Patient_ID", how="left")

    # Fit imputation statistics on TRAIN only, then apply to val/test
    # (prevents leakage of val/test medians into the imputed features).
    tr = _merge(train_df)
    train_medians = tr[feat_cols].median(numeric_only=True)

    def _Xy(m):
        X = m[feat_cols].fillna(train_medians).values
        y = m["y"].astype(int).values
        return X, y

    X_tr, y_tr = _Xy(tr)
    X_va, y_va = _Xy(_merge(val_df))
    X_te, y_te = _Xy(_merge(test_df))

    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_w, eval_metric="logloss",
        use_label_encoder=False, random_state=42,
        n_jobs=4, device="cpu",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    prob_val  = model.predict_proba(X_va)[:, 1]
    prob_test = model.predict_proba(X_te)[:, 1]

    tau_star = _tune_tau(y_va, prob_val)
    m_default = _metrics(y_te, prob_test, tau=0.5)
    m_tuned   = _metrics(y_te, prob_test, tau=tau_star)

    n, n_pos = len(y_te), int(y_te.sum())
    row = {
        "tag": tag, "n_features": len(feat_cols),
        "n": n, "n_pos": n_pos, "n_neg": n - n_pos,
        "tau_star": tau_star,
        **{f"{k}_default": v for k, v in m_default.items()},
        **{f"{k}_tuned":   v for k, v in m_tuned.items()},
    }
    print(f"    AUROC={m_default['AUROC']:.4f}  "
          f"F1@0.5={m_default['Macro_F1']:.4f}  "
          f"F1@τ*{tau_star}={m_tuned['Macro_F1']:.4f}  "
          f"Sens={m_default['Sensitivity']:.3f} "
          f"Spec={m_default['Specificity']:.3f}")

    imp = sorted(zip(feat_cols, model.feature_importances_),
                 key=lambda x: -x[1])[:10]
    print("    Top 5 feats:", [(f, round(w, 3)) for f, w in imp[:5]])

    out_dir = RESULT_D / tag
    out_dir.mkdir(exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    pd.DataFrame(imp, columns=["feature", "importance"]).to_csv(
        out_dir / "feature_importance.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(row, f, indent=2)
    return row


# ===========================================================================
# First_Recur: 4 experiments × 1 model
# ===========================================================================
def run_first_recur(out_rows: list):
    print(f"\n{'='*60}\n  Project: FirstRecur\n{'='*60}")

    train_df = pd.read_csv(FR_SPLITS / "Train.csv")
    val_df   = pd.read_csv(FR_SPLITS / "Validation.csv")
    test_df  = pd.read_csv(FR_SPLITS / "Test.csv")

    clin = _encode(pd.read_csv(FR_DATA / "clean_clinical.csv"))

    avail_demo = [c for c in DEMOGRAPHIC_COLS    if c in clin.columns]
    avail_diag = [c for c in DIAGNOSIS_COLS      if c in clin.columns]
    avail_mol  = [c for c in MOLECULAR_COLS      if c in clin.columns]
    avail_tx   = [c for c in FR_TREATMENT_COLS   if c in clin.columns]

    base_cols = ["Patient_ID"] + list(set(avail_demo + avail_diag + avail_mol + avail_tx))
    merged = clin[base_cols]

    experiments = {
        "Exp1": avail_demo + avail_diag,
        "Exp2": avail_demo + avail_diag + avail_mol,
        "Exp3": avail_demo + avail_diag + avail_tx,
        "Exp4": avail_demo + avail_diag + avail_mol + avail_tx,
    }

    for exp_name, feat_cols in experiments.items():
        tag = f"XGB__FirstRecur__{exp_name}"
        print(f"\n  [{exp_name}]  ({len(feat_cols)} features)  tag={tag}")
        row = _train_one_cell(tag, feat_cols, train_df, val_df, test_df, merged)
        row["project"] = "FirstRecur"
        row["experiment"] = exp_name
        out_rows.append(row)


# ===========================================================================
# Second_Recur: 3 experiments × 1 model
# ===========================================================================
def run_second_recur(out_rows: list):
    print(f"\n{'='*60}\n  Project: SecondRecur\n{'='*60}")

    train_df = pd.read_csv(SR_SPLITS / "Train.csv")
    val_df   = pd.read_csv(SR_SPLITS / "Validation.csv")
    test_df  = pd.read_csv(SR_SPLITS / "Test.csv")

    # NOTE: cadence cap MUST run before _encode so the cap operates on the
    # original Landmark_day column (which _encode does not modify, but we
    # apply the cap explicitly first to keep this step auditable).
    clin_raw = pd.read_csv(SR_DATA / "clean_clinical.csv")
    clin     = _encode(_cap_mri_days_at_landmark(clin_raw))
    rad      = pd.read_csv(SR_DATA / "radiomic_features.csv")
    landmark_map = dict(zip(clin_raw["Patient_ID"],
                            pd.to_numeric(clin_raw["Landmark_day"], errors="coerce")))
    vlm      = _build_vlm_table(SR_DATA / "mri_captions_v3_structured.csv",
                                landmark_map=landmark_map)

    avail_demo  = [c for c in DEMOGRAPHIC_COLS    if c in clin.columns]
    avail_diag  = [c for c in DIAGNOSIS_COLS      if c in clin.columns]
    avail_mol   = [c for c in MOLECULAR_COLS      if c in clin.columns]
    avail_init  = [c for c in SR_INITIAL_TX_COLS  if c in clin.columns]
    avail_salv  = [c for c in SR_SALVAGE_TX_COLS  if c in clin.columns]
    avail_tp    = [c for c in SR_TIMEPOINT_COLS   if c in clin.columns]
    avail_rad   = [c for c in RADIOMIC_COLS       if c in rad.columns]
    avail_vlm   = [c for c in VLM_V3_FIELDS       if c in vlm.columns]

    expA_no_mol_cols = avail_demo + avail_diag + avail_init + avail_salv
    expA_cols = expA_no_mol_cols + avail_mol
    expB_cols = expA_cols + avail_tp + avail_rad
    expC_cols = expA_cols + avail_tp + avail_vlm
    expD_cols = expA_cols + avail_tp + avail_rad + avail_vlm

    # Build a single merged feature table with everything
    keep_clin = ["Patient_ID"] + list(set(expA_cols + avail_tp))
    merged = (
        clin[keep_clin]
        .merge(rad[["Patient_ID"] + avail_rad], on="Patient_ID", how="left")
        .merge(vlm,                              on="Patient_ID", how="left")
    )

    experiments = {
        "ExpA_TxNoMol":    expA_no_mol_cols,
        "ExpA_Tx":         expA_cols,
        "ExpB_TxRadiomic": expB_cols,
        "ExpC_TxVLM":      expC_cols,
        "ExpD_TxRadVLM":   expD_cols,
    }

    for exp_name, feat_cols in experiments.items():
        tag = f"XGB__SecondRecur__{exp_name}"
        print(f"\n  [{exp_name}]  ({len(feat_cols)} features)  tag={tag}")
        row = _train_one_cell(tag, feat_cols, train_df, val_df, test_df, merged)
        row["project"] = "SecondRecur"
        row["experiment"] = exp_name
        out_rows.append(row)


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="both",
                    choices=["second_recur", "first_recur", "both"])
    args = ap.parse_args()

    out_rows: list[dict] = []
    if args.project in ("first_recur", "both"):
        run_first_recur(out_rows)
    if args.project in ("second_recur", "both"):
        run_second_recur(out_rows)

    if not out_rows:
        return
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nResults → {OUT_CSV}")

    print("\n" + "="*70)
    print(f"{'tag':<45} {'AUROC':>7}  {'F1@0.5':>7}  {'F1(τ*)':>7}")
    print("="*70)
    for _, r in df_out.iterrows():
        print(f"  {r['tag']:<43} "
              f"{r['AUROC_default']:7.4f}  "
              f"{r['Macro_F1_default']:7.4f}  "
              f"{r['Macro_F1_tuned']:7.4f}")


if __name__ == "__main__":
    main()
