"""
run_finetuned_mistral_xgb.py

Hybrid First_Recur imbalance experiment:

    RAG prompt -> fine-tuned Mistral P(Yes) -> class-weighted XGBoost

Important leakage guard
-----------------------
Do not train XGBoost on `predictions_train.jsonl` from the same LoRA adapter
that was fine-tuned on those patients. Those are in-sample predictions and can
be near memorized labels. By default this script requires out-of-fold (OOF)
train scores:

    Imbalance/oof_predictions/<cell_tag>/predictions_train_oof.jsonl

Validation/test scores can reuse the normal fine-tuned LoRA inference outputs
written by `First_Recur/Model/code/infer.py`:

    Model/results/<cell_tag>/predictions_valid.jsonl
    Model/results/<cell_tag>/predictions_test.jsonl

For quick ablations only, pass `--allow-in-sample-train-scores`. Results from
that mode should be labelled as optimistic/stacking-leakage-prone and should
not be used as the main submission result.

Usage:
    cd 24083155

    # Safe mode, requires OOF train scores:
    python Imbalance/run_finetuned_mistral_xgb.py --cell-tag Exp4__beep__beep

    # Quick diagnostic only, leakage-prone:
    python Imbalance/run_finetuned_mistral_xgb.py --cell-tag Exp4__beep__beep --allow-in-sample-train-scores

    # Run all cells that have result directories:
    python Imbalance/run_finetuned_mistral_xgb.py --all --run-missing-infer
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise SystemExit(
        "xgboost is not installed in this Python environment. Run with the "
        "project environment, for example: "
        "python first/Imbalance/run_finetuned_mistral_xgb.py --cell-tag <tag>"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "dataset" / "first"
MODEL_RESULTS = ROOT / "Model" / "results"
MODEL_CODE = ROOT / "Model" / "code"
SPLIT_DIR = DATA_ROOT / "splits"
OOF_DIR = ROOT / "Imbalance" / "oof_predictions"
OUT_DIR = ROOT / "Imbalance" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "aggregate_finetuned_mistral_xgb.csv"

SPLIT_FILE = {
    "train": "predictions_train.jsonl",
    "valid": "predictions_valid.jsonl",
    "test": "predictions_test.jsonl",
}


class MissingOOFTrainScoresError(FileNotFoundError):
    """Raised when safe stacking cannot run because OOF train scores are absent."""


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_prediction_file(cell_tag: str, split: str, run_missing_infer: bool,
                            allow_in_sample_train_scores: bool) -> Path:
    if split == "train":
        oof_path = OOF_DIR / cell_tag / "predictions_train_oof.jsonl"
        if oof_path.exists():
            return oof_path
        if not allow_in_sample_train_scores:
            raise MissingOOFTrainScoresError(
                f"Missing safe OOF train scores: {oof_path}\n"
                "Using Model/results/<cell>/predictions_train.jsonl would be "
                "stacking leakage because the LoRA adapter was trained on those "
                "patients. Create OOF train predictions first, or pass "
                "--allow-in-sample-train-scores only for a clearly-labelled "
                "diagnostic run."
            )

    path = MODEL_RESULTS / cell_tag / SPLIT_FILE[split]
    if path.exists():
        if split == "train":
            print(
                "[warning] using in-sample train Mistral scores. "
                "This is leakage-prone and should not be a main result."
            )
        return path
    if not run_missing_infer:
        raise FileNotFoundError(
            f"Missing {path}. Re-run with --run-missing-infer to generate it."
        )
    if split == "train" and not allow_in_sample_train_scores:
        raise MissingOOFTrainScoresError(
            "Refusing to generate in-sample train predictions in safe mode. "
            "Use OOF predictions for train, or pass --allow-in-sample-train-scores "
            "for a diagnostic run."
        )

    sys.path.insert(0, str(MODEL_CODE))
    import infer as infer_mod

    print(f"[infer] missing {split} predictions for {cell_tag}; generating now")
    infer_mod.infer(cell_tag, split=split)
    if not path.exists():
        raise FileNotFoundError(f"infer.py finished but {path} was not created")
    return path


def _load_mistral_scores(cell_tag: str, split: str, run_missing_infer: bool,
                         allow_in_sample_train_scores: bool) -> pd.DataFrame:
    path = _ensure_prediction_file(
        cell_tag, split, run_missing_infer, allow_in_sample_train_scores
    )
    rows = _read_jsonl(path)
    df = pd.DataFrame(rows)
    if "patient_id" not in df.columns or "label" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path} must contain patient_id, label, and score")

    p_yes = pd.to_numeric(df["score"], errors="coerce").clip(1e-6, 1 - 1e-6)
    out = pd.DataFrame({
        "Patient_ID": df["patient_id"].astype(str),
        "label": df["label"].astype(int),
        "mistral_p_yes": p_yes,
        "mistral_p_no": 1.0 - p_yes,
        "mistral_logit_margin": np.log(p_yes / (1.0 - p_yes)),
    })
    if out["mistral_p_yes"].isna().any():
        raise ValueError(f"{path} contains NaN scores")
    return out


def _load_clinical_split(split: str) -> pd.DataFrame:
    name = {"train": "Train", "valid": "Validation", "test": "Test"}[split]
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
    if "y" not in df.columns:
        raise ValueError(f"{name}.csv must contain y")
    df = df.rename(columns={"y": "label"})
    return df


def _merge_scores_with_clinical(score_df: pd.DataFrame, split: str,
                                feature_mode: str,
                                llm_score_features: str) -> pd.DataFrame:
    clinical = _load_clinical_split(split)
    merged = clinical.merge(score_df, on=["Patient_ID", "label"], how="inner")
    if len(merged) != len(score_df) or len(merged) != len(clinical):
        raise AssertionError(
            f"{split} merge mismatch: scores={len(score_df)}, "
            f"clinical={len(clinical)}, merged={len(merged)}"
        )

    if feature_mode == "llm":
        keep = ["Patient_ID", "label", *_llm_feature_columns(llm_score_features)]
        return merged[keep]
    if feature_mode == "clinical":
        drop = ["mistral_p_yes", "mistral_p_no", "mistral_logit_margin"]
        return merged.drop(columns=drop)
    if feature_mode == "combined":
        if llm_score_features == "margin":
            merged = merged.drop(columns=["mistral_p_yes", "mistral_p_no"])
        return merged
    raise ValueError(feature_mode)


def _llm_feature_columns(llm_score_features: str) -> list[str]:
    if llm_score_features == "margin":
        return ["mistral_logit_margin"]
    if llm_score_features == "all":
        return ["mistral_p_yes", "mistral_p_no", "mistral_logit_margin"]
    raise ValueError(llm_score_features)


def _make_design_matrices(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                          test_df: pd.DataFrame):
    y_train = train_df["label"].astype(int).to_numpy()
    y_valid = valid_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    pid_test = test_df["Patient_ID"].astype(str).tolist()

    feature_cols = [c for c in train_df.columns if c not in {"Patient_ID", "label"}]
    X_train_raw = train_df[feature_cols].copy()
    X_valid_raw = valid_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    all_raw = pd.concat(
        [X_train_raw, X_valid_raw, X_test_raw],
        axis=0,
        keys=["train", "valid", "test"],
    )
    all_raw = all_raw.replace({pd.NA: np.nan})
    all_enc = pd.get_dummies(all_raw, dummy_na=True)
    all_enc = all_enc.apply(pd.to_numeric, errors="coerce").fillna(-999.0)

    X_train = all_enc.loc["train"].to_numpy(dtype=np.float32)
    X_valid = all_enc.loc["valid"].to_numpy(dtype=np.float32)
    X_test = all_enc.loc["test"].to_numpy(dtype=np.float32)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, pid_test, list(all_enc.columns)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().to_dict()
    n = len(y)
    k = len(counts)
    return np.asarray([n / (k * counts[int(v)]) for v in y], dtype=np.float32)


def _bootstrap_ci(metric_fn, y, p, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _metrics(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = (p >= tau).astype(int)
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    auroc_lo, auroc_hi = _bootstrap_ci(roc_auc_score, y, p)
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    auprc_lo, auprc_hi = _bootstrap_ci(average_precision_score, y, p)
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y, pred))
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1) if (tp + fp) else 0.0
    npv = tn / max(tn + fn, 1) if (tn + fn) else 0.0
    mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0
    brier = float(np.mean((p - y) ** 2))
    return {
        "AUROC": round(auroc, 4),
        "AUROC_lo": round(auroc_lo, 4),
        "AUROC_hi": round(auroc_hi, 4),
        "AUPRC": round(auprc, 4),
        "AUPRC_lo": round(auprc_lo, 4),
        "AUPRC_hi": round(auprc_hi, 4),
        "Macro_F1": round(f1, 4),
        "Accuracy": round(acc, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "PPV": round(ppv, 4),
        "NPV": round(npv, 4),
        "MCC": round(mcc, 4),
        "Brier": round(brier, 4),
    }


def _tune_tau(y_valid: np.ndarray, p_valid: np.ndarray) -> tuple[float, float]:
    best_tau = 0.5
    best_f1 = -math.inf
    for tau in np.arange(0.05, 0.96, 0.01):
        pred = (p_valid >= tau).astype(int)
        score = f1_score(y_valid, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_tau = float(tau)
    return round(best_tau, 2), round(best_f1, 4)


def run_one_cell(cell_tag: str, feature_mode: str,
                 run_missing_infer: bool = False,
                 allow_in_sample_train_scores: bool = False,
                 llm_score_features: str = "margin") -> dict:
    split_frames = {}
    for split in ("train", "valid", "test"):
        scores = _load_mistral_scores(
            cell_tag, split, run_missing_infer, allow_in_sample_train_scores
        )
        split_frames[split] = _merge_scores_with_clinical(
            scores, split, feature_mode, llm_score_features
        )

    X_tr, y_tr, X_va, y_va, X_te, y_te, pid_te, feature_names = _make_design_matrices(
        split_frames["train"], split_frames["valid"], split_frames["test"]
    )
    weights = _balanced_sample_weights(y_tr)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=4,
        device="cpu",
    )
    model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_va, y_va)], verbose=False)

    p_valid = model.predict_proba(X_va)[:, 1]
    p_test = model.predict_proba(X_te)[:, 1]
    tau_star, valid_macro_f1 = _tune_tau(y_va, p_valid)
    m_default = _metrics(y_te, p_test, tau=0.5)
    m_tuned = _metrics(y_te, p_test, tau=tau_star)

    out_cell = OUT_DIR / "finetuned_mistral_xgb" / f"{feature_mode}__{cell_tag}"
    out_cell.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Patient_ID": pid_te,
        "label": y_te,
        "score": p_test,
        "pred_default": (p_test >= 0.5).astype(int),
        "pred_tuned": (p_test >= tau_star).astype(int),
    }).to_csv(out_cell / "predictions_test.csv", index=False)

    row = {
        "cell_tag": cell_tag,
        "feature_mode": feature_mode,
        "train_score_source": (
            "in_sample_leakage_prone"
            if allow_in_sample_train_scores else
            "oof"
        ),
        "llm_score_features": llm_score_features,
        "n_features": int(X_tr.shape[1]),
        "n_train": int(len(y_tr)),
        "n_train_pos": int((y_tr == 1).sum()),
        "n_train_neg": int((y_tr == 0).sum()),
        "weight_label0": round(float(weights[y_tr == 0][0]), 4) if np.any(y_tr == 0) else float("nan"),
        "weight_label1": round(float(weights[y_tr == 1][0]), 4) if np.any(y_tr == 1) else float("nan"),
        "tau_star": tau_star,
        "valid_macro_f1_at_tau_star": valid_macro_f1,
        **{f"{k}_default": v for k, v in m_default.items()},
        **{f"{k}_tuned": v for k, v in m_tuned.items()},
    }
    (out_cell / "metrics.json").write_text(json.dumps(row, indent=2))

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.head(100).to_csv(out_cell / "feature_importance_top100.csv", index=False)

    print(
        f"[{feature_mode}/{cell_tag}] AUROC={m_default['AUROC']:.4f} "
        f"F1@0.5={m_default['Macro_F1']:.4f} "
        f"F1@tau*({tau_star})={m_tuned['Macro_F1']:.4f} "
        f"Spec={m_tuned['Specificity']:.3f} Sens={m_tuned['Sensitivity']:.3f}"
    )
    return row


def list_candidate_cells(require_train: bool = False) -> list[str]:
    cells = []
    for d in sorted(MODEL_RESULTS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        needed = ["predictions_valid.jsonl", "predictions_test.jsonl"]
        if require_train:
            needed.append("predictions_train.jsonl")
        if all((d / f).exists() for f in needed):
            cells.append(d.name)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-tag", help="single cell tag, e.g. Exp4__beep__beep")
    ap.add_argument("--all", action="store_true",
                    help="run all cells with valid/test prediction files")
    ap.add_argument("--feature-mode", default="combined",
                    choices=["llm", "clinical", "combined"],
                    help="llm=P(Yes)/P(No)/margin only; clinical=structured only; combined=both")
    ap.add_argument("--llm-score-features", default="margin",
                    choices=["margin", "all"],
                    help="margin uses only logit(P_yes/P_no); all also includes P(Yes) and P(No)")
    ap.add_argument("--run-missing-infer", action="store_true",
                    help="run fine-tuned Mistral inference for missing valid/test predictions")
    ap.add_argument("--allow-in-sample-train-scores", action="store_true",
                    help="diagnostic only: allow leakage-prone train predictions from the same LoRA adapter")
    args = ap.parse_args()

    if not args.cell_tag and not args.all:
        ap.error("supply --cell-tag or --all")
    cells = list_candidate_cells(require_train=False) if args.all else [args.cell_tag]
    print(f"Running fine-tuned-Mistral -> weighted XGBoost for {len(cells)} cell(s)")
    for c in cells:
        print("  -", c)

    rows = []
    failures = 0
    for cell in cells:
        try:
            rows.append(run_one_cell(
                cell,
                args.feature_mode,
                args.run_missing_infer,
                args.allow_in_sample_train_scores,
                args.llm_score_features,
            ))
        except Exception as e:
            failures += 1
            print(f"[FAIL] {cell}: {type(e).__name__}: {e}")

    if rows:
        new = pd.DataFrame(rows)
        if OUT_CSV.exists():
            old = pd.read_csv(OUT_CSV)
            key_new = set(new[["cell_tag", "feature_mode"]].apply(tuple, axis=1))
            keep = ~old[["cell_tag", "feature_mode"]].apply(tuple, axis=1).isin(key_new)
            combined = pd.concat([old[keep], new], ignore_index=True)
        else:
            combined = new
        combined.to_csv(OUT_CSV, index=False)
        print(f"Aggregate -> {OUT_CSV} ({len(combined)} rows)")
    elif failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
