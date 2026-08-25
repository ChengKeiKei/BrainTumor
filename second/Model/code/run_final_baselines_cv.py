"""Leakage-safe 5-fold conventional baselines for Second Recurrence.

Each fold uses its own train/validation/test partition. Imputation and scaling
are fit on fold-train only; Platt scaling is fit on fold-validation only; the
reported predictions are pooled out-of-fold test predictions for all 147
patients. This is the conventional baseline companion to ``run_cv_grid.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import run_xgboost as common


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT.parent / "dataset" / "second"
DATA = DATASET_ROOT / "Processed"
FOLDS = DATASET_ROOT / "splits_cv"
OUT_DIR = ROOT / "Model" / "results" / "final_baselines_cv"
OUT_CSV = ROOT / "Model" / "results" / "aggregate_final_baselines_cv.csv"
EPS = 1e-6


def _feature_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    clinical_raw = pd.read_csv(DATA / "clean_clinical.csv")
    clinical = common._encode(common._cap_mri_days_at_landmark(clinical_raw))
    radiomic = pd.read_csv(DATA / "radiomic_features.csv")
    landmark = dict(zip(
        clinical_raw["Patient_ID"],
        pd.to_numeric(clinical_raw["Landmark_day"], errors="coerce"),
    ))
    vlm = common._build_vlm_table(
        DATA / "mri_captions_v3_structured.csv", landmark_map=landmark
    )

    def available(cols: list[str], frame: pd.DataFrame) -> list[str]:
        return [col for col in cols if col in frame.columns]

    demo = available(common.DEMOGRAPHIC_COLS, clinical)
    diagnosis = available(common.DIAGNOSIS_COLS, clinical)
    molecular = available(common.MOLECULAR_COLS, clinical)
    initial = available(common.SR_INITIAL_TX_COLS, clinical)
    salvage = available(common.SR_SALVAGE_TX_COLS, clinical)
    timepoints = available(common.SR_TIMEPOINT_COLS, clinical)
    rad = available(common.RADIOMIC_COLS, radiomic)
    vlm_cols = available(common.VLM_V3_FIELDS, vlm)

    tx_no_mol = demo + diagnosis + initial + salvage
    tx = tx_no_mol + molecular
    experiments = {
        "ExpA_TxNoMol": tx_no_mol,
        "ExpA_Tx": tx,
        "ExpB_TxRadiomic": tx + timepoints + rad,
        "ExpC_TxVLM": tx + timepoints + vlm_cols,
        "ExpD_TxRadVLM": tx + timepoints + rad + vlm_cols,
    }

    clinical_cols = list(dict.fromkeys(tx + timepoints))
    merged = (
        clinical[["Patient_ID"] + clinical_cols]
        .merge(radiomic[["Patient_ID"] + rad], on="Patient_ID", how="left")
        .merge(vlm[["Patient_ID"] + vlm_cols], on="Patient_ID", how="left")
    )
    return merged, experiments


def _platt(y_valid: np.ndarray, valid_prob: np.ndarray, test_prob: np.ndarray) -> np.ndarray:
    valid_prob = np.clip(valid_prob, EPS, 1 - EPS)
    test_prob = np.clip(test_prob, EPS, 1 - EPS)
    valid_logit = np.log(valid_prob / (1 - valid_prob)).reshape(-1, 1)
    test_logit = np.log(test_prob / (1 - test_prob)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
    calibrator.fit(valid_logit, y_valid)
    return calibrator.predict_proba(test_logit)[:, 1]


def _models(seed: int, pos_weight: float) -> dict[str, object]:
    return {
        "Majority": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
            device="cpu",
        ),
        "MLP_64x32": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            batch_size=16,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=seed,
        ),
    }


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, metric, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(metric(y[index], p[index]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prediction = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    auc_lo, auc_hi = _bootstrap_ci(y, p, roc_auc_score)
    f1_metric = lambda yy, pp: f1_score(
        yy, (pp >= 0.5).astype(int), average="macro", zero_division=0
    )
    f1_lo, f1_hi = _bootstrap_ci(y, p, f1_metric, seed=43)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUROC_lo": auc_lo,
        "AUROC_hi": auc_hi,
        "AUPRC": average_precision_score(y, p),
        "Macro_F1": f1_score(y, prediction, average="macro", zero_division=0),
        "Macro_F1_lo": f1_lo,
        "Macro_F1_hi": f1_hi,
        "Accuracy": accuracy_score(y, prediction),
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "MCC": matthews_corrcoef(y, prediction),
        "Brier": brier_score_loss(y, p),
    }


def run(experiments_to_run: set[str] | None = None) -> pd.DataFrame:
    features, experiments = _feature_table()
    if experiments_to_run:
        experiments = {k: v for k, v in experiments.items() if k in experiments_to_run}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "feature_columns.json").write_text(json.dumps(experiments, indent=2))

    pooled: dict[tuple[str, str], list[dict]] = {}
    for fold_id in range(5):
        fold = FOLDS / f"fold{fold_id}"
        train = pd.read_csv(fold / "Train.csv")
        valid = pd.read_csv(fold / "Validation.csv")
        test = pd.read_csv(fold / "Test.csv")

        for experiment, columns in experiments.items():
            def joined(split: pd.DataFrame) -> pd.DataFrame:
                return split[["Patient_ID", "y"]].merge(features, on="Patient_ID", how="left")

            tr, va, te = joined(train), joined(valid), joined(test)
            medians = tr[columns].median(numeric_only=True).reindex(columns).fillna(0.0)
            x_train = tr[columns].fillna(medians).to_numpy(dtype=float)
            x_valid = va[columns].fillna(medians).to_numpy(dtype=float)
            x_test = te[columns].fillna(medians).to_numpy(dtype=float)
            y_train = tr["y"].to_numpy(dtype=int)
            y_valid = va["y"].to_numpy(dtype=int)
            y_test = te["y"].to_numpy(dtype=int)

            scaler = StandardScaler().fit(x_train)
            scaled = {
                "train": scaler.transform(x_train),
                "valid": scaler.transform(x_valid),
                "test": scaler.transform(x_test),
            }
            pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

            for model_name, model in _models(42 + fold_id, pos_weight).items():
                use_scaled = model_name in {"LogisticRegression", "MLP_64x32"}
                fit_x = scaled["train"] if use_scaled else x_train
                valid_x = scaled["valid"] if use_scaled else x_valid
                test_x = scaled["test"] if use_scaled else x_test
                model.fit(fit_x, y_train)
                valid_raw = model.predict_proba(valid_x)[:, 1]
                test_raw = model.predict_proba(test_x)[:, 1]
                if model_name == "Majority":
                    # A true non-discriminating reference must have one score
                    # for every patient. Fold-specific prevalence scores can
                    # otherwise create a spurious pooled OOF AUROC.
                    test_raw = np.full(len(y_test), 0.5)
                    test_platt = test_raw.copy()
                else:
                    test_platt = _platt(y_valid, valid_raw, test_raw)

                key = (experiment, model_name)
                for patient_id, label, raw, calibrated in zip(
                    te["Patient_ID"], y_test, test_raw, test_platt
                ):
                    pooled.setdefault(key, []).append({
                        "Patient_ID": patient_id,
                        "label": int(label),
                        "fold": fold_id,
                        "score_raw": float(raw),
                        "score_platt": float(calibrated),
                    })

    summaries = []
    for (experiment, model_name), rows in sorted(pooled.items()):
        frame = pd.DataFrame(rows)
        if frame["Patient_ID"].nunique() != 147 or len(frame) != 147:
            raise RuntimeError(f"{experiment}/{model_name} does not contain 147 unique OOF patients")
        frame.to_csv(OUT_DIR / f"{experiment}__{model_name}.csv", index=False)
        y = frame["label"].to_numpy(dtype=int)
        raw = _metrics(y, frame["score_raw"].to_numpy(dtype=float))
        calibrated = _metrics(y, frame["score_platt"].to_numpy(dtype=float))
        summaries.append({
            "experiment": experiment,
            "model": model_name,
            "n_oof": len(frame),
            "n_features": len(experiments[experiment]),
            **{f"raw_{key}": value for key, value in raw.items()},
            **{f"platt_{key}": value for key, value in calibrated.items()},
        })

    result = pd.DataFrame(summaries)
    result.to_csv(OUT_CSV, index=False)
    print(result[[
        "experiment", "model", "platt_AUROC", "platt_AUPRC",
        "platt_Macro_F1", "platt_Sensitivity", "platt_Specificity",
    ]].to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_DIR / 'feature_columns.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=None)
    args = parser.parse_args()
    run(set(args.experiments) if args.experiments else None)


if __name__ == "__main__":
    main()
