"""Prospective fixed-landmark clinical sensitivity baselines.

Prediction time is TTP1 + 90 days and the outcome is second progression in
the next 180 days. Patients without adequate follow-up are excluded. Existing
radiomic and VLM aggregates are deliberately not used because they were gated
with the original outcome-dependent landmark and cannot be safely re-gated
from their patient-level summaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT.parent / "dataset" / "second"
sys.path.insert(0, str(DATASET_ROOT))

import _landmark as landmark  # noqa: E402
import audit_followup_censoring as followup  # noqa: E402
import run_final_baselines_cv as baseline  # noqa: E402
import run_xgboost as common  # noqa: E402


OUT_DIR = ROOT / "Model" / "results" / "fixed_landmark_baselines_cv"
OUT_CSV = ROOT / "Model" / "results" / "aggregate_fixed_landmark_baselines_cv.csv"

TIMING_GROUPS = [
    (
        "Initial Chemo Therapy",
        " Number of days from Diagnosis to Initial Chemo Therapy Start date",
        [" Number of days from Diagnosis to Initial Chemo Therapy end date", "Name of Initial Chemo Therapy"],
    ),
    (
        "Radiation Therapy",
        "Number of days from Diagnosis to Radiation Therapy Start date",
        ["Number of days from Diagnosis to Radiation Therapy end date", "Dose", "Number of Fractions"],
    ),
    (
        "Additional Therapy",
        "Number of Days from Diagnosis to Starting Additional Therapy ",
        [
            "Cycle length of Additional Therapy (q days)",
            "Number of Days from Diagnosis to Complete Additional Therapy ",
            "Number of Cycles of Additional Therapy",
        ],
    ),
    (
        "Immuno therapy",
        "Number of Days from Diagnosis to Start Immunotherapy ",
        [
            "Cycle length of Immunotherapy (q days)",
            "Number of Days from Diagnosis to Complete Immunotherapy ",
            "Number of Cycles of Immunotherapy",
        ],
    ),
    (
        "Brachy therapy",
        "Number of Days from Diagnosis to the day of Insertion of Brachytherapy ",
        [],
    ),
    (
        "Other Types of Therapy (LITT, more chemo, proton therapy)",
        "Number of Days from Diagnosis to Start Other Additional Therapy ",
        ["Number of Days from Diagnosis to Complete Other Additional Therapy "],
    ),
]

CYCLE_GROUPS = [
    (
        "Number of Days from Diagnosis to Starting Additional Therapy ",
        "Cycle length of Additional Therapy (q days)",
        "Number of Cycles of Additional Therapy",
    ),
    (
        "Number of Days from Diagnosis to Start Immunotherapy ",
        "Cycle length of Immunotherapy (q days)",
        "Number of Cycles of Immunotherapy",
    ),
]


def _fixed_clinical(anchor_offset: int, horizon: int) -> pd.DataFrame:
    audit, _ = followup.build(anchor_offset=anchor_offset, horizon=horizon)
    selected = audit.loc[audit["prospective_eligible"], [
        "Patient_ID", "fixed_landmark_day", "prospective_y"
    ]].copy()

    # Start from the already tier-screened clinical table, then re-mask every
    # time-stamped treatment against the earlier fixed landmark.
    clinical = pd.read_csv(DATASET_ROOT / "Processed" / "clean_clinical.csv")
    clinical = clinical.drop(columns=["y", "Landmark_day"], errors="ignore").merge(
        selected, on="Patient_ID", how="inner", validate="one_to_one"
    )
    clinical = clinical.rename(columns={
        "fixed_landmark_day": "Landmark_day",
        "prospective_y": "y",
    })
    clinical["y"] = clinical["y"].astype(int)

    for flag, start_col, companions in TIMING_GROUPS:
        if start_col not in clinical.columns:
            continue
        start = pd.to_numeric(clinical[start_col], errors="coerce")
        post = start.notna() & (start >= clinical["Landmark_day"])
        for column in [flag, start_col, *companions]:
            if column in clinical.columns:
                clinical.loc[post, column] = np.nan
        for column in companions:
            if "Complete" not in column and "end date" not in column:
                continue
            if column in clinical.columns:
                end = pd.to_numeric(clinical[column], errors="coerce")
                clinical.loc[end > clinical["Landmark_day"], column] = np.nan

    # A treatment that began before the landmark may still have a cycle count
    # recorded from later follow-up. Keep only cycles that could have finished
    # by the prediction date.
    for start_col, cycle_length_col, cycle_count_col in CYCLE_GROUPS:
        if not all(column in clinical.columns for column in (
            start_col, cycle_length_col, cycle_count_col
        )):
            continue
        start = pd.to_numeric(clinical[start_col], errors="coerce")
        cycle_length = pd.to_numeric(clinical[cycle_length_col], errors="coerce")
        cycle_count = pd.to_numeric(clinical[cycle_count_col], errors="coerce")
        possible = np.floor((clinical["Landmark_day"] - start) / cycle_length).clip(lower=0)
        cap = (
            start.notna()
            & cycle_length.notna()
            & (cycle_length > 0)
            & cycle_count.notna()
            & (cycle_count > possible)
        )
        clinical.loc[cap, cycle_count_col] = possible.loc[cap]

    if len(clinical) != 65 or clinical["Patient_ID"].nunique() != 65:
        raise RuntimeError("fixed-landmark cohort must contain 65 unique patients")
    return clinical


def _feature_table(anchor_offset: int, horizon: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    clinical = common._encode(_fixed_clinical(anchor_offset, horizon))

    def available(columns: list[str]) -> list[str]:
        return [column for column in columns if column in clinical.columns]

    no_molecular = (
        available(common.DEMOGRAPHIC_COLS)
        + available(common.DIAGNOSIS_COLS)
        + available(common.SR_INITIAL_TX_COLS)
        + available(common.SR_SALVAGE_TX_COLS)
    )
    with_molecular = no_molecular + available(common.MOLECULAR_COLS)
    experiments = {
        "ExpA_TxNoMol": no_molecular,
        "ExpA_Tx": with_molecular,
    }
    keep = list(dict.fromkeys([column for cols in experiments.values() for column in cols]))
    return clinical[["Patient_ID", "y"] + keep], experiments


def run(anchor_offset: int = 90, horizon: int = 180) -> pd.DataFrame:
    frame, experiments = _feature_table(anchor_offset, horizon)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "feature_columns.json").write_text(json.dumps(experiments, indent=2))
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pooled: dict[tuple[str, str], list[dict]] = {}

    for fold_id, (development_idx, test_idx) in enumerate(split.split(frame, frame["y"])):
        development = frame.iloc[development_idx]
        test = frame.iloc[test_idx]
        train_idx, valid_idx = train_test_split(
            np.arange(len(development)),
            test_size=0.20,
            stratify=development["y"],
            random_state=100 + fold_id,
        )
        train = development.iloc[train_idx]
        valid = development.iloc[valid_idx]

        for experiment, columns in experiments.items():
            medians = train[columns].median(numeric_only=True).reindex(columns).fillna(0.0)
            x_train = train[columns].fillna(medians).to_numpy(dtype=float)
            x_valid = valid[columns].fillna(medians).to_numpy(dtype=float)
            x_test = test[columns].fillna(medians).to_numpy(dtype=float)
            y_train = train["y"].to_numpy(dtype=int)
            y_valid = valid["y"].to_numpy(dtype=int)
            y_test = test["y"].to_numpy(dtype=int)
            scaler = StandardScaler().fit(x_train)
            scaled = {
                "train": scaler.transform(x_train),
                "valid": scaler.transform(x_valid),
                "test": scaler.transform(x_test),
            }
            pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

            for model_name, model in baseline._models(42 + fold_id, pos_weight).items():
                use_scaled = model_name in {"LogisticRegression", "MLP_64x32"}
                model.fit(scaled["train"] if use_scaled else x_train, y_train)
                valid_prob = model.predict_proba(scaled["valid"] if use_scaled else x_valid)[:, 1]
                test_prob = model.predict_proba(scaled["test"] if use_scaled else x_test)[:, 1]
                if model_name == "Majority":
                    test_prob = np.full(len(y_test), 0.5)
                    calibrated = test_prob.copy()
                else:
                    calibrated = baseline._platt(y_valid, valid_prob, test_prob)

                for patient_id, label, raw, platt in zip(
                    test["Patient_ID"], y_test, test_prob, calibrated
                ):
                    pooled.setdefault((experiment, model_name), []).append({
                        "Patient_ID": patient_id,
                        "label": int(label),
                        "fold": fold_id,
                        "score_raw": float(raw),
                        "score_platt": float(platt),
                    })

    summaries = []
    for (experiment, model_name), rows in sorted(pooled.items()):
        predictions = pd.DataFrame(rows)
        if len(predictions) != 65 or predictions["Patient_ID"].nunique() != 65:
            raise RuntimeError(f"{experiment}/{model_name} does not have 65 unique OOF patients")
        predictions.to_csv(OUT_DIR / f"{experiment}__{model_name}.csv", index=False)
        y = predictions["label"].to_numpy(dtype=int)
        raw = baseline._metrics(y, predictions["score_raw"].to_numpy(dtype=float))
        platt = baseline._metrics(y, predictions["score_platt"].to_numpy(dtype=float))
        summaries.append({
            "experiment": experiment,
            "model": model_name,
            "anchor_offset_days": anchor_offset,
            "horizon_days": horizon,
            "n_oof": len(predictions),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "n_features": len(experiments[experiment]),
            **{f"raw_{key}": value for key, value in raw.items()},
            **{f"platt_{key}": value for key, value in platt.items()},
        })

    result = pd.DataFrame(summaries)
    result.to_csv(OUT_CSV, index=False)
    metadata = {
        "prediction_landmark": f"TTP1 + {anchor_offset} days",
        "prediction_horizon_days": horizon,
        "n": 65,
        "n_positive": 31,
        "n_negative": 34,
        "modalities": "clinical only",
        "reason_modalities_excluded": (
            "Existing radiomic/VLM patient aggregates were gated by the original "
            "outcome-dependent landmark and cannot be safely re-gated."
        ),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(result[[
        "experiment", "model", "platt_AUROC", "platt_AUPRC",
        "platt_Macro_F1", "platt_Sensitivity", "platt_Specificity",
    ]].to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_DIR / 'feature_columns.json'}")
    return result


if __name__ == "__main__":
    run()
