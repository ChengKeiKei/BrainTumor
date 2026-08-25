"""10-case stress test: is First Recurrence probability fixed, or a code bug?"""

from __future__ import annotations

from math import exp
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from src.encoding import encode_ui_clinical_row
from src.hybrid_inference import (
    DEFAULT_ARTIFACT,
    TRAINING_EMB_DIR,
    _clinical_row_from_ui,
    _encode_clinical,
    _iteration_kwargs,
    predict_first_hybrid,
)
from src.literature import LiteratureDoc


CASES: dict[str, dict] = {
    "01_GBM_G4_IDHwt_MGMTun_RT": {
        "sex": "Male",
        "race": "White",
        "age": 68,
        "primary_diagnosis": "Glioblastoma",
        "grade": "4",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "No",
        "idh1": "Wildtype / negative",
        "idh2": "Wildtype / negative",
        "codeletion_1p19q": "Wildtype / negative",
        "mgmt": "Unmethylated",
        "tert": "Mutant / positive",
        "egfr": "Mutant / positive",
        "first_surgery_day": 14,
        "initial_chemo": "Yes",
        "initial_chemo_name": "Temozolomide",
        "initial_chemo_start_day": 30,
        "initial_chemo_end_day": 210,
        "radiotherapy": "Yes",
        "rt_start_day": 30,
        "rt_end_day": 72,
        "rt_dose": 60,
        "rt_fractions": 30,
    },
    "02_Oligo_G2_IDHmut_codeleted_MGMTm": {
        "sex": "Female",
        "race": "Asian",
        "age": 32,
        "primary_diagnosis": "Oligodendroglioma",
        "grade": "2",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "No",
        "idh1": "Mutant / positive",
        "idh2": "Wildtype / negative",
        "codeletion_1p19q": "Mutant / positive",
        "mgmt": "Methylated",
        "tert": "Wildtype / negative",
        "egfr": "Wildtype / negative",
        "first_surgery_day": 10,
        "initial_chemo": "Yes",
        "initial_chemo_name": "PCV",
        "initial_chemo_start_day": 40,
        "initial_chemo_end_day": 200,
        "radiotherapy": "Yes",
        "rt_start_day": 40,
        "rt_end_day": 80,
        "rt_dose": 54,
        "rt_fractions": 30,
    },
    "03_Astro_G3_IDHmut_noRT": {
        "sex": "Female",
        "race": "Chinese",
        "age": 45,
        "primary_diagnosis": "Astrocytoma",
        "grade": "3",
        "biopsy_before_resection": "Yes",
        "previous_brain_tumor": "No",
        "idh1": "Mutant / positive",
        "mgmt": "Methylated",
        "codeletion_1p19q": "Wildtype / negative",
        "first_surgery_day": 7,
        "initial_chemo": "Yes",
        "initial_chemo_name": "Temozolomide",
        "initial_chemo_start_day": 20,
        "initial_chemo_end_day": 180,
        "radiotherapy": "No",
    },
    "04_Diffuse_G2_all_unknown_mol": {
        "sex": "Male",
        "race": "Malay",
        "age": 50,
        "primary_diagnosis": "Diffuse glioma",
        "grade": "2",
        "biopsy_before_resection": "Unknown / not available",
        "previous_brain_tumor": "No",
        "idh1": "Unknown / not tested",
        "idh2": "Unknown / not tested",
        "codeletion_1p19q": "Unknown / not tested",
        "mgmt": "Unknown / not tested",
        "atrx": "Unknown / not tested",
        "tert": "Unknown / not tested",
        "egfr": "Unknown / not tested",
        "first_surgery_day": 30,
        "initial_chemo": "Unknown / not available",
        "radiotherapy": "Unknown / not available",
    },
    "05_GBM_G4_no_treatment": {
        "sex": "Male",
        "race": "White",
        "age": 72,
        "primary_diagnosis": "Glioblastoma",
        "grade": "4",
        "biopsy_before_resection": "Yes",
        "previous_brain_tumor": "No",
        "idh1": "Wildtype / negative",
        "mgmt": "Unmethylated",
        "first_surgery_day": 5,
        "initial_chemo": "No",
        "radiotherapy": "No",
    },
    "06_Pilocytic_G1_young": {
        "sex": "Female",
        "race": "Indian",
        "age": 18,
        "primary_diagnosis": "Pilocytic astrocytoma",
        "grade": "1",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "No",
        "idh1": "Wildtype / negative",
        "mgmt": "Unknown / not tested",
        "first_surgery_day": 3,
        "initial_chemo": "No",
        "radiotherapy": "No",
    },
    "07_GBM_prev_tumor_yes": {
        "sex": "Male",
        "race": "Other",
        "age": 60,
        "primary_diagnosis": "Glioblastoma",
        "grade": "4",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "Yes",
        "idh1": "Wildtype / negative",
        "mgmt": "Methylated",
        "egfr": "Mutant / positive",
        "first_surgery_day": 14,
        "initial_chemo": "Yes",
        "initial_chemo_name": "Temozolomide",
        "initial_chemo_start_day": 30,
        "initial_chemo_end_day": 210,
        "radiotherapy": "Yes",
        "rt_start_day": 30,
        "rt_end_day": 72,
        "rt_dose": 60,
        "rt_fractions": 30,
    },
    "08_Astro_G2_IDHwt": {
        "sex": "Female",
        "race": "White",
        "age": 55,
        "primary_diagnosis": "Astrocytoma",
        "grade": "2",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "No",
        "idh1": "Wildtype / negative",
        "mgmt": "Unmethylated",
        "first_surgery_day": 20,
        "initial_chemo": "Yes",
        "initial_chemo_name": "Temozolomide",
        "radiotherapy": "Yes",
        "rt_dose": 50,
        "rt_fractions": 28,
        "rt_start_day": 25,
        "rt_end_day": 60,
    },
    "09_defaults_almost_empty": {
        "sex": "Unknown",
        "race": "Unknown",
        "age": None,
        "primary_diagnosis": "Unknown",
        "grade": "Unknown",
        "biopsy_before_resection": "Unknown / not available",
        "previous_brain_tumor": "Unknown / not available",
        "idh1": "Unknown / not tested",
        "mgmt": "Unknown / not tested",
        "first_surgery_day": None,
        "initial_chemo": "Unknown / not available",
        "radiotherapy": "Unknown / not available",
    },
    "10_GBM_extreme_age_90": {
        "sex": "Male",
        "race": "White",
        "age": 90,
        "primary_diagnosis": "Glioblastoma",
        "grade": "4",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "No",
        "idh1": "Wildtype / negative",
        "mgmt": "Unmethylated",
        "tert": "Mutant / positive",
        "first_surgery_day": 1,
        "initial_chemo": "Yes",
        "initial_chemo_name": "Temozolomide",
        "radiotherapy": "Yes",
        "rt_dose": 40,
        "rt_fractions": 15,
        "rt_start_day": 10,
        "rt_end_day": 30,
    },
}


def _docs() -> list[LiteratureDoc]:
    return [
        LiteratureDoc(
            "1",
            "GBM prognosis",
            "Glioblastoma IDH-wildtype has poor prognosis and high recurrence.",
            "stub",
            0.9,
            "2023",
        )
    ]


def run_smoke_battery() -> pd.DataFrame:
    artifact = joblib.load(DEFAULT_ARTIFACT)
    model = artifact["model"]
    booster = model.get_booster()
    mean_emb = artifact["mean_embedding"]
    docs = _docs()
    rows = []
    for name, data in CASES.items():
        pred = predict_first_hybrid(data, docs, use_real_llm=False)
        mapped = encode_ui_clinical_row(data)
        z = artifact["pca"].transform(mean_emb.reshape(1, -1))
        c, _ = _encode_clinical(_clinical_row_from_ui(data), artifact["clinical_encoder"])
        X = np.concatenate([z, c], axis=1)
        dm = xgb.DMatrix(X, feature_names=artifact["feature_names"])
        kwargs = _iteration_kwargs(model)
        contrib = booster.predict(dm, pred_contribs=True, **kwargs)[0]
        recon = 1.0 / (1.0 + exp(-float(contrib.sum())))
        rows.append(
            {
                "case": name,
                "prob_%": round(pred.probability * 100, 2),
                "diagnosis_mapped": mapped["Primary Diagnosis"],
                "idh1": mapped["IDH1 mutation"],
                "mgmt": mapped["MGMT methylation"],
                "rt": mapped["Radiation Therapy"],
                "xai_ok": abs(recon - pred.probability) < 1e-6,
            }
        )
    return pd.DataFrame(rows)


def test_ten_cases_not_hardcoded() -> None:
    df = run_smoke_battery()
    assert len(df) == 10
    assert df["xai_ok"].all()
    probs = df["prob_%"].tolist()
    assert len(set(probs)) >= 2, "Probability appears hardcoded to a single value"
    # Exp3 hybrid does not use molecular columns, so smoke-mode (mean embedding)
    # span is narrower than Exp4. Still must move with clinical/treatment inputs.
    assert max(probs) - min(probs) >= 5.0, "Smoke-mode span unexpectedly tiny"


def test_real_embeddings_move_probability() -> None:
    """Prove the model is not stuck when the LLM embedding changes."""
    artifact = joblib.load(DEFAULT_ARTIFACT)
    model = artifact["model"]
    Xtr = np.load(TRAINING_EMB_DIR / "train.npy").astype(np.float32)
    base = CASES["01_GBM_G4_IDHwt_MGMTun_RT"]
    probs = []
    for i in range(10):
        z = artifact["pca"].transform(Xtr[i].reshape(1, -1))
        c, _ = _encode_clinical(_clinical_row_from_ui(base), artifact["clinical_encoder"])
        X = np.concatenate([z, c], axis=1)
        probs.append(float(model.predict_proba(X)[0, 1]))
    assert max(probs) - min(probs) > 0.4


if __name__ == "__main__":
    df = run_smoke_battery()
    print(df.to_string(index=False))
    probs = df["prob_%"].tolist()
    print(f"\nUnique probs: {sorted(set(probs))}")
    print(f"Range: {min(probs):.2f}% .. {max(probs):.2f}%  span={max(probs)-min(probs):.2f} pp")
    print(f"XAI ok all 10: {bool(df['xai_ok'].all())}")

    artifact = joblib.load(DEFAULT_ARTIFACT)
    model = artifact["model"]
    Xtr = np.load(TRAINING_EMB_DIR / "train.npy").astype(np.float32)
    base = CASES["01_GBM_G4_IDHwt_MGMTun_RT"]
    real = []
    for i in range(10):
        z = artifact["pca"].transform(Xtr[i].reshape(1, -1))
        c, _ = _encode_clinical(_clinical_row_from_ui(base), artifact["clinical_encoder"])
        X = np.concatenate([z, c], axis=1)
        real.append(float(model.predict_proba(X)[0, 1]) * 100)
    print(f"Real-embedding range (same clinical): {min(real):.1f}% .. {max(real):.1f}% span={max(real)-min(real):.1f} pp")

    test_ten_cases_not_hardcoded()
    test_real_embeddings_move_probability()
    print("\nAll 10 stress checks passed.")
