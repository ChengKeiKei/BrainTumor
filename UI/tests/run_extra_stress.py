"""10 extra First + 10 extra Second stress cases.

Focus areas beyond run_10x10_stress.py:
- the newly added previous-tumor fields (type / year / grade);
- chemo-name aliases (TMZ, CCNU) and radiotherapy answered "No";
- date edge cases: surgery before diagnosis, same-day events, missing anchor;
- every one-hot categorical value must land inside the MU training vocabulary.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import joblib
import pandas as pd

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from src.encoding import encode_ui_clinical_row
from src.fields import apply_date_offsets
from src.hybrid_inference import DEFAULT_ARTIFACT, predict_first_hybrid
from src.literature import LiteratureDoc
from src.second_inference import predict_second

DOCS = [LiteratureDoc("1", "GBM prognosis", "Glioblastoma IDH-wildtype recurrence.", "stub", 0.9, "2023")]
DX = date(2023, 6, 1)

FR_DAY_KEYS = ("first_surgery_day", "initial_chemo_start_day", "initial_chemo_end_day", "rt_start_day", "rt_end_day")
SR_DAY_KEYS = (
    "time_to_first_progression",
    "additional_therapy_start",
    "additional_therapy_end",
    "immunotherapy_start",
    "immunotherapy_end",
    "latest_mri_day",
)

SR_REQUIRED = {
    "patient_id": "Patient ID / RN",
    "age": "Age at diagnosis",
    "sex": "Sex at birth",
    "primary_diagnosis": "Primary diagnosis",
    "grade": "WHO grade of primary brain tumor",
    "time_to_first_progression": "Date of first recurrence/progression",
    "type_first_progression": "Type of first progression",
    "multiple_surgeries": "Multiple surgeries",
    "additional_therapy": "Additional/salvage therapy",
    "latest_mri_day": "Latest eligible MRI date",
    "mri_report": "MRI report",
}

FR_EXTRA = {
    "11_prev_GBM_2019_grade3": {
        "sex": "Male", "race": "Chinese", "age": 61,
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "biopsy_before_resection": "No", "previous_brain_tumor": "Yes",
        "type_previous_brain_tumor": "Glioblastoma (GBM)",
        "year_previous_surgery": 2019, "grade_previous_brain_tumor": "3",
        "first_surgery_day": 12, "initial_chemo": "Yes", "initial_chemo_name": "TMZ",
        "initial_chemo_start_day": 35, "initial_chemo_end_day": 215,
        "radiotherapy": "Yes", "rt_start_day": 35, "rt_end_day": 77, "rt_dose": 60, "rt_fractions": 30,
    },
    "12_prev_astro_grade2": {
        "sex": "Female", "race": "Malay", "age": 39,
        "primary_diagnosis": "Astrocytoma", "grade": "3",
        "biopsy_before_resection": "Yes", "previous_brain_tumor": "Yes",
        "type_previous_brain_tumor": "Astrocytoma",
        "year_previous_surgery": 2015, "grade_previous_brain_tumor": "2",
        "first_surgery_day": 8, "initial_chemo": "Yes", "initial_chemo_name": "Temodal",
        "initial_chemo_start_day": 25, "initial_chemo_end_day": 190,
        "radiotherapy": "Yes", "rt_start_day": 25, "rt_end_day": 65, "rt_dose": 54, "rt_fractions": 30,
    },
    "13_prev_neurocytoma_grade4_now": {
        "sex": "Male", "race": "Indian", "age": 47,
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "biopsy_before_resection": "No", "previous_brain_tumor": "Yes",
        "type_previous_brain_tumor": "Neurocytoma",
        "year_previous_surgery": 2010, "grade_previous_brain_tumor": "4",
        "first_surgery_day": 20, "initial_chemo": "No",
        "radiotherapy": "No",
    },
    "14_prev_other_type_unmapped": {
        "sex": "Female", "race": "Other", "age": 55,
        "primary_diagnosis": "Diffuse glioma", "grade": "2",
        "biopsy_before_resection": "Unknown / not available", "previous_brain_tumor": "Yes",
        "type_previous_brain_tumor": "Other",
        "year_previous_surgery": 2021, "grade_previous_brain_tumor": "Unknown / not applicable",
        "first_surgery_day": 5, "initial_chemo": "Yes", "initial_chemo_name": "CCNU",
        "initial_chemo_start_day": 30, "initial_chemo_end_day": 120,
        "radiotherapy": "Yes", "rt_start_day": 40, "rt_end_day": 82, "rt_dose": 59.4, "rt_fractions": 33,
    },
    "15_rt_no_but_dose_entered": {
        "sex": "Male", "race": "White", "age": 72,
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "biopsy_before_resection": "No", "previous_brain_tumor": "No",
        "first_surgery_day": 3, "initial_chemo": "No",
        "radiotherapy": "No", "rt_dose": 60, "rt_fractions": 30,
    },
    "16_surgery_before_diagnosis": {
        "sex": "Female", "race": "Chinese", "age": 63,
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "biopsy_before_resection": "Yes", "previous_brain_tumor": "No",
        "first_surgery_day": -21, "initial_chemo": "Yes", "initial_chemo_name": "temozolomide",
        "initial_chemo_start_day": 14, "initial_chemo_end_day": 194,
        "radiotherapy": "Yes", "rt_start_day": 14, "rt_end_day": 56, "rt_dose": 40, "rt_fractions": 15,
    },
    "17_same_day_everything": {
        "sex": "Male", "race": "Asian", "age": 50,
        "primary_diagnosis": "Astrocytoma", "grade": "3",
        "biopsy_before_resection": "No", "previous_brain_tumor": "No",
        "first_surgery_day": 0, "initial_chemo": "Yes", "initial_chemo_name": "Lomustine",
        "initial_chemo_start_day": 0, "initial_chemo_end_day": 0,
        "radiotherapy": "Yes", "rt_start_day": 0, "rt_end_day": 0, "rt_dose": 34, "rt_fractions": 10,
    },
    "18_unknown_chemo_name": {
        "sex": "Female", "race": "Unknown", "age": 29,
        "primary_diagnosis": "Pilocytic astrocytoma", "grade": "1",
        "biopsy_before_resection": "No", "previous_brain_tumor": "No",
        "first_surgery_day": 45, "initial_chemo": "Yes", "initial_chemo_name": "Bevacizumab",
        "initial_chemo_start_day": 60, "initial_chemo_end_day": 240,
        "radiotherapy": "Unknown / not available",
    },
    "19_extreme_late_treatment": {
        "sex": "Male", "race": "Black or African American", "age": 80,
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "biopsy_before_resection": "No", "previous_brain_tumor": "Yes",
        "type_previous_brain_tumor": "Unknown / not applicable",
        "first_surgery_day": 400, "initial_chemo": "Yes", "initial_chemo_name": "TMZ",
        "initial_chemo_start_day": 430, "initial_chemo_end_day": 800,
        "radiotherapy": "Yes", "rt_start_day": 430, "rt_end_day": 472, "rt_dose": 50, "rt_fractions": 25,
    },
    "20_minimal_required_only": {
        "sex": "Female", "race": "Indian", "age": 44,
        "primary_diagnosis": "Oligodendroglioma", "grade": "2",
        "biopsy_before_resection": "No", "previous_brain_tumor": "No",
        "first_surgery_day": 9, "initial_chemo": "Unknown / not available",
        "radiotherapy": "Unknown / not available",
    },
}

SR_EXTRA = {
    "11_very_early_recur_salvage": {
        "patient_id": "SRX-11", "age": 66, "sex": "Male",
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "time_to_first_progression": 120, "type_first_progression": "Local",
        "multiple_surgeries": "Yes", "additional_therapy": "Yes",
        "additional_therapy_start": 140, "additional_therapy_end": 260, "additional_cycles": 6,
        "immunotherapy": "No", "latest_mri_day": 300,
        "mri_report": "Enlarging enhancing lesion with central necrosis, progressive disease.",
    },
    "12_late_recur_stable": {
        "patient_id": "SRX-12", "age": 35, "sex": "Female",
        "primary_diagnosis": "Oligodendroglioma", "grade": "2",
        "time_to_first_progression": 1800, "type_first_progression": "Local",
        "multiple_surgeries": "No", "additional_therapy": "Yes",
        "additional_therapy_start": 1830, "additional_therapy_end": 2000, "additional_cycles": 4,
        "immunotherapy": "No", "latest_mri_day": 2100,
        "mri_report": "Stable postoperative changes, no new enhancement.",
        "idh1": "Mutant / positive", "codeletion_1p19q": "Mutant / positive", "mgmt": "Methylated",
    },
    "13_distant_progression": {
        "patient_id": "SRX-13", "age": 58, "sex": "Male",
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "time_to_first_progression": 250, "type_first_progression": "Distant",
        "multiple_surgeries": "No", "additional_therapy": "Yes",
        "additional_therapy_start": 270, "additional_therapy_end": 380, "additional_cycles": 3,
        "immunotherapy": "Yes", "immunotherapy_start": 290, "immunotherapy_end": 400,
        "latest_mri_day": 420,
        "mri_report": "New distant enhancing focus in contralateral hemisphere.",
    },
    "14_leptomeningeal": {
        "patient_id": "SRX-14", "age": 49, "sex": "Female",
        "primary_diagnosis": "Astrocytoma", "grade": "3",
        "time_to_first_progression": 500, "type_first_progression": "Leptomeningeal",
        "multiple_surgeries": "No", "additional_therapy": "No",
        "latest_mri_day": 540,
        "mri_report": "Leptomeningeal enhancement along cerebellar folia, progressive.",
    },
    "15_no_salvage_no_immuno": {
        "patient_id": "SRX-15", "age": 77, "sex": "Male",
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "time_to_first_progression": 200, "type_first_progression": "Multifocal",
        "multiple_surgeries": "No", "additional_therapy": "No",
        "latest_mri_day": 230,
        "mri_report": "Multifocal enhancing disease, marked progression.",
    },
    "16_same_day_mri_and_recur": {
        "patient_id": "SRX-16", "age": 52, "sex": "Female",
        "primary_diagnosis": "Diffuse glioma", "grade": "2",
        "time_to_first_progression": 365, "type_first_progression": "Local",
        "multiple_surgeries": "No", "additional_therapy": "Yes",
        "additional_therapy_start": 365, "additional_therapy_end": 500, "additional_cycles": 5,
        "immunotherapy": "No", "latest_mri_day": 365,
        "mri_report": "Mild interval increase in FLAIR signal without new enhancement.",
    },
    "17_clinical_progression_only": {
        "patient_id": "SRX-17", "age": 43, "sex": "Male",
        "primary_diagnosis": "Astrocytoma", "grade": "2",
        "time_to_first_progression": 900, "type_first_progression": "Clinical progression only",
        "multiple_surgeries": "No", "additional_therapy": "Unknown / not available",
        "latest_mri_day": 940,
        "mri_report": "No definite radiological progression; clinical seizures worsening.",
    },
    "18_missing_optional_all": {
        "patient_id": "SRX-18", "age": 60, "sex": "Female",
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "time_to_first_progression": 300, "type_first_progression": "Local",
        "multiple_surgeries": "Unknown / not available", "additional_therapy": "Yes",
        "latest_mri_day": 350,
        "mri_report": "Progressive enhancement at resection margin.",
    },
    "19_young_favorable": {
        "patient_id": "SRX-19", "age": 24, "sex": "Male",
        "primary_diagnosis": "Pilocytic astrocytoma", "grade": "1",
        "time_to_first_progression": 1500, "type_first_progression": "Local",
        "multiple_surgeries": "No", "additional_therapy": "Yes",
        "additional_therapy_start": 1520, "additional_therapy_end": 1600, "additional_cycles": 2,
        "immunotherapy": "No", "latest_mri_day": 1700,
        "mri_report": "Small stable residual, no interval change.",
        "enhancing_volume": 0.8, "edema_volume": 1.5,
    },
    "20_borderline_full_data": {
        "patient_id": "SRX-20", "age": 55, "sex": "Female",
        "primary_diagnosis": "Glioblastoma", "grade": "4",
        "time_to_first_progression": 600, "type_first_progression": "Local",
        "multiple_surgeries": "Yes", "additional_therapy": "Yes",
        "additional_therapy_start": 620, "additional_therapy_end": 750, "additional_cycles": 6,
        "immunotherapy": "Yes", "immunotherapy_start": 640, "immunotherapy_end": 760,
        "latest_mri_day": 800,
        "mri_report": "Equivocal enhancement, possible treatment effect versus progression.",
        "enhancing_volume": 4.2, "edema_volume": 12.0, "idh1": "Wildtype / negative", "mgmt": "Unmethylated",
    },
}


def _with_dates(numeric: dict, day_keys: tuple[str, ...]) -> dict:
    out = dict(numeric)
    out["diagnosis_date"] = DX
    for key in day_keys:
        days = numeric.get(key)
        if isinstance(days, (int, float)):
            out[key] = DX + timedelta(days=int(days))
    return apply_date_offsets(out)


def check_vocab(data: dict, enc: dict) -> list[str]:
    """Return one-hot columns produced by this case that the training vocab lacks."""
    row = encode_ui_clinical_row(data)
    df = pd.DataFrame([{c: row.get(c) for c in enc["categorical_cols"]}])
    cat = pd.get_dummies(df, dummy_na=True)
    active = [c for c in cat.columns if cat[c].iloc[0]]
    return [c for c in active if c not in enc["cat_columns"]]


def main() -> None:
    art = joblib.load(DEFAULT_ARTIFACT)
    enc = art["clinical_encoder"]

    print("=" * 78)
    print("FIRST RECURRENCE — 10 extra cases (new prev-tumor fields, aliases, date edges)")
    print("=" * 78)
    fr_rows = []
    for name, case in FR_EXTRA.items():
        numeric = dict(case)
        numeric["diagnosis_date"] = DX
        p_num = predict_first_hybrid(numeric, DOCS, use_real_llm=False).probability
        dated = _with_dates(case, FR_DAY_KEYS)
        p_date = predict_first_hybrid(dated, DOCS, use_real_llm=False).probability
        bad_vocab = check_vocab(dated, enc)
        fr_rows.append({
            "case": name,
            "prob_%": round(p_num * 100, 2),
            "date_path_%": round(p_date * 100, 2),
            "dates_match": abs(p_num - p_date) < 1e-9,
            "vocab_ok": not bad_vocab,
            "bad_vocab": bad_vocab or "",
        })
    fr = pd.DataFrame(fr_rows)
    print(fr.to_string(index=False))
    assert fr["dates_match"].all(), "FIRST: date path diverged from numeric path"
    assert fr["vocab_ok"].all(), "FIRST: one-hot value outside training vocabulary"
    assert fr["prob_%"].between(0, 100).all()

    # Missing anchor date: day offsets stay None, prediction must still run.
    no_anchor = dict(FR_EXTRA["20_minimal_required_only"])
    no_anchor["first_surgery_day"] = date(2023, 6, 10)
    converted = apply_date_offsets(no_anchor)
    assert converted["first_surgery_day"] is None, "surgery date without diagnosis date must become None"
    p = predict_first_hybrid(converted, DOCS, use_real_llm=False).probability
    assert 0.0 < p < 1.0
    print(f"\nFIRST no-anchor-date case: surgery day -> None, prob {p*100:.2f}% (ok)")

    print()
    print("=" * 78)
    print("SECOND RECURRENCE — 10 extra cases")
    print("=" * 78)
    sr_rows = []
    for name, case in SR_EXTRA.items():
        numeric = dict(case)
        numeric["diagnosis_date"] = DX
        pred_num = predict_second(numeric, SR_REQUIRED, DOCS, use_real_llm=False)
        dated = _with_dates(case, SR_DAY_KEYS)
        pred_date = predict_second(dated, SR_REQUIRED, DOCS, use_real_llm=False)
        sr_rows.append({
            "case": name,
            "prob_%": round(pred_num.probability * 100, 2),
            "date_path_%": round(pred_date.probability * 100, 2),
            "dates_match": abs(pred_num.probability - pred_date.probability) < 1e-9,
            "risk": pred_num.risk_level,
            "prompt_ok": "Salvage treatment" in pred_num.evidence_prompt,
        })
    sr = pd.DataFrame(sr_rows)
    print(sr.to_string(index=False))
    assert sr["dates_match"].all(), "SECOND: date path diverged from numeric path"
    assert sr["prompt_ok"].all(), "SECOND: prompt missing salvage-treatment block"
    probs = sr["prob_%"].tolist()
    assert len(set(probs)) >= 8, f"SECOND: probabilities look hardcoded: {probs}"

    print(f"\nSECOND span: {min(probs):.2f}%..{max(probs):.2f}% ({len(set(probs))} unique)")
    print("\nAll 10 extra FIRST and 10 extra SECOND stress cases passed.")


if __name__ == "__main__":
    main()
