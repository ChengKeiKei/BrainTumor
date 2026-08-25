"""Smoke checks for counterfactual helpers and imaging caption fallback."""

from __future__ import annotations

from pathlib import Path
import sys

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from src.counterfactual import (
    first_recurrence_scenarios,
    run_counterfactuals,
    second_recurrence_scenarios,
)
from src.fields import SECOND_RECURRENCE_FIELDS
from src.imaging import generate_caption_from_uploads, radfm_available
from src.hybrid_inference import predict_first_hybrid
from src.literature import LiteratureDoc
from src.risk_engine import predict_second_recurrence


def test_first_counterfactuals_run() -> None:
    docs = [LiteratureDoc("0", "t", "a", "s", 0.1, "2020")]
    base = {
        "sex": "Male",
        "race": "White",
        "age": 60,
        "primary_diagnosis": "Glioblastoma",
        "grade": "4",
        "biopsy_before_resection": "No",
        "previous_brain_tumor": "No",
        "idh1": "Wildtype / negative",
        "mgmt": "Unmethylated",
        "first_surgery_day": 14,
        "initial_chemo": "Yes",
        "initial_chemo_name": "Temozolomide",
        "radiotherapy": "Yes",
        "rt_dose": 60,
        "rt_fractions": 30,
        "rt_start_day": 30,
        "rt_end_day": 72,
    }
    base_prob = predict_first_hybrid(base, docs, use_real_llm=False).probability

    def predict(candidate: dict) -> float:
        return predict_first_hybrid(candidate, docs, use_real_llm=False).probability

    df = run_counterfactuals(base, first_recurrence_scenarios(base), predict, baseline_prob=base_prob)
    assert not df.empty
    assert "delta_pp" in df.columns
    assert df["baseline_prob_%"].nunique() == 1


def test_second_counterfactuals_only() -> None:
    required = {f.key: f.label for f in SECOND_RECURRENCE_FIELDS if f.required}
    base = {
        "patient_id": "CF1",
        "age": 55,
        "sex": "Male",
        "primary_diagnosis": "Astrocytoma",
        "grade": "3",
        "time_to_first_progression": 400,
        "type_first_progression": "Local",
        "multiple_surgeries": "No",
        "additional_therapy": "Yes",
        "additional_cycles": 3,
        "latest_mri_day": 430,
        "mri_report": "Mild residual enhancement, otherwise stable",
        "idh1": "Mutant / positive",
        "mgmt": "Methylated",
    }
    base_prob = predict_second_recurrence(base, required).probability

    def predict(candidate: dict) -> float:
        return predict_second_recurrence(candidate, required).probability

    df = run_counterfactuals(base, second_recurrence_scenarios(base), predict, baseline_prob=base_prob)
    assert not df.empty
    # Stable vs progressing captions should move probability in opposite directions.
    progressing = df[df["counterfactual"].str.contains("progressing", case=False)].iloc[0]
    stable = df[df["counterfactual"].str.contains("stable", case=False)].iloc[0]
    assert progressing["new_prob_%"] > stable["new_prob_%"]


def test_date_offsets_from_diagnosis() -> None:
    from datetime import date

    from src.fields import apply_date_offsets, days_from_diagnosis

    assert days_from_diagnosis(date(2024, 1, 1), date(2024, 1, 15)) == 14
    converted = apply_date_offsets(
        {
            "diagnosis_date": date(2024, 1, 1),
            "first_surgery_day": date(2024, 1, 15),
            "time_to_first_progression": date(2024, 7, 1),
            "age": 60,
        }
    )
    assert converted["first_surgery_day"] == 14
    assert converted["time_to_first_progression"] == 182
    assert converted["first_surgery_day_date"] == date(2024, 1, 15)
    # Numeric test inputs are left unchanged.
    numeric = apply_date_offsets({"diagnosis_date": date(2024, 1, 1), "first_surgery_day": 14})
    assert numeric["first_surgery_day"] == 14


def test_imaging_fallback_without_upload() -> None:
    result = generate_caption_from_uploads(None)
    assert result.caption == ""
    assert "No MRI" in result.warning


if __name__ == "__main__":
    test_first_counterfactuals_run()
    test_second_counterfactuals_only()
    test_imaging_fallback_without_upload()
    test_date_offsets_from_diagnosis()
    print(f"RadFM available: {radfm_available()}")
    print("Counterfactual + imaging smoke checks passed.")
