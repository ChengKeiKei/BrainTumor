from __future__ import annotations

import re
from dataclasses import dataclass
from math import exp
from typing import Any


@dataclass(frozen=True)
class Prediction:
    probability: float
    risk_level: str
    evidence_completeness: str
    drivers: list[str]
    contributions: list[dict[str, float | str]]
    missing_required: list[str]
    evidence_prompt: str
    mode: str = ""
    model_name: str = ""
    warning: str = ""
    checkpoint_paths: dict[str, str] | None = None


def _sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == "Unknown" or value == "Unknown / not available"


def _risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "High recurrence risk"
    if probability >= 0.45:
        return "Intermediate recurrence risk"
    return "Lower recurrence risk"


def _confidence(missing_required: list[str], optional_missing: int) -> str:
    if missing_required:
        return "Low: required inputs are incomplete"
    if optional_missing >= 5:
        return "Moderate: core inputs present, optional evidence limited"
    return "Higher: core inputs present"


def _add_contribution(
    contributions: list[dict[str, float | str]],
    drivers: list[str],
    feature: str,
    effect: float,
    explanation: str,
) -> None:
    contributions.append({"feature": feature, "effect": effect, "explanation": explanation})
    if effect > 0:
        drivers.append(explanation)


def predict_first_recurrence(data: dict[str, Any], required_labels: dict[str, str]) -> Prediction:
    missing_required = [label for key, label in required_labels.items() if _missing(data.get(key))]
    logit = 0.55
    drivers: list[str] = []
    contributions: list[dict[str, float | str]] = [
        {"feature": "Cohort baseline", "effect": 0.55, "explanation": "First Recurrence cohort baseline risk"}
    ]

    age = data.get("age")
    if isinstance(age, (int, float)):
        if age >= 60:
            effect = 0.35
            logit += effect
            _add_contribution(contributions, drivers, "Age at diagnosis", effect, "Age at diagnosis is 60 years or above")
        elif age < 35:
            effect = -0.20
            logit += effect
            _add_contribution(contributions, drivers, "Age at diagnosis", effect, "Younger age lowers the demo risk score")

    diagnosis = str(data.get("primary_diagnosis", "")).lower()
    if "glioblastoma" in diagnosis:
        effect = 0.75
        logit += effect
        _add_contribution(contributions, drivers, "Primary diagnosis", effect, "Primary diagnosis is glioblastoma")
    elif "oligodendroglioma" in diagnosis:
        effect = -0.25
        logit += effect
        _add_contribution(contributions, drivers, "Primary diagnosis", effect, "Oligodendroglioma lowers the demo risk score")

    grade = str(data.get("grade", ""))
    if grade == "4":
        effect = 0.65
        logit += effect
        _add_contribution(contributions, drivers, "WHO grade", effect, "WHO grade 4 tumor")
    elif grade == "3":
        effect = 0.35
        logit += effect
        _add_contribution(contributions, drivers, "WHO grade", effect, "WHO grade 3 tumor")
    elif grade in {"1", "2"}:
        effect = -0.25
        logit += effect
        _add_contribution(contributions, drivers, "WHO grade", effect, "Lower WHO grade lowers the demo risk score")

    if data.get("previous_brain_tumor") == "Yes":
        effect = 0.35
        logit += effect
        _add_contribution(contributions, drivers, "Previous brain tumor", effect, "Previous brain tumor history")

    if data.get("initial_chemo") == "No":
        effect = 0.20
        logit += effect
        _add_contribution(contributions, drivers, "Initial chemotherapy", effect, "No initial chemotherapy recorded")
    if data.get("radiotherapy") == "No":
        effect = 0.20
        logit += effect
        _add_contribution(contributions, drivers, "Radiation therapy", effect, "No radiation therapy recorded")

    rt_dose = data.get("rt_dose")
    if isinstance(rt_dose, (int, float)) and rt_dose > 0:
        if rt_dose < 45:
            effect = 0.18
            logit += effect
            _add_contribution(contributions, drivers, "Radiation dose", effect, "Radiation dose below common radical-treatment range")
        elif rt_dose >= 54:
            effect = -0.12
            logit += effect
            _add_contribution(contributions, drivers, "Radiation dose", effect, "Radiation dose in common radical-treatment range")

    if data.get("idh1") == "Mutant / positive" or data.get("idh2") == "Mutant / positive":
        effect = -0.35
        logit += effect
        _add_contribution(contributions, drivers, "IDH mutation", effect, "IDH mutation present")
    if data.get("codeletion_1p19q") == "Mutant / positive":
        effect = -0.25
        logit += effect
        _add_contribution(contributions, drivers, "1p/19q", effect, "1p/19q codeletion present")
    if data.get("mgmt") == "Methylated":
        effect = -0.12
        logit += effect
        _add_contribution(contributions, drivers, "MGMT", effect, "MGMT methylation present")

    if missing_required:
        effect = 0.12 * len(missing_required)
        logit += effect
        _add_contribution(contributions, drivers, "Missing required inputs", effect, "Required fields are incomplete")

    probability = max(0.02, min(0.98, _sigmoid(logit)))
    optional_missing = sum(1 for key in ["idh1", "idh2", "codeletion_1p19q", "mgmt", "atrx", "tert", "egfr"] if _missing(data.get(key)))
    prompt = build_first_recurrence_prompt(data)
    if not drivers:
        drivers = ["No strong single risk driver detected from the entered fields"]
    return Prediction(
        probability,
        _risk_level(probability),
        _confidence(missing_required, optional_missing),
        drivers[:6],
        sorted(contributions, key=lambda item: abs(float(item["effect"])), reverse=True)[:8],
        missing_required,
        prompt,
    )


def predict_second_recurrence(data: dict[str, Any], required_labels: dict[str, str]) -> Prediction:
    missing_required = [label for key, label in required_labels.items() if _missing(data.get(key))]
    logit = 0.15
    drivers: list[str] = []
    contributions: list[dict[str, float | str]] = [
        {"feature": "Cohort baseline", "effect": 0.15, "explanation": "Second Recurrence cohort baseline risk"}
    ]

    ttp1 = data.get("time_to_first_progression")
    if isinstance(ttp1, (int, float)):
        if ttp1 <= 365:
            effect = 0.65
            logit += effect
            _add_contribution(contributions, drivers, "Time to first recurrence", effect, "First recurrence occurred within 1 year")
        elif ttp1 <= 730:
            effect = 0.30
            logit += effect
            _add_contribution(contributions, drivers, "Time to first recurrence", effect, "First recurrence occurred within 2 years")
        else:
            effect = -0.20
            logit += effect
            _add_contribution(contributions, drivers, "Time to first recurrence", effect, "Longer time to first recurrence lowers the demo risk score")

    diagnosis = str(data.get("primary_diagnosis", "")).lower()
    grade = str(data.get("grade", ""))
    if "glioblastoma" in diagnosis or grade == "4":
        effect = 0.45
        logit += effect
        _add_contribution(contributions, drivers, "Diagnosis / grade", effect, "Aggressive baseline diagnosis/grade")

    if data.get("multiple_surgeries") == "Yes":
        effect = 0.25
        logit += effect
        _add_contribution(contributions, drivers, "Multiple surgeries", effect, "Multiple surgeries recorded before prediction")

    if data.get("additional_therapy") == "No":
        effect = 0.25
        logit += effect
        _add_contribution(contributions, drivers, "Salvage therapy", effect, "No salvage/additional therapy recorded")

    cycles = data.get("additional_cycles")
    if isinstance(cycles, (int, float)):
        if cycles <= 1:
            effect = 0.18
            logit += effect
            _add_contribution(contributions, drivers, "Salvage therapy cycles", effect, "Low number of salvage therapy cycles")
        elif cycles >= 6:
            effect = -0.12
            logit += effect
            _add_contribution(contributions, drivers, "Salvage therapy cycles", effect, "Higher number of salvage therapy cycles")

    report = str(data.get("mri_report", "")).lower()
    concerning_terms = [
        "progression",
        "increase",
        "enlarg",
        "new lesion",
        "enhancing",
        "edema",
        "mass effect",
    ]
    reassuring_terms = ["stable", "no progression", "unchanged", "decrease", "reduced"]
    concern_hits = [term for term in concerning_terms if term in report]
    # "no progression" should not count as a progression hit.
    if "no progression" in report:
        concern_hits = [term for term in concern_hits if term != "progression"]
    # Neutral morphology after decrease/reduced should not count as concerning.
    if re.search(r"(decrease\w*|reduced)\s+(in\s+)?(edema|enhancing)", report):
        concern_hits = [term for term in concern_hits if term not in {"edema", "enhancing"}]
    if "no progression" in report or "stable" in report:
        concern_hits = [term for term in concern_hits if term not in {"edema", "enhancing"}]
    reassuring_hits = [term for term in reassuring_terms if term in report]
    if concern_hits:
        effect = min(0.55, 0.12 * len(concern_hits))
        logit += effect
        _add_contribution(contributions, drivers, "MRI/RadFM text", effect, "MRI/RadFM text contains progression-related terms")
    if reassuring_hits:
        effect = -min(0.35, 0.10 * len(reassuring_hits))
        logit += effect
        _add_contribution(contributions, drivers, "MRI/RadFM text", effect, "MRI/RadFM text contains stability-related terms")

    enh = data.get("enhancing_volume")
    edema = data.get("edema_volume")
    if isinstance(enh, (int, float)) and enh > 20:
        effect = 0.20
        logit += effect
        _add_contribution(contributions, drivers, "Enhancing volume", effect, "Enhancing tumor volume is elevated")
    if isinstance(edema, (int, float)) and edema > 40:
        effect = 0.16
        logit += effect
        _add_contribution(contributions, drivers, "Edema/FLAIR volume", effect, "Edema/FLAIR abnormality volume is elevated")

    if data.get("idh1") == "Mutant / positive":
        effect = -0.18
        logit += effect
        _add_contribution(contributions, drivers, "IDH1 mutation", effect, "IDH1 mutation present")
    if data.get("mgmt") == "Methylated":
        effect = -0.08
        logit += effect
        _add_contribution(contributions, drivers, "MGMT", effect, "MGMT methylation present")

    if missing_required:
        effect = 0.12 * len(missing_required)
        logit += effect
        _add_contribution(contributions, drivers, "Missing required inputs", effect, "Required fields are incomplete")

    probability = max(0.02, min(0.98, _sigmoid(logit)))
    optional_missing = sum(1 for key in ["enhancing_volume", "edema_volume", "radiomic_summary", "idh1", "codeletion_1p19q", "mgmt"] if _missing(data.get(key)))
    prompt = build_second_recurrence_prompt(data)
    if not drivers:
        drivers = ["No strong single risk driver detected from the entered fields"]
    return Prediction(
        probability,
        _risk_level(probability),
        _confidence(missing_required, optional_missing),
        drivers[:6],
        sorted(contributions, key=lambda item: abs(float(item["effect"])), reverse=True)[:8],
        missing_required,
        prompt,
    )


def build_first_recurrence_prompt(data: dict[str, Any]) -> str:
    """Exp3-style prompt: demographics, diagnosis, treatment. No molecular block."""
    from .encoding import (
        map_diagnosis,
        map_dose,
        map_race,
        map_sex,
        map_yes_no_binary,
        map_yes_no_or_nan,
    )

    dx = map_diagnosis(data.get("primary_diagnosis")) or data.get("primary_diagnosis") or "Unknown"
    sex = map_sex(data.get("sex")) or data.get("sex") or "Unknown"
    race = map_race(data.get("race")) or data.get("race") or "Unknown"
    biopsy = map_yes_no_binary(data.get("biopsy_before_resection"))
    biopsy_txt = "Unknown" if biopsy is None else str(biopsy)
    prev = map_yes_no_or_nan(data.get("previous_brain_tumor")) or data.get("previous_brain_tumor") or "Unknown"
    chemo = map_yes_no_or_nan(data.get("initial_chemo")) or data.get("initial_chemo") or "Unknown"
    rt = map_yes_no_or_nan(data.get("radiotherapy")) or data.get("radiotherapy") or "Unknown"
    dose = map_dose(data.get("rt_dose")) or data.get("rt_dose") or "Unknown"
    return f"""You are a clinical outcome prediction assistant.
Given the following post-treatment glioma patient information, predict whether the patient will experience recurrence/progression.

Demographics: Sex at Birth: {sex}; Race: {race}; Age at diagnosis: {data.get("age", "Unknown")}.
Diagnosis: Primary Diagnosis: {dx}; Grade of Primary Brain Tumor: {data.get("grade", "Unknown")}; Stereotactic Biopsy before Surgical Resection: {biopsy_txt}; Previous Brain Tumor: {prev}.
Treatment: Days to surgery: {data.get("first_surgery_day", "Unknown")}; Initial chemotherapy: {chemo}; Chemo agent: {data.get("initial_chemo_name", "Unknown")}; Days to chemo start: {data.get("initial_chemo_start_day", "Unknown")}; Days to chemo end: {data.get("initial_chemo_end_day", "Unknown")}; Radiation therapy: {rt}; Days to radiation start: {data.get("rt_start_day", "Unknown")}; Days to radiation end: {data.get("rt_end_day", "Unknown")}; Radiation dose: {dose}; Radiation fractions: {data.get("rt_fractions", "Unknown")}.
"""


def build_second_recurrence_prompt(data: dict[str, Any]) -> str:
    """ExpC_v3-style clinical block used by the BioMistral + PubMedBERT LoRA path."""
    from .encoding import map_diagnosis, map_sex

    dx = map_diagnosis(data.get("primary_diagnosis")) or data.get("primary_diagnosis") or "Unknown"
    sex = map_sex(data.get("sex")) or data.get("sex") or "Unknown"
    idh = data.get("idh1", "Unknown")
    codeletion = data.get("codeletion_1p19q", "Unknown")
    mgmt = data.get("mgmt", "Unknown")
    return f"""You are a clinical outcome prediction assistant.
Given the following post-treatment glioma patient information, predict whether the patient will experience recurrence/progression.

Demographics: Sex at Birth: {sex}; Age at diagnosis: {data.get("age", "Unknown")}.
Diagnosis: Primary Diagnosis: {dx}; Grade of Primary Brain Tumor: {data.get("grade", "Unknown")}.
Molecular markers: IDH1: {idh}; 1p/19q: {codeletion}; MGMT: {mgmt}.
First recurrence landmark: Days to first progression: {data.get("time_to_first_progression", "Unknown")}; Type of first progression: {data.get("type_first_progression", "Unknown")}; Multiple surgeries: {data.get("multiple_surgeries", "Unknown")}.
Salvage treatment (between 1st progression and landmark): Additional/salvage therapy: {data.get("additional_therapy", "Unknown")}; Days to salvage start: {data.get("additional_therapy_start", "Unknown")}; Days to salvage end: {data.get("additional_therapy_end", "Unknown")}; Salvage cycles completed: {data.get("additional_cycles", "Unknown")}; Immunotherapy: {data.get("immunotherapy", "Unknown")}; Days to immuno start: {data.get("immunotherapy_start", "Unknown")}; Days to immuno end: {data.get("immunotherapy_end", "Unknown")}.
MRI cadence (pre-landmark only): Latest eligible MRI day: {data.get("latest_mri_day", "Unknown")}.
MRI radiology findings:
{data.get("mri_report") or "Unknown"}
Enhancing volume: {data.get("enhancing_volume", "Unknown")}; Edema/FLAIR volume: {data.get("edema_volume", "Unknown")}; Radiomic summary: {data.get("radiomic_summary", "Unknown")}.
"""
