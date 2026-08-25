"""Map doctor-facing UI values onto the MU training vocabulary used by XGBoost."""

from __future__ import annotations

from typing import Any


_MISSING_TEXT = {
    "",
    "unknown",
    "unknown / not available",
    "unknown / not tested",
    "not available",
    "not tested",
    "none",
    "nan",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return _norm(value) in _MISSING_TEXT


def map_yes_no_binary(value: Any) -> int | None:
    text = _norm(value)
    if text in _MISSING_TEXT:
        return None
    if text in {"yes", "y", "true", "1"}:
        return 1
    if text in {"no", "n", "false", "0"}:
        return 0
    return None


def map_yes_no_or_nan(value: Any) -> str | None:
    text = _norm(value)
    if text in _MISSING_TEXT:
        return None
    if text.startswith("yes"):
        return "Yes"
    if text == "no":
        return "No"
    return str(value).strip() if value not in {None, ""} else None


def map_diagnosis(value: Any) -> str | None:
    text = _norm(value)
    if text in _MISSING_TEXT:
        return None
    mapping = {
        "glioblastoma": "GBM",
        "gbm": "GBM",
        "astrocytoma": "Astrocytoma",
        "oligodendroglioma": "Oligodendro-glioma",
        "oligodendro-glioma": "Oligodendro-glioma",
        "diffuse glioma": "Diffuse glioma",
        "pilocytic astrocytoma": "Pilocytic astrocytoma",
        "glioma w/ gbm features": "Glioma w/ GBM features",
        "other glioma": None,
    }
    return mapping.get(text, str(value).strip())


def map_race(value: Any) -> str | None:
    text = _norm(value)
    if text in _MISSING_TEXT:
        return "Unknown"
    mapping = {
        "white": "White",
        "black or african american": "Black or African American",
        "black": "Black or African American",
        "asian": "Asian",
        "malay": "Asian",
        "chinese": "Asian",
        "indian": "Asian",
        "other": "Unknown",
        "unknown": "Unknown",
    }
    return mapping.get(text, "Unknown")


def map_sex(value: Any) -> str | None:
    text = _norm(value)
    if text in _MISSING_TEXT:
        return None
    if text in {"male", "m"}:
        return "Male"
    if text in {"female", "f"}:
        return "Female"
    return None


def map_mutation_code(value: Any, *, unknown_code: int = 2) -> int | None:
    """Map UI mutation labels to MU numeric codes (0=wildtype/neg, 1=mutant/pos, unknown=code)."""
    if value is None or (isinstance(value, float) and value != value):
        return unknown_code
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = _norm(value)
    if text in _MISSING_TEXT:
        return unknown_code
    if any(token in text for token in ("mutant", "positive", "mutated", "amplified", "deleted", "altered", "codelet")):
        return 1
    if any(token in text for token in ("wildtype", "negative", "intact", "not amplified", "unaltered", "no ")):
        return 0
    return unknown_code


def map_mgmt(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = _norm(value)
    if text in _MISSING_TEXT:
        return 4
    if "unmethyl" in text:
        return 0
    if "methyl" in text:
        return 1
    if "indeterminate" in text:
        return 2
    return 4


def map_codeletion_1p19q(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = _norm(value)
    if text in _MISSING_TEXT:
        return 10
    if "codelet" in text or "mutant" in text or "positive" in text:
        return 1
    if "intact" in text or "wildtype" in text or "negative" in text or text.startswith("no "):
        return 0
    return 10


def map_chr7_10(value: Any) -> int | None:
    binary = map_yes_no_binary(value)
    if binary is not None:
        return binary
    text = _norm(value)
    if text in _MISSING_TEXT:
        return 2
    if "+7" in text or "present" in text:
        return 1
    return 2


def map_yes_or_nan(value: Any) -> str | None:
    """MU training data only contains 'Yes' (or blank) for chemo/RT status.

    Mapping UI 'No' to missing keeps the one-hot row identical to how
    untreated patients appeared during training.
    """
    text = _norm(value)
    if text.startswith("yes"):
        return "Yes"
    return None


def map_previous_tumor_type(value: Any) -> str | None:
    """Map to the exact MU vocabulary: Astrocytoma, GBM, Neurocytoma."""
    text = _norm(value)
    if text in _MISSING_TEXT or "not applicable" in text:
        return None
    if "glioblastoma" in text or text == "gbm" or "(gbm)" in text:
        return "GBM"
    if "astrocytoma" in text:
        return "Astrocytoma"
    if "neurocytoma" in text:
        return "Neurocytoma"
    return None


def map_previous_tumor_grade(value: Any) -> str | None:
    """Map UI grade to the exact MU one-hot strings (note trailing spaces)."""
    text = _norm(value).removeprefix("grade").strip()
    if text in _MISSING_TEXT or "not applicable" in text:
        return None
    # Training categories: 'Grade 2', 'Grade 2 ', 'Grade 3 ', 'Grade 4'.
    mapping = {"2": "Grade 2", "3": "Grade 3 ", "4": "Grade 4"}
    return mapping.get(text)


def map_chemo_name(value: Any) -> str | None:
    """Normalise free-text chemo names to the MU vocabulary."""
    text = _norm(value)
    if text in _MISSING_TEXT:
        return None
    if "temozolomide" in text or text in {"tmz", "temodal", "temodar"}:
        return "Temozolomide"
    if "lomustine" in text or text in {"ccnu", "ceenu"}:
        return "Lomustine"
    return None


def map_dose(value: Any) -> str | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value)} Gy"
        return f"{value} Gy"
    text = str(value).strip()
    if text.lower().endswith("gy"):
        return text
    return f"{text} Gy"


def encode_ui_clinical_row(data: dict[str, Any]) -> dict[str, Any]:
    """Convert Streamlit form values into the MU clinical column schema.

    The First Recurrence hybrid uses the Exp3 metadata+treatment subset;
    molecular keys are mapped here but are not fed to that XGBoost model.
    """
    return {
        "Sex at Birth": map_sex(data.get("sex")),
        "Race": map_race(data.get("race")),
        "Age at diagnosis": data.get("age"),
        "Primary Diagnosis": map_diagnosis(data.get("primary_diagnosis")),
        "Grade of Primary Brain Tumor": None if is_missing(data.get("grade")) else str(data.get("grade")),
        "Stereotactic Biopsy before Surgical Resection": map_yes_no_binary(data.get("biopsy_before_resection")),
        "Previous Brain Tumor": map_yes_no_or_nan(data.get("previous_brain_tumor")),
        "Type of previous brain tumor": map_previous_tumor_type(data.get("type_previous_brain_tumor")),
        "Year of previous surgery": data.get("year_previous_surgery"),
        "Grade of Previous brain tumor": map_previous_tumor_grade(data.get("grade_previous_brain_tumor")),
        "IDH1 mutation": map_mutation_code(data.get("idh1"), unknown_code=2),
        "IDH2 mutation": map_mutation_code(data.get("idh2"), unknown_code=2),
        "1p/19q": map_codeletion_1p19q(data.get("codeletion_1p19q")),
        "ATRX mutation": map_mutation_code(data.get("atrx"), unknown_code=4),
        "MGMT methylation": map_mgmt(data.get("mgmt")),
        "BRAF V600E mutation": map_mutation_code(data.get("braf"), unknown_code=2),
        "TERT promoter mutation": map_mutation_code(data.get("tert"), unknown_code=2),
        "Chromosome 7 gain and Chromosome 10 loss": map_chr7_10(data.get("chr7_10")),
        "H3-3A mutation": map_mutation_code(data.get("h3_3a"), unknown_code=2),
        "EGFR amplification": map_mutation_code(data.get("egfr"), unknown_code=2),
        "PTEN mutation": map_mutation_code(data.get("pten"), unknown_code=0),
        "CDKN2A/B deletion": map_mutation_code(data.get("cdkn2ab"), unknown_code=0),
        "TP53 alteration": map_mutation_code(data.get("tp53"), unknown_code=0),
        "Other mutations/alterations": None if is_missing(data.get("other_molecular")) else data.get("other_molecular"),
        "Number of days from Diagnosis to First surgery or procedure ": data.get("first_surgery_day"),
        "Initial Chemo Therapy": map_yes_or_nan(data.get("initial_chemo")),
        "Name of Initial Chemo Therapy": map_chemo_name(data.get("initial_chemo_name")),
        " Number of days from Diagnosis to Initial Chemo Therapy Start date": data.get("initial_chemo_start_day"),
        " Number of days from Diagnosis to Initial Chemo Therapy end date": data.get("initial_chemo_end_day"),
        "Radiation Therapy": map_yes_or_nan(data.get("radiotherapy")),
        "Number of days from Diagnosis to Radiation Therapy Start date": data.get("rt_start_day"),
        "Number of days from Diagnosis to Radiation Therapy end date": data.get("rt_end_day"),
        "Dose": map_dose(data.get("rt_dose")),
        "Number of Fractions": data.get("rt_fractions"),
    }
