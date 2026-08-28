from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


FieldType = Literal["text", "number", "select", "textarea", "date"]


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    field_type: FieldType
    group: str
    required: bool = False
    options: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    default: str | float | int | None = None
    help_text: str = ""
    # Only render this field when another field currently equals a value,
    # e.g. ("previous_brain_tumor", "Yes"). Hidden fields stay missing.
    depends_on: tuple[str, str] | None = None

    @property
    def display_label(self) -> str:
        return f"{self.label} *" if self.required else self.label


# UI date keys that are converted to model day-offsets from diagnosis_date.
DATE_OFFSET_KEYS: tuple[str, ...] = (
    "first_surgery_day",
    "initial_chemo_start_day",
    "initial_chemo_end_day",
    "rt_start_day",
    "rt_end_day",
    "time_to_first_progression",
    "additional_therapy_start",
    "additional_therapy_end",
    "immunotherapy_start",
    "immunotherapy_end",
    "latest_mri_day",
)


def _as_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return date(value.year, value.month, value.day)
    return None


def days_from_diagnosis(diagnosis: Any, event: Any) -> int | None:
    start = _as_date(diagnosis)
    end = _as_date(event)
    if start is None or end is None:
        return None
    return (end - start).days


def apply_date_offsets(data: dict[str, Any]) -> dict[str, Any]:
    """Keep doctor-entered dates, and add model day-offsets from diagnosis_date."""
    out = dict(data)
    diagnosis = out.get("diagnosis_date")
    out["diagnosis_date"] = _as_date(diagnosis)
    for key in DATE_OFFSET_KEYS:
        raw = out.get(key)
        if isinstance(raw, date):
            out[f"{key}_date"] = date(raw.year, raw.month, raw.day)
            out[key] = days_from_diagnosis(out["diagnosis_date"], raw)
        elif raw in {"", None}:
            out[key] = None
    return out


YES_NO_UNKNOWN = ("Unknown / not available", "Yes", "No")
MUTATION_OPTIONS = ("Unknown / not tested", "Mutant / positive", "Wildtype / negative")
METHYLATION_OPTIONS = ("Unknown / not tested", "Methylated", "Unmethylated")
DIAGNOSIS_OPTIONS = (
    "Unknown",
    "Glioblastoma",
    "Astrocytoma",
    "Oligodendroglioma",
    "Diffuse glioma",
    "Pilocytic astrocytoma",
    "Other glioma",
)


# The Exp3 hybrid XGBoost does not consume molecular columns; say so in the
# section title itself so doctors know before they fill anything in.
FR_MOLECULAR_GROUP = "Optional molecular record (for case notes only — NOT used by this model's prediction)"


FIRST_RECURRENCE_FIELDS: tuple[Field, ...] = (
    Field("patient_id", "Patient ID / RN", "text", "Patient information", True),
    Field(
        "diagnosis_date",
        "Date of diagnosis",
        "date",
        "Patient information",
        True,
        help_text="Anchor date. All treatment dates below are converted to days from this date for the model.",
    ),
    Field("age", "Age at diagnosis", "number", "Required clinical features", True, min_value=0, max_value=100),
    Field("sex", "Sex at birth", "select", "Required clinical features", True, ("Unknown", "Female", "Male")),
    Field(
        "race",
        "Race",
        "select",
        "Required clinical features",
        True,
        ("Unknown", "Malay", "Chinese", "Indian", "Other", "White", "Black or African American", "Asian"),
        help_text="Local labels are mapped to the MU training race vocabulary before XGBoost inference.",
    ),
    Field(
        "primary_diagnosis",
        "Primary diagnosis",
        "select",
        "Required clinical features",
        True,
        DIAGNOSIS_OPTIONS,
        help_text="Glioblastoma maps to GBM; Oligodendroglioma maps to Oligodendro-glioma in the trained artifact.",
    ),
    Field("grade", "WHO grade of primary brain tumor", "select", "Required clinical features", True, ("Unknown", "1", "2", "3", "4")),
    Field("biopsy_before_resection", "Stereotactic biopsy before surgical resection", "select", "Required clinical features", True, YES_NO_UNKNOWN),
    Field("previous_brain_tumor", "Previous brain tumor history", "select", "Required clinical features", True, YES_NO_UNKNOWN),
    Field(
        "type_previous_brain_tumor",
        "Type of previous brain tumor",
        "select",
        "Required clinical features",
        False,
        ("Unknown / not applicable", "Astrocytoma", "Glioblastoma (GBM)", "Neurocytoma", "Other"),
        help_text="Shown because previous brain tumor is Yes.",
        depends_on=("previous_brain_tumor", "Yes"),
    ),
    Field(
        "year_previous_surgery",
        "Year of previous surgery",
        "number",
        "Required clinical features",
        False,
        min_value=1950,
        max_value=2026,
        help_text="Shown because previous brain tumor is Yes.",
        depends_on=("previous_brain_tumor", "Yes"),
    ),
    Field(
        "grade_previous_brain_tumor",
        "WHO grade of previous brain tumor",
        "select",
        "Required clinical features",
        False,
        ("Unknown / not applicable", "2", "3", "4"),
        help_text="Shown because previous brain tumor is Yes.",
        depends_on=("previous_brain_tumor", "Yes"),
    ),
    Field(
        "first_surgery_day",
        "Date of first surgery/procedure",
        "date",
        "Required treatment features",
        True,
        help_text="Enter the calendar date. The model uses days from diagnosis, calculated automatically.",
    ),
    Field("initial_chemo", "Initial chemotherapy given", "select", "Required treatment features", True, YES_NO_UNKNOWN),
    Field(
        "initial_chemo_name",
        "Name of initial chemotherapy",
        "text",
        "Required treatment features",
        help_text="e.g. Temozolomide (TMZ) or Lomustine (CCNU).",
        depends_on=("initial_chemo", "Yes"),
    ),
    Field(
        "initial_chemo_start_day",
        "Chemotherapy start date",
        "date",
        "Required treatment features",
        help_text="Enter the calendar date. The model uses days from diagnosis, calculated automatically.",
        depends_on=("initial_chemo", "Yes"),
    ),
    Field(
        "initial_chemo_end_day",
        "Chemotherapy end date",
        "date",
        "Required treatment features",
        help_text="Enter the calendar date. The model uses days from diagnosis, calculated automatically.",
        depends_on=("initial_chemo", "Yes"),
    ),
    Field("radiotherapy", "Radiation therapy given", "select", "Required treatment features", True, YES_NO_UNKNOWN),
    Field(
        "rt_start_day",
        "Radiation start date",
        "date",
        "Required treatment features",
        help_text="Enter the calendar date. The model uses days from diagnosis, calculated automatically.",
        depends_on=("radiotherapy", "Yes"),
    ),
    Field(
        "rt_end_day",
        "Radiation end date",
        "date",
        "Required treatment features",
        help_text="Enter the calendar date. The model uses days from diagnosis, calculated automatically.",
        depends_on=("radiotherapy", "Yes"),
    ),
    Field("rt_dose", "Radiation dose (Gy)", "number", "Required treatment features", min_value=0, max_value=120, depends_on=("radiotherapy", "Yes")),
    Field("rt_fractions", "Number of radiation fractions", "number", "Required treatment features", min_value=0, max_value=80, depends_on=("radiotherapy", "Yes")),
    Field(
        "idh1",
        "IDH1 mutation",
        "select",
        FR_MOLECULAR_GROUP,
        False,
        MUTATION_OPTIONS,
        help_text="Not used by the Exp3 hybrid XGBoost. Kept for case records only.",
    ),
    Field("idh2", "IDH2 mutation", "select", FR_MOLECULAR_GROUP, False, MUTATION_OPTIONS),
    Field("codeletion_1p19q", "1p/19q codeletion", "select", FR_MOLECULAR_GROUP, False, MUTATION_OPTIONS),
    Field("mgmt", "MGMT methylation", "select", FR_MOLECULAR_GROUP, False, METHYLATION_OPTIONS),
    Field("atrx", "ATRX mutation/loss", "select", FR_MOLECULAR_GROUP, False, MUTATION_OPTIONS),
    Field("tert", "TERT promoter mutation", "select", FR_MOLECULAR_GROUP, False, MUTATION_OPTIONS),
    Field("egfr", "EGFR amplification", "select", FR_MOLECULAR_GROUP, False, MUTATION_OPTIONS),
    Field("other_molecular", "Other mutations / alterations", "textarea", FR_MOLECULAR_GROUP),
)


SECOND_RECURRENCE_FIELDS: tuple[Field, ...] = (
    Field("patient_id", "Patient ID / RN", "text", "Patient information", True),
    Field(
        "diagnosis_date",
        "Date of diagnosis",
        "date",
        "Patient information",
        True,
        help_text="Anchor date. Recurrence, salvage and MRI dates are converted to days from this date for the model.",
    ),
    Field("age", "Age at diagnosis", "number", "Required baseline features", True, min_value=0, max_value=100),
    Field("sex", "Sex at birth", "select", "Required baseline features", True, ("Unknown", "Female", "Male")),
    Field("primary_diagnosis", "Primary diagnosis", "select", "Required baseline features", True, DIAGNOSIS_OPTIONS),
    Field("grade", "WHO grade of primary brain tumor", "select", "Required baseline features", True, ("Unknown", "1", "2", "3", "4")),
    Field(
        "time_to_first_progression",
        "Date of first recurrence/progression",
        "date",
        "Required first-recurrence landmark",
        True,
        help_text="Enter the calendar date, not the number of days.",
    ),
    Field("type_first_progression", "Type of first progression", "select", "Required first-recurrence landmark", True, ("Unknown", "Local", "Distant", "Leptomeningeal", "Multifocal", "Clinical progression only")),
    Field("multiple_surgeries", "Multiple surgeries before second-recurrence prediction", "select", "Required first-recurrence landmark", True, YES_NO_UNKNOWN),
    Field("additional_therapy", "Additional/salvage therapy given after first recurrence", "select", "Required salvage treatment features", True, YES_NO_UNKNOWN),
    Field(
        "additional_therapy_start",
        "Salvage therapy start date",
        "date",
        "Required salvage treatment features",
        help_text="Enter the calendar date, not the number of days.",
        depends_on=("additional_therapy", "Yes"),
    ),
    Field(
        "additional_therapy_end",
        "Salvage therapy end date",
        "date",
        "Required salvage treatment features",
        help_text="Enter the calendar date, not the number of days.",
        depends_on=("additional_therapy", "Yes"),
    ),
    Field("additional_cycles", "Number of salvage therapy cycles", "number", "Required salvage treatment features", min_value=0, max_value=100, depends_on=("additional_therapy", "Yes")),
    Field("immunotherapy", "Immunotherapy given", "select", "Required salvage treatment features", False, YES_NO_UNKNOWN),
    Field(
        "immunotherapy_start",
        "Immunotherapy start date",
        "date",
        "Required salvage treatment features",
        help_text="Enter the calendar date, not the number of days.",
        depends_on=("immunotherapy", "Yes"),
    ),
    Field(
        "immunotherapy_end",
        "Immunotherapy end date",
        "date",
        "Required salvage treatment features",
        help_text="Enter the calendar date, not the number of days.",
        depends_on=("immunotherapy", "Yes"),
    ),
    Field(
        "latest_mri_day",
        "Latest eligible MRI date",
        "date",
        "Required imaging evidence",
        True,
        help_text="Enter the calendar date of the last MRI before second-recurrence prediction.",
    ),
    Field("mri_report", "MRI / RadFM report text before second-recurrence prediction", "textarea", "Required imaging evidence", True),
    Field("enhancing_volume", "Enhancing tumor volume / radiomic volume", "number", "Optional radiomic features", min_value=0, max_value=100000),
    Field("edema_volume", "Edema / FLAIR abnormality volume", "number", "Optional radiomic features", min_value=0, max_value=100000),
    Field("radiomic_summary", "Other radiomic feature summary", "textarea", "Optional radiomic features"),
    Field("idh1", "IDH1 mutation", "select", "Optional molecular features", False, MUTATION_OPTIONS),
    Field("codeletion_1p19q", "1p/19q codeletion", "select", "Optional molecular features", False, MUTATION_OPTIONS),
    Field("mgmt", "MGMT methylation", "select", "Optional molecular features", False, METHYLATION_OPTIONS),
)


FIRST_RECURRENCE_GUIDE = {
    "Must have": [
        "Patient identifier and date of diagnosis",
        "Age, sex, race",
        "Primary diagnosis and WHO grade",
        "Previous brain tumor history",
        "First surgery/procedure date",
        "Initial chemotherapy and radiotherapy status/dates",
    ],
    "Helpful if available": [
        "Exact chemotherapy name and radiation dose/fractions",
        "Molecular markers (not used by the Exp3 hybrid XGBoost; live embedding prompt also omits them to match training)",
    ],
    "Do not use": [
        "Follow-up MRI after the first recurrence event",
        "Salvage treatment after first recurrence",
        "Death date or post-event outcome notes",
    ],
}


SECOND_RECURRENCE_GUIDE = {
    "Must have": [
        "Date of diagnosis and date of first recurrence/progression",
        "Type of first progression",
        "Salvage/additional therapy status and dates",
        "Latest eligible pre-second-event MRI date",
        "MRI report or RadFM caption before the prediction point",
    ],
    "Helpful if available": [
        "Radiomic tumor volume / edema volume",
        "Molecular markers",
        "Immunotherapy and other salvage treatment details",
    ],
    "Do not use": [
        "MRI acquired after second recurrence",
        "Treatment started after second recurrence",
        "Death or post-second-event notes",
    ],
}
