"""
feature_render.py — Convert one row of `Dataset/splits/{Train,Validation,Test}.csv`
into the BEEP-style EHR narrative the LLM sees (Demographics / Diagnosis /
Molecular / Treatment).

The four experiments use the column subsets defined in
`Dataset/Processed/feature_groups.json`:

    Exp1: metadata only
    Exp2: metadata + molecular
    Exp3: metadata + treatment   (no salvage, no Immunotherapy/Brachytherapy)
    Exp4: metadata + molecular + treatment
    Exp5: PPUM-aligned core features only
    Exp6: PPUM-aligned core + IDH1 and radiotherapy details

Numeric molecular codes (e.g. MGMT methylation = 4) are de-coded to plain
English using `Model/configs/molecular_codes.json`, so the LLM never sees
opaque integers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]            # BrainTumor/first/
DATASET_DIR = ROOT.parent / "dataset" / "first"
SPLITS_DIR = DATASET_DIR / "splits"
GROUPS_PATH = DATASET_DIR / "Processed" / "feature_groups.json"
CODES_PATH = ROOT / "Model" / "configs" / "molecular_codes.json"

LABEL_COL = "y"
ID_COL = "Patient_ID"

# ---------- column → block layout ---------------------------------------------
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
    "TP53 alteration", "Other mutations/alterations",
]

TREATMENT_COLS = [
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

PPUM_ALIGNED_COLS = [
    "Sex at Birth",
    "Race",
    "Age at diagnosis",
    "Primary Diagnosis",
    "Grade of Primary Brain Tumor",
    "Stereotactic Biopsy before Surgical Resection",
]

PPUM_ALIGNED_PLUS_COLS = PPUM_ALIGNED_COLS + [
    "IDH1 mutation",
    "Radiation Therapy",
    "Dose",
    "Number of Fractions",
]

EXPERIMENT_BLOCKS: dict[str, list[str]] = {
    "Exp1": ["Demographics", "Diagnosis"],
    "Exp2": ["Demographics", "Diagnosis", "Molecular"],
    "Exp3": ["Demographics", "Diagnosis", "Treatment"],
    "Exp4": ["Demographics", "Diagnosis", "Molecular", "Treatment"],
    "Exp5": ["PPUM-aligned core"],
    "Exp6": ["PPUM-aligned plus"],
}

BLOCK_TO_COLS = {
    "Demographics": DEMOGRAPHIC_COLS,
    "Diagnosis":    DIAGNOSIS_COLS,
    "Molecular":    MOLECULAR_COLS,
    "Treatment":    TREATMENT_COLS,
    "PPUM-aligned core": PPUM_ALIGNED_COLS,
    "PPUM-aligned plus": PPUM_ALIGNED_PLUS_COLS,
}

PRETTY_TREATMENT_NAME = {
    "Number of days from Diagnosis to First surgery or procedure ": "Days to surgery",
    "Initial Chemo Therapy": "Initial chemotherapy",
    "Name of Initial Chemo Therapy": "Chemo agent",
    " Number of days from Diagnosis to Initial Chemo Therapy Start date": "Days to chemo start",
    " Number of days from Diagnosis to Initial Chemo Therapy end date": "Days to chemo end",
    "Radiation Therapy": "Radiation therapy",
    "Number of days from Diagnosis to Radiation Therapy Start date": "Days to radiation start",
    "Number of days from Diagnosis to Radiation Therapy end date": "Days to radiation end",
    "Dose": "Radiation dose",
    "Number of Fractions": "Radiation fractions",
}

# ---------- low-level helpers --------------------------------------------------
_MOL_CODES: dict[str, dict[str, str]] | None = None


def _load_codes() -> dict[str, dict[str, str]]:
    global _MOL_CODES
    if _MOL_CODES is None:
        with CODES_PATH.open() as f:
            raw = json.load(f)
        _MOL_CODES = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _MOL_CODES


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _pretty_age(v) -> str:
    try:
        return f"{int(round(float(v)))}"
    except Exception:
        return str(v)


def _decode_molecular(col: str, raw_value) -> str | None:
    """Numeric code → human string (or pass through string-typed values)."""
    if _is_missing(raw_value):
        return None
    codes = _load_codes()
    mapping = codes.get(col)
    if mapping is None:
        return str(raw_value).strip()
    key = str(int(raw_value)) if isinstance(raw_value, (int, float)) and not _is_missing(raw_value) else str(raw_value).strip()
    return mapping.get(key, f"{col}: {raw_value}")


def _format_value(col: str, raw_value) -> str | None:
    if _is_missing(raw_value):
        return None
    if col == "Age at diagnosis":
        return _pretty_age(raw_value)
    if isinstance(raw_value, float) and raw_value.is_integer():
        return str(int(raw_value))
    return str(raw_value).strip()


# ---------- block builders -----------------------------------------------------
def _block_demographics(row: pd.Series) -> str | None:
    parts: list[str] = []
    for col in DEMOGRAPHIC_COLS:
        val = _format_value(col, row.get(col))
        if val is not None:
            parts.append(f"{col}: {val}")
    return "Demographics: " + "; ".join(parts) + "." if parts else None


def _block_diagnosis(row: pd.Series) -> str | None:
    parts: list[str] = []
    for col in DIAGNOSIS_COLS:
        val = _format_value(col, row.get(col))
        if val is not None:
            parts.append(f"{col}: {val}")
    return "Diagnosis: " + "; ".join(parts) + "." if parts else None


def _block_molecular(row: pd.Series) -> str | None:
    parts: list[str] = []
    for col in MOLECULAR_COLS:
        if col == "Other mutations/alterations":
            val = _format_value(col, row.get(col))
            if val is not None:
                parts.append(f"Other alterations: {val}")
            continue
        decoded = _decode_molecular(col, row.get(col))
        if decoded is not None:
            parts.append(decoded)
    if not parts:
        return None
    return "Molecular markers: " + "; ".join(parts) + "."


def _block_treatment(row: pd.Series) -> str | None:
    parts: list[str] = []
    for col in TREATMENT_COLS:
        val = _format_value(col, row.get(col))
        if val is None:
            continue
        label = PRETTY_TREATMENT_NAME.get(col, col.strip())
        parts.append(f"{label}: {val}")
    if not parts:
        return None
    return "Treatment: " + "; ".join(parts) + "."


def _normalise_core_diagnosis(value) -> str | None:
    """Harmonise hospital-specific diagnosis spelling for Exp5 only."""
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"-", "unknown", "not recorded"}:
        return None
    if "glioblastoma" in text or text == "gbm" or "gliosarcoma" in text \
            or "gbm feature" in text:
        return "GBM-spectrum glioma"
    if "oligodendro" in text:
        return "Oligodendroglioma"
    if "pilocytic" in text or "pilomyxoid" in text:
        return "Pilocytic/pilomyxoid astrocytoma"
    if "astrocytoma" in text:
        return "Astrocytoma"
    if "glioma" in text:
        return "Diffuse/other glioma"
    return "Other/non-glioma diagnosis"


def _normalise_core_grade(value) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"unknown", "not tested", "nan", "-"}:
        return None
    if "3" in text and "4" in text:
        return "3-4"
    for grade in (1, 2, 3, 4):
        try:
            if float(text) == grade:
                return str(grade)
        except ValueError:
            if str(grade) in text:
                return str(grade)
    return text


def _normalise_core_biopsy(value) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true"}:
        return "Yes"
    if text in {"0", "no", "n", "false"}:
        return "No"
    if text in {"unknown", "not recorded", "-"}:
        return None
    return str(value).strip()


def _block_ppum_aligned(row: pd.Series) -> str | None:
    parts: list[str] = []
    sex = _format_value("Sex at Birth", row.get("Sex at Birth"))
    race = _format_value("Race", row.get("Race"))
    age = _format_value("Age at diagnosis", row.get("Age at diagnosis"))
    diagnosis = _normalise_core_diagnosis(row.get("Primary Diagnosis"))
    grade = _normalise_core_grade(row.get("Grade of Primary Brain Tumor"))
    biopsy = _normalise_core_biopsy(
        row.get("Stereotactic Biopsy before Surgical Resection"))
    for label, value in (
        ("Sex at birth", sex),
        ("Race", race),
        ("Age at diagnosis", age),
        ("Primary diagnosis group", diagnosis),
        ("WHO grade", grade),
        ("Stereotactic biopsy before resection", biopsy),
    ):
        if value is not None:
            parts.append(f"{label}: {value}")
    return "PPUM-aligned baseline: " + "; ".join(parts) + "." if parts else None


def _normalise_radiation(value) -> str:
    if _is_missing(value):
        return "Not documented as given"
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "proton therapy"}:
        return "Given"
    if text in {"no", "n", "0", "false", "unknown", "-"}:
        return "Not documented as given"
    return str(value).strip()


def _normalise_dose(value) -> str:
    if _is_missing(value):
        return "Not available"
    text = str(value).strip()
    try:
        number = float(text.lower().replace("gy", "").strip())
        return f"{number:g} Gy"
    except ValueError:
        return text


def _normalise_fractions(value) -> str:
    if _is_missing(value):
        return "Not available"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value).strip()


def _block_ppum_aligned_plus(row: pd.Series) -> str | None:
    core = _block_ppum_aligned(row)
    idh1 = _decode_molecular("IDH1 mutation", row.get("IDH1 mutation"))
    if idh1 is None:
        idh1 = "IDH1 unknown"
    extension = (
        "Aligned molecular and radiotherapy: "
        f"IDH1 status: {idh1}; "
        f"Radiation therapy: {_normalise_radiation(row.get('Radiation Therapy'))}; "
        f"Radiation dose: {_normalise_dose(row.get('Dose'))}; "
        f"Radiation fractions: {_normalise_fractions(row.get('Number of Fractions'))}."
    )
    return "\n".join(part for part in (core, extension) if part)


_BLOCK_FN = {
    "Demographics": _block_demographics,
    "Diagnosis":    _block_diagnosis,
    "Molecular":    _block_molecular,
    "Treatment":    _block_treatment,
    "PPUM-aligned core": _block_ppum_aligned,
    "PPUM-aligned plus": _block_ppum_aligned_plus,
}


# ---------- public API ---------------------------------------------------------
def render_patient(row: pd.Series, experiment: str) -> str:
    """Return the EHR narrative the LLM sees for this patient × experiment."""
    if experiment not in EXPERIMENT_BLOCKS:
        raise ValueError(f"Unknown experiment {experiment!r}. Pick from {list(EXPERIMENT_BLOCKS)}")
    blocks = []
    for name in EXPERIMENT_BLOCKS[experiment]:
        text = _BLOCK_FN[name](row)
        if text:
            blocks.append(text)
    return "\n".join(blocks)


def render_split(experiment: str, split: str = "Train") -> pd.DataFrame:
    """Render every patient in the given split. Returns DataFrame with
    `Patient_ID`, `text`, `label` columns."""
    csv_path = SPLITS_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        ID_COL: df[ID_COL],
        "text":  [render_patient(r, experiment) for _, r in df.iterrows()],
        "label": df[LABEL_COL].astype(int),
    })
    return out


def list_experiments() -> Iterable[str]:
    return list(EXPERIMENT_BLOCKS.keys())


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="Exp1", choices=list(EXPERIMENT_BLOCKS))
    ap.add_argument("--split", default="Train", choices=["Train", "Validation", "Test"])
    ap.add_argument("--n", type=int, default=2, help="How many sample patients to dump.")
    args = ap.parse_args()

    df = render_split(args.exp, args.split)
    print(f"# {args.split} / {args.exp}: {len(df)} patients (positive rate = {df['label'].mean():.1%})")
    for _, row in df.head(args.n).iterrows():
        print("=" * 60)
        print(f"{row[ID_COL]}  (label = {row['label']})")
        print(row["text"])
    sys.exit(0)
