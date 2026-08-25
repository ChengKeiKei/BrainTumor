"""
feature_render.py — Convert one row of `Dataset/Splits/{Train,Validation,Test}.csv`
into the BEEP-style EHR narrative the LLM sees, for the **Second Progression**
prediction task. Also produces a separate, keyword-style retrieval query
used by BM25 + dense retrieval (see `render_retrieval_query`).

Experiment lineup (current, locked in with the user — see Dataset/README.md
and Model/configs/grid.yaml):

    ExpA_TxNoMol     : Demographics + Diagnosis + InitialTx + SalvageTx
    ExpA_Tx          : ExpA_TxNoMol + Molecular
    ExpB_TxRadiomic  : ExpA_Tx + Timepoints (MRI day columns) + Radiomic features
    ExpC_TxVLM       : ExpA_Tx + Timepoints + VLM (RadFM image captions)
    ExpD_TxRadVLM    : ExpA_Tx + Timepoints + Radiomic + VLM

(Legacy Exp1-Exp4 names are kept in `EXPERIMENT_BLOCKS` for backward
compatibility with the pre-renumber checkpoints under
`Model/results/_pre_renumber_*/`, but no new code path uses them.)

All three current experiments include Demographics + Diagnosis as the
base patient context; this is the parity choice the user made after the
First_Recur run, so the LLM always sees age/sex/diagnosis even when the
experiment is supposed to isolate molecular or treatment information.

Treatment columns are landmark-gated upstream in
`Dataset/Clinical_preprocessing.ipynb` (every salvage therapy whose
start day ≥ `Landmark_day` is set to NaN before splits are written),
so the defensive `_block_salvage_tx` guard is only a belt-and-braces
check.

Radiomic and VLM blocks read live data from
`Dataset/Processed/radiomic_features.csv` and `mri_captions.csv` (or
`mri_captions_v2_context.csv` when present — see prompt-version logic
in `_block_vlm`).

Numeric molecular codes (e.g. MGMT methylation = 4) are de-coded to plain
English using `Model/configs/molecular_codes.json`, so the LLM never sees
opaque integers.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT        = Path(__file__).resolve().parents[2]            # 24083155/second/
DATASET_DIR = ROOT.parent / "dataset" / "second"
SPLITS_DIR  = DATASET_DIR / "splits"
PROCESSED_D = DATASET_DIR / "Processed"
GROUPS_PATH = PROCESSED_D / "feature_groups.json"
CODES_PATH  = ROOT / "Model" / "configs" / "molecular_codes.json"

LABEL_COL = "y"
ID_COL    = "Patient_ID"

# --------------------------------------------------------------------------- #
# Column groupings (exact column names match Dataset/Processed/clean_clinical.csv
# and the Train/Validation/Test split CSVs).
# --------------------------------------------------------------------------- #
# Demographics block (kept identical to First_Recur for cross-task consistency).
# All three columns have 100% coverage in Second_Recur's clean_clinical.csv.
DEMOGRAPHIC_COLS = ["Sex at Birth", "Race", "Age at diagnosis"]

# Diagnosis block (Primary Dx, Grade, prior tumor history). The "previous"
# fields are sparsely populated (~5-7%) but informative when present.
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

# Initial therapy block (Days 0–~90).
INITIAL_TX_COLS = [
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

# Salvage therapy block (between TTP1 and T₂; preprocessing has already
# nulled out anything dated after T₂, so we don't re-mask here).
SALVAGE_TX_COLS = [
    "Additional Therapy",
    "Cycle length of Additional Therapy (q days)",
    "Number of Days from Diagnosis to Starting Additional Therapy ",
    "Number of Days from Diagnosis to Complete Additional Therapy ",
    "Number of Cycles of Additional Therapy",
    "Immuno therapy",
    "Cycle length of Immunotherapy (q days)",
    "Number of Days from Diagnosis to Start Immunotherapy ",
    "Number of Days from Diagnosis to Complete Immunotherapy ",
    "Number of Cycles of Immunotherapy",
    "Brachy therapy",
    "Number of Days from Diagnosis to the day of Insertion of Brachytherapy ",
    "Other Types of Therapy (LITT, more chemo, proton therapy)",
    "Number of Days from Diagnosis to Start Other Additional Therapy ",
    "Number of Days from Diagnosis to Complete Other Additional Therapy ",
]

# MRI timepoint days — only kept for Exp3/Exp4 to give the LLM a sense of
# imaging cadence; preprocessing nulled out any timepoint dated >= T₂.
TIMEPOINT_COLS = [
    "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ",
    "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ",
    "Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ",
    "Number of Days from Diagnosis to 4th MRI (Timepoint_4) ",
    "Number of Days from Diagnosis to 5th MRI (Timepoint_5) ",
    "Number of Days from Diagnosis to 6th MRI (Timepoint_6) ",
]

# What ends up in each experiment's prompt.
EXPERIMENT_BLOCKS: dict[str, list[str]] = {
    # --- Current canonical experiments (Apr 2026 onward) ----------------------
    # The old Exp1 (Demographics + Diagnosis + Molecular) was dropped because it
    # duplicates First_Recur Exp2 on a smaller cohort and a harder label. The
    # Second_Recur ablation now centres on dynamic features (Salvage Treatment,
    # MRI follow-up timepoints, Radiomic / VLM imaging summaries), with
    # Demographics + Diagnosis + Molecular kept as patient-anchor base blocks.
    "ExpA_TxNoMol":     ["Demographics", "Diagnosis",
                         "InitialTx", "SalvageTx"],
    "ExpA_Tx":          ["Demographics", "Diagnosis", "Molecular",
                         "InitialTx", "SalvageTx"],
    "ExpB_TxRadiomic":  ["Demographics", "Diagnosis", "Molecular",
                         "InitialTx", "SalvageTx", "Timepoints", "Radiomic"],
    "ExpC_TxVLM":       ["Demographics", "Diagnosis", "Molecular",
                         "InitialTx", "SalvageTx", "Timepoints", "VLM"],
    "ExpD_TxRadVLM":    ["Demographics", "Diagnosis", "Molecular",
                         "InitialTx", "SalvageTx", "Timepoints", "Radiomic", "VLM"],
    # --- Legacy aliases (kept so older artifacts in results/_pre_renumber/
    # still render against the same code path, and so audit scripts that
    # reference Exp1-4 continue to work). DO NOT use for new runs.
    "Exp1": ["Demographics", "Diagnosis", "Molecular"],
    "Exp2": ["Demographics", "Diagnosis", "Molecular", "InitialTx", "SalvageTx"],
    "Exp3": ["Demographics", "Diagnosis", "Molecular", "InitialTx", "SalvageTx",
             "Timepoints", "Radiomic"],
    "Exp4": ["Demographics", "Diagnosis", "Molecular", "InitialTx", "SalvageTx",
             "Timepoints", "VLM"],
}

# Pretty labels for treatment-related columns (the raw column names are
# verbose and contain trailing spaces; the LLM doesn't need that noise).
PRETTY_TX_LABEL = {
    "Number of days from Diagnosis to First surgery or procedure ":   "Days to surgery",
    "Initial Chemo Therapy":                                          "Initial chemotherapy",
    "Name of Initial Chemo Therapy":                                  "Chemo agent",
    " Number of days from Diagnosis to Initial Chemo Therapy Start date": "Days to chemo start",
    " Number of days from Diagnosis to Initial Chemo Therapy end date":   "Days to chemo end",
    "Radiation Therapy":                                              "Radiation therapy",
    "Number of days from Diagnosis to Radiation Therapy Start date":  "Days to radiation start",
    "Number of days from Diagnosis to Radiation Therapy end date":    "Days to radiation end",
    "Dose":                                                           "Radiation dose",
    "Number of Fractions":                                            "Radiation fractions",
    "Additional Therapy":                                             "Salvage chemotherapy",
    "Cycle length of Additional Therapy (q days)":                    "Salvage cycle length (days)",
    "Number of Days from Diagnosis to Starting Additional Therapy ":  "Days to salvage start",
    "Number of Days from Diagnosis to Complete Additional Therapy ":  "Days to salvage end",
    "Number of Cycles of Additional Therapy":                         "Salvage cycles completed",
    "Immuno therapy":                                                 "Immunotherapy",
    "Cycle length of Immunotherapy (q days)":                         "Immuno cycle length (days)",
    "Number of Days from Diagnosis to Start Immunotherapy ":          "Days to immuno start",
    "Number of Days from Diagnosis to Complete Immunotherapy ":       "Days to immuno end",
    "Number of Cycles of Immunotherapy":                              "Immuno cycles completed",
    "Brachy therapy":                                                 "Brachytherapy",
    "Number of Days from Diagnosis to the day of Insertion of Brachytherapy ": "Days to brachy insertion",
    "Other Types of Therapy (LITT, more chemo, proton therapy)":      "Other salvage therapy",
    "Number of Days from Diagnosis to Start Other Additional Therapy ":   "Days to other-Tx start",
    "Number of Days from Diagnosis to Complete Other Additional Therapy ":"Days to other-Tx end",
}

PRETTY_TIMEPOINT_LABEL = {
    "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ": "MRI #1 day",
    "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ": "MRI #2 day",
    "Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ": "MRI #3 day",
    "Number of Days from Diagnosis to 4th MRI (Timepoint_4) ": "MRI #4 day",
    "Number of Days from Diagnosis to 5th MRI (Timepoint_5) ": "MRI #5 day",
    "Number of Days from Diagnosis to 6th MRI (Timepoint_6) ": "MRI #6 day",
}


# --------------------------------------------------------------------------- #
# Low-level helpers (re-used from First_Recur for cross-task consistency)
# --------------------------------------------------------------------------- #
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


def _decode_molecular(col: str, raw_value) -> str | None:
    """Numeric code → human string (or pass through string-typed values)."""
    if _is_missing(raw_value):
        return None
    codes = _load_codes()
    mapping = codes.get(col)
    if mapping is None:
        return str(raw_value).strip()
    if isinstance(raw_value, (int, float)) and not _is_missing(raw_value):
        key = str(int(raw_value))
    else:
        key = str(raw_value).strip()
    return mapping.get(key, f"{col}: {raw_value}")


def _format_value(raw_value) -> str | None:
    if _is_missing(raw_value):
        return None
    # Coerce numeric-looking integers (incl. numpy float64, pandas Int64)
    try:
        v = float(raw_value)
        if not math.isnan(v) and float(v).is_integer():
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return str(raw_value).strip()


# --------------------------------------------------------------------------- #
# Per-block builders
# --------------------------------------------------------------------------- #
def _block_demographics(row: pd.Series) -> str | None:
    """Sex, Race, Age at diagnosis. All 100 % populated in clean_clinical.csv."""
    parts: list[str] = []
    for col in DEMOGRAPHIC_COLS:
        val = _format_value(row.get(col))
        if val is None:
            continue
        if col == "Age at diagnosis":
            try:
                val = str(int(round(float(val))))
            except (TypeError, ValueError):
                pass
        parts.append(f"{col}: {val}")
    return "Demographics: " + "; ".join(parts) + "." if parts else None


def _block_diagnosis(row: pd.Series) -> str | None:
    """Primary Dx, Grade, biopsy & previous-tumor history."""
    parts: list[str] = []
    for col in DIAGNOSIS_COLS:
        val = _format_value(row.get(col))
        if val is not None:
            parts.append(f"{col}: {val}")
    return "Diagnosis: " + "; ".join(parts) + "." if parts else None


def _block_molecular(row: pd.Series) -> str | None:
    parts: list[str] = []
    for col in MOLECULAR_COLS:
        if col == "Other mutations/alterations":
            val = _format_value(row.get(col))
            if val is not None:
                parts.append(f"Other alterations: {val}")
            continue
        decoded = _decode_molecular(col, row.get(col))
        if decoded is not None:
            parts.append(decoded)
    if not parts:
        return None
    return "Molecular markers: " + "; ".join(parts) + "."


def _block_initial_tx(row: pd.Series) -> str | None:
    parts: list[str] = []
    for col in INITIAL_TX_COLS:
        val = _format_value(row.get(col))
        if val is None:
            continue
        parts.append(f"{PRETTY_TX_LABEL.get(col, col.strip())}: {val}")
    if not parts:
        return None
    return "Initial treatment: " + "; ".join(parts) + "."


def _block_salvage_tx(row: pd.Series) -> str | None:
    """Render salvage treatments between 1st progression and the prediction
    landmark.

    Defensive guards (belt-and-braces vs `_build_clinical_preprocessing.py`):
      1. Any start-day cell whose value >= landmark is dropped.
      2. Any end-day cell whose value > landmark is dropped.
      3. The `Number of Cycles of <Salvage Tx>` cell is capped to whatever
         could have been completed by `Landmark_day` given the recorded
         start day and cycle length — prevents post-landmark cycle counts
         from leaking into the prompt.
    """
    landmark = row.get("Landmark_day")

    def _f(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(v) else v

    L = _f(landmark)

    # ----- recompute capped cycle counts ------------------------------------
    cap_specs = [
        ("Number of Cycles of Additional Therapy",
         "Number of Days from Diagnosis to Starting Additional Therapy ",
         "Cycle length of Additional Therapy (q days)"),
        ("Number of Cycles of Immunotherapy",
         "Number of Days from Diagnosis to Start Immunotherapy ",
         "Cycle length of Immunotherapy (q days)"),
    ]
    capped_n = {}
    for ncyc_col, start_col, clen_col in cap_specs:
        s, cl = _f(row.get(start_col)), _f(row.get(clen_col))
        n     = _f(row.get(ncyc_col))
        if None in (L, s, cl, n) or cl <= 0:
            continue
        max_n = max(0, math.floor((L - s) / cl))
        if n > max_n:
            capped_n[ncyc_col] = max_n   # may be 0 → "no cycles by landmark"

    parts: list[str] = []
    for col in SALVAGE_TX_COLS:
        val_raw = row.get(col)
        val     = _format_value(val_raw)
        if val is None:
            continue
        # (1) drop any start-day cell at/after landmark
        if (col.endswith("Start Other Additional Therapy ") or
            col.endswith("Starting Additional Therapy ") or
            col.endswith("Start Immunotherapy ") or
            col.endswith("Insertion of Brachytherapy ")):
            v = _f(val_raw)
            if L is not None and v is not None and v >= L:
                continue
        # (2) drop any end-day cell after landmark
        if (col.endswith("Complete Additional Therapy ") or
            col.endswith("Complete Immunotherapy ") or
            col.endswith("Complete Other Additional Therapy ")):
            v = _f(val_raw)
            if L is not None and v is not None and v > L:
                continue
        # (3) override cycle count with landmark-capped value
        if col in capped_n:
            val = str(int(capped_n[col]))
        parts.append(f"{PRETTY_TX_LABEL.get(col, col.strip())}: {val}")
    if not parts:
        return None
    return ("Salvage treatment (between 1st progression and Landmark_day): "
            + "; ".join(parts) + ".")


def _block_timepoints(row: pd.Series) -> str | None:
    """MRI cadence: only render scan days strictly before Landmark_day."""
    landmark = row.get("Landmark_day")
    try:
        L = float(landmark)
        if math.isnan(L):
            L = None
    except (TypeError, ValueError):
        L = None

    parts: list[str] = []
    n_pre, latest = 0, None
    for col in TIMEPOINT_COLS:
        val_raw = row.get(col)
        try:
            v = float(val_raw)
            if math.isnan(v):
                v = None
        except (TypeError, ValueError):
            v = None
        if v is None:
            continue
        if L is not None and v >= L:
            continue
        n_pre += 1
        latest = v if latest is None else max(latest, v)
        val_fmt = _format_value(val_raw)
        if val_fmt is None:
            continue
        parts.append(f"{PRETTY_TIMEPOINT_LABEL.get(col, col.strip())}: {val_fmt}")

    if not parts:
        return None
    head = "MRI cadence (days from diagnosis, pre-landmark only): " + "; ".join(parts) + "."
    if L is not None and latest is not None:
        head += f" ({n_pre} pre-landmark scan(s); latest at day {int(latest)}, {int(L - latest)} d before Landmark_day.)"
    return head


# Live data blocks — read from Processed/radiomic_features.csv and Processed/
# mri_captions{,_v2_context}.csv. Both fall back to None (= block omitted)
# when the artefact is missing for this patient, so the prompt never renders
# a misleading "unknown" placeholder for an unobserved modality.
def _block_radiomic(row: pd.Series) -> str | None:
    """Pull the patient's baseline (Timepoint-1) radiomic profile from
    Dataset/Processed/radiomic_features.csv if available, else return None."""
    try:
        df = _radiomic_table()
    except FileNotFoundError:
        return None
    sub = df[df[ID_COL] == row.get(ID_COL)]
    if sub.empty:
        return None
    parts: list[str] = []
    for col in [c for c in sub.columns if c != ID_COL]:
        val = _format_value(sub.iloc[0][col])
        if val is not None:
            parts.append(f"{col}: {val}")
    if not parts:
        return None
    return "Pre-landmark radiomic profile (latest scan before prediction): " + "; ".join(parts) + "."


def _filter_captions_pre_landmark(sub: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Drop caption rows whose ``Day_from_diag >= Landmark_day``.

    Defence-in-depth: the current caption CSVs happen to be clean by data
    construction, but if a future caption file ever contains a post-landmark
    scan, this filter prevents it from ever appearing in a prompt, retrieval
    query, or XGBoost feature.
    """
    landmark = row.get("Landmark_day")
    try:
        L = float(landmark)
        if math.isnan(L):
            return sub
    except (TypeError, ValueError):
        return sub
    if "Day_from_diag" not in sub.columns:
        return sub
    days = pd.to_numeric(sub["Day_from_diag"], errors="coerce")
    return sub[days.notna() & (days < L)]


def _block_vlm(row: pd.Series) -> str | None:
    """Pull the patient's pre-landmark MRI captions from
    Dataset/Processed/mri_captions.csv if available, else return None.
    We concatenate captions in chronological order (earliest first).

    All captions whose ``Day_from_diag >= Landmark_day`` are filtered out
    before rendering (see ``_filter_captions_pre_landmark``).
    """
    try:
        df = _captions_table()
    except FileNotFoundError:
        return None
    sub = df[df[ID_COL] == row.get(ID_COL)].sort_values("Timepoint")
    sub = _filter_captions_pre_landmark(sub, row)
    if sub.empty:
        return None
    parts: list[str] = []
    for _, r in sub.iterrows():
        cap = r.get("caption")
        if _is_missing(cap):
            continue
        parts.append(f"[Timepoint {int(r['Timepoint'])}, day {int(r['Day_from_diag'])}] {cap}")
    if not parts:
        return None
    return "MRI radiology findings:\n" + "\n".join(parts)


_BLOCK_FN = {
    "Demographics": _block_demographics,
    "Diagnosis":    _block_diagnosis,
    "Molecular":    _block_molecular,
    "InitialTx":    _block_initial_tx,
    "SalvageTx":    _block_salvage_tx,
    "Timepoints":   _block_timepoints,
    "Radiomic":     _block_radiomic,
    "VLM":          _block_vlm,
}


# --------------------------------------------------------------------------- #
# Retrieval-query builders (separate from prompt narrative).
#
# WHY THIS EXISTS:
#   The narrative blocks above are written for the LLM to read — they include
#   structured labels ("Demographics:", "Molecular markers:") and "unknown"
#   placeholders for missing markers. When that text is fed verbatim to BM25,
#   the constant labels dominate term frequencies, retrieval converges to the
#   same generic GBM reviews for every patient (we measured: 14 unique 3-doc
#   sets across 101 Exp1 patients on Second_Recur). To get patient-specific
#   retrieval we build a separate, keyword-style query that contains ONLY the
#   patient-distinctive clinical findings (positive molecular markers, named
#   drugs, presence/absence of recorded therapies, salvage-Tx status, imaging
#   findings). This matches BEEP §3.2's use of the patient phenotype as the
#   query, and is what a clinician would type into PubMed.
# --------------------------------------------------------------------------- #
# Constant cohort-level terms that are TRUE for every patient in our cohort
# and therefore carry zero discriminative signal for retrieval. Including
# them collapses BM25 onto the same 3 generic GBM reviews for everyone.
_CONSTANT_COHORT_TERMS = {
    "primary diagnosis", "gbm", "grade 4", "grade of primary brain tumor",
    "glioblastoma multiforme",
}

# Tokens that mean "we don't know" — pure noise to a retriever.
_NULL_MARKERS = {"unknown", "n/a", "na", "none", "not available", "not reported"}


def _query_demographics(row: pd.Series) -> str | None:
    """Age + sex only. Race is rarely discriminative for glioma literature."""
    parts: list[str] = []
    sex = _format_value(row.get("Sex at Birth"))
    age = _format_value(row.get("Age at diagnosis"))
    if age is not None:
        try:
            age_i = int(round(float(age)))
            parts.append(f"age {age_i}")
        except (TypeError, ValueError):
            pass
    if sex is not None:
        parts.append(sex.lower())
    return " ".join(parts) if parts else None


def _query_diagnosis(row: pd.Series) -> str | None:
    """Skip the constant 'Primary Diagnosis: GBM; Grade 4' — every cohort
    patient has it. Keep only previous-tumor history when present."""
    prev = _format_value(row.get("Previous Brain Tumor"))
    if prev and prev.lower() not in _NULL_MARKERS and prev.lower() != "no":
        prev_type = _format_value(row.get("Type of previous brain tumor"))
        return f"prior brain tumor {prev_type or ''}".strip()
    return None


def _query_molecular(row: pd.Series) -> str | None:
    """Keep only KNOWN molecular markers (drop 'unknown'). Each marker is
    rendered as 'mgmt methylated' / 'idh1 mutant' etc.

    These are exactly the terms a clinician would search PubMed with.
    """
    parts: list[str] = []
    for col in MOLECULAR_COLS:
        if col == "Other mutations/alterations":
            v = _format_value(row.get(col))
            if v and v.lower() not in _NULL_MARKERS:
                parts.append(v.lower())
            continue
        decoded = _decode_molecular(col, row.get(col))
        if decoded is None:
            continue
        d = decoded.lower()
        if any(nm in d for nm in _NULL_MARKERS):
            continue
        parts.append(d)
    return " ".join(parts) if parts else None


def _query_initial_tx(row: pd.Series) -> str | None:
    """Initial therapy — keep agent names + presence flags, skip day numbers.
    Numerics don't help BM25 and dilute the embedding."""
    parts: list[str] = []
    chemo_yn = _format_value(row.get("Initial Chemo Therapy"))
    if chemo_yn and chemo_yn != "0":
        agent = _format_value(row.get("Name of Initial Chemo Therapy"))
        parts.append(f"chemotherapy {agent.lower()}" if agent else "chemotherapy")
    rt_yn = _format_value(row.get("Radiation Therapy"))
    if rt_yn and rt_yn != "0":
        parts.append("radiotherapy")
    return " ".join(parts) if parts else None


def _query_salvage_tx(row: pd.Series) -> str | None:
    """Salvage therapy — same idea, keep named regimens, skip day numbers."""
    parts: list[str] = []
    add = _format_value(row.get("Additional Therapy"))
    if add and add.lower() not in _NULL_MARKERS and add != "0":
        parts.append(f"salvage chemotherapy {add.lower()}")
    immuno = _format_value(row.get("Immuno therapy"))
    if immuno and immuno.lower() not in _NULL_MARKERS and immuno != "0":
        parts.append(f"immunotherapy {immuno.lower()}")
    brachy = _format_value(row.get("Brachy therapy"))
    if brachy and brachy.lower() not in _NULL_MARKERS and brachy != "0":
        parts.append("brachytherapy")
    other = _format_value(row.get("Other Types of Therapy (LITT, more chemo, proton therapy)"))
    if other and other.lower() not in _NULL_MARKERS:
        parts.append(other.lower())
    if parts:
        parts.insert(0, "recurrent glioblastoma after first progression")
    return " ".join(parts) if parts else None


def _query_timepoints(row: pd.Series) -> str | None:
    """Skip MRI day numbers — they don't help PubMed retrieval."""
    return None


def _query_radiomic(row: pd.Series) -> str | None:
    """Radiomic features are numeric (volume, sphericity). They won't match
    PubMed text. Reduce to a coarse 'tumor volume' keyword if present."""
    try:
        df = _radiomic_table()
    except FileNotFoundError:
        return None
    sub = df[df[ID_COL] == row.get(ID_COL)]
    return "tumor volume morphology radiomic" if not sub.empty else None


def _query_vlm(row: pd.Series) -> str | None:
    """VLM captions ARE patient-specific natural language. Use them
    directly — they are the most discriminative retrieval cue.

    Captions at or after Landmark_day are excluded by
    ``_filter_captions_pre_landmark`` so the retriever only ever sees
    pre-prediction information.
    """
    try:
        df = _captions_table()
    except FileNotFoundError:
        return None
    sub = df[df[ID_COL] == row.get(ID_COL)].sort_values("Timepoint")
    sub = _filter_captions_pre_landmark(sub, row)
    if sub.empty:
        return None
    caps = [str(c) for c in sub["caption"].tolist() if not _is_missing(c)]
    return " ".join(caps) if caps else None


_QUERY_FN = {
    "Demographics": _query_demographics,
    "Diagnosis":    _query_diagnosis,
    "Molecular":    _query_molecular,
    "InitialTx":    _query_initial_tx,
    "SalvageTx":    _query_salvage_tx,
    "Timepoints":   _query_timepoints,
    "Radiomic":     _query_radiomic,
    "VLM":          _query_vlm,
}


def render_retrieval_query(row: pd.Series, experiment: str) -> str:
    """Build a clean keyword-style query for BM25 + dense retrieval.

    Distinct from `render_patient` (which produces the human-readable
    narrative the LLM sees). The retrieval query contains only patient-
    distinctive clinical findings — no structured labels, no 'unknown'
    placeholders, no constant cohort-level terms.

    Always anchored with 'glioblastoma' so retrieval stays on-topic
    even when most molecular markers are unknown.
    """
    if experiment not in EXPERIMENT_BLOCKS:
        raise ValueError(f"Unknown experiment {experiment!r}")
    parts = ["glioblastoma"]
    seen = set(parts)
    for name in EXPERIMENT_BLOCKS[experiment]:
        fn = _QUERY_FN.get(name)
        if fn is None:
            continue
        text = fn(row)
        if not text:
            continue
        for tok in text.split():
            t = tok.strip(",.;:").lower()
            if not t or t in seen or t in _CONSTANT_COHORT_TERMS:
                continue
            seen.add(t)
            parts.append(t)
    return " ".join(parts)

# Cached lookups so we hit disk once per process, not once per patient.
_RADIOMIC_CACHE: pd.DataFrame | None = None
_CAPTIONS_CACHE: pd.DataFrame | None = None


def _radiomic_table() -> pd.DataFrame:
    global _RADIOMIC_CACHE
    if _RADIOMIC_CACHE is None:
        path = PROCESSED_D / "radiomic_features.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        _RADIOMIC_CACHE = pd.read_csv(path)
    return _RADIOMIC_CACHE


def _captions_table() -> pd.DataFrame:
    """Return the MRI captions table for the prompt-version requested by the
    caller.

    Selection precedence:
      1. RAG_CAPTIONS_VERSION env var ("v1", "v2_context", "v3_structured", ...)
      2. Default = "v1" (preserves backward-compatible behaviour for old runs).

    File-name convention matches `VLM/run_radfm_captions.py`:
      v1            → Dataset/Processed/mri_captions.csv
      v2_context    → Dataset/Processed/mri_captions_v2_context.csv
      v3_structured → Dataset/Processed/mri_captions_v3_structured.csv
    """
    global _CAPTIONS_CACHE
    if _CAPTIONS_CACHE is None:
        version = os.environ.get("RAG_CAPTIONS_VERSION", "v1").strip() or "v1"
        suffix  = "" if version == "v1" else f"_{version}"
        path    = PROCESSED_D / f"mri_captions{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} (RAG_CAPTIONS_VERSION={version!r}). "
                f"Run `python VLM/run_radfm_captions.py --prompt-version {version}` first."
            )
        _CAPTIONS_CACHE = pd.read_csv(path)
        print(f"[feature_render] captions: loaded {len(_CAPTIONS_CACHE)} rows from {path.name}")
    return _CAPTIONS_CACHE


# --------------------------------------------------------------------------- #
# Public API (signature matches First_Recur so build_dataset.py needs no edit)
# --------------------------------------------------------------------------- #
def render_patient(row: pd.Series, experiment: str) -> str:
    if experiment not in EXPERIMENT_BLOCKS:
        raise ValueError(f"Unknown experiment {experiment!r}. "
                         f"Pick from {list(EXPERIMENT_BLOCKS)}")
    blocks = []
    for name in EXPERIMENT_BLOCKS[experiment]:
        text = _BLOCK_FN[name](row)
        if text:
            blocks.append(text)
    return "\n".join(blocks)


def render_split(experiment: str, split: str = "Train") -> pd.DataFrame:
    csv_path = SPLITS_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        ID_COL: df[ID_COL],
        "text":           [render_patient(r, experiment)            for _, r in df.iterrows()],
        "retrieval_query": [render_retrieval_query(r, experiment)   for _, r in df.iterrows()],
        "label":          df[LABEL_COL].astype(int),
    })
    return out


def list_experiments() -> Iterable[str]:
    return list(EXPERIMENT_BLOCKS.keys())


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",   default="Exp1", choices=list(EXPERIMENT_BLOCKS))
    ap.add_argument("--split", default="Train", choices=["Train","Validation","Test"])
    ap.add_argument("--n",     type=int, default=2)
    args = ap.parse_args()

    df = render_split(args.exp, args.split)
    print(f"# {args.split} / {args.exp}: {len(df)} patients "
          f"(positive rate = {df['label'].mean():.1%})")
    for _, row in df.head(args.n).iterrows():
        print("=" * 60)
        print(f"{row[ID_COL]}  (label = {row['label']})")
        print(row["text"])
    sys.exit(0)
