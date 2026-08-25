"""
Map the manually-collected PPUM cohort (Collect Data (4).xlsx) onto the exact
MU-Glioma-Post schema (Dataset/splits/*.csv) so it can be scored by ALL models
(clinical XGBoost/LR, LLM no-RAG, LLM+RAG) as an EXTERNAL VALIDATION set.

Default output: PPUM/generated/PPUM.csv
(same columns as MU Train.csv, Patient_ID = PPUM_<StudyID>)

Mapping decisions (documented for the write-up / limitations):
  * Molecular text -> MU numeric codes (inverse of Model/configs/molecular_codes.json).
    Missing marker -> MU 'unknown' code where one exists (MU itself codes not-tested as
    unknown, e.g. IDH=2); markers with no unknown code (PTEN/CDKN2A-B/TP53) left blank.
  * Treatment dates -> "Number of days from Diagnosis to X" (MU style). Diagnosis anchor is
    year-only for most PPUM rows -> anchored to 1 Jan of that year (best-effort; a documented
    source of noise for this external set). Free-text dates ("6 weeks") -> missing.
  * Dose -> "<n> Gy" string to match MU's string dtype. Fractions -> numeric.
  * Primary Diagnosis normalised to MU vocabulary where unambiguous (glioblastoma->GBM etc.).
  * Previous Brain Tumor blank -> 'No'.
  * Label y: First Recurrence / Progression  Yes->1, No->0.
"""
from __future__ import annotations
import argparse, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = Path(__file__).resolve().parent / "input" / "Collect Data (4).xlsx"
MU_TRAIN = SUBMISSION_ROOT / "dataset" / "first" / "splits" / "Train.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "PPUM.csv"

# ---- molecular: PPUM text -> MU integer code ----
MOL_MAP = {
    "IDH1 mutation":      ({"mutant": 1, "wildtype": 0}, 2),
    "IDH2 mutation":      ({"mutant": 1, "wildtype": 0}, 2),
    "1p/19q":             ({"codeleted": 1, "not codeleted": 0}, 10),
    "ATRX mutation":      ({"mutant": 1, "lost": 1, "wildtype": 0, "retained": 0}, 4),
    "MGMT methylation":   ({"methylated": 1, "unmethylated": 0}, 4),
    "BRAF V600E mutation":({"mutant": 1, "wildtype": 0}, 2),
    "TERT promoter mutation": ({"mutant": 1, "wildtype": 0}, 2),
    "Chromosome 7 gain and Chromosome 10 loss": ({"present": 1, "absent": 0}, 2),
    "H3-3A mutation":     ({"mutant": 1, "wildtype": 0}, 2),
    "EGFR amplification": ({"amplified": 1, "not amplified": 0}, 2),
    "PTEN mutation":      ({"mutant": 1, "wildtype": 0}, None),      # no unknown code
    "CDKN2A/B deletion":  ({"deleted": 1, "not deleted": 0}, None),
    "TP53 alteration":    ({"altered": 1, "not altered": 0}, None),
}
PPUM_MOL_COL = {  # MU col -> PPUM col
    "1p/19q": "1p/19q codeletion",
    "Chromosome 7 gain and Chromosome 10 loss": "Chromosome 7 gain & Chromosome 10 loss",
}

def parse_date(v):
    if pd.isna(v):
        return pd.NaT
    if isinstance(v, pd.Timestamp):
        return v
    s = str(v).strip()
    if re.fullmatch(r"\d{4}", s):                    # year only -> 1 Jan
        return pd.Timestamp(int(s), 1, 1)
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(s, dayfirst=dayfirst, errors="raise")
        except Exception:
            continue
    return pd.NaT                                    # "6 weeks", "every 28 days", etc.

def days_between(diag, event):
    d0, d1 = parse_date(diag), parse_date(event)
    if pd.isna(d0) or pd.isna(d1):
        return np.nan
    return int((d1 - d0).days)

def norm_diag(s):
    if pd.isna(s): return np.nan
    t = str(s).strip().lower()
    if t in ("-", ""): return np.nan
    if "glioblastoma" in t or t == "gbm": return "GBM"
    if "oligodendro" in t: return "Oligodendro-glioma"
    if "pilocytic" in t: return "Pilocytic astrocytoma"
    if "astrocytoma" in t: return "Astrocytoma"
    return str(s).strip()

def yn(v):
    if pd.isna(v): return np.nan
    return "Yes" if str(v).strip().lower().startswith("y") else ("No" if str(v).strip().lower().startswith("n") else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX,
                        help="Source PPUM workbook.")
    parser.add_argument("--mu-train", type=Path, default=MU_TRAIN,
                        help="First-recurrence Train.csv used as the target schema.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Destination PPUM CSV.")
    args = parser.parse_args()

    mu_cols = list(pd.read_csv(args.mu_train).columns)
    df = pd.read_excel(args.xlsx, sheet_name="PPUM_First_Recurrence", header=1).iloc[1:].reset_index(drop=True)
    df = df[df["Study ID"].notna()].reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        diag = r["Date of Diagnosis"]
        out = {c: np.nan for c in mu_cols}
        out["Patient_ID"] = f"PPUM_{str(r['Study ID']).strip()}"
        out["Sex at Birth"] = r["Sex at Birth"]
        out["Race"] = r["Race"]
        out["Age at diagnosis"] = r["Age at diagnosis (years)"]
        out["Primary Diagnosis"] = norm_diag(r["Primary Diagnosis"])
        g = r["Grade of Primary Brain Tumor"]
        out["Grade of Primary Brain Tumor"] = (np.nan if pd.isna(g) or str(g).strip().lower() in ("unknown","nan","-")
                                               else str(g).strip().replace(".0",""))
        out["Stereotactic Biopsy before Surgical Resection"] = yn(r["Stereotactic Biopsy before Surgical Resection"])
        out["Previous Brain Tumor"] = yn(r["Previous Brain Tumor"]) or "No"
        out["Type of previous brain tumor"] = r.get("Type of previous brain tumor")
        out["Year of previous surgery"] = r.get("Year of previous surgery")
        gp = r.get("Grade of Previous brain tumor")
        out["Grade of Previous brain tumor"] = (f"Grade {str(gp).strip().replace('.0','')}" if pd.notna(gp) else np.nan)

        # molecular
        for mu_col, (mapping, unk) in MOL_MAP.items():
            ppum_col = PPUM_MOL_COL.get(mu_col, mu_col)
            raw = r.get(ppum_col)
            if pd.isna(raw):
                if unk is not None:
                    out[mu_col] = unk
            else:
                out[mu_col] = mapping.get(str(raw).strip().lower(), unk if unk is not None else np.nan)
        out["Other mutations/alterations"] = np.nan

        # treatment  (negative day-offsets are impossible -> NaN; year-only diagnosis
        # anchor makes cross-year offsets noisy, documented as an external-set limitation)
        def days_pos(ev):
            d = days_between(diag, ev)
            return np.nan if (pd.isna(d) or d < 0) else d
        out["Number of days from Diagnosis to First surgery or procedure "] = days_pos(r["Date of First Surgery or Procedure"])
        out["Initial Chemo Therapy"] = yn(r["Initial Chemo Therapy"])
        out["Name of Initial Chemo Therapy"] = r["Name of Initial Chemo Therapy"] if pd.notna(r["Name of Initial Chemo Therapy"]) else np.nan
        out[" Number of days from Diagnosis to Initial Chemo Therapy Start date"] = days_pos(r["Date Initial Chemo Start"])
        out[" Number of days from Diagnosis to Initial Chemo Therapy end date"] = days_pos(r["Date Initial Chemo End"])
        out["Radiation Therapy"] = yn(r["Radiation Therapy"])
        out["Number of days from Diagnosis to Radiation Therapy Start date"] = days_pos(r["Date Radiation Therapy Start"])
        out["Number of days from Diagnosis to Radiation Therapy end date"] = days_pos(r["Date Radiation Therapy End"])
        dose = r["Dose (Gy)"]
        out["Dose"] = (f"{int(float(dose))} Gy" if pd.notna(dose) and str(dose).strip() != "" else np.nan)
        nf = r["Number of Fractions"]
        out["Number of Fractions"] = float(nf) if pd.notna(nf) and str(nf).strip() != "" else np.nan

        # label
        out["y"] = 1 if str(r["First Recurrence / Progression"]).strip().lower().startswith("y") else 0
        rows.append(out)

    ppum = pd.DataFrame(rows)[mu_cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ppum.to_csv(args.out, index=False)

    print(f"PPUM external set -> {args.out}")
    print(f"n={len(ppum)}  pos(recur)={int(ppum['y'].sum())}  neg={int((ppum['y']==0).sum())}")
    print("\nSanity — first 5 rows (key cols):")
    show = ["Patient_ID","Sex at Birth","Age at diagnosis","Primary Diagnosis",
            "Grade of Primary Brain Tumor","IDH1 mutation","MGMT methylation",
            "Number of days from Diagnosis to First surgery or procedure ",
            "Radiation Therapy","Dose","Number of Fractions","y"]
    print(ppum[show].head(5).to_string(index=False))
    print("\nMissing per treatment-days col (of 31):")
    for c in [c for c in mu_cols if "Number of days" in c]:
        print(f"  {c.strip()}: {int(ppum[c].isna().sum())} missing")


if __name__ == "__main__":
    main()
