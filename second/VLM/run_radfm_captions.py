#!/usr/bin/env python
"""Caption every (patient × pre-landmark) scan with a medical VLM.

Tries the medical-VLM fallback ladder in order; the first model that
loads + runs successfully wins.  The model actually used is recorded
in `Dataset/Processed/mri_captions.meta.json` so the proposal write-up
matches reality.

Usage:
    python run_radfm_captions.py [--max N]   # caption first N scans (default = all)
    python run_radfm_captions.py --prompt-version v2_context   # anatomy-grounded
                                                                # patient-context prompt

Output CSV path is suffixed by --prompt-version so v1 and v2 captions can
coexist on disk:
    v1          → Dataset/Processed/mri_captions.csv
    v2_context  → Dataset/Processed/mri_captions_v2_context.csv
"""
from __future__ import annotations
import argparse, json, sys, time, importlib
from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT.parent / "dataset" / "second"
DATA_CSV  = DATASET_ROOT / "Processed" / "mri_paths.csv"
CLINICAL_CSV = DATASET_ROOT / "Processed" / "clean_clinical.csv"

def _output_paths(prompt_version: str) -> tuple[Path, Path]:
    """Different prompt versions write to different files so we never
    silently overwrite the v1 cache."""
    suffix = "" if prompt_version == "v1" else f"_{prompt_version}"
    out_csv  = DATASET_ROOT / "Processed" / f"mri_captions{suffix}.csv"
    meta_json = DATASET_ROOT / "Processed" / f"mri_captions{suffix}.meta.json"
    return out_csv, meta_json

LADDER = [
    ("RadFM",       "vlm_backends.radfm_backend",       "caption_volumes"),
    ("LLaVA-Med",   "vlm_backends.llava_med_backend",   "caption_volumes"),
    ("MAIRA-2",     "vlm_backends.maira2_backend",      "caption_volumes"),
    ("Med-Flamingo","vlm_backends.medflamingo_backend", "caption_volumes"),
]


def try_load(model_name, modpath, fnname):
    try:
        mod = importlib.import_module(modpath)
        fn  = getattr(mod, fnname)
        # Probe-load the model (this triggers downloads / weight init):
        _ = mod.load_model()
        return fn
    except Exception as e:
        print(f"  [{model_name}] unavailable: {type(e).__name__}: {e}")
        return None


def _load_existing(out_csv: Path) -> tuple[pd.DataFrame, set[tuple]]:
    """Load existing captions (for resume).  Migrate the legacy `T2` column
    name to `Landmark_day` if encountered."""
    if not out_csv.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(out_csv)
    if "T2" in df.columns and "Landmark_day" not in df.columns:
        df = df.rename(columns={"T2": "Landmark_day"})
    done = {(r["Patient_ID"], int(r["Timepoint"])) for _, r in df.iterrows()}
    return df, done


# --------------------------------------------------------------------------- #
# Per-scan clinical-context builder (used by --prompt-version v2_context).
# Mirrors the BEEP-style anatomy-grounded prompting approach: feed the VLM
# the patient's already-known clinical state so it doesn't hallucinate the
# diagnosis (e.g. v1 captions sometimes labelled glioma cohort scans as
# "meningioma" / "extra-axial mass", which is medically wrong here).
# --------------------------------------------------------------------------- #
def _build_clinical_context(scan_row: pd.Series, clin_df: pd.DataFrame) -> str:
    """Return a short, structured patient-context block for one scan.

    The block contains:
      - Demographics (age, sex)
      - Confirmed diagnosis (locked: GBM grade 4)
      - Days since diagnosis (so RadFM knows this is follow-up, not pre-op)
      - Initial therapy (chemo agent, radiation Y/N) so the model expects
        post-treatment changes (resection cavity, radiation effect, etc.)
      - Salvage therapy if any was given before this scan day
    """
    pid = scan_row["Patient_ID"]
    sub = clin_df[clin_df["Patient_ID"] == pid]
    if sub.empty:
        return ""
    p = sub.iloc[0]
    day = int(scan_row["Day_from_diag"])

    parts: list[str] = []

    # demographics
    sex = str(p.get("Sex at Birth", "")).strip()
    age = p.get("Age at diagnosis")
    if pd.notna(age):
        try:
            parts.append(f"Age {int(round(float(age)))} {sex.lower()}")
        except (TypeError, ValueError):
            pass

    # locked diagnosis (this is the anchor that prevents 'meningioma' hallucinations)
    parts.append("Confirmed glioblastoma (WHO grade 4)")

    # known molecular markers (only positives, skip 'unknown')
    mol_bits = []
    idh1 = p.get("IDH1 mutation")
    if pd.notna(idh1):
        mol_bits.append("IDH-wildtype" if str(idh1).lower() in ("0","wildtype","wt") else "IDH-mutant")
    mgmt = p.get("MGMT methylation")
    if pd.notna(mgmt) and str(mgmt).lower() not in ("unknown","nan",""):
        try:
            v = int(float(mgmt))
            if v == 1:   mol_bits.append("MGMT methylated")
            elif v == 0: mol_bits.append("MGMT unmethylated")
        except (TypeError, ValueError):
            pass
    if mol_bits:
        parts.append("; ".join(mol_bits))

    # initial therapy summary
    chemo = p.get("Initial Chemo Therapy")
    rt    = p.get("Radiation Therapy")
    init = []
    if pd.notna(chemo) and str(chemo).strip() not in ("0","nan",""):
        agent = p.get("Name of Initial Chemo Therapy")
        if pd.notna(agent) and str(agent).strip():
            init.append(f"chemotherapy ({str(agent).strip()})")
        else:
            init.append("chemotherapy")
    if pd.notna(rt) and str(rt).strip() not in ("0","nan",""):
        init.append("radiotherapy")
    if init:
        parts.append("Received " + " + ".join(init))

    # this scan's day from diagnosis (so model knows it's follow-up not pre-op)
    parts.append(f"Imaging day: {day} days after diagnosis (post-treatment follow-up)")

    # salvage therapy that started BEFORE this scan day
    add = p.get("Additional Therapy")
    add_start = p.get("Number of Days from Diagnosis to Starting Additional Therapy ")
    if (pd.notna(add) and str(add).strip() not in ("0","nan","")
            and pd.notna(add_start) and float(add_start) < day):
        parts.append(f"Salvage chemotherapy ({str(add).strip()}) started day {int(float(add_start))}")

    immuno = p.get("Immuno therapy")
    immuno_start = p.get("Number of Days from Diagnosis to Start Immunotherapy ")
    if (pd.notna(immuno) and str(immuno).strip() not in ("0","nan","")
            and pd.notna(immuno_start) and float(immuno_start) < day):
        parts.append(f"Immunotherapy ({str(immuno).strip()}) started day {int(float(immuno_start))}")

    return ". ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="0 = all rows")
    ap.add_argument("--flush-every", type=int, default=10,
                    help="checkpoint to disk every N scans (default 10)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore existing captions and recaption from scratch")
    ap.add_argument("--prompt-version", default="v1",
                    choices=["v1", "v2_context", "v3_structured"],
                    help="v1 = original (no patient context); "
                         "v2_context = anatomy-grounded with patient demographics, "
                         "diagnosis lock, and treatment history (recommended for "
                         "post-treatment glioma follow-up scans); "
                         "v3_structured = same patient anchor as v2 but a fixed "
                         "7-item YES/NO/UNCLEAR checklist (enhancement, necrosis, "
                         "hemorrhage, edema, mass effect, multifocal, size vs "
                         "baseline). Output is auto-parsed into a deterministic "
                         "structured block — bypasses RadFM's weak free-text "
                         "concepts (lobe accuracy 23 %, necrosis recall 12 %).")
    args = ap.parse_args()

    out_csv, meta_json = _output_paths(args.prompt_version)

    df = pd.read_csv(DATA_CSV)
    if "Landmark_day" not in df.columns and "T2" in df.columns:
        df = df.rename(columns={"T2": "Landmark_day"})

    # Load clinical CSV only if we need it (avoid surprising failures for v1).
    clin_df = pd.DataFrame()
    if args.prompt_version in ("v2_context", "v3_structured"):
        if not CLINICAL_CSV.exists():
            sys.exit(f"ERROR: --prompt-version {args.prompt_version} requires "
                     f"{CLINICAL_CSV} (produced by Dataset/Clinical_preprocessing.ipynb).")
        clin_df = pd.read_csv(CLINICAL_CSV)
        print(f"clinical context loaded: {len(clin_df)} patients from {CLINICAL_CSV.name}")

    # Resume: drop rows whose (Patient_ID, Timepoint) already has a caption
    existing, done = (pd.DataFrame(), set()) if args.restart else _load_existing(out_csv)
    if done:
        before = len(df)
        df = df[~df.apply(lambda r: (r["Patient_ID"], int(r["Timepoint"])) in done, axis=1)]
        print(f"resume: {len(done)} captions already on disk → {len(df)}/{before} remaining")

    if args.max > 0:
        df = df.head(args.max)
    print(f"caption {len(df)} new scans → {out_csv}  (prompt_version={args.prompt_version})")

    if len(df) == 0:
        print("nothing to do."); return

    caption_fn, model_used = None, None
    for name, modpath, fnname in LADDER:
        print(f"trying {name} ...")
        fn = try_load(name, modpath, fnname)
        if fn is not None:
            caption_fn, model_used = fn, name
            print(f"  ✓ using {name}")
            break
    if caption_fn is None:
        sys.exit("ERROR: no medical VLM in the fallback ladder loaded successfully. "
                 "Check pip installs and HF model availability.")

    t0 = time.time(); new_rows = []
    n_total = len(df)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        scan_t0 = time.time()
        err = None
        ctx = (_build_clinical_context(r, clin_df)
               if args.prompt_version in ("v2_context", "v3_structured") else "")
        try:
            cap = caption_fn(t1c=r["brain_t1c"], t1n=r["brain_t1n"],
                             t2f=r["brain_t2f"], t2w=r["brain_t2w"],
                             clinical_context=ctx,
                             prompt_version=args.prompt_version)
        except TypeError:
            # Backward compat: backend without v2 kwargs (LLaVA-Med, etc.)
            try:
                cap = caption_fn(t1c=r["brain_t1c"], t1n=r["brain_t1n"],
                                 t2f=r["brain_t2f"], t2w=r["brain_t2w"])
            except Exception as e:
                cap = ""
                err = f"{type(e).__name__}: {e}"
        except Exception as e:
            cap = ""
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - scan_t0
        new_rows.append({"Patient_ID":      r["Patient_ID"],
                         "Timepoint":       int(r["Timepoint"]),
                         "Day_from_diag":   float(r["Day_from_diag"]),
                         "Landmark_day":    float(r["Landmark_day"]),
                         "y":               int(r["y"]),
                         "model":           model_used,
                         "prompt_version":  args.prompt_version,
                         "elapsed_sec":     round(elapsed, 2),
                         "caption":         cap,
                         "error":           err})

        # Periodic flush so a crash doesn't lose work
        if i % args.flush_every == 0 or i == n_total:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_csv(out_csv, index=False)
            mean_per = (time.time() - t0) / i
            eta = (n_total - i) * mean_per / 60
            print(f"  [{i}/{n_total}]  avg {mean_per:5.1f}s/scan  "
                  f"ETA {eta:5.1f} min  flushed → {out_csv.name}", flush=True)

    meta_json.write_text(json.dumps({
        "model_used":          model_used,
        "prompt_version":      args.prompt_version,
        "fallback_ladder":     [n for n, _, _ in LADDER],
        "n_scans_captioned":   len(existing) + len(new_rows),
        "n_eligible_patients": int(pd.concat([existing, pd.DataFrame(new_rows)],
                                              ignore_index=True)["Patient_ID"].nunique()),
        "elapsed_minutes":     round((time.time() - t0) / 60, 2),
    }, indent=2))
    print(f"\nDONE — wrote {out_csv}\n        and {meta_json}")


if __name__ == "__main__":
    main()
