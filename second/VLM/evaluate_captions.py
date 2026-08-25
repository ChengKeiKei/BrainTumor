"""
evaluate_captions.py — Multi-axis evaluation of RadFM-generated MRI captions.

We have NO gold radiology reports for MU-Glioma-Post, so reference-based
NLG metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) are inapplicable.
Instead this script reports four families that are appropriate for
report-free evaluation of clinical VLM output:

  1. STRUCTURAL / LEXICAL
     - length distribution
     - lexical diversity (type-token ratio, distinct-2)
     - per-patient caption diversity (timepoint-vs-timepoint similarity)
     - top template phrases (repetition)
     - empty / very-short rate

  2. CLINICAL-CONCEPT COVERAGE
     - presence rate of standard glioma report terms (anatomy, signal,
       enhancement pattern, mass effect, edema, necrosis, hemorrhage, ...)
     - per-caption concept count

  3. SEGMENTATION-GROUNDED FACTUAL ACCURACY
     For each caption we have the matching tumorMask NIfTI; we compute:
        * tumor lobe (frontal / parietal / temporal / occipital / cerebellar)
          via centroid in voxel space, then check whether the caption
          mentions the same lobe.
        * tumor size bucket (small/medium/large) via voxel volume,
          then check whether the caption mentions a compatible size term.
        * necrosis presence (BraTS label 1 voxels > threshold) vs
          caption mention of "necrosis" / "necrotic" / "central necrosis".
        * enhancement presence (BraTS label 3 voxels > threshold) vs
          caption mention of "enhanc"/"contrast"/"ring".

  4. TEMPORAL-LEAKAGE PROBE
     Captions are generated from MRI taken BEFORE the prediction landmark.
     If a caption mentions a future-outcome term (progression, recurrence,
     died, death, expired) NOT in a negation, that's a temporal leak.

Outputs (under Second_Recur/VLM/eval_radfm/):
    summary.json               - all aggregate numbers
    per_caption.csv            - per-caption flags (so we can audit any row)
    template_phrases.txt       - most common 6-grams
    plots/length_hist.png      - char length distribution
    plots/concept_coverage.png - bar chart of concept presence rates
    plots/lobe_confusion.png   - confusion matrix (segmentation lobe vs caption lobe)
    plots/size_confusion.png   - confusion matrix (segmentation bucket vs caption mention)
    REPORT.md                  - human-readable summary table
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parent / "dataset" / "second"
DATA_CSV = DATASET_ROOT / "Processed" / "mri_paths.csv"
CAP_CSV  = DATASET_ROOT / "Processed" / "mri_captions.csv"
OUT_DIR  = ROOT / "VLM" / "eval_radfm"
PLOT_DIR = OUT_DIR / "plots"

# ---------------------------------------------------------------------------
# concept lexicons (lower-cased word stems)
# ---------------------------------------------------------------------------
LOBE_TERMS = {
    "frontal":   ["frontal"],
    "parietal":  ["parietal"],
    "temporal":  ["temporal"],
    "occipital": ["occipital"],
    "cerebellar":["cerebell", "infratentorial", "posterior fossa"],
    "thalamic":  ["thalam", "basal ganglia", "deep gray"],
    "brainstem": ["brainstem", "pons", "midbrain", "medulla"],
}

SIZE_TERMS = {
    "small":  ["small", "tiny", "minute", "<1 cm", "< 1 cm"],
    "medium": ["medium", "moderate", "intermediate"],
    "large":  ["large", "bulky", "extensive", "massive", "huge"],
}

CLINICAL_CONCEPTS = {
    "enhancement":     ["enhanc", "contrast", "ring"],
    "edema":           ["edema", "oedema", "vasogenic", "peritumoral"],
    "necrosis":        ["necro"],
    "hemorrhage":      ["hemorrhag", "haemorrhag", "bleed", "blood prod"],
    "mass_effect":     ["mass effect", "midline shift", "herniation",
                        "compression", "displac"],
    "cyst":            ["cyst"],
    "T1_signal":       ["t1", "hypointens", "isointens"],
    "T2_signal":       ["t2", "hyperintens", "flair"],
    "shape_size":      ["measur", "diameter", " mm", " cm", "centimeter",
                        "millimeter", "size"],
    "infiltration":    ["infiltrat", "diffus", "extends", "extension",
                        "involve", "crosses"],
    "ventricle":       ["ventric"],
    "midline":         ["midline", "corpus callosum", "falx", "shift"],
}

LEAK_TERMS = ["progression", "progress", "recurr", "died", "death",
              "expired", "deceased"]
NEG_PREFIX = re.compile(
    r"\b(no|not|without|absent|negative|free of|denies|deny)\b[^\.]{0,30}",
    re.I,
)

SIZE_VOXEL_THRESHOLDS = (5_000, 30_000)  # tumor voxel-count buckets

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-]+", s.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _contains_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def _is_leak(caption: str) -> bool:
    """True iff caption contains a future-outcome term NOT inside a negation."""
    text = caption.lower()
    for term in LEAK_TERMS:
        for m in re.finditer(rf"\b{term}\w*", text):
            window_start = max(0, m.start() - 40)
            ctx = text[window_start:m.start()]
            if NEG_PREFIX.search(ctx):
                continue
            return True
    return False


def _lobe_from_centroid(mask_arr: np.ndarray, affine: np.ndarray) -> str:
    """
    Lobe assignment via centroid in patient (RAS) world coordinates,
    normalized against the SRI24 brain bounding box.
    MU-Glioma-Post is SRI24-registered (240x240x155) and skull-stripped,
    so the image volume IS the brain bounding box. We map the eight image
    corners to RAS world coords, take their bbox, then put the tumor
    centroid in that frame.
    """
    import nibabel as nib

    nonzero_vox = np.argwhere(mask_arr > 0)
    if len(nonzero_vox) == 0:
        return "unknown"

    Sx, Sy, Sz = mask_arr.shape
    corners_vox = np.array([[0, 0, 0], [Sx-1, 0, 0], [0, Sy-1, 0],
                            [0, 0, Sz-1], [Sx-1, Sy-1, 0],
                            [Sx-1, 0, Sz-1], [0, Sy-1, Sz-1],
                            [Sx-1, Sy-1, Sz-1]])
    def vox_to_ras(vox: np.ndarray) -> np.ndarray:
        ones = np.ones((vox.shape[0], 1))
        world = (affine @ np.hstack([vox, ones]).T).T[:, :3]
        axc = nib.aff2axcodes(affine)
        sign = {"R":  1, "L": -1, "A":  1, "P": -1, "S":  1, "I": -1}
        return world * np.array([[sign[axc[0]], sign[axc[1]], sign[axc[2]]]])

    brain_ras  = vox_to_ras(corners_vox)
    brain_lo   = brain_ras.min(axis=0)
    brain_hi   = brain_ras.max(axis=0)
    centroid   = vox_to_ras(nonzero_vox.mean(axis=0, keepdims=True))[0]
    rel        = (centroid - brain_lo) / np.maximum(brain_hi - brain_lo, 1)
    rx, ry, rz = rel  # 0 = L/P/I edge, 1 = R/A/S edge

    # cerebellum / brainstem: inferior 30% of brain
    if rz < 0.30:
        return "cerebellar" if ry < 0.45 else "brainstem"
    # frontal-occipital axis (anterior-posterior)
    if ry > 0.62:
        return "frontal"
    if ry < 0.30:
        return "occipital"
    # middle third: parietal (upper) vs temporal (lower)
    return "parietal" if rz > 0.60 else "temporal"


def _caption_lobe(caption: str) -> str | None:
    """Return the first lobe explicitly mentioned in the caption (or None)."""
    text = caption.lower()
    for lobe, terms in LOBE_TERMS.items():
        if any(t in text for t in terms):
            return lobe
    return None


def _size_bucket_from_voxels(n: int) -> str:
    s, m = SIZE_VOXEL_THRESHOLDS
    if n < s:
        return "small"
    if n < m:
        return "medium"
    return "large"


def _caption_size(caption: str) -> str | None:
    for bucket, terms in SIZE_TERMS.items():
        if any(t in caption.lower() for t in terms):
            return bucket
    return None


def _seg_stats(path: str) -> dict:
    """Compute lobe / size bucket / necrosis flag / enhancement flag."""
    import nibabel as nib
    try:
        img = nib.load(path)
        arr = img.get_fdata().astype(np.int16)
    except Exception as e:
        return {"error": repr(e)}
    tumor = arr > 0
    n_tumor = int(tumor.sum())
    n_necr  = int((arr == 1).sum())   # BraTS NCR
    n_enh   = int((arr == 3).sum())   # BraTS ET
    return {
        "lobe_seg":      _lobe_from_centroid(tumor, img.affine),
        "size_seg":      _size_bucket_from_voxels(n_tumor),
        "n_voxels":      n_tumor,
        "n_necr_voxels": n_necr,
        "n_enh_voxels":  n_enh,
        "has_necr":      n_necr > 50,
        "has_enh":       n_enh  > 50,
    }


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------
def evaluate(limit: int | None = None, no_seg: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cap = pd.read_csv(CAP_CSV)
    paths = pd.read_csv(DATA_CSV)
    df = cap.merge(paths[["Patient_ID", "Timepoint", "tumorMask"]],
                   on=["Patient_ID", "Timepoint"], how="left")
    if limit:
        df = df.head(limit)

    # 1) STRUCTURAL ----------------------------------------------------------
    df["len_chars"]   = df["caption"].str.len()
    df["len_tokens"]  = df["caption"].fillna("").apply(lambda s: len(_tokens(s)))
    df["distinct_2"]  = df["caption"].fillna("").apply(
        lambda s: (len(set(_ngrams(_tokens(s), 2))) /
                   max(1, len(_ngrams(_tokens(s), 2)))))

    all_tokens   = [t for s in df["caption"].fillna("") for t in _tokens(s)]
    type_token   = len(set(all_tokens)) / max(1, len(all_tokens))
    six_grams    = Counter(g for s in df["caption"].fillna("")
                            for g in _ngrams(_tokens(s), 6))
    top_phrases  = six_grams.most_common(20)

    short_rate = float((df["len_tokens"] < 30).mean())
    empty_rate = float((df["len_tokens"] == 0).mean())

    # per-patient diversity: distinct-2 across all that patient's captions
    per_patient_div = (
        df.groupby("Patient_ID")["caption"]
          .apply(lambda s: len(set(_ngrams(_tokens(" ".join(s.dropna())), 2))) /
                          max(1, len(_ngrams(_tokens(" ".join(s.dropna())), 2))))
          .mean()
    )

    # 2) CONCEPT COVERAGE ----------------------------------------------------
    concept_hits = {c: 0 for c in CLINICAL_CONCEPTS}
    per_cap_concepts = []
    for s in df["caption"].fillna(""):
        n_present = 0
        for concept, terms in CLINICAL_CONCEPTS.items():
            if _contains_any(s, terms):
                concept_hits[concept] += 1
                n_present += 1
        per_cap_concepts.append(n_present)
    df["n_concepts"] = per_cap_concepts
    coverage = {c: round(h / len(df), 3) for c, h in concept_hits.items()}

    # 3) SEGMENTATION-GROUNDED ----------------------------------------------
    if not no_seg:
        print(f"[eval] computing segmentation stats for {len(df)} scans...")
        seg_rows = []
        for i, r in df.iterrows():
            if i % 25 == 0:
                print(f"  [{i}/{len(df)}] segmentation analysis ...")
            stats = _seg_stats(r["tumorMask"]) if r["tumorMask"] else {}
            seg_rows.append(stats)
        seg = pd.DataFrame(seg_rows, index=df.index)
        df = pd.concat([df, seg], axis=1)
        df["lobe_cap"]  = df["caption"].fillna("").apply(_caption_lobe)
        df["size_cap"]  = df["caption"].fillna("").apply(_caption_size)
        df["mention_necr"] = df["caption"].fillna("").str.lower().str.contains("necro")
        df["mention_enh"]  = df["caption"].fillna("").str.lower().str.contains(
            "enhanc|contrast|ring")

        # accuracy when caption commits to a lobe / size at all
        cap_committed = df["lobe_cap"].notna() & df["lobe_seg"].notna() & \
                        (df["lobe_seg"] != "unknown")
        lobe_acc = float((df.loc[cap_committed, "lobe_cap"] ==
                          df.loc[cap_committed, "lobe_seg"]).mean()) \
                   if cap_committed.any() else None
        lobe_commit_rate = float(df["lobe_cap"].notna().mean())

        size_committed = df["size_cap"].notna() & df["size_seg"].notna()
        size_acc = float((df.loc[size_committed, "size_cap"] ==
                          df.loc[size_committed, "size_seg"]).mean()) \
                   if size_committed.any() else None
        size_commit_rate = float(df["size_cap"].notna().mean())

        # necrosis & enhancement: classification-style metrics
        def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
            tp = int(((y_true) & (y_pred)).sum())
            fp = int(((~y_true) & (y_pred)).sum())
            fn = int(((y_true) & (~y_pred)).sum())
            tn = int(((~y_true) & (~y_pred)).sum())
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            f1   = 2 * prec * rec / max(1e-9, prec + rec)
            return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "precision": round(prec, 3),
                    "recall":    round(rec, 3),
                    "f1":        round(f1, 3),
                    "support_pos": int(y_true.sum())}
        necr_metrics = _binary_metrics(df["has_necr"].fillna(False),
                                       df["mention_necr"].fillna(False))
        enh_metrics  = _binary_metrics(df["has_enh"].fillna(False),
                                       df["mention_enh"].fillna(False))
    else:
        lobe_acc = size_acc = lobe_commit_rate = size_commit_rate = None
        necr_metrics = enh_metrics = None

    # 4) LEAKAGE -------------------------------------------------------------
    df["leak"] = df["caption"].fillna("").apply(_is_leak)
    n_leaks = int(df["leak"].sum())

    # write per-caption CSV (audit trail) -----------------------------------
    keep_cols = ["Patient_ID", "Timepoint", "y", "len_tokens", "distinct_2",
                 "n_concepts", "leak"]
    if not no_seg:
        keep_cols += ["n_voxels", "lobe_seg", "lobe_cap",
                      "size_seg", "size_cap",
                      "has_necr", "mention_necr",
                      "has_enh", "mention_enh"]
    df[keep_cols + ["caption"]].to_csv(OUT_DIR / "per_caption.csv", index=False)

    # write template phrases ------------------------------------------------
    with (OUT_DIR / "template_phrases.txt").open("w") as f:
        f.write("# top 20 most-repeated 6-grams across all captions\n")
        for phrase, count in top_phrases:
            f.write(f"{count:4d}  {' '.join(phrase)}\n")

    # plots -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.hist(df["len_chars"].dropna(), bins=30, color="#3a7ca5", edgecolor="white")
    plt.axvline(df["len_chars"].median(), color="red", linestyle="--",
                label=f"median={int(df['len_chars'].median())}")
    plt.xlabel("Caption length (chars)"); plt.ylabel("# captions")
    plt.title("RadFM caption length distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_DIR / "length_hist.png", dpi=110); plt.close()

    plt.figure(figsize=(8, 4.5))
    items = sorted(coverage.items(), key=lambda kv: kv[1], reverse=True)
    keys = [k for k, _ in items]; vals = [v for _, v in items]
    bars = plt.barh(keys, vals, color="#5e9c76", edgecolor="white")
    for b, v in zip(bars, vals):
        plt.text(v + 0.01, b.get_y() + b.get_height()/2,
                 f"{v:.0%}", va="center", fontsize=8)
    plt.xlim(0, 1.05); plt.xlabel("Fraction of captions mentioning")
    plt.title("Clinical-concept coverage in RadFM captions")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "concept_coverage.png", dpi=110); plt.close()

    if not no_seg:
        # lobe confusion
        seg_lobes = ["frontal", "parietal", "temporal", "occipital",
                     "cerebellar", "thalamic", "brainstem"]
        cap_lobes = seg_lobes + ["(none)"]
        cm = np.zeros((len(seg_lobes), len(cap_lobes)), dtype=int)
        for _, r in df.iterrows():
            if r["lobe_seg"] in seg_lobes:
                col = (cap_lobes.index(r["lobe_cap"])
                       if r["lobe_cap"] in seg_lobes else len(seg_lobes))
                cm[seg_lobes.index(r["lobe_seg"])][col] += 1

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cap_lobes))); ax.set_xticklabels(cap_lobes, rotation=40, ha="right")
        ax.set_yticks(range(len(seg_lobes))); ax.set_yticklabels(seg_lobes)
        for i in range(len(seg_lobes)):
            for j in range(len(cap_lobes)):
                if cm[i][j]:
                    ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                            color="white" if cm[i][j] > cm.max()/2 else "black",
                            fontsize=9)
        ax.set_xlabel("Caption mentions"); ax.set_ylabel("Segmentation centroid")
        ax.set_title(f"Lobe confusion (acc on committed = "
                     f"{lobe_acc:.0%}, commit rate = {lobe_commit_rate:.0%})")
        plt.colorbar(im, fraction=0.04)
        plt.tight_layout(); plt.savefig(PLOT_DIR / "lobe_confusion.png", dpi=110); plt.close()

        # size confusion
        seg_sizes = ["small", "medium", "large"]
        cap_sizes = seg_sizes + ["(none)"]
        cm2 = np.zeros((len(seg_sizes), len(cap_sizes)), dtype=int)
        for _, r in df.iterrows():
            if r["size_seg"] in seg_sizes:
                col = (cap_sizes.index(r["size_cap"])
                       if r["size_cap"] in seg_sizes else len(seg_sizes))
                cm2[seg_sizes.index(r["size_seg"])][col] += 1
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm2, cmap="Oranges")
        ax.set_xticks(range(len(cap_sizes))); ax.set_xticklabels(cap_sizes)
        ax.set_yticks(range(len(seg_sizes))); ax.set_yticklabels(seg_sizes)
        for i in range(len(seg_sizes)):
            for j in range(len(cap_sizes)):
                if cm2[i][j]:
                    ax.text(j, i, str(cm2[i][j]), ha="center", va="center",
                            color="white" if cm2[i][j] > cm2.max()/2 else "black")
        ax.set_xlabel("Caption mentions"); ax.set_ylabel("Segmentation bucket")
        ax.set_title(f"Size confusion (acc on committed = "
                     f"{size_acc:.0%}, commit rate = {size_commit_rate:.0%})")
        plt.colorbar(im, fraction=0.04)
        plt.tight_layout(); plt.savefig(PLOT_DIR / "size_confusion.png", dpi=110); plt.close()

    # summary JSON -----------------------------------------------------------
    summary = {
        "n_captions": len(df),
        "n_patients": int(df["Patient_ID"].nunique()),
        "structural": {
            "len_chars_mean":   round(float(df["len_chars"].mean()), 1),
            "len_chars_median": int(df["len_chars"].median()),
            "len_chars_min":    int(df["len_chars"].min()),
            "len_chars_max":    int(df["len_chars"].max()),
            "len_tokens_mean":  round(float(df["len_tokens"].mean()), 1),
            "type_token_ratio_corpus": round(type_token, 3),
            "distinct_2_per_caption":  round(float(df["distinct_2"].mean()), 3),
            "per_patient_distinct_2":  round(float(per_patient_div), 3),
            "short_caption_rate_lt30tokens": round(short_rate, 3),
            "empty_rate":      round(empty_rate, 3),
        },
        "concept_coverage": coverage,
        "concept_per_caption_mean": round(float(np.mean(per_cap_concepts)), 2),
        "leakage": {
            "n_leaks": n_leaks,
            "leak_rate": round(n_leaks / len(df), 4),
        },
    }
    if not no_seg:
        summary["segmentation_grounded"] = {
            "lobe_commit_rate":       round(lobe_commit_rate, 3),
            "lobe_accuracy_on_committed": round(lobe_acc, 3) if lobe_acc is not None else None,
            "size_commit_rate":       round(size_commit_rate, 3),
            "size_accuracy_on_committed": round(size_acc, 3) if size_acc is not None else None,
            "necrosis": necr_metrics,
            "enhancement": enh_metrics,
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # human report ----------------------------------------------------------
    md = ["# RadFM Caption Evaluation\n",
          f"Captions evaluated: **{summary['n_captions']}** "
          f"({summary['n_patients']} patients)\n",
          "## 1. Structural / lexical\n",
          f"| Metric | Value |", "|---|---|"]
    for k, v in summary["structural"].items():
        md.append(f"| {k} | {v} |")

    md += ["\n## 2. Clinical-concept coverage\n",
           f"Mean concepts mentioned per caption: "
           f"**{summary['concept_per_caption_mean']}** (out of "
           f"{len(CLINICAL_CONCEPTS)})",
           "\n| Concept | Coverage |", "|---|---|"]
    for k, v in sorted(coverage.items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v:.0%} |")

    if not no_seg:
        sg = summary["segmentation_grounded"]
        md += [
            "\n## 3. Segmentation-grounded factual accuracy\n",
            f"- **Lobe**: commit rate = {sg['lobe_commit_rate']:.0%}, "
            f"accuracy when committed = {sg['lobe_accuracy_on_committed']:.0%}",
            f"- **Size**: commit rate = {sg['size_commit_rate']:.0%}, "
            f"accuracy when committed = {sg['size_accuracy_on_committed']:.0%}",
            f"- **Necrosis (binary, support {sg['necrosis']['support_pos']}/"
            f"{summary['n_captions']})**: "
            f"P={sg['necrosis']['precision']}, R={sg['necrosis']['recall']}, "
            f"F1={sg['necrosis']['f1']}",
            f"- **Enhancement (binary, support {sg['enhancement']['support_pos']}/"
            f"{summary['n_captions']})**: "
            f"P={sg['enhancement']['precision']}, R={sg['enhancement']['recall']}, "
            f"F1={sg['enhancement']['f1']}",
            "\n![](plots/lobe_confusion.png)",
            "\n![](plots/size_confusion.png)",
        ]

    md += [
        "\n## 4. Temporal-leakage probe\n",
        f"- Future-outcome term (progression / recurrence / death) outside "
        f"a negation: **{n_leaks}/{summary['n_captions']}** "
        f"({summary['leakage']['leak_rate']:.1%})",
        "\n## 5. Top template phrases (6-grams)\n",
    ]
    for phrase, count in top_phrases[:10]:
        md.append(f"- `{' '.join(phrase)}`  ({count}x)")

    md += [
        "\n## 6. Files\n",
        "- `summary.json` — all aggregate numbers",
        "- `per_caption.csv` — per-row flags for audit",
        "- `template_phrases.txt` — full top-20 list",
        "- `plots/length_hist.png`, `plots/concept_coverage.png`, "
        "`plots/lobe_confusion.png`, `plots/size_confusion.png`",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(md))

    print("\n" + "=" * 70)
    print(f"Done. Report → {OUT_DIR / 'REPORT.md'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only first N captions (for smoke testing)")
    ap.add_argument("--no-seg", action="store_true",
                    help="skip segmentation-grounded checks (much faster)")
    args = ap.parse_args()
    evaluate(limit=args.limit, no_seg=args.no_seg)
