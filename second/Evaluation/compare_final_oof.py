"""Paired patient-level bootstrap comparisons for final OOF models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Model" / "results"
OUT_DIR = ROOT / "Evaluation" / "generated"
OUT = OUT_DIR / "paired_oof_comparisons.csv"
OUT_MD = OUT_DIR / "PAIRED_OOF_COMPARISONS.md"


def read_csv_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[["Patient_ID", "label", "score_raw", "score_platt"]]


def read_jsonl_predictions(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    frame = pd.DataFrame(rows).rename(columns={"patient_id": "Patient_ID"})
    return frame[["Patient_ID", "label", "score_raw", "score"]].rename(
        columns={"score": "score_platt"}
    )


def candidates() -> dict[str, tuple[str, pd.DataFrame]]:
    specs = {
        "XGBoost_ExpC_VLM": (
            "retrospective_147",
            RESULTS / "final_baselines_cv" / "ExpC_TxVLM__XGBoost.csv",
            "csv",
        ),
        "Mistral_ExpA_noRAG": (
            "retrospective_147",
            RESULTS / "ExpA_Tx__baseline__mistral__cv" / "oof_predictions_platt.jsonl",
            "jsonl",
        ),
        "Mistral_ExpC_ColBERT": (
            "retrospective_147",
            RESULTS / "ExpC_TxVLM__beep__colbert__mistral__cv__v3_structured"
            / "oof_predictions_platt.jsonl",
            "jsonl",
        ),
        "XGBoost_ExpA_fixed_landmark": (
            "fixed_landmark_65",
            RESULTS / "fixed_landmark_baselines_cv" / "ExpA_Tx__XGBoost.csv",
            "csv",
        ),
        "XGBoost_ExpA_noMol_fixed_landmark": (
            "fixed_landmark_65",
            RESULTS / "fixed_landmark_baselines_cv" / "ExpA_TxNoMol__XGBoost.csv",
            "csv",
        ),
    }
    loaded = {}
    for name, (group, path, kind) in specs.items():
        if not path.exists():
            continue
        frame = read_csv_predictions(path) if kind == "csv" else read_jsonl_predictions(path)
        if len(frame) != frame["Patient_ID"].nunique():
            raise RuntimeError(f"duplicate patients in {path}")
        loaded[name] = (group, frame)
    return loaded


def paired_compare(
    name_a: str,
    frame_a: pd.DataFrame,
    name_b: str,
    frame_b: pd.DataFrame,
    seed: int = 42,
    iterations: int = 10000,
) -> dict:
    merged = frame_a.merge(frame_b, on="Patient_ID", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(frame_a) or len(merged) != len(frame_b):
        raise RuntimeError(f"patient sets differ for {name_a} and {name_b}")
    if not np.array_equal(merged["label_a"], merged["label_b"]):
        raise RuntimeError(f"labels differ for {name_a} and {name_b}")
    y = merged["label_a"].to_numpy(dtype=int)
    raw_a = merged["score_raw_a"].to_numpy(dtype=float)
    raw_b = merged["score_raw_b"].to_numpy(dtype=float)
    calibrated_a = merged["score_platt_a"].to_numpy(dtype=float)
    calibrated_b = merged["score_platt_b"].to_numpy(dtype=float)

    metrics = {
        "AUROC": (lambda yy, pp: roc_auc_score(yy, pp), raw_a, raw_b),
        "Macro_F1": (
            lambda yy, pp: f1_score(
                yy, (pp >= 0.5).astype(int), average="macro", zero_division=0
            ),
            calibrated_a,
            calibrated_b,
        ),
    }
    rng = np.random.default_rng(seed)
    output = {"model_a": name_a, "model_b": name_b, "n": len(y)}
    for metric_name, (metric_fn, a, b) in metrics.items():
        observed_a = metric_fn(y, a)
        observed_b = metric_fn(y, b)
        differences = []
        for _ in range(iterations):
            index = rng.integers(0, len(y), len(y))
            if len(np.unique(y[index])) < 2:
                continue
            differences.append(metric_fn(y[index], a[index]) - metric_fn(y[index], b[index]))
        values = np.asarray(differences)
        p_value = min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))
        output.update({
            f"{metric_name}_a": observed_a,
            f"{metric_name}_b": observed_b,
            f"delta_{metric_name}": observed_a - observed_b,
            f"delta_{metric_name}_lo": float(np.percentile(values, 2.5)),
            f"delta_{metric_name}_hi": float(np.percentile(values, 97.5)),
            f"p_{metric_name}_paired_bootstrap": float(p_value),
        })
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = candidates()
    rows = []
    names = sorted(loaded)
    for index, name_a in enumerate(names):
        group_a, frame_a = loaded[name_a]
        for name_b in names[index + 1:]:
            group_b, frame_b = loaded[name_b]
            if group_a != group_b:
                continue
            rows.append(paired_compare(name_a, frame_a, name_b, frame_b))

    columns = [
        "model_a", "model_b", "n",
        "AUROC_a", "AUROC_b", "delta_AUROC", "delta_AUROC_lo", "delta_AUROC_hi",
        "p_AUROC_paired_bootstrap",
        "Macro_F1_a", "Macro_F1_b", "delta_Macro_F1", "delta_Macro_F1_lo",
        "delta_Macro_F1_hi", "p_Macro_F1_paired_bootstrap",
    ]
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Paired OOF Comparisons",
        "",
        "Positive deltas favour model A. Intervals and two-sided p-values use "
        "10,000 paired patient-level bootstrap samples.",
        "",
        "| Model A | Model B | N | Delta AUROC (95% CI) | p | Delta Macro-F1 (95% CI) | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | {row['n']} | "
            f"{row['delta_AUROC']:.3f} [{row['delta_AUROC_lo']:.3f}, {row['delta_AUROC_hi']:.3f}] | "
            f"{row['p_AUROC_paired_bootstrap']:.3f} | "
            f"{row['delta_Macro_F1']:.3f} [{row['delta_Macro_F1_lo']:.3f}, {row['delta_Macro_F1_hi']:.3f}] | "
            f"{row['p_Macro_F1_paired_bootstrap']:.3f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({len(rows)} comparisons)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
