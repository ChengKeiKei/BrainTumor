"""Build the final shared-adapter reranker table and paired comparisons."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Model" / "results"
OUT_DIR = ROOT / "Evaluation" / "generated"
RERANKERS = ("beep", "minilm", "medcpt", "colbert", "bge_m3")
LLMS = ("mistral", "biomistral")
PREFIX = "ExpC_TxVLM__beep"
SUFFIX = "sharedcv__v3_structured"


def tag(llm: str, reranker: str) -> str:
    return f"{PREFIX}__{reranker}__{llm}__{SUFFIX}"


def load_predictions(path: Path, score_key: str = "score") -> dict[str, tuple[int, float]]:
    rows = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            rows[str(row["patient_id"])] = (int(row["label"]), float(row[score_key]))
    return rows


def percentile_ci(values: list[float]) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(lo), float(hi)


def paired_comparison(llm: str, reranker: str, n_boot: int = 5000) -> dict:
    candidate_dir = RESULTS / tag(llm, reranker)
    reference_dir = RESULTS / tag(llm, "colbert")
    cand_raw = load_predictions(candidate_dir / "oof_predictions_raw.jsonl")
    ref_raw = load_predictions(reference_dir / "oof_predictions_raw.jsonl")
    cand_cal = load_predictions(candidate_dir / "oof_predictions_platt.jsonl")
    ref_cal = load_predictions(reference_dir / "oof_predictions_platt.jsonl")
    ids = sorted(ref_raw)
    if set(ids) != set(cand_raw) or set(ids) != set(cand_cal) or set(ids) != set(ref_cal):
        raise RuntimeError(f"OOF patient mismatch for {llm}/{reranker}")

    y = np.asarray([ref_raw[pid][0] for pid in ids])
    cand_raw_s = np.asarray([cand_raw[pid][1] for pid in ids])
    ref_raw_s = np.asarray([ref_raw[pid][1] for pid in ids])
    cand_cal_s = np.asarray([cand_cal[pid][1] for pid in ids])
    ref_cal_s = np.asarray([ref_cal[pid][1] for pid in ids])

    delta_auc = roc_auc_score(y, cand_raw_s) - roc_auc_score(y, ref_raw_s)
    delta_f1 = (
        f1_score(y, cand_cal_s >= 0.5, average="macro")
        - f1_score(y, ref_cal_s >= 0.5, average="macro")
    )
    rng = np.random.default_rng(20260716)
    auc_boot: list[float] = []
    f1_boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        auc_boot.append(
            roc_auc_score(y[idx], cand_raw_s[idx])
            - roc_auc_score(y[idx], ref_raw_s[idx])
        )
        f1_boot.append(
            f1_score(y[idx], cand_cal_s[idx] >= 0.5, average="macro")
            - f1_score(y[idx], ref_cal_s[idx] >= 0.5, average="macro")
        )
    auc_lo, auc_hi = percentile_ci(auc_boot)
    f1_lo, f1_hi = percentile_ci(f1_boot)
    return {
        "llm": llm,
        "reranker": reranker,
        "reference": "colbert",
        "delta_raw_AUROC": delta_auc,
        "delta_raw_AUROC_lo": auc_lo,
        "delta_raw_AUROC_hi": auc_hi,
        "delta_calibrated_Macro_F1": delta_f1,
        "delta_calibrated_Macro_F1_lo": f1_lo,
        "delta_calibrated_Macro_F1_hi": f1_hi,
        "bootstrap_samples": len(auc_boot),
    }


def fmt(value: str | float) -> str:
    return f"{float(value):.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_path = RESULTS / "aggregate_shared_reranker_cv.csv"
    with aggregate_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 aggregate rows, found {len(rows)}")

    comparisons = [
        paired_comparison(llm, reranker)
        for llm in LLMS
        for reranker in RERANKERS
        if reranker != "colbert"
    ]
    comparison_path = OUT_DIR / "paired_shared_reranker_comparisons.csv"
    with comparison_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# Shared-Adapter Five-Reranker Cross-Validation",
        "",
        "## Method",
        "",
        "For each patient-level fold, one LoRA adapter per backbone was trained on a "
        "deterministic class-balanced mixture of contexts from the five rerankers. The "
        "adapter was held fixed while BEEP, MiniLM, MedCPT, ColBERT, and BGE-M3 were "
        "evaluated separately. Platt scaling was fitted on each fold's validation "
        "patients only; all reported metrics pool 147 out-of-fold predictions.",
        "",
        "Training used 4-bit Mistral/BioMistral, LoRA rank 16, batch size 4, 200 "
        "iterations, maximum sequence length 3072, and gradient checkpointing.",
        "",
        "## Results",
        "",
        "| Backbone | Reranker | Raw AUROC (95% CI) | Raw AUPRC | Calibrated Macro-F1 | Accuracy | Brier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for llm in LLMS:
        subset = sorted(
            (row for row in rows if row["llm"] == llm),
            key=lambda row: float(row["raw_AUROC"]),
            reverse=True,
        )
        for row in subset:
            lines.append(
                f"| {llm} | {row['reranker']} | {fmt(row['raw_AUROC'])} "
                f"[{fmt(row['raw_AUROC_lo'])}, {fmt(row['raw_AUROC_hi'])}] | "
                f"{fmt(row['raw_AUPRC'])} | {fmt(row['calibrated_Macro_F1'])} | "
                f"{fmt(row['calibrated_Accuracy'])} | {fmt(row['calibrated_Brier'])} |"
            )

    lines.extend([
        "",
        "## Paired Comparison",
        "",
        "Differences below are candidate minus ColBERT using paired patient bootstrap "
        "samples. A 95% interval containing zero is not statistically conclusive.",
        "",
        "| Backbone | Candidate | Delta raw AUROC (95% CI) | Delta Macro-F1 (95% CI) |",
        "|---|---|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['llm']} | {row['reranker']} | "
            f"{fmt(row['delta_raw_AUROC'])} "
            f"[{fmt(row['delta_raw_AUROC_lo'])}, {fmt(row['delta_raw_AUROC_hi'])}] | "
            f"{fmt(row['delta_calibrated_Macro_F1'])} "
            f"[{fmt(row['delta_calibrated_Macro_F1_lo'])}, "
            f"{fmt(row['delta_calibrated_Macro_F1_hi'])}] |"
        )

    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "Select the reranker primarily by raw AUROC and its paired uncertainty. Use "
        "calibrated Macro-F1, Brier score, and operational sensitivity/specificity as "
        "secondary criteria. Do not claim superiority when the paired confidence "
        "interval includes zero.",
        "",
    ])
    summary_path = OUT_DIR / "SHARED_RERANKER_CV_RESULTS.md"
    summary_path.write_text("\n".join(lines))
    print(f"summary: {summary_path}")
    print(f"paired comparisons: {comparison_path}")


if __name__ == "__main__":
    main()
