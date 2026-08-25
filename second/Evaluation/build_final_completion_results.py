"""Build the final, protocol-aware Second-Recurrence evidence package.

This intentionally keeps fixed-split, pooled cross-validation, and censoring
analyses separate. Combining those estimates into a single leaderboard would
make small-sample test-set exploration look confirmatory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_RESULTS = ROOT / "Model" / "results"
OUT = ROOT / "Evaluation" / "generated"

FEATURE_LABELS = {
    "ExpA_TxNoMol": "Demographics + diagnosis + initial/salvage treatment",
    "ExpA_Tx": "ExpA + molecular markers",
    "ExpB_TxRadiomic": "ExpA + timepoints + radiomics",
    "ExpC_TxVLM": "ExpA + timepoints + structured RadFM captions",
    "ExpD_TxRadVLM": "ExpA + timepoints + radiomics + structured RadFM captions",
}
RERANKER_LABELS = {
    "beep": "PubMedBERT",
    "minilm": "MiniLM",
    "medcpt": "MedCPT",
    "colbert": "ColBERTv2",
    "bge_m3": "BGE-M3",
}

SUMMARY_COLUMNS = [
    "task", "evaluation", "selection_status", "feature_set", "features",
    "model", "retrieval", "calibration", "threshold", "n", "n_pos",
    "n_neg", "AUROC", "AUROC_lo", "AUROC_hi", "AUPRC", "Macro_F1",
    "Accuracy", "Sensitivity", "Specificity", "MCC", "Brier", "source",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def metric(row: dict, key: str, default: str = "") -> str:
    value = row.get(key, default)
    return "" if value is None else str(value)


def conventional_rows() -> list[dict]:
    path = MODEL_RESULTS / "aggregate_final_baselines_cv.csv"
    rows = []
    for item in read_csv(path):
        rows.append({
            "task": "Second recurrence",
            "evaluation": "Pooled 5-fold patient-level OOF",
            "selection_status": "Confirmatory baseline",
            "feature_set": item["experiment"],
            "features": FEATURE_LABELS[item["experiment"]],
            "model": item["model"],
            "retrieval": "None",
            "calibration": "Raw discrimination; fold-local Platt classification" if item["model"] != "Majority" else "None",
            "threshold": "0.5",
            "n": item["n_oof"],
            "n_pos": "78",
            "n_neg": "69",
            "AUROC": item["raw_AUROC"],
            "AUROC_lo": item["raw_AUROC_lo"],
            "AUROC_hi": item["raw_AUROC_hi"],
            "AUPRC": item["raw_AUPRC"],
            "Macro_F1": item["platt_Macro_F1"],
            "Accuracy": item["platt_Accuracy"],
            "Sensitivity": item["platt_Sensitivity"],
            "Specificity": item["platt_Specificity"],
            "MCC": item["platt_MCC"],
            "Brier": item["platt_Brier"],
            "source": str(path),
        })
    return rows


def llm_cv_rows() -> list[dict]:
    path = MODEL_RESULTS / "aggregate_cv.csv"
    rows = []
    for item in read_csv(path):
        tag = item["tag"]
        exp = tag.split("__", 1)[0]
        retrieval = "None" if "__baseline__" in tag else "BEEP + ColBERT"
        rows.append({
            "task": "Second recurrence",
            "evaluation": "Pooled 5-fold patient-level OOF",
            "selection_status": "Confirmatory LLM",
            "feature_set": exp,
            "features": FEATURE_LABELS.get(exp, exp),
            "model": "Mistral-7B-Instruct-v0.3 4-bit" if "__mistral" in tag else "BioMistral-7B 4-bit",
            "retrieval": retrieval,
            "calibration": item["calibration"],
            "threshold": item["threshold"],
            "n": item["n_oof"],
            "n_pos": item["n_pos"],
            "n_neg": item["n_neg"],
            "AUROC": item.get("raw_AUROC", item.get("AUROC", "")),
            "AUROC_lo": item.get("raw_AUROC_lo", item.get("AUROC_lo", "")),
            "AUROC_hi": item.get("raw_AUROC_hi", item.get("AUROC_hi", "")),
            "AUPRC": item.get("raw_AUPRC", item.get("AUPRC", "")),
            "Macro_F1": item.get("calibrated_Macro_F1", item.get("Macro_F1", "")),
            "Accuracy": item.get("calibrated_Accuracy", ""),
            "Sensitivity": item.get("calibrated_Sensitivity", item.get("Sensitivity", "")),
            "Specificity": item.get("calibrated_Specificity", item.get("Specificity", "")),
            "MCC": item.get("calibrated_MCC", item.get("MCC", "")),
            "Brier": item.get("calibrated_Brier", ""),
            "source": str(path),
        })
    return rows


def fixed_landmark_rows() -> list[dict]:
    path = MODEL_RESULTS / "aggregate_fixed_landmark_baselines_cv.csv"
    rows = []
    for item in read_csv(path):
        rows.append({
            "task": "Second recurrence",
            "evaluation": "Fixed-landmark pooled 5-fold OOF",
            "selection_status": "Prospective sensitivity baseline",
            "feature_set": item["experiment"],
            "features": FEATURE_LABELS[item["experiment"]],
            "model": item["model"],
            "retrieval": "None",
            "calibration": "Raw discrimination; fold-local Platt classification" if item["model"] != "Majority" else "None",
            "threshold": "0.5",
            "n": item["n_oof"],
            "n_pos": item["n_pos"],
            "n_neg": item["n_neg"],
            "AUROC": item["raw_AUROC"],
            "AUROC_lo": item["raw_AUROC_lo"],
            "AUROC_hi": item["raw_AUROC_hi"],
            "AUPRC": item["raw_AUPRC"],
            "Macro_F1": item["platt_Macro_F1"],
            "Accuracy": item["platt_Accuracy"],
            "Sensitivity": item["platt_Sensitivity"],
            "Specificity": item["platt_Specificity"],
            "MCC": item["platt_MCC"],
            "Brier": item["platt_Brier"],
            "source": str(path),
        })
    return rows


def fixed_split_rows() -> list[dict]:
    path = MODEL_RESULTS / "aggregate_platt_0p5.csv"
    rows = []
    for item in read_csv(path):
        tag = item["tag"]
        exp = tag.split("__", 1)[0]
        rows.append({
            "task": "Second recurrence",
            "evaluation": "Original fixed test split",
            "selection_status": "Sensitivity analysis",
            "feature_set": exp,
            "features": FEATURE_LABELS.get(exp, exp),
            "model": "Mistral-7B-Instruct-v0.3 4-bit",
            "retrieval": "None",
            "calibration": item["calibration"],
            "threshold": item["threshold"],
            "n": item["n"],
            "n_pos": item["n_pos"],
            "n_neg": item["n_neg"],
            "AUROC": item["AUROC"],
            "AUROC_lo": "",
            "AUROC_hi": "",
            "AUPRC": item["AUPRC"],
            "Macro_F1": item["Macro_F1"],
            "Accuracy": item["Accuracy"],
            "Sensitivity": item["Sensitivity"],
            "Specificity": item["Specificity"],
            "MCC": item["MCC"],
            "Brier": item["Brier"],
            "source": str(path),
        })
    return rows


def fixed_rag_rows() -> list[dict]:
    path = MODEL_RESULTS / "aggregate.csv"
    rows = []
    for item in read_csv(path):
        tag = item.get("tag", "")
        parts = tag.split("__")
        if len(parts) not in {4, 5} or parts[1] != "beep":
            continue
        exp, _, reranker, llm = parts[:4]
        caption_suffix = f" ({parts[4]})" if len(parts) == 5 else ""
        rows.append({
            "task": "Second recurrence",
            "evaluation": "Original fixed test split",
            "selection_status": "Exploratory fixed-split reranker grid",
            "feature_set": exp + caption_suffix,
            "features": FEATURE_LABELS.get(exp, exp),
            "model": "Mistral-7B-Instruct-v0.3 4-bit" if llm == "mistral" else "BioMistral-7B 4-bit",
            "retrieval": f"BEEP + {RERANKER_LABELS.get(reranker, reranker)}",
            "calibration": "Raw model score",
            "threshold": "0.5",
            "n": metric(item, "n"),
            "n_pos": metric(item, "n_pos"),
            "n_neg": metric(item, "n_neg"),
            "AUROC": metric(item, "AUROC"),
            "AUROC_lo": metric(item, "AUROC_lo"),
            "AUROC_hi": metric(item, "AUROC_hi"),
            "AUPRC": metric(item, "AUPRC"),
            "Macro_F1": metric(item, "Macro_F1"),
            "Accuracy": metric(item, "Accuracy"),
            "Sensitivity": metric(item, "Sensitivity"),
            "Specificity": metric(item, "Specificity"),
            "MCC": metric(item, "MCC"),
            "Brier": metric(item, "Brier"),
            "source": str(path),
        })
    return rows


def as_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def fmt(value: str | float) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "pending"


def write_csv(rows: list[dict]) -> Path:
    path = OUT / "Final_Summary.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(rows: list[dict]) -> Path:
    cv = [r for r in rows if r["evaluation"].startswith("Pooled")]
    conventional = [r for r in cv if r["selection_status"] == "Confirmatory baseline"]
    llm = [r for r in cv if r["selection_status"] == "Confirmatory LLM"]
    fixed = [
        r for r in rows
        if r["evaluation"].startswith("Original") and r["retrieval"] == "None"
    ]
    prospective = [r for r in rows if r["evaluation"].startswith("Fixed-landmark")]
    best_conv = max(conventional, key=lambda r: as_float(r, "AUROC")) if conventional else None
    best_fixed = max(fixed, key=lambda r: as_float(r, "AUROC")) if fixed else None
    best_prospective = max(prospective, key=lambda r: as_float(r, "AUROC")) if prospective else None
    censor_path = OUT / "followup_censoring_audit" / "summary.json"
    censor = json.loads(censor_path.read_text()) if censor_path.exists() else {}
    paired = read_csv(OUT / "paired_oof_comparisons.csv")

    lines = [
        "# Final Second-Recurrence Completion Results",
        "",
        "## What is complete",
        "",
        "- Five feature levels are implemented, including the molecular-ablation and radiomics+VLM fusion experiments.",
        "- Majority, logistic regression, XGBoost, and two-layer MLP baselines were evaluated with pooled five-fold patient-level out-of-fold predictions.",
        "- All five no-RAG Mistral fixed-split experiments were trained and evaluated with validation-only Platt scaling at threshold 0.5.",
        "- Confirmatory LLM cross-validation uses a separate validation calibrator inside every training fold.",
        "- Follow-up adequacy and the outcome-dependent landmark definition were audited.",
        "- A clinical-only fixed-landmark five-fold sensitivity experiment was completed on adequately followed patients.",
        "",
        "## Primary pooled cross-validation",
        "",
        "| Feature set | Model | Retrieval | AUROC | AUPRC | Macro-F1 | Accuracy | Sens. | Spec. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(conventional + llm, key=lambda r: as_float(r, "AUROC"), reverse=True):
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {row['retrieval']} | "
            f"{fmt(row['AUROC'])} | {fmt(row['AUPRC'])} | {fmt(row['Macro_F1'])} | "
            f"{fmt(row['Accuracy'])} | {fmt(row['Sensitivity'])} | {fmt(row['Specificity'])} |"
        )
    if not llm:
        lines.append("| ExpA_Tx | Mistral (five-fold run) | None | pending | pending | pending | pending | pending | pending |")
    lines.extend([
        "",
        "AUROC and AUPRC use raw out-of-fold scores to measure discrimination. "
        "Macro-F1, sensitivity, and specificity use fold-local Platt-calibrated probabilities at threshold 0.5.",
    ])
    rag_pair = next((
        row for row in paired
        if row.get("model_a") == "Mistral_ExpA_noRAG"
        and row.get("model_b") == "Mistral_ExpC_ColBERT"
    ), None)
    if rag_pair:
        rag_auc = -float(rag_pair["delta_AUROC"])
        rag_auc_lo = -float(rag_pair["delta_AUROC_hi"])
        rag_auc_hi = -float(rag_pair["delta_AUROC_lo"])
        rag_f1 = -float(rag_pair["delta_Macro_F1"])
        rag_f1_lo = -float(rag_pair["delta_Macro_F1_hi"])
        rag_f1_hi = -float(rag_pair["delta_Macro_F1_lo"])
        lines.extend([
            "",
            f"ColBERT is the proposed retrospective model by point estimate. Versus no-RAG, "
            f"delta AUROC is +{rag_auc:.3f} [{rag_auc_lo:.3f}, {rag_auc_hi:.3f}] "
            f"(paired-bootstrap p={float(rag_pair['p_AUROC_paired_bootstrap']):.3f}) and "
            f"delta Macro-F1 is +{rag_f1:.3f} [{rag_f1_lo:.3f}, {rag_f1_hi:.3f}] "
            f"(p={float(rag_pair['p_Macro_F1_paired_bootstrap']):.3f}). The gain is "
            "promising but not statistically conclusive.",
        ])

    lines.extend([
        "",
        "## Fixed-landmark prospective sensitivity analysis",
        "",
        "| Feature set | Model | N | AUROC | AUPRC | Macro-F1 | Accuracy | Sens. | Spec. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(prospective, key=lambda r: as_float(r, "AUROC"), reverse=True):
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {row['n']} | "
            f"{fmt(row['AUROC'])} | {fmt(row['AUPRC'])} | {fmt(row['Macro_F1'])} | "
            f"{fmt(row['Accuracy'])} | {fmt(row['Sensitivity'])} | {fmt(row['Specificity'])} |"
        )
    if best_prospective:
        lines.extend([
            "",
            f"The best clinically prospective sensitivity model was {best_prospective['model']} with "
            f"{best_prospective['feature_set']} (AUROC {fmt(best_prospective['AUROC'])}, "
            f"Macro-F1 {fmt(best_prospective['Macro_F1'])}). Radiomics and VLM were excluded because their "
            "existing patient-level aggregates cannot be safely re-gated to the earlier fixed landmark.",
        ])
    molecular_pair = next((
        row for row in paired
        if row.get("model_a") == "XGBoost_ExpA_fixed_landmark"
        and row.get("model_b") == "XGBoost_ExpA_noMol_fixed_landmark"
    ), None)
    if molecular_pair:
        lines.append(
            f"The paired molecular-marker ablation favoured ExpA_Tx by delta AUROC "
            f"{float(molecular_pair['delta_AUROC']):.3f} "
            f"[{float(molecular_pair['delta_AUROC_lo']):.3f}, "
            f"{float(molecular_pair['delta_AUROC_hi']):.3f}] "
            f"(paired-bootstrap p={float(molecular_pair['p_AUROC_paired_bootstrap']):.3f})."
        )

    lines.extend(["", "## Fixed-split sensitivity analysis", ""])
    if best_fixed:
        lines.append(
            f"The strongest no-RAG fixed-split discrimination was {best_fixed['feature_set']} "
            f"(AUROC {fmt(best_fixed['AUROC'])}, AUPRC {fmt(best_fixed['AUPRC'])}). "
            "These n=23 results are secondary because validation and test prevalence drifted after cohort subsetting."
        )
    if best_conv:
        lines.append(
            f"The strongest conventional pooled-CV model by AUROC was {best_conv['model']} "
            f"with {best_conv['feature_set']} (AUROC {fmt(best_conv['AUROC'])}, "
            f"Macro-F1 {fmt(best_conv['Macro_F1'])})."
        )

    lines.extend([
        "",
        "## Clinical validity warning",
        "",
        f"The original cohort has n={censor.get('original_n', 'unknown')}. A prospective sensitivity definition using "
        f"TTP1 + 90 days and a 180-day prediction horizon retains only n={censor.get('prospective_n', 'unknown')} "
        f"({censor.get('prospective_positive', 'unknown')} positive, {censor.get('prospective_negative', 'unknown')} negative).",
        "",
        f"It excludes {censor.get('excluded_event_before_landmark', 'unknown')} events that occurred before the fixed landmark and "
        f"{censor.get('excluded_censored_before_horizon', 'unknown')} patients without adequate horizon follow-up. "
        "The original Landmark_day uses the event day for positives and last follow-up for negatives, so the current models must be described as retrospective association models until confirmed on a fixed-landmark or survival-analysis cohort.",
        "",
        "## Final interpretation",
        "",
        "First Recurrence can be frozen as the completed primary study. Second Recurrence now has the required baselines, calibration framework, locked ColBERT confirmation, and a clinical fixed-landmark sensitivity analysis. ExpC_TxVLM Mistral plus ColBERT is the proposed retrospective model by point estimate, while no-RAG remains the simpler LLM baseline because the paired gain is not conclusive. A definitive prospective multimodal claim requires rebuilding scan-level radiomics and captions at the fixed landmark on a larger adequately followed cohort. More reranker searching on the original outcome-dependent cohort cannot resolve that limitation.",
        "",
        "The machine-readable evidence table is `Final_Summary.csv`. Detailed excluded-patient reasons are in `followup_censoring_audit/patient_followup_audit.csv`.",
    ])
    path = OUT / "FINAL_COMPLETION_RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = (
        conventional_rows()
        + llm_cv_rows()
        + fixed_landmark_rows()
        + fixed_split_rows()
        + fixed_rag_rows()
    )
    csv_path = write_csv(rows)
    report_path = write_report(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
