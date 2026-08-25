"""
re_infer_and_eval.py — Re-run inference + evaluation for one or more
already-trained cells, *without* re-training the LoRA adapter.

Why this script exists
----------------------
The original `infer.py::get_yes_no_ids` resolved the Yes/No token IDs in
isolation (`tokenizer.encode("Yes")`), but Mistral's BPE assigns *different*
IDs after `[/INST]` (in-context tokenization). Training used the
in-context IDs (via apply_chat_template), so the LoRA adapters are
correct; only the inference-time logit lookup was reading from the wrong
vocab slots. We patched `infer.py`, but the cells already in
`aggregate.csv` (Exp1/2/3 baselines, all Exp4 cells
ablations) were scored under the buggy logic and need to be re-scored.

This script:
  * re-runs inference on the held-out test split using the existing,
    fully-trained adapter at `Model/checkpoints/<tag>/adapters/`
  * re-runs `evaluate.py` on the new predictions
  * rewrites the corresponding row in `Model/results/aggregate.csv`

It is idempotent and safe to invoke at any time. It does *not* touch
prompt JSONL or adapters.

Usage:
    python code/re_infer_and_eval.py --tags Exp4__baseline Exp1__baseline ...
    python code/re_infer_and_eval.py --all-existing      # picks up every adapter on disk
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "Model"
CKPT_D    = MODEL_DIR / "checkpoints"
RESULT_D  = MODEL_DIR / "results"

sys.path.insert(0, str(MODEL_DIR / "code"))


def find_existing_tags() -> list[str]:
    tags = []
    for d in sorted(CKPT_D.iterdir()):
        if (d / "adapters" / "adapter_config.json").exists() or \
           (d / "adapters" / "adapters.safetensors").exists():
            tags.append(d.name)
    return tags


def update_aggregate_row(tag: str, metrics: dict) -> None:
    """Replace (or append) the row for `tag` in aggregate.csv, preserving
    the existing column order."""
    agg_path = RESULT_D / "aggregate.csv"
    keys = ["tag", "n", "n_pos", "n_neg",
            "AUROC", "AUROC_lo", "AUROC_hi",
            "AUPRC", "AUPRC_lo", "AUPRC_hi",
            "Macro_F1", "Accuracy", "Sensitivity", "Specificity",
            "MCC", "Brier", "ECE"]
    new_row = {
        "tag":         tag,
        "n":           metrics["n"],
        "n_pos":       metrics["n_pos"],
        "n_neg":       metrics["n_neg"],
        "AUROC":       metrics["AUROC"],
        "AUROC_lo":    metrics["AUROC_95CI"][0],
        "AUROC_hi":    metrics["AUROC_95CI"][1],
        "AUPRC":       metrics["AUPRC"],
        "AUPRC_lo":    metrics["AUPRC_95CI"][0],
        "AUPRC_hi":    metrics["AUPRC_95CI"][1],
        "Macro_F1":    metrics["Macro_F1"],
        "Accuracy":    metrics["Accuracy"],
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
        "MCC":         metrics["MCC"],
        "Brier":       metrics["Brier"],
        "ECE":         metrics["ECE"],
    }
    rows = []
    if agg_path.exists():
        with agg_path.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["tag"] != tag:
                    rows.append(r)
    rows.append({k: new_row[k] for k in keys})
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def reinfer_one(tag: str) -> dict | None:
    from infer    import infer
    from evaluate import evaluate

    adapter = CKPT_D / tag / "adapters"
    if not (adapter / "adapter_config.json").exists() and \
       not (adapter / "adapters.safetensors").exists():
        print(f"[re-infer] SKIP {tag}: no adapter on disk")
        return None

    t0 = time.perf_counter()
    print(f"\n{'='*70}\n[re-infer] {tag}\n{'='*70}")
    pred_path = infer(tag, split="test")
    metrics   = evaluate(str(pred_path))
    update_aggregate_row(tag, metrics)
    print(f"[re-infer] {tag}: AUROC={metrics['AUROC']:.4f}  "
          f"({time.perf_counter()-t0:.1f}s)")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", help="Cells to re-infer.")
    ap.add_argument("--all-existing", action="store_true",
                    help="Re-infer every cell that has a trained adapter.")
    args = ap.parse_args()

    if args.all_existing:
        tags = find_existing_tags()
    elif args.tags:
        tags = args.tags
    else:
        ap.error("Pass --tags or --all-existing")

    print(f"[re-infer] Will re-infer {len(tags)} cells:")
    for t in tags:
        print(f"  - {t}")

    summary: dict[str, dict] = {}
    for t in tags:
        try:
            m = reinfer_one(t)
            if m is not None:
                summary[t] = {"AUROC": m["AUROC"], "AUPRC": m["AUPRC"],
                              "MCC": m["MCC"]}
        except Exception as err:  # pylint: disable=broad-except
            print(f"[re-infer] FAILED {t}: {err}")
            summary[t] = {"error": str(err)}

    out = RESULT_D / "reinfer_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[re-infer] summary -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
