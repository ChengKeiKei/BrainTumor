"""Evaluate an earlier saved adapter step from a completed five-fold CV run.

This creates lightweight hard-linked adapter views, runs validation/test
inference, fits fold-local Platt scaling, and aggregates 147 OOF predictions.
It is intended for choosing a shorter training schedule without retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import evaluate
import infer
import run_cv_grid

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "Model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
PROMPT_DIR = MODEL_DIR / "prompts"
RESULT_DIR = MODEL_DIR / "results"

BASE = "ExpC_TxVLM__beep__colbert__mistral__cv__fold{fold}__v3_structured"
MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def adapter_view(fold: int, step: int) -> str:
    source_tag = BASE.format(fold=fold)
    view_tag = f"{source_tag}__step{step}_sensitivity"
    source = CHECKPOINT_DIR / source_tag / "adapters"
    target = CHECKPOINT_DIR / view_tag / "adapters"
    target.mkdir(parents=True, exist_ok=True)

    source_weights = source / f"{step:07d}_adapters.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    shutil.copy2(source / "adapter_config.json", target / "adapter_config.json")
    target_weights = target / "adapters.safetensors"
    if target_weights.exists() or target_weights.is_symlink():
        target_weights.unlink()
    os.link(source_weights, target_weights)
    (CHECKPOINT_DIR / view_tag / "model_id.txt").write_text(MODEL_ID + "\n")
    return view_tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=150)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    calibrated_rows: list[dict] = []
    for fold in range(5):
        source_tag = BASE.format(fold=fold)
        view_tag = adapter_view(fold, args.step)
        output_tag = f"{source_tag}__step{args.step}_sensitivity_eval"
        output_dir = RESULT_DIR / output_tag
        calibrated = output_dir / "predictions_test_platt.jsonl"
        if not (args.skip_existing and calibrated.exists()):
            prompts = str(PROMPT_DIR / source_tag.replace("__mistral", ""))
            valid = infer.infer(view_tag, "valid", model_id=MODEL_ID,
                                prompts_dir=prompts, output_tag=output_tag)
            test = infer.infer(view_tag, "test", model_id=MODEL_ID,
                               prompts_dir=prompts, output_tag=output_tag)
            calibrated = run_cv_grid._platt_calibrate(valid, test, output_dir)
            evaluate.evaluate(calibrated, threshold=0.5, bootstrap=1000)
        calibrated_rows.extend(read_jsonl(calibrated))

    ids = [str(row["patient_id"]) for row in calibrated_rows]
    if len(calibrated_rows) != 147 or len(set(ids)) != 147:
        raise RuntimeError(f"expected 147 unique OOF rows, got {len(ids)}/{len(set(ids))}")

    out = RESULT_DIR / f"ExpC_TxVLM__beep__colbert__mistral__cv__v3_structured__step{args.step}_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    calibrated_path = out / "oof_predictions_platt.jsonl"
    write_jsonl(calibrated_path, calibrated_rows)
    calibrated_metrics = evaluate.evaluate(calibrated_path, threshold=0.5, bootstrap=2000)
    (out / "oof_metrics.json").write_text(json.dumps(calibrated_metrics, indent=2))

    raw_rows = []
    for row in calibrated_rows:
        raw = dict(row)
        raw["score"] = float(row["score_raw"])
        raw_rows.append(raw)
    raw_path = out / "oof_predictions_raw.jsonl"
    write_jsonl(raw_path, raw_rows)
    raw_metrics = evaluate.evaluate(raw_path, threshold=0.5, bootstrap=2000)
    (out / "oof_metrics_raw.json").write_text(json.dumps(raw_metrics, indent=2))

    print(json.dumps({
        "step": args.step,
        "raw_AUROC": raw_metrics["AUROC"],
        "raw_AUPRC": raw_metrics["AUPRC"],
        "calibrated_Macro_F1": calibrated_metrics["Macro_F1"],
        "calibrated_Accuracy": calibrated_metrics["Accuracy"],
        "calibrated_Brier": calibrated_metrics["Brier"],
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
