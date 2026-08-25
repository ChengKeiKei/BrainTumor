"""
run_cv_grid.py — Stratified patient-level 5-fold cross-validation for Second_Recur.

Why
---
The held-out test set is only 23 patients. AUROC point estimates with such
a small n have wide CIs (±0.1 or worse), and F1 is even more brittle.
Reviewers will rightly call this fragile. Cross-validation pools all 147
patients into 5 stratified folds, trains 5 separate LoRA adapters, and
reports OOF (out-of-fold) metrics over all 147 predictions.

Trade-off: 5× the training time of a single cell. ~30–60 min/fold on M-series
chips. For the full ExpC × 5 rerankers × 3 caption variants × 5 folds = 75
cells, that's ~30 h. So this script is designed to CV ONE cell at a time
(default = your best cell, but configurable via --tag or the (--exp/--ret/
--rrk/--captions-version) quartet).

Output layout (mirrors the test-only pipeline, with `__cv` and `__foldN` suffixes):

    Dataset/splits_cv/foldN/{Train,Validation,Test}.csv   # 1 fold split
    Model/prompts/<tag>__cv__foldN/{train,valid,test}.jsonl
    Model/checkpoints/<tag>__cv__foldN/adapters/
    Model/results/<tag>__cv__foldN/predictions_test.jsonl
    Model/results/<tag>__cv__foldN/predictions_test.metrics.json

Then the script aggregates all 5 folds into:

    Model/results/<tag>__cv/oof_predictions.jsonl   # 147 OOF rows
    Model/results/<tag>__cv/oof_metrics.json        # AUROC + F1 over OOF
    Model/results/aggregate_cv.csv                  # one row per CV'd cell

Usage:
    # CV the best cell (ExpC_TxVLM__beep__medcpt__biomistral__v3_structured)
    python Model/code/run_cv_grid.py --tag ExpC_TxVLM__beep__medcpt__biomistral \\
                                     --captions-version v3_structured

    # CV ExpA_Tx baseline with medcpt (no captions)
    python Model/code/run_cv_grid.py --exp ExpA_Tx --rrk medcpt

    # Just generate the fold split CSVs without training
    python Model/code/run_cv_grid.py --make-folds-only

This script is meant to be run AFTER the held-out pipeline (run_grid.py) is
already finalised. The held-out test results stay as the primary reported
number; CV provides confirmation with smaller error bars.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

ROOT      = Path(__file__).resolve().parents[2]    # Second_Recur/
DATASET_D = ROOT.parent / "dataset" / "second"
SPLITS_D  = DATASET_D / "splits"
CV_DIR    = DATASET_D / "splits_cv"               # one folder per fold
MODEL_DIR = ROOT / "Model"
RESULT_D  = MODEL_DIR / "results"
PROMPT_D  = MODEL_DIR / "prompts"

DEFAULT_FOLDS = 5
DEFAULT_SEED  = 42


# ---------------------------------------------------------------------------
# Step 1 — fold split generation
# ---------------------------------------------------------------------------
def make_folds(n_folds: int = DEFAULT_FOLDS, seed: int = DEFAULT_SEED) -> Path:
    """Pool Train+Validation+Test, then write n_folds stratified splits.

    Within each fold we further peel off ~15% as validation (used for
    threshold-tuning / early-stop), so each fold writes Train / Validation
    / Test CSVs. Test = the fold itself; Train+Validation = the remainder.
    """
    train = pd.read_csv(SPLITS_D / "Train.csv")
    valid = pd.read_csv(SPLITS_D / "Validation.csv")
    test  = pd.read_csv(SPLITS_D / "Test.csv")
    pool  = pd.concat([train, valid, test], ignore_index=True)
    print(f"[make_folds] pool size: {len(pool)} patients "
          f"({int(pool.y.sum())} y=1 / {int((pool.y==0).sum())} y=0)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pool_idx = pool.index.to_numpy()
    pool_y   = pool["y"].astype(int).to_numpy()

    if CV_DIR.exists():
        shutil.rmtree(CV_DIR)
    CV_DIR.mkdir(parents=True)

    rng = np.random.default_rng(seed)

    for fold_id, (rest_idx, test_idx) in enumerate(skf.split(pool_idx, pool_y)):
        rest = pool.iloc[rest_idx].reset_index(drop=True)
        # Stratified val carve-out from the rest (~15% of the rest).
        rest_pos = rest[rest.y == 1].index.to_numpy()
        rest_neg = rest[rest.y == 0].index.to_numpy()
        n_val_pos = max(1, int(round(0.15 * len(rest_pos))))
        n_val_neg = max(1, int(round(0.15 * len(rest_neg))))
        val_pos   = rng.choice(rest_pos, size=n_val_pos, replace=False)
        val_neg   = rng.choice(rest_neg, size=n_val_neg, replace=False)
        val_mask  = np.zeros(len(rest), dtype=bool)
        val_mask[np.concatenate([val_pos, val_neg])] = True

        fold_dir = CV_DIR / f"fold{fold_id}"
        fold_dir.mkdir(parents=True)
        rest[~val_mask].to_csv(fold_dir / "Train.csv",      index=False)
        rest[val_mask] .to_csv(fold_dir / "Validation.csv", index=False)
        pool.iloc[test_idx].to_csv(fold_dir / "Test.csv",   index=False)

        n_tr = (~val_mask).sum(); n_vl = val_mask.sum(); n_te = len(test_idx)
        n_tr_pos = int(rest.iloc[~val_mask][rest.iloc[~val_mask, :].columns.get_loc("y")].sum() if False else rest[~val_mask].y.sum())
        n_te_pos = int(pool.iloc[test_idx].y.sum())
        print(f"  fold{fold_id}: train={n_tr} (pos={n_tr_pos})  "
              f"val={n_vl}  test={n_te} (pos={n_te_pos})")
    print(f"[make_folds] wrote → {CV_DIR}/foldN/(Train|Validation|Test).csv")
    return CV_DIR


# ---------------------------------------------------------------------------
# Step 2 — CV one cell
# ---------------------------------------------------------------------------
def _cv_tag(base_tag: str, fold_id: int) -> str:
    return f"{base_tag}__cv__fold{fold_id}"


def _build_one_fold(fold_id: int, exp: str, ret: str, rrk: str,
                    captions_version: str = "v1",
                    top_k_sparse: int = 10, top_k_dense: int = 10,
                    final_top_k: int = 3) -> str:
    """Repoint feature_render.SPLITS_DIR at the fold dir, then rebuild prompts.

    Returns the *prompt* tag for the fold (LLM-agnostic, mirrors how
    ``run_grid.py`` names its prompt directories: no ``__biomistral`` suffix).
    The caller is responsible for the LLM-suffixed *results* tag used by
    train/infer.
    """
    import build_dataset
    import feature_render

    fold_split = CV_DIR / f"fold{fold_id}"
    if not fold_split.exists():
        raise FileNotFoundError(f"Fold split not found: {fold_split}. "
                                f"Run --make-folds-only first.")

    feature_render.SPLITS_DIR = fold_split
    os.environ["RAG_CAPTIONS_VERSION"] = captions_version

    base_prompt_tag = build_dataset.config_tag(exp, ret, rrk)
    prompt_tag = _cv_tag(base_prompt_tag, fold_id)
    if captions_version != "v1":
        prompt_tag = f"{prompt_tag}__{captions_version}"

    # build_one_config writes prompts under PROMPT_D / <unsuffixed base_prompt_tag>.
    # Move it to the fold-suffixed name so subsequent fold rebuilds don't clobber.
    build_dataset.build_one_config(
        exp, ret, rrk,
        top_k_sparse=top_k_sparse, top_k_dense=top_k_dense,
        final_top_k=final_top_k,
    )
    src = PROMPT_D / base_prompt_tag
    dst = PROMPT_D / prompt_tag
    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)
    return prompt_tag


def _platt_calibrate(valid_path: Path, test_path: Path, out_dir: Path) -> Path:
    """Fit Platt scaling on this fold's validation patients only."""
    def _load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.open()]

    valid_rows = _load(valid_path)
    test_rows = _load(test_path)
    y_valid = np.asarray([int(row["label"]) for row in valid_rows])
    if len(np.unique(y_valid)) != 2:
        raise ValueError("fold validation set must contain both classes for Platt scaling")

    eps = 1e-6
    valid_raw = np.clip(np.asarray([float(row["score"]) for row in valid_rows]), eps, 1 - eps)
    test_raw = np.clip(np.asarray([float(row["score"]) for row in test_rows]), eps, 1 - eps)
    valid_logit = np.log(valid_raw / (1 - valid_raw)).reshape(-1, 1)
    test_logit = np.log(test_raw / (1 - test_raw)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
    calibrator.fit(valid_logit, y_valid)
    valid_cal = calibrator.predict_proba(valid_logit)[:, 1]
    test_cal = calibrator.predict_proba(test_logit)[:, 1]

    out_path = out_dir / "predictions_test_platt.jsonl"
    with out_path.open("w") as fh:
        for row, raw, calibrated in zip(test_rows, test_raw, test_cal):
            result = dict(row)
            result["score_raw"] = float(raw)
            result["score"] = float(calibrated)
            result["calibration"] = "Platt_fold_validation"
            fh.write(json.dumps(result) + "\n")

    calibration = {
        "method": "Platt scaling",
        "fit_split": "fold validation only",
        "valid_n": int(len(y_valid)),
        "valid_n_pos": int(y_valid.sum()),
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "valid_brier_raw": float(brier_score_loss(y_valid, valid_raw)),
        "valid_brier_platt": float(brier_score_loss(y_valid, valid_cal)),
    }
    (out_dir / "platt_calibration.json").write_text(json.dumps(calibration, indent=2))
    return out_path


def _train_and_infer(results_tag: str, prompt_tag: str,
                     model_alias: str, train_kw: dict) -> Path:
    """Train + infer on a fold.

    ``prompt_tag`` is LLM-agnostic and points at PROMPT_D / prompt_tag for the
    JSONL prompt files. ``results_tag`` is LLM-suffixed and is what train.py /
    infer.py use to name the output directories (checkpoints/<results_tag>,
    results/<results_tag>). This separation mirrors run_grid.py.
    """
    import train as train_mod
    import infer as infer_mod
    import evaluate as evaluate_mod

    if model_alias == "biomistral":
        model_id = (MODEL_DIR / "local_models" / "BioMistral-7B-DARE-4bit").as_posix()
    elif model_alias == "mistral":
        model_id = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    else:
        raise ValueError(f"unsupported model alias: {model_alias}")

    prompts_path = str(PROMPT_D / prompt_tag)

    print(f"\n  >>> training {results_tag}  (prompts: {prompt_tag})")
    t0 = time.time()
    train_mod.train(
        tag=results_tag,
        prompts_dir=prompts_path,
        steps=train_kw.get("steps", 250),
        batch_size=train_kw.get("batch_size", 4),
        lr=float(train_kw.get("lr", 2e-5)),
        save_steps=train_kw.get("save_steps", 50),
        eval_steps=train_kw.get("eval_steps", 50),
        logging_steps=train_kw.get("logging_steps", 10),
        model_id=model_id,
    )
    print(f"  train wall = {time.time()-t0:.1f}s")

    valid_path = infer_mod.infer(results_tag, split="valid", model_id=model_id,
                                 prompts_dir=prompts_path)
    raw_test_path = infer_mod.infer(results_tag, split="test",
                                    model_id=model_id,
                                    prompts_dir=prompts_path)
    pred_path = _platt_calibrate(valid_path, raw_test_path, RESULT_D / results_tag)
    metrics = evaluate_mod.evaluate(pred_path, threshold=0.5, bootstrap=1000)
    print(f"  fold Platt AUROC={metrics['AUROC']:.3f}  F1@0.5={metrics['Macro_F1']:.3f}")
    return pred_path


# ---------------------------------------------------------------------------
# Step 3 — aggregate OOF metrics
# ---------------------------------------------------------------------------
def _aggregate_oof(base_tag: str, captions_version: str, n_folds: int) -> dict:
    """Pool predictions_test.jsonl across all folds → 147 OOF rows."""
    rows = []
    for fold_id in range(n_folds):
        target_tag = _cv_tag(base_tag, fold_id)
        if captions_version != "v1":
            target_tag = f"{target_tag}__{captions_version}"
        p = RESULT_D / target_tag / "predictions_test_platt.jsonl"
        if not p.exists():
            print(f"  [skip] missing {p}")
            continue
        for line in p.open():
            rows.append(json.loads(line))

    if not rows:
        return {}
    patient_ids = [r.get("patient_id", r.get("Patient_ID")) for r in rows]
    if len(rows) != 147 or len(set(patient_ids)) != 147:
        raise RuntimeError(
            f"incomplete OOF pool: expected 147 unique patients, got "
            f"{len(rows)} rows / {len(set(patient_ids))} unique"
        )

    out_dir = RESULT_D / f"{base_tag}__cv" if captions_version == "v1" \
              else RESULT_D / f"{base_tag}__cv__{captions_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / "oof_predictions_platt.jsonl"
    with oof_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import evaluate as evaluate_mod
    metrics = evaluate_mod.evaluate(oof_path, threshold=0.5, bootstrap=2000)
    metrics_path = out_dir / "oof_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    raw_path = out_dir / "oof_predictions_raw.jsonl"
    with raw_path.open("w") as f:
        for r in rows:
            raw = dict(r)
            raw["score"] = float(r["score_raw"])
            raw["calibration"] = "None"
            f.write(json.dumps(raw) + "\n")
    raw_metrics = evaluate_mod.evaluate(raw_path, threshold=0.5, bootstrap=2000)
    (out_dir / "oof_metrics_raw.json").write_text(json.dumps(raw_metrics, indent=2))

    # Upsert into aggregate_cv.csv so resumed runs remain idempotent.
    agg_csv = RESULT_D / "aggregate_cv.csv"
    headers = [
        "tag", "calibration", "threshold", "n_oof", "n_pos", "n_neg",
        "raw_AUROC", "raw_AUROC_lo", "raw_AUROC_hi", "raw_AUPRC",
        "calibrated_AUROC", "calibrated_AUPRC", "calibrated_Macro_F1",
        "calibrated_Accuracy", "calibrated_Sensitivity", "calibrated_Specificity",
        "calibrated_MCC", "calibrated_Brier",
    ]
    cv_tag = f"{base_tag}__cv" + (f"__{captions_version}" if captions_version != "v1" else "")
    row = [
        cv_tag, "Platt_fold_validation", 0.5,
        metrics["n"], metrics["n_pos"], metrics["n_neg"],
        raw_metrics["AUROC"], raw_metrics["AUROC_95CI"][0], raw_metrics["AUROC_95CI"][1],
        raw_metrics["AUPRC"], metrics["AUROC"], metrics["AUPRC"], metrics["Macro_F1"],
        metrics["Accuracy"], metrics["Sensitivity"], metrics["Specificity"],
        metrics["MCC"], metrics["Brier"],
    ]
    existing = []
    if agg_csv.exists():
        with agg_csv.open(newline="") as f:
            existing = [r for r in csv.reader(f) if r and r[0] != "tag" and r[0] != cv_tag]
    with agg_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(existing)
        w.writerow(row)
    print(f"\nOOF metrics → {metrics_path}")
    print(f"  Raw AUROC = {raw_metrics['AUROC']:.3f}  "
          f"95% CI [{raw_metrics['AUROC_95CI'][0]:.3f}, {raw_metrics['AUROC_95CI'][1]:.3f}]")
    print(f"  Macro F1 @ calibrated 0.5 = {metrics['Macro_F1']:.3f}")
    return {"raw": raw_metrics, "calibrated": metrics}


# ---------------------------------------------------------------------------
# Step 4 — driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ExpC_TxVLM",
                    help="Experiment block (default: ExpC_TxVLM, the best cell)")
    ap.add_argument("--ret", default="beep")
    ap.add_argument("--rrk", default="medcpt",
                    help="Reranker (default: medcpt, the best on held-out test)")
    ap.add_argument("--llm", default="biomistral", choices=["biomistral", "mistral"])
    ap.add_argument("--captions-version", default="v3_structured",
                    choices=["v1", "v2_context", "v3_structured"])
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--seed",  type=int, default=DEFAULT_SEED)
    ap.add_argument("--make-folds-only", action="store_true",
                    help="Just create Dataset/splits_cv/foldN/ and exit.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip folds whose predictions_test.jsonl already exists.")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    # Step 1 — make folds (idempotent; only re-runs when --make-folds-only set)
    if not (CV_DIR.exists() and (CV_DIR / "fold0").exists()):
        make_folds(n_folds=args.folds, seed=args.seed)
    if args.make_folds_only:
        return

    train_kw = {"steps": args.steps, "batch_size": args.batch_size,
                "lr": args.lr, "save_steps": 50, "eval_steps": 50,
                "logging_steps": 10}

    import build_dataset
    base_prompt_tag  = build_dataset.config_tag(args.exp, args.ret, args.rrk)
    base_results_tag = f"{base_prompt_tag}__{args.llm}"

    print(f"\nCV {args.folds} folds for cell:")
    print(f"  exp={args.exp}  ret={args.ret}  rrk={args.rrk}  "
          f"llm={args.llm}  captions={args.captions_version}")
    print(f"  base_prompt_tag  = {base_prompt_tag}")
    print(f"  base_results_tag = {base_results_tag}")

    cap_suffix = "" if args.captions_version == "v1" else f"__{args.captions_version}"

    for fold_id in range(args.folds):
        prompt_tag  = _cv_tag(base_prompt_tag,  fold_id) + cap_suffix
        results_tag = _cv_tag(base_results_tag, fold_id) + cap_suffix

        if args.skip_existing and (RESULT_D / results_tag / "predictions_test_platt.jsonl").exists():
            print(f"\n  [skip] fold{fold_id} already complete: {results_tag}")
            continue

        print(f"\n{'='*70}\n>>> fold {fold_id+1}/{args.folds}: {results_tag}\n{'='*70}")
        built_prompt_tag = _build_one_fold(
            fold_id, args.exp, args.ret, args.rrk,
            captions_version=args.captions_version,
        )
        assert built_prompt_tag == prompt_tag, \
            f"prompt-tag mismatch: built={built_prompt_tag} expected={prompt_tag}"
        _train_and_infer(results_tag, prompt_tag, args.llm, train_kw)

    print(f"\n{'='*70}\nAggregating OOF metrics across {args.folds} folds...\n{'='*70}")
    _aggregate_oof(base_results_tag, args.captions_version, args.folds)


if __name__ == "__main__":
    main()
