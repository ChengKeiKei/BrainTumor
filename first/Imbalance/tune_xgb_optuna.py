"""Bayesian hyperparameter search for hybrid LLM + XGBoost cells.

For each (encoder, pooling, cell_tag) cell, this script:

  1. Loads the frozen LLM embedding + structured clinical features.
  2. Runs Optuna Bayesian optimisation (TPE sampler) over 7 hyperparameters
     including `pca_dim`. The objective is **validation logloss**.
     PCA is re-fit on train only for every trial; valid/test are never used
     to fit transforms.
  3. Retrains XGBoost with the best config on (train), early-stopped on
     (valid), then reports **test-set** metrics. Test never enters the
     search loop.

Outputs:

    results/tuning/<encoder>__<pooling>/<cell_tag>__study.db   (Optuna study)
    results/tuning/<encoder>__<pooling>/<cell_tag>__best.json  (best config + test metrics)
    results/tuning_summary.csv                                  (one row per cell)

Notes:

* Catastrophic mean-pool cells (Mistral Exp4 ColBERT/MiniLM) are skipped
  by default because their embeddings have float16 overflow Infs.
  Override with `--include-bad-mean`.

* Run examples:

    # quick smoke test (1 cell, 5 trials)
    python tune_xgb_optuna.py --encoder mistral --pooling last \\
        --cell-tag Exp3__beep__medcpt --n-trials 5

    # full run across every available cell (background)
    python tune_xgb_optuna.py --all --n-trials 30
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# Reuse the leakage-safe helpers from run_llm_xgb so the data pipeline is
# identical to the headline experiments.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llm_xgb import (  # noqa: E402
    EMB_ROOT,
    OUT_DIR,
    _balanced_sample_weights,
    _encode_clinical,
    _feature_group_for_cell,
    _load_clinical_features,
    _load_split,
    _metrics,
    _resolve_emb_dir,
    _tune_tau,
    _verify_no_split_drift,
    list_extracted_cells,
)

TUNE_DIR = OUT_DIR / "tuning"
TUNE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "tuning_summary.csv"

# These cells have float16 overflow Infs in the mean-pool extraction. Skip
# until float32 re-extraction is done.
BAD_MEAN_CELLS = {
    ("mistral", "mean", "Exp4__beep__colbert"),
    ("mistral", "mean", "Exp4__beep__minilm"),
}

POOLINGS = ("last", "mean", "max")
ENCODERS = ("mistral", "biomistral")


# ----------------------------------------------------------------------------
# Data loader (one (encoder, pooling, cell) -> arrays for train/valid/test)
# ----------------------------------------------------------------------------
def load_cell_arrays(encoder: str, pooling: str, cell_tag: str):
    emb_dir = _resolve_emb_dir(encoder, cell_tag, pooling)
    Xe_tr, y_tr, pid_tr = _load_split(emb_dir, "train")
    Xe_va, y_va, pid_va = _load_split(emb_dir, "valid")
    Xe_te, y_te, pid_te = _load_split(emb_dir, "test")
    _verify_no_split_drift(pid_tr, pid_va, pid_te)

    # Always include clinical features (this is the hybrid setup).
    clinical_group, feature_cols = _feature_group_for_cell(cell_tag)
    clin_tr = _load_clinical_features(pid_tr, "train", feature_cols)
    clin_va = _load_clinical_features(pid_va, "valid", feature_cols)
    clin_te = _load_clinical_features(pid_te, "test", feature_cols)
    Xc_tr, Xc_va, Xc_te, _ = _encode_clinical(clin_tr, clin_va, clin_te)

    return {
        "Xe_tr": Xe_tr, "Xe_va": Xe_va, "Xe_te": Xe_te,
        "Xc_tr": Xc_tr, "Xc_va": Xc_va, "Xc_te": Xc_te,
        "y_tr": y_tr, "y_va": y_va, "y_te": y_te,
        "pid_te": pid_te,
        "clinical_group": clinical_group,
    }


# ----------------------------------------------------------------------------
# Optuna objective
# ----------------------------------------------------------------------------
def make_objective(data: dict):
    """Closure over loaded arrays so each trial only re-runs PCA + XGBoost."""

    Xe_tr, Xe_va = data["Xe_tr"], data["Xe_va"]
    Xc_tr, Xc_va = data["Xc_tr"], data["Xc_va"]
    y_tr, y_va = data["y_tr"], data["y_va"]
    weights_tr = _balanced_sample_weights(y_tr)

    def objective(trial: optuna.Trial) -> float:
        pca_dim = trial.suggest_categorical("pca_dim", [16, 32, 64, 96, 128])
        max_depth = trial.suggest_int("max_depth", 2, 6)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.5, 10.0, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 8)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        # Fit PCA on train embeddings only.
        dim = min(int(pca_dim), Xe_tr.shape[0] - 1, Xe_tr.shape[1])
        pca = PCA(n_components=dim, random_state=42)
        Pe_tr = pca.fit_transform(Xe_tr)
        Pe_va = pca.transform(Xe_va)

        X_tr = np.concatenate([Pe_tr, Xc_tr], axis=1)
        X_va = np.concatenate([Pe_va, Xc_va], axis=1)

        model = XGBClassifier(
            n_estimators=400,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=4,
            device="cpu",
            verbosity=0,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight=weights_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        p_va = model.predict_proba(X_va)[:, 1]
        # Tiny clipping to avoid log(0); logloss is the search objective.
        p_va = np.clip(p_va, 1e-6, 1 - 1e-6)
        loss = float(log_loss(y_va, p_va))

        trial.set_user_attr("best_iteration", int(model.best_iteration or 0))
        return loss

    return objective


# ----------------------------------------------------------------------------
# Final retrain + test-set evaluation with best config
# ----------------------------------------------------------------------------
def fit_with_best(data: dict, best: dict) -> dict:
    Xe_tr, Xe_va, Xe_te = data["Xe_tr"], data["Xe_va"], data["Xe_te"]
    Xc_tr, Xc_va, Xc_te = data["Xc_tr"], data["Xc_va"], data["Xc_te"]
    y_tr, y_va, y_te = data["y_tr"], data["y_va"], data["y_te"]
    weights_tr = _balanced_sample_weights(y_tr)

    dim = min(int(best["pca_dim"]), Xe_tr.shape[0] - 1, Xe_tr.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    Pe_tr = pca.fit_transform(Xe_tr)
    Pe_va = pca.transform(Xe_va)
    Pe_te = pca.transform(Xe_te)

    X_tr = np.concatenate([Pe_tr, Xc_tr], axis=1)
    X_va = np.concatenate([Pe_va, Xc_va], axis=1)
    X_te = np.concatenate([Pe_te, Xc_te], axis=1)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        min_child_weight=int(best["min_child_weight"]),
        reg_lambda=float(best["reg_lambda"]),
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=4,
        device="cpu",
        verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=weights_tr,
              eval_set=[(X_va, y_va)], verbose=False)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    tau_star, valid_f1 = _tune_tau(y_va, p_va)
    m_def = _metrics(y_te, p_te, tau=0.5)
    m_tun = _metrics(y_te, p_te, tau=tau_star)

    return {
        "best_iteration": int(model.best_iteration or 0),
        "tau_star": tau_star,
        "valid_macro_f1_at_tau_star": valid_f1,
        "pca_dim_effective": int(dim),
        **{f"{k}_default": v for k, v in m_def.items()},
        **{f"{k}_tuned": v for k, v in m_tun.items()},
    }


# ----------------------------------------------------------------------------
# Per-cell driver
# ----------------------------------------------------------------------------
def tune_one_cell(encoder: str, pooling: str, cell_tag: str,
                  n_trials: int, seed: int = 42, timeout_sec: int | None = None) -> dict:
    print(f"\n=== TUNE  encoder={encoder}  pooling={pooling}  cell={cell_tag} "
          f"  trials={n_trials} ===")
    data = load_cell_arrays(encoder, pooling, cell_tag)
    n_tr = len(data["y_tr"])
    n_va = len(data["y_va"])
    n_te = len(data["y_te"])
    print(f"  n_train={n_tr}  n_valid={n_va}  n_test={n_te}  "
          f"emb_dim={data['Xe_tr'].shape[1]}  clinical_dim={data['Xc_tr'].shape[1]}")

    out_dir = TUNE_DIR / f"{encoder}__{pooling}"
    out_dir.mkdir(parents=True, exist_ok=True)
    study_path = out_dir / f"{cell_tag}__study.db"
    storage = f"sqlite:///{study_path}"

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed, n_startup_trials=8),
        storage=storage,
        study_name=f"{encoder}__{pooling}__{cell_tag}",
        load_if_exists=True,
    )

    objective = make_objective(data)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    best_params = study.best_trial.params
    best_loss = float(study.best_value)
    final = fit_with_best(data, best_params)

    row = {
        "encoder": encoder,
        "pooling": pooling,
        "cell_tag": cell_tag,
        "clinical_group": data["clinical_group"],
        "n_train": n_tr, "n_valid": n_va, "n_test": n_te,
        "best_valid_logloss": round(best_loss, 6),
        **{f"best_{k}": v for k, v in best_params.items()},
        **final,
    }

    best_json = out_dir / f"{cell_tag}__best.json"
    best_json.write_text(json.dumps(row, indent=2, default=str))
    print(f"  -> best valid logloss = {best_loss:.4f}")
    print(f"  -> test AUROC default = {final['AUROC_default']}  tuned = {final['AUROC_tuned']}")
    print(f"  -> wrote {best_json}")
    return row


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def discover_cells(encoders, poolings, include_bad_mean=False):
    out = []
    for enc in encoders:
        for pool in poolings:
            cells = list_extracted_cells(enc, pool)
            for c in cells:
                if (not include_bad_mean) and (enc, pool, c) in BAD_MEAN_CELLS:
                    print(f"[SKIP bad-mean] {enc}/{pool}/{c}")
                    continue
                out.append((enc, pool, c))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default=None, choices=[*ENCODERS])
    ap.add_argument("--pooling", default=None, choices=[*POOLINGS])
    ap.add_argument("--cell-tag", default=None)
    ap.add_argument("--all", action="store_true",
                    help="run across every (encoder, pooling, cell) combination")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout-sec", type=int, default=None,
                    help="per-cell wall-time cap")
    ap.add_argument("--include-bad-mean", action="store_true",
                    help="do NOT skip catastrophic mean-pool cells")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all:
        targets = discover_cells(ENCODERS, POOLINGS, args.include_bad_mean)
    else:
        if not (args.encoder and args.pooling and args.cell_tag):
            ap.error("Provide --encoder, --pooling, --cell-tag OR use --all.")
        targets = [(args.encoder, args.pooling, args.cell_tag)]
        if (not args.include_bad_mean) and tuple(targets[0]) in BAD_MEAN_CELLS:
            print(f"[SKIP bad-mean] {targets[0]}; pass --include-bad-mean to override.")
            return

    print(f"Will tune {len(targets)} cell(s); {args.n_trials} trials each.")
    rows = []
    failures = 0
    for i, (enc, pool, cell) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}]")
        try:
            rows.append(tune_one_cell(enc, pool, cell,
                                      n_trials=args.n_trials,
                                      timeout_sec=args.timeout_sec,
                                      seed=args.seed))
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {enc}/{pool}/{cell}: {type(exc).__name__}: {exc}")

    if rows:
        new_df = pd.DataFrame(rows)
        if SUMMARY_CSV.exists():
            old = pd.read_csv(SUMMARY_CSV)
            key = ["encoder", "pooling", "cell_tag"]
            new_keys = set(new_df[key].apply(tuple, axis=1))
            keep = ~old[key].apply(tuple, axis=1).isin(new_keys)
            combined = pd.concat([old[keep], new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(SUMMARY_CSV, index=False)
        print(f"\nAggregate tuning summary -> {SUMMARY_CSV}  ({len(combined)} rows)")
    print(f"DONE  rows={len(rows)}  failures={failures}")


if __name__ == "__main__":
    main()
