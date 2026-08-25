"""
Train class-weighted XGBoost on frozen LLM embeddings.

This is the recommended leakage-safe LLM-aware imbalance ablation:

    RAG prompt -> frozen Mistral/BioMistral embedding -> weighted XGBoost

No OOF predictions are needed because the frozen encoder was not trained on
First_Recur labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise SystemExit(
        "xgboost is not installed in this Python environment. Use "
        "the environment documented in README.md."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "dataset" / "first"
EMB_ROOT = ROOT / "Imbalance" / "embeddings"
OUT_DIR = ROOT / "Imbalance" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "aggregate_llm_xgb.csv"
SPLIT_DIR = DATA_ROOT / "splits"
SPLIT_ASSIGN = SPLIT_DIR / "split_assignments.csv"
CLINICAL_FILE = {"train": "Train.csv", "valid": "Validation.csv", "test": "Test.csv"}
FEATURE_GROUPS_JSON = DATA_ROOT / "Processed" / "feature_groups.json"


def _feature_group_for_cell(cell_tag: str) -> tuple[str, list[str]]:
    """Return the intended clinical feature group for a cell tag."""
    if cell_tag.startswith("Exp1__"):
        group = "Exp1_metadata"
    elif cell_tag.startswith("Exp2__"):
        group = "Exp2_metadata_molecular"
    elif cell_tag.startswith("Exp3__"):
        group = "Exp3_metadata_treatment"
    elif cell_tag.startswith("Exp4__"):
        group = "Exp4_metadata_molecular_treatment"
    else:
        raise ValueError(f"Cannot infer feature group from cell_tag={cell_tag!r}")
    groups = json.loads(FEATURE_GROUPS_JSON.read_text())
    return group, groups[group]


def _load_clinical_features(pids: list[str], split: str, feature_cols: list[str]) -> np.ndarray:
    """Load structured clinical features aligned to `pids` order."""
    df = pd.read_csv(SPLIT_DIR / CLINICAL_FILE[split])
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df = df.set_index("Patient_ID")
    if "y" in df.columns:
        df = df.drop(columns=["y"])
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Clinical feature columns missing from {split}: {missing}")
    aligned = df.loc[pids, feature_cols].copy()
    return aligned


def _encode_clinical(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit one-hot feature columns on train, then align valid/test.

    This avoids peeking at validation/test categories when defining the
    structured clinical feature space.
    """
    train_raw = train_df.replace({pd.NA: np.nan})
    valid_raw = valid_df.replace({pd.NA: np.nan})
    test_raw = test_df.replace({pd.NA: np.nan})

    numeric_cols = [
        c for c in train_raw.columns
        if pd.api.types.is_numeric_dtype(train_raw[c])
    ]
    categorical_cols = [c for c in train_raw.columns if c not in numeric_cols]

    train_num = train_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(-999.0)
    valid_num = valid_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(-999.0)
    test_num = test_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(-999.0)

    train_cat = pd.get_dummies(train_raw[categorical_cols], dummy_na=True)
    cat_columns = list(train_cat.columns)
    valid_cat = pd.get_dummies(valid_raw[categorical_cols], dummy_na=True)
    valid_cat = valid_cat.reindex(columns=cat_columns, fill_value=0.0)
    test_cat = pd.get_dummies(test_raw[categorical_cols], dummy_na=True)
    test_cat = test_cat.reindex(columns=cat_columns, fill_value=0.0)

    train_enc = pd.concat([train_num, train_cat], axis=1)
    valid_enc = pd.concat([valid_num, valid_cat], axis=1)
    test_enc = pd.concat([test_num, test_cat], axis=1)
    columns = list(train_enc.columns)

    return (
        train_enc.to_numpy(dtype=np.float32),
        valid_enc.to_numpy(dtype=np.float32),
        test_enc.to_numpy(dtype=np.float32),
        columns,
    )


def _load_split(emb_dir: Path, split: str):
    arr = np.load(emb_dir / f"{split}.npy").astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        print(f"[WARN] {emb_dir.name}/{split}: replacing {n_bad} non-finite embedding values with 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    ids = pd.read_csv(emb_dir / f"{split}_ids.csv")
    return arr, ids["label"].astype(int).to_numpy(), ids["Patient_ID"].astype(str).tolist()


def _verify_no_split_drift(pids_train, pids_valid, pids_test):
    if not SPLIT_ASSIGN.exists():
        return
    sa = pd.read_csv(SPLIT_ASSIGN)
    sa_map = dict(zip(sa["Patient_ID"].astype(str), sa["split"]))
    for pids, expected in [
        (pids_train, "Train"),
        (pids_valid, "Validation"),
        (pids_test, "Test"),
    ]:
        bad = [p for p in pids if sa_map.get(p) != expected]
        if bad:
            raise AssertionError(
                f"Split drift in {expected}: {bad[:5]} have unexpected split assignments."
            )


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().to_dict()
    n = len(y)
    k = len(counts)
    return np.asarray([n / (k * counts[int(v)]) for v in y], dtype=np.float32)


def _bootstrap_ci(metric_fn, y, p, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _metrics(y, p, tau):
    pred = (p >= tau).astype(int)
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    auroc_lo, auroc_hi = _bootstrap_ci(roc_auc_score, y, p)
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y, pred))
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1) if (tp + fp) else 0.0
    mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0
    brier = float(np.mean((p - y) ** 2))
    return {
        "AUROC": round(auroc, 4),
        "AUROC_lo": round(auroc_lo, 4),
        "AUROC_hi": round(auroc_hi, 4),
        "AUPRC": round(auprc, 4),
        "Macro_F1": round(f1, 4),
        "Accuracy": round(acc, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "PPV": round(ppv, 4),
        "MCC": round(mcc, 4),
        "Brier": round(brier, 4),
    }


def _tune_tau(y_val, p_val) -> tuple[float, float]:
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.arange(0.05, 0.96, 0.01):
        score = f1_score(y_val, (p_val >= tau).astype(int),
                         average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_tau = float(score), float(tau)
    return round(best_tau, 2), round(best_f1, 4)


def _maybe_pca(X_tr, X_va, X_te, pca_dim: int):
    if pca_dim <= 0:
        return X_tr, X_va, X_te, "none", X_tr.shape[1]
    dim = min(pca_dim, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    return (
        pca.fit_transform(X_tr),
        pca.transform(X_va),
        pca.transform(X_te),
        f"pca_{dim}",
        dim,
    )


def _resolve_emb_dir(encoder: str, cell_tag: str, pooling: str) -> Path:
    """Return the embedding dir for (encoder, cell, pooling).

    New layout: embeddings/<encoder>/<pooling>/<cell_tag>/
    Legacy fallback (only for pooling='last'): embeddings/<encoder>/<cell_tag>/
    """
    new_dir = EMB_ROOT / encoder / pooling / cell_tag
    if new_dir.exists():
        return new_dir
    if pooling == "last":
        legacy = EMB_ROOT / encoder / cell_tag
        if legacy.exists():
            return legacy
    raise FileNotFoundError(
        f"No embeddings for {encoder}/{pooling}/{cell_tag}. "
        f"Run extract_embeddings.py --pooling {pooling} (or --pooling all) first."
    )


def run_one_cell(cell_tag: str, encoder: str, pca_dim: int = 64,
                 include_clinical: bool = False,
                 pooling: str = "last") -> dict:
    emb_dir = _resolve_emb_dir(encoder, cell_tag, pooling)
    X_tr, y_tr, pid_tr = _load_split(emb_dir, "train")
    X_va, y_va, pid_va = _load_split(emb_dir, "valid")
    X_te, y_te, pid_te = _load_split(emb_dir, "test")
    _verify_no_split_drift(pid_tr, pid_va, pid_te)

    raw_dim = int(X_tr.shape[1])
    X_tr, X_va, X_te, reduction, final_dim = _maybe_pca(X_tr, X_va, X_te, pca_dim)

    feature_mode = "embedding"
    n_clinical_features = 0
    clinical_group = ""
    if include_clinical:
        clinical_group, feature_cols = _feature_group_for_cell(cell_tag)
        clin_tr = _load_clinical_features(pid_tr, "train", feature_cols)
        clin_va = _load_clinical_features(pid_va, "valid", feature_cols)
        clin_te = _load_clinical_features(pid_te, "test", feature_cols)
        Xc_tr, Xc_va, Xc_te, clin_cols = _encode_clinical(clin_tr, clin_va, clin_te)
        n_clinical_features = Xc_tr.shape[1]
        X_tr = np.concatenate([X_tr, Xc_tr], axis=1)
        X_va = np.concatenate([X_va, Xc_va], axis=1)
        X_te = np.concatenate([X_te, Xc_te], axis=1)
        final_dim = int(X_tr.shape[1])
        feature_mode = "embedding+clinical"

    weights = _balanced_sample_weights(y_tr)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=4,
        device="cpu",
    )
    model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_va, y_va)], verbose=False)

    p_val = model.predict_proba(X_va)[:, 1]
    p_test = model.predict_proba(X_te)[:, 1]
    tau_star, valid_f1 = _tune_tau(y_va, p_val)
    m_default = _metrics(y_te, p_test, tau=0.5)
    m_tuned = _metrics(y_te, p_test, tau=tau_star)

    row = {
        "cell_tag": cell_tag,
        "encoder": encoder,
        "pooling": pooling,
        "feature_mode": feature_mode,
        "clinical_group": clinical_group,
        "train_score_source": "frozen_embedding_no_oof_needed",
        "reduction": reduction,
        "n_features_raw": raw_dim,
        "n_clinical_features": int(n_clinical_features),
        "n_features": int(final_dim),
        "n_train": int(len(y_tr)),
        "n_train_pos": int((y_tr == 1).sum()),
        "n_train_neg": int((y_tr == 0).sum()),
        "tau_star": tau_star,
        "valid_macro_f1_at_tau_star": valid_f1,
        **{f"{k}_default": v for k, v in m_default.items()},
        **{f"{k}_tuned": v for k, v in m_tuned.items()},
    }

    out_suffix = "__plus_clinical" if include_clinical else ""
    pool_suffix = "" if pooling == "last" else f"__{pooling}"
    out_cell = OUT_DIR / f"llm_xgb_{encoder}{pool_suffix}" / f"{cell_tag}{out_suffix}"
    out_cell.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Patient_ID": pid_te,
        "label": y_te,
        "score": p_test,
        "pred_default": (p_test >= 0.5).astype(int),
        "pred_tuned": (p_test >= tau_star).astype(int),
    }).to_csv(out_cell / "predictions_test.csv", index=False)
    (out_cell / "metrics.json").write_text(json.dumps(row, indent=2))

    print(
        f"[{encoder}/{pooling}/{cell_tag}] AUROC={m_default['AUROC']:.4f} "
        f"F1@0.5={m_default['Macro_F1']:.4f} "
        f"F1@tau*({tau_star})={m_tuned['Macro_F1']:.4f} "
        f"Spec={m_tuned['Specificity']:.3f} Sens={m_tuned['Sensitivity']:.3f}"
    )
    return row


def list_extracted_cells(encoder: str, pooling: str = "last") -> list[str]:
    """Cells that have a test.npy under either the new or legacy layout."""
    base_new = EMB_ROOT / encoder / pooling
    base_legacy = EMB_ROOT / encoder
    cells: set[str] = set()
    if base_new.exists():
        cells.update(d.name for d in base_new.iterdir()
                     if d.is_dir() and (d / "test.npy").exists())
    if pooling == "last" and base_legacy.exists():
        cells.update(d.name for d in base_legacy.iterdir()
                     if d.is_dir() and d.name not in {"last", "mean", "max"}
                     and (d / "test.npy").exists())
    return sorted(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--encoder", default="mistral",
                    help="encoder slug matching a subdir under Imbalance/embeddings/ "
                         "(e.g. mistral, biomistral, gemma-3-9b-it, qwen3-8b)")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="0 disables PCA; default mitigates 4096-d overfit risk")
    ap.add_argument("--include-clinical", action="store_true",
                    help="concatenate structured clinical features (post-PCA) for hybrid: "
                         "RAG -> frozen embedding + clinical -> weighted XGBoost")
    ap.add_argument("--pooling", default="last", choices=["last", "mean", "max"],
                    help="which pooled LLM embedding to feed XGBoost. Run separately "
                         "for each to compare last/mean/max as an ablation.")
    args = ap.parse_args()

    if not args.cell_tag and not args.all:
        ap.error("supply --cell-tag or --all")
    cells = (list_extracted_cells(args.encoder, args.pooling) if args.all
             else [args.cell_tag])
    print(f"Running weighted XGBoost on {args.encoder} embeddings "
          f"(pooling={args.pooling}) for {len(cells)} cell(s)")

    rows = []
    failures = 0
    for cell in cells:
        try:
            rows.append(run_one_cell(cell, args.encoder, args.pca_dim,
                                     include_clinical=args.include_clinical,
                                     pooling=args.pooling))
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {cell}: {type(exc).__name__}: {exc}")

    if rows:
        new = pd.DataFrame(rows)
        key_cols = ["cell_tag", "encoder", "pooling", "feature_mode", "reduction"]
        if OUT_CSV.exists():
            old = pd.read_csv(OUT_CSV)
            for col in key_cols:
                if col not in old.columns:
                    if col == "feature_mode":
                        old[col] = "embedding"
                    elif col == "pooling":
                        # Existing rows in the aggregate were last-token by definition.
                        old[col] = "last"
                    else:
                        old[col] = ""
            key_new = set(new[key_cols].apply(tuple, axis=1))
            keep = ~old[key_cols].apply(tuple, axis=1).isin(key_new)
            combined = pd.concat([old[keep], new], ignore_index=True)
        else:
            combined = new
        combined.to_csv(OUT_CSV, index=False)
        print(f"Aggregate -> {OUT_CSV} ({len(combined)} rows)")
    elif failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
