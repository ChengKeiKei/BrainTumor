"""Utilities for First_Recur imbalance analysis.

This module intentionally avoids SMOTE/T-SMOTE because the First_Recur
clinical features include temporal day variables. The safer default is to keep
real patient rows unchanged and use training-time class/sample weights.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL prompt file."""
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_label_counts(path: Path) -> dict[int, int]:
    """Count labels in a prompt JSONL file."""
    records = read_jsonl(path)
    return dict(Counter(int(r["label"]) for r in records))


def balanced_sample_weights(labels: Iterable[int]) -> np.ndarray:
    """Return inverse-frequency sample weights with mean weight near 1.

    For binary labels, each class receives total weight n_samples / n_classes.
    These weights can be passed to XGBoost/LightGBM/sklearn estimators without
    changing the original patient rows or timeline values.
    """
    y = pd.Series([int(v) for v in labels])
    counts = y.value_counts().to_dict()
    n = len(y)
    k = len(counts)
    return y.map({cls: n / (k * cnt) for cls, cnt in counts.items()}).to_numpy()


def xgboost_scale_pos_weight(labels: Iterable[int], positive_label: int = 1) -> float:
    """Return XGBoost's scale_pos_weight = n_negative / n_positive."""
    y = pd.Series([int(v) for v in labels])
    n_pos = int((y == positive_label).sum())
    n_neg = int((y != positive_label).sum())
    if n_pos == 0:
        raise ValueError("Cannot compute scale_pos_weight with zero positive labels.")
    return n_neg / n_pos


def split_class_table(split_dir: Path) -> pd.DataFrame:
    """Summarise class counts in First_Recur split CSVs."""
    rows = []
    for split in ("Train", "Validation", "Test"):
        df = pd.read_csv(Path(split_dir) / f"{split}.csv")
        y = df["y"].astype(int)
        rows.append({
            "split": split,
            "n": len(df),
            "y_no": int((y == 0).sum()),
            "y_yes": int((y == 1).sum()),
            "positive_rate": float(y.mean()),
            "minority_ratio_no_to_yes": (
                float((y == 0).sum() / max((y == 1).sum(), 1))
            ),
        })
    return pd.DataFrame(rows)
