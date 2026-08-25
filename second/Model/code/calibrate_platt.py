"""Fit Platt scaling on validation predictions and evaluate test at 0.5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import evaluate as evaluate_mod


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Model" / "results"
EPS = 1e-6


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def calibrate(tag: str, reinfer: bool = False) -> dict:
    result_dir = RESULTS / tag
    valid_path = result_dir / "predictions_valid.jsonl"
    test_path = result_dir / "predictions_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not valid_path.exists():
        if not reinfer:
            raise FileNotFoundError(valid_path)
        import infer as infer_mod
        valid_path = infer_mod.infer(tag, split="valid")

    valid_rows, test_rows = _load(valid_path), _load(test_path)
    y_valid = np.asarray([int(row["label"]) for row in valid_rows])
    valid_raw = np.clip(np.asarray([float(row["score"]) for row in valid_rows]), EPS, 1 - EPS)
    test_raw = np.clip(np.asarray([float(row["score"]) for row in test_rows]), EPS, 1 - EPS)
    if len(np.unique(y_valid)) != 2:
        raise ValueError(f"{tag}: validation predictions contain only one class")

    logit_valid = np.log(valid_raw / (1 - valid_raw)).reshape(-1, 1)
    logit_test = np.log(test_raw / (1 - test_raw)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
    model.fit(logit_valid, y_valid)
    calibrated = model.predict_proba(logit_test)[:, 1]

    out_path = result_dir / "predictions_test_platt.jsonl"
    with out_path.open("w") as fh:
        for row, raw, score in zip(test_rows, test_raw, calibrated):
            output = dict(row)
            output["score_raw"] = float(raw)
            output["score"] = float(score)
            output["calibration"] = "Platt_validation"
            fh.write(json.dumps(output) + "\n")

    metrics = evaluate_mod.evaluate(out_path, threshold=0.5, bootstrap=2000)
    metadata = {
        "tag": tag,
        "calibration": "Platt_validation",
        "threshold": 0.5,
        "valid_n": int(len(y_valid)),
        "valid_n_pos": int(y_valid.sum()),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        **metrics,
    }
    (result_dir / "predictions_test.metrics_platt_0p5.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--reinfer", action="store_true")
    parser.add_argument("--out", default=str(RESULTS / "aggregate_platt_0p5.csv"))
    args = parser.parse_args()
    rows = []
    for tag in args.tags:
        try:
            rows.append(calibrate(tag, reinfer=args.reinfer))
        except Exception as exc:
            print(f"[FAIL] {tag}: {type(exc).__name__}: {exc}")
    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
