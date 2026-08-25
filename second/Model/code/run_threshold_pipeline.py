"""
run_threshold_pipeline.py — Re-infer validation predictions for every
completed cell, then threshold-tune τ on validation and report the
tuned test metrics.

Writes:
    results/<tag>/predictions_valid.jsonl                  (one per cell)
    results/<tag>/predictions_test.metrics_tuned.json      (one per cell)
    results/aggregate_threshold_tuned.csv                  (summary across cells)

The default-τ aggregate (`results/aggregate.csv`) is left untouched —
the milestone report quotes BOTH numbers transparently.

Usage:
    python code/run_threshold_pipeline.py
    python code/run_threshold_pipeline.py --skip-infer    # if valid preds exist
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling code modules importable
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ROOT      = THIS_DIR.parent.parent              # Second_Recur/
RESULT_D  = ROOT / "Model" / "results"


def list_completed_tags() -> list[str]:
    """Every cell that has a non-archived test prediction file."""
    tags = []
    for d in sorted(RESULT_D.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "predictions_test.jsonl").exists():
            tags.append(d.name)
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-infer", action="store_true",
                    help="Skip Phase 1; assume predictions_valid.jsonl exists.")
    args = ap.parse_args()

    tags = list_completed_tags()
    print(f"Found {len(tags)} completed cells:")
    for t in tags:
        v_done = "(valid OK)" if (RESULT_D / t / "predictions_valid.jsonl").exists() else "(no valid)"
        print(f"  - {t}   {v_done}")

    if not args.skip_infer:
        import infer as infer_mod
        print("\n" + "=" * 60)
        print("Phase 1: Validation inference")
        print("=" * 60)
        for i, tag in enumerate(tags, 1):
            valid_path = RESULT_D / tag / "predictions_valid.jsonl"
            if valid_path.exists():
                print(f"[{i}/{len(tags)}] [skip] {tag} — predictions_valid.jsonl exists")
                continue
            print(f"\n[{i}/{len(tags)}] {tag} → infer split=valid")
            t0 = time.time()
            try:
                infer_mod.infer(tag, split="valid")
                print(f"   done in {time.time() - t0:.1f}s")
            except Exception as exc:
                print(f"   FAILED: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print("Phase 2: Threshold tuning")
    print("=" * 60)
    import threshold_tune as tt
    rows = []
    for tag in tags:
        try:
            rows.append(tt.tune_one(tag, reinfer=False))
        except FileNotFoundError as e:
            print(f"  [skip] {tag}: {e}")
        except Exception as e:
            print(f"  [FAIL] {tag}: {type(e).__name__}: {e}")
    if rows:
        out_csv = RESULT_D / "aggregate_threshold_tuned.csv"
        tt.write_aggregate(rows, out_csv)
        print(f"\nDONE. Aggregate → {out_csv}")
    else:
        print("\nNo cells were tuned successfully.")


if __name__ == "__main__":
    main()
