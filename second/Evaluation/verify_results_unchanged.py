#!/usr/bin/env python3
"""Fingerprint canonical Second_Recur LoRA outputs for drift checks.

Run after training or before a paper freeze, then re-run later and diff:

  python Result/verify_results_unchanged.py
  diff Result/last_results_digest.txt <(python Result/verify_results_unchanged.py -)

Use -q for machine-readable one-line SHA256 of aggregate + all metrics files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Model" / "results"
SUMMARY = ROOT / "Evaluation" / "generated" / "Summary_Result.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print single combined hash only")
    args = ap.parse_args()

    lines: list[str] = []
    agg = RESULTS / "aggregate.csv"
    if not agg.is_file():
        print("missing", agg, file=sys.stderr)
        return 2

    lines.append(f"aggregate.csv sha256={_sha256(agg)}")

    # Duplicate tags in aggregate (append-only CSV hazard)
    with agg.open(newline="") as f:
        rows = list(csv.DictReader(f))
    tags = [r["tag"] for r in rows]
    dupes = [t for t, c in Counter(tags).items() if c > 1]
    if dupes:
        lines.append(f"WARNING duplicate tag rows in aggregate.csv: {dupes}")
        for t in dupes:
            lines.append(f"  {t!r} appears {tags.count(t)} times (last row wins for truth)")

    # Per-cell metrics on disk vs aggregate (last row per tag)
    last_by_tag: dict[str, dict] = {}
    for r in rows:
        last_by_tag[r["tag"]] = r

    mismatches: list[str] = []
    cell_dirs = sorted(
        p for p in RESULTS.iterdir()
        if p.is_dir() and "__biomistral" in p.name and not p.name.startswith("_")
        and "__cv__" not in p.name
    )
    metrics_paths: list[Path] = []
    for d in cell_dirs:
        mpath = d / "predictions_test.metrics.json"
        if not mpath.is_file():
            continue
        metrics_paths.append(mpath)
        tag = d.name
        disk = json.loads(mpath.read_text())
        row = last_by_tag.get(tag)
        if row is None:
            mismatches.append(f"{tag}: on disk but no row in aggregate.csv")
            continue
        a_disk = float(disk["AUROC"])
        a_agg = float(row["AUROC"])
        if abs(a_disk - a_agg) > 1e-9:
            mismatches.append(
                f"{tag}: AUROC disk={a_disk} aggregate_last={a_agg}"
            )
        lines.append(f"{tag} predictions_test.metrics.json sha256={_sha256(mpath)}")

    tuned = RESULTS / "aggregate_threshold_tuned.csv"
    if tuned.is_file():
        lines.append(f"aggregate_threshold_tuned.csv sha256={_sha256(tuned)}")

    if SUMMARY.is_file():
        lines.append(f"Summary_Result.csv sha256={_sha256(SUMMARY)}")

    if args.quiet:
        h = hashlib.sha256()
        h.update(agg.read_bytes())
        for p in sorted(metrics_paths):
            h.update(p.read_bytes())
        print(h.hexdigest())
        return 1 if mismatches else 0

    out_path = ROOT / "Evaluation" / "generated" / "last_results_digest.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\n"
    if mismatches:
        body += "\nMISMATCHES (fix aggregate or re-run eval):\n" + "\n".join(mismatches) + "\n"

    out_path.write_text(body)
    print(body)
    print(f"Wrote {out_path}")
    return 1 if (mismatches or dupes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
