"""
ensemble.py — BEEP §3.4 evidence-aggregation ensembles for inference time.

Given per-document P(Yes) scores (each obtained by running the LoRA
Mistral with one retrieved abstract at a time), aggregate them into a
single per-patient probability. We support the three aggregation rules
discussed in BEEP:

    "uniform"  — simple mean over the top-K docs   (Naik et al. eq. 4)
    "softmax"  — softmax-weighted mean over docs    (Naik et al. eq. 5)
    "max"      — take the doc that fired hardest    (sanity baseline)

When `predictions_jsonl` already contains one row per patient (the
default `infer.py` mode that puts the whole top-K block into one prompt),
this module is a no-op and just rewrites the file as-is.

Format of the per-doc predictions JSONL expected here (alternative to
`infer.py`):

    {"patient_id":"...", "doc_rank":1, "label":1, "score":0.83, "doc_pmid":"..."}
    {"patient_id":"...", "doc_rank":2, ...}
    ...
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s  = sum(es)
    return [e / s for e in es]


def aggregate(per_doc_jsonl: str | Path, mode: str = "uniform") -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with Path(per_doc_jsonl).open() as f:
        for line in f:
            r = json.loads(line)
            grouped[r["patient_id"]].append(r)

    out = []
    for pid, rows in grouped.items():
        rows.sort(key=lambda r: r.get("doc_rank", 0))
        scores = [float(r["score"]) for r in rows]
        labels = [int(r["label"])  for r in rows]
        label  = labels[0]

        if mode == "uniform":
            agg = sum(scores) / len(scores)
        elif mode == "softmax":
            ws = _softmax(scores)
            agg = sum(w * s for w, s in zip(ws, scores))
        elif mode == "max":
            agg = max(scores)
        else:
            raise ValueError(f"Unknown ensemble mode: {mode!r}")

        out.append({"patient_id": pid, "label": label, "score": agg, "k": len(rows)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc-jsonl", required=True,
                    help="Per-document predictions JSONL (one row per (patient, doc_rank))")
    ap.add_argument("--out", required=True,
                    help="Aggregated per-patient predictions JSONL")
    ap.add_argument("--mode", default="uniform",
                    choices=["uniform", "softmax", "max"])
    args = ap.parse_args()

    rows = aggregate(args.per_doc_jsonl, mode=args.mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[ensemble.py] mode={args.mode}  patients={len(rows)}  → {args.out}")


if __name__ == "__main__":
    main()
