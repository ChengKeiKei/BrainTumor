"""Smoke-test every reranker on a fixed brain-tumor-recurrence toy query.

Run from the submission root with:
    python database/Reranker/tests/smoke_test.py [--only beep,minilm,...]

The toy corpus is deliberately small (5 docs, 1 obvious ground-truth
pair). We check:
  * the reranker loads without error
  * the top-1 doc is the ground-truth PMID (or at least appears in top-2)
  * `rerank` is deterministic across two calls on MPS/CPU
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from rerankers import RERANKER_REGISTRY, get_reranker  # noqa: E402


TOY_QUERY = "IDH-wildtype glioblastoma recurrence after radiotherapy and temozolomide"

TOY_DOCS = [
    {
        "pmid": "A1",
        "title": "Temozolomide and radiotherapy outcomes in IDH-wildtype glioblastoma",
        "abstract": (
            "We analyse progression-free survival of 312 IDH-wildtype "
            "glioblastoma patients treated with concurrent radiation and "
            "temozolomide. Median time to first recurrence was 7.1 months; "
            "MGMT methylation and residual tumour volume were the strongest "
            "predictors of early progression."
        ),
    },
    {
        "pmid": "A2",
        "title": "MGMT methylation status predicts temozolomide response in glioma",
        "abstract": (
            "MGMT promoter methylation remains the principal biomarker of "
            "alkylating-agent sensitivity in high-grade glioma, though its "
            "prognostic role differs between IDH-mutant and IDH-wildtype "
            "tumours."
        ),
    },
    {
        "pmid": "B1",
        "title": "A review of breast cancer hormone therapy escalation strategies",
        "abstract": (
            "Hormone receptor positive breast cancer patients receiving "
            "extended aromatase inhibitor therapy benefit from escalation "
            "in the presence of CYP2D6 poor-metaboliser genotype."
        ),
    },
    {
        "pmid": "B2",
        "title": "Dietary salt intake and essential hypertension in adolescents",
        "abstract": (
            "A prospective cohort of 4,500 adolescents demonstrated a "
            "dose-response relationship between daily sodium intake and "
            "systolic blood pressure, with the strongest effect seen "
            "among obese participants."
        ),
    },
    {
        "pmid": "B3",
        "title": "Nosocomial infections in ICU patients with sepsis",
        "abstract": (
            "Rates of catheter-associated bloodstream infection in adult "
            "intensive-care units declined with the introduction of "
            "chlorhexidine bathing and bundled insertion protocols."
        ),
    },
]
EXPECTED_TOP_PMIDS = {"A1", "A2"}


def score_result(ranked):
    if not ranked:
        return "FAIL (no output)"
    top1 = ranked[0]["pmid"]
    top2 = ranked[1]["pmid"] if len(ranked) > 1 else None
    if top1 in EXPECTED_TOP_PMIDS:
        return f"PASS (top1={top1})"
    if top2 in EXPECTED_TOP_PMIDS:
        return f"SOFT PASS (top1={top1}, top2={top2})"
    return f"FAIL (top1={top1}, top2={top2})"


def run_one(name: str) -> dict:
    print("\n" + "=" * 72)
    print(f"  Reranker: {name}")
    print("=" * 72)
    t0 = time.time()
    try:
        rr = get_reranker(name)
    except Exception as err:  # pylint: disable=broad-except
        print(f"  LOAD FAILED: {type(err).__name__}: {err}")
        return {"name": name, "loaded": False, "error": str(err)}
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s")

    ranked = rr.rerank(TOY_QUERY, TOY_DOCS, top_k=3)
    rerank_s = time.time() - (t0 + load_s)
    print(f"  reranked 5 docs in {rerank_s:.2f}s")
    for i, d in enumerate(ranked, 1):
        print(f"    [{i}] {d['pmid']}  {d['title'][:72]}")

    # Determinism check (same input → same order)
    ranked2 = rr.rerank(TOY_QUERY, TOY_DOCS, top_k=3)
    deterministic = [d["pmid"] for d in ranked] == [d["pmid"] for d in ranked2]
    verdict = score_result(ranked)
    print(f"  verdict: {verdict}   deterministic: {deterministic}")
    return {
        "name": name,
        "loaded": True,
        "load_s": round(load_s, 2),
        "rerank_s": round(rerank_s, 3),
        "top1": ranked[0]["pmid"],
        "top2": ranked[1]["pmid"] if len(ranked) > 1 else None,
        "verdict": verdict,
        "deterministic": deterministic,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of " + ",".join(RERANKER_REGISTRY),
    )
    args = parser.parse_args()
    targets = (
        [t.strip() for t in args.only.split(",") if t.strip()]
        if args.only
        else list(RERANKER_REGISTRY)
    )
    unknown = [t for t in targets if t not in RERANKER_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown reranker(s): {unknown}")

    results = [run_one(name) for name in targets]

    print("\n" + "#" * 72)
    print("# Summary")
    print("#" * 72)
    for r in results:
        if r["loaded"]:
            print(
                f"  {r['name']:<8}  load {r['load_s']:>5.1f}s  "
                f"rerank {r['rerank_s']:>5.2f}s  "
                f"top1={r['top1']:<3} {r['verdict']}"
            )
        else:
            print(f"  {r['name']:<8}  LOAD FAILED: {r['error'][:80]}")


if __name__ == "__main__":
    main()
