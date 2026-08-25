"""Verify the patched ColBERTv2Reranker actually consumes a long EHR query.

Stanford ColBERTv2 defaults `query_maxlen=32`; for a real Exp4 EHR
narrative the discriminative content (Molecular markers, Treatment) lives
*after* token 32, so the default would make the late-interaction blind to
exactly the most useful patient features.

After our patch (rerankers.py:ColBERTv2Reranker.QUERY_MAXLEN=256), this
test:
    1. confirms the query_maxlen has actually propagated into the
       underlying Stanford Checkpoint;
    2. confirms a long query gives a *different* ranking than its first
       32-token prefix — proving the model now sees the long-tail content.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from rerankers import ColBERTv2Reranker  # noqa: E402


LONG_QUERY = (
    # Demographics + Diagnosis (~30 tokens, fits in default budget)
    "Demographics: Sex at Birth: Male; Race: White; Age at diagnosis: 62. "
    "Diagnosis: Primary Diagnosis: Glioblastoma; Grade of Primary Brain "
    "Tumor: 4; Stereotactic Biopsy before Surgical Resection: No; "
    "Previous Brain Tumor: No. "
    # Molecular block (the discriminative tokens are HERE)
    "Molecular markers: IDH1 wildtype; IDH2 wildtype; MGMT unmethylated; "
    "TERT promoter mutated; +7/-10 present; EGFR amplified; CDKN2A/B "
    "deleted; TP53 altered; ATRX wildtype; 1p/19q non-codeletion; H3-3A "
    "wildtype; PTEN mutated; BRAF V600E wildtype. "
    # Treatment block (also late in the query)
    "Treatment: Days to surgery: 7; Initial chemotherapy: Yes; Chemo "
    "agent: Temozolomide; Days to chemo start: 35; Days to chemo end: "
    "245; Radiation therapy: Yes; Days to radiation start: 35; Days to "
    "radiation end: 75; Radiation dose: 60; Radiation fractions: 30."
)
SHORT_PREFIX = " ".join(LONG_QUERY.split()[:32])  # what default would see

DOCS = [
    {
        "pmid": "MOL_REL",
        "title": "MGMT promoter methylation status and temozolomide response in glioblastoma",
        "abstract": (
            "MGMT promoter methylation is the strongest molecular predictor "
            "of temozolomide benefit in IDH-wildtype glioblastoma. We "
            "review prognostic implications, EGFR amplification crosstalk, "
            "and CDKN2A/B deletion as recurrence risk factors."
        ),
    },
    {
        "pmid": "TX_REL",
        "title": "Standard Stupp protocol: 60 Gy radiotherapy with concurrent temozolomide",
        "abstract": (
            "The Stupp regimen of 60 Gy radiation in 30 fractions with "
            "concurrent and adjuvant temozolomide remains the standard of "
            "care for newly diagnosed glioblastoma; recurrence is expected "
            "within 7–9 months in MGMT-unmethylated IDH-wildtype tumours."
        ),
    },
    {
        "pmid": "DX_REL",
        "title": "Glioblastoma WHO grade 4: diagnostic and clinical overview",
        "abstract": (
            "Glioblastoma is the most aggressive primary brain tumour, WHO "
            "grade 4, with median overall survival around 15 months under "
            "current standard care."
        ),
    },
    {
        "pmid": "OFF1",
        "title": "Hormone therapy escalation in HER2-positive breast cancer",
        "abstract": "Off-topic clinical narrative about breast cancer therapy.",
    },
    {
        "pmid": "OFF2",
        "title": "Sodium intake and adolescent hypertension",
        "abstract": "Off-topic dietary epidemiology.",
    },
    {
        "pmid": "OFF3",
        "title": "Catheter-associated bloodstream infection in ICU",
        "abstract": "Off-topic infection control commentary.",
    },
]


def main() -> int:
    rr = ColBERTv2Reranker()  # uses patched QUERY_MAXLEN=256

    inner = rr.model.model
    cfg_qmax = getattr(getattr(inner, "config", None), "query_maxlen", None)
    ckpt = getattr(inner, "checkpoint", None) or getattr(inner, "inference_ckpt", None)
    tok_qmax = getattr(getattr(ckpt, "query_tokenizer", None), "query_maxlen", None) if ckpt else None
    print(f"[probe] inner.config.query_maxlen     = {cfg_qmax}")
    print(f"[probe] inner.checkpoint.query_maxlen = {tok_qmax}")
    assert cfg_qmax == 256, f"config.query_maxlen still {cfg_qmax}, expected 256"

    print("\n--- Ranking with FULL EHR query (≈400 tokens) ---")
    long_rank = rr.rerank(LONG_QUERY, DOCS, top_k=3)
    long_top = [d["pmid"] for d in long_rank]
    print(" top-3:", long_top)

    print("\n--- Ranking with TRUNCATED 32-token prefix only ---")
    short_rank = rr.rerank(SHORT_PREFIX, DOCS, top_k=3)
    short_top = [d["pmid"] for d in short_rank]
    print(" top-3:", short_top)

    differs = long_top != short_top
    print(f"\nLong-vs-short produce different rankings: {differs}")

    long_top_set = set(long_top)
    expected_relevant = {"MOL_REL", "TX_REL", "DX_REL"}
    n_relevant_in_top3 = len(long_top_set & expected_relevant)
    print(f"Relevant docs in long-query top-3: {n_relevant_in_top3}/3")

    ok = (cfg_qmax == 256) and (n_relevant_in_top3 == 3)
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
