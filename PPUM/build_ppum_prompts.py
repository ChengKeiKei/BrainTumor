"""
Build LLM prompts for the PPUM external set, for Exp1-4, two arms:
  * no-RAG  -> tag Exp{i}__baseline__ppum
  * RAG k3  -> tag Exp{i}__beep__beep__ppum  (BM25 + BEEP dense + BEEP rerank, top_k=3)

Prompts are written under PPUM/generated/prompts/<tag>/test.jsonl so nothing in the main
Model/prompts or Model/results tree is touched. Reuses the exact project code
(feature_render, prompts, retrieval_pipeline) so PPUM patients are rendered identically
to MU patients.

Run: python PPUM/build_ppum_prompts.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
CODE = SUBMISSION_ROOT / "first" / "Model" / "code"
sys.path.insert(0, str(CODE))
import feature_render as FR                       # noqa: E402
from prompts import build_chat_record             # noqa: E402
from retrieval_pipeline import RetrievalConfig, retrieve_and_rerank  # noqa: E402

PPUM_ROOT = Path(__file__).resolve().parent
PPUM_CSV = PPUM_ROOT / "generated" / "PPUM.csv"
OUTROOT = PPUM_ROOT / "generated" / "prompts"
EXPS = ["Exp1", "Exp2", "Exp3", "Exp4"]


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    df = pd.read_csv(PPUM_CSV)
    cfg = RetrievalConfig(retriever="beep", reranker="beep",
                          top_k_sparse=10, top_k_dense=10, final_top_k=3)

    for exp in EXPS:
        # --- no-RAG ---
        norag = []
        for _, row in df.iterrows():
            text = FR.render_patient(row, exp)
            norag.append(build_chat_record(text, int(row["y"]), docs=None,
                                           patient_id=str(row["Patient_ID"])))
        write_jsonl(OUTROOT / f"{exp}__baseline__ppum" / "test.jsonl", norag)
        print(f"[{exp}] no-RAG: {len(norag)} prompts")

        # --- RAG k3 (heavy: retrieval per patient) ---
        rag = []
        t0 = time.time()
        for _, row in df.iterrows():
            text = FR.render_patient(row, exp)
            docs = retrieve_and_rerank(text, cfg)
            rag.append(build_chat_record(text, int(row["y"]), docs=docs,
                                         patient_id=str(row["Patient_ID"])))
        write_jsonl(OUTROOT / f"{exp}__beep__beep__ppum" / "test.jsonl", rag)
        print(f"[{exp}] RAG k3 : {len(rag)} prompts  ({time.time()-t0:.1f}s)")

    print("\nDONE building PPUM prompts ->", OUTROOT)


if __name__ == "__main__":
    main()
