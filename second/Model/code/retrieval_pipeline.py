"""
retrieval_pipeline.py — Bridge from `Model/` to `Retrieval/` and `Reranker/`.

For one patient EHR string we:

  1. Sparse-retrieve top-K_sparse PubMed docs with BM25 over the 10K corpus.
  2. Dense-retrieve top-K_dense docs with one of:
        - "beep"  : BEEP-faithful PubMedBERT biencoder + IndexFlatL2
                    (Retrieval/indexes/dense_beep/), prefixed with the BEEP
                    outcome question.
        - "nomic" : nomic-embed-text-v1.5 + cosine-style L2
                    (Retrieval/indexes/dense_nomic/), prefixed with
                    "search_query: ".
  3. Union-merge the two candidate lists (dedup on pmid) — BEEP §3.2.
  4. Cross-encode-rerank with one of the five rerankers in
     `First_Recur/Reranker/code/rerankers.py` and keep top_k.

Models are heavy: every constructor caches the loaded artefacts in a
module-level dict so the grid driver can sweep many (retriever, reranker)
configs without paying the load-time cost more than once per process.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Import order matters on macOS arm64: torch must be imported before faiss,
# otherwise faiss-cpu's BLAS conflicts with torch's and crashes (segfault)
# inside `model.resize_token_embeddings(...)` when we later add [ENTSEP].
import numpy as np
import torch
import faiss

ROOT = Path(__file__).resolve().parents[2]                # BrainTumor/second/
DATABASE_DIR = ROOT.parent / "database"
RETRIEVAL_DIR = DATABASE_DIR / "Retrieval"
RERANKER_CODE = DATABASE_DIR / "Reranker" / "code"

CORPUS_PATH   = RETRIEVAL_DIR / "corpus" / "brain_tumor_recurrence_shared_10k.jsonl"
BM25_PATH     = RETRIEVAL_DIR / "indexes" / "bm25" / "bm25.pkl"
DENSE_BEEP    = RETRIEVAL_DIR / "indexes" / "dense_beep"
DENSE_NOMIC   = RETRIEVAL_DIR / "indexes" / "dense_nomic"

OUTCOME_QUESTION = "What is the probability of brain tumor recurrence?"

# Make Reranker/code importable without polluting the user's PYTHONPATH.
if str(RERANKER_CODE) not in sys.path:
    sys.path.insert(0, str(RERANKER_CODE))

# --------------------------------------------------------------------------- #
# Caches                                                                      #
# --------------------------------------------------------------------------- #
_CORPUS:        list[dict]            | None = None
_PMID_TO_IDX:   dict[str, int]        | None = None
_BM25_DATA:     tuple                 | None = None  # (bm25, corpus_subset)
_DENSE_CACHE:   dict[str, dict]              = {}
_RERANKER_CACHE: dict[str, object]            = {}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_corpus() -> list[dict]:
    global _CORPUS, _PMID_TO_IDX
    if _CORPUS is None:
        docs = []
        with CORPUS_PATH.open() as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        _CORPUS = docs
        _PMID_TO_IDX = {d["pmid"]: i for i, d in enumerate(docs)}
    return _CORPUS


def _load_bm25():
    global _BM25_DATA
    if _BM25_DATA is None:
        with BM25_PATH.open("rb") as f:
            data = pickle.load(f)
        _BM25_DATA = (data["bm25"], data["corpus"])
    return _BM25_DATA


def _load_dense(name: str):
    """name in {'beep','nomic'}."""
    if name in _DENSE_CACHE:
        return _DENSE_CACHE[name]

    if name == "beep":
        idx = faiss.read_index(str(DENSE_BEEP / "dense.faiss"))
        with (DENSE_BEEP / "mapping.json").open() as f:
            pmids = json.load(f)
        encoder = _build_beep_encoder()
        encode_fn = lambda q: _encode_beep(encoder, OUTCOME_QUESTION + " " + q)

    elif name == "nomic":
        from sentence_transformers import SentenceTransformer
        idx = faiss.read_index(str(DENSE_NOMIC / "dense.faiss"))
        with (DENSE_NOMIC / "mapping.json").open() as f:
            pmids = json.load(f)
        st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                                 trust_remote_code=True, device=_device())
        st.max_seq_length = 512
        def encode_fn(q: str) -> np.ndarray:
            v = st.encode(["search_query: " + q], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(v)
            return v
    else:
        raise ValueError(f"Unknown dense encoder: {name!r}")

    pmid_to_doc = {d["pmid"]: d for d in _load_corpus()}
    _DENSE_CACHE[name] = {"index": idx, "pmids": pmids,
                          "encode": encode_fn, "pmid_to_doc": pmid_to_doc}
    return _DENSE_CACHE[name]


def _build_beep_encoder():
    """Loads the BEEP PubMedBERT biencoder for query-time encoding."""
    from transformers import AutoModel, AutoTokenizer
    base   = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    ckpt   = RETRIEVAL_DIR / "biencoder" / "biencoder-dense-retriever.pt"
    device = _device()
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["[ENTSEP]"]})
    model = AutoModel.from_pretrained(base)
    model.resize_token_embeddings(len(tok))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return {"model": model, "tokenizer": tok, "device": device}


@torch.no_grad()
def _encode_beep(enc, text: str) -> np.ndarray:
    inp = enc["tokenizer"]([text], padding="max_length", truncation=True,
                           max_length=512, return_tensors="pt")
    inp = {k: v.to(enc["device"]) for k, v in inp.items()}
    out = enc["model"](**inp)
    return out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")


def _load_reranker(name: str):
    if name in _RERANKER_CACHE:
        return _RERANKER_CACHE[name]
    from rerankers import get_reranker  # type: ignore  # in Reranker/code/
    rr = get_reranker(name)
    _RERANKER_CACHE[name] = rr
    return rr


# --------------------------------------------------------------------------- #
# Hybrid retrieval + rerank                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    retriever: str = "beep"          # "beep" | "nomic" | "none" (skip dense)
    reranker:  str = "beep"          # "beep" | "minilm" | "medcpt" | "colbert" | "bge_m3" | "none"
    top_k_sparse: int = 10
    top_k_dense:  int = 10
    final_top_k:  int = 5


def hybrid_retrieve(query_text: str, cfg: RetrievalConfig) -> list[dict]:
    """Return up to (top_k_sparse + top_k_dense) candidate docs (deduped by pmid)."""
    candidates: list[dict] = []
    seen: set[str] = set()

    if cfg.top_k_sparse > 0:
        bm25, corpus_bm25 = _load_bm25()
        toks = query_text.lower().split()
        scores = bm25.get_scores(toks)
        top_idx = np.argsort(scores)[::-1][: cfg.top_k_sparse]
        for i in top_idx:
            doc = corpus_bm25[i]
            pmid = doc.get("pmid")
            if pmid in seen:
                continue
            seen.add(pmid)
            candidates.append(doc)

    if cfg.retriever != "none" and cfg.top_k_dense > 0:
        dense = _load_dense(cfg.retriever)
        q_vec = dense["encode"](query_text)                       # (1, dim)
        dists, idxs = dense["index"].search(q_vec, cfg.top_k_dense)
        for j in idxs[0]:
            if j == -1:
                continue
            pmid = dense["pmids"][j]
            if pmid in seen:
                continue
            seen.add(pmid)
            doc = dense["pmid_to_doc"].get(pmid)
            if doc is not None:
                candidates.append(doc)

    return candidates


def retrieve_and_rerank(query_text: str, cfg: RetrievalConfig) -> list[dict]:
    """Full BEEP §3.2 → §3.3 pipeline: hybrid + rerank → top_k."""
    cands = hybrid_retrieve(query_text, cfg)
    if not cands:
        return []
    if cfg.reranker == "none":
        return cands[: cfg.final_top_k]
    rr = _load_reranker(cfg.reranker)
    return rr.rerank(query_text, cands, top_k=cfg.final_top_k)


# --------------------------------------------------------------------------- #
# CLI smoke                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="70-year-old IDH-wildtype glioblastoma, MGMT unmethylated, post-Stupp temozolomide")
    ap.add_argument("--retriever", default="beep", choices=["beep", "nomic", "none"])
    ap.add_argument("--reranker",  default="beep",
                    choices=["beep", "minilm", "medcpt", "colbert", "bge_m3", "none"])
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    cfg = RetrievalConfig(retriever=args.retriever, reranker=args.reranker,
                          final_top_k=args.top_k)
    t0 = time.time()
    results = retrieve_and_rerank(args.query, cfg)
    dt = time.time() - t0
    print(f"\n[{args.retriever} + {args.reranker}]  top-{args.top_k}  in {dt:.2f}s")
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip().replace("\n", " ")[:120]
        print(f"  {i}. pmid={r.get('pmid','')}  {title}")
