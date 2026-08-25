"""Build the BEEP-faithful dense FAISS index over the 10K PubMed corpus.

Follows BEEP (Naik et al., EMNLP 2022 Findings) `text_triplet_bireranker.py`:

    base model : microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    extra token : [ENTSEP]  (then resize_token_embeddings)
    checkpoint  : biencoder-dense-retriever.pt   (state_dict of the PubMedBERT encoder)
    pooling     : last_hidden_state[:, 0, :]      (CLS token)
    similarity  : L2 (euclidean) — matches the original training loss.

Query side uses the same model with a leading "outcome question" prefix,
which for glioma recurrence we formulate as:

    "What is the probability of brain tumor recurrence?"

Usage
-----
    python dense_retrieval_beep.py build            # encode corpus + build FAISS
    python dense_retrieval_beep.py query "IDH mut."  # smoke-test retrieval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faiss
from transformers import AutoModel, AutoTokenizer

RETRIEVAL_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH   = RETRIEVAL_DIR / "corpus" / "brain_tumor_recurrence_shared_10k.jsonl"
BIENCODER_PT  = RETRIEVAL_DIR / "biencoder" / "biencoder-dense-retriever.pt"
OUT_DIR       = RETRIEVAL_DIR / "indexes" / "dense_beep"
OUT_FAISS     = OUT_DIR / "dense.faiss"
OUT_EMB       = OUT_DIR / "embeddings.npy"
OUT_MAP       = OUT_DIR / "mapping.json"

BASE_MODEL   = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
SPECIAL_TOK  = "[ENTSEP]"
MAX_LEN      = 512
BATCH_SIZE   = 16

OUTCOME_QUESTION = "What is the probability of brain tumor recurrence?"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_beep_biencoder(device: torch.device):
    print(f"Loading base model: {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOK]})

    model = AutoModel.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading BEEP biencoder checkpoint: {BIENCODER_PT}", flush=True)
    state = torch.load(BIENCODER_PT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})", flush=True)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})", flush=True)

    model.eval().to(device)
    return model, tokenizer


@torch.no_grad()
def _encode_batch(texts, model, tokenizer, device) -> np.ndarray:
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype("float32")
    return cls


def _load_corpus() -> list[dict]:
    corpus = []
    with open(CORPUS_PATH) as fh:
        for line in fh:
            corpus.append(json.loads(line))
    return corpus


def build_index(batch_size: int = BATCH_SIZE, force: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    if OUT_FAISS.exists() and not force:
        print(f"Index already exists at {OUT_FAISS}  (use --force to rebuild)")
        return

    device = _device()
    print(f"Device: {device}")

    model, tokenizer = _load_beep_biencoder(device)
    corpus = _load_corpus()
    print(f"Loaded {len(corpus)} documents from {CORPUS_PATH.name}")

    texts = [f"{d.get('title','')} {d.get('abstract','')}".strip() for d in corpus]
    pmids = [d["pmid"] for d in corpus]

    if OUT_EMB.exists() and not force:
        print(f"Found cached embeddings at {OUT_EMB} — reusing")
        embeddings = np.load(OUT_EMB)
    else:
        print(f"Encoding {len(texts)} docs (batch={batch_size}, max_len={MAX_LEN})...")
        chunks: list[np.ndarray] = []
        t0 = time.time()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            chunks.append(_encode_batch(batch, model, tokenizer, device))
            done = i + len(batch)
            if done % (batch_size * 10) == 0 or done == len(texts):
                elapsed = time.time() - t0
                eta = elapsed * (len(texts) - done) / max(done, 1)
                print(f"  encoded {done}/{len(texts)}   elapsed {elapsed:5.1f}s   eta {eta:5.1f}s",
                      flush=True)
        embeddings = np.concatenate(chunks, axis=0).astype("float32")
        np.save(OUT_EMB, embeddings)
        print(f"Saved embeddings: {OUT_EMB}  shape={embeddings.shape}")

    print("Building FAISS IndexFlatL2 (BEEP uses euclidean distance)...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"Saved FAISS index: {OUT_FAISS}  ntotal={index.ntotal}")

    with open(OUT_MAP, "w") as f:
        json.dump(pmids, f)
    print(f"Saved pmid mapping: {OUT_MAP}  len={len(pmids)}")


def query(query_text: str, top_k: int = 10) -> None:
    device = _device()
    model, tokenizer = _load_beep_biencoder(device)

    q_text = f"{OUTCOME_QUESTION} {query_text}"
    q_emb  = _encode_batch([q_text], model, tokenizer, device)  # (1,768)

    index = faiss.read_index(str(OUT_FAISS))
    dists, idxs = index.search(q_emb.astype("float32"), top_k)
    pmids = json.load(open(OUT_MAP))
    corpus = {d["pmid"]: d for d in _load_corpus()}

    print(f"\nQuery: {q_text!r}\nTop-{top_k} results (L2 distance, lower = better):\n")
    for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0]), 1):
        pmid  = pmids[idx]
        title = corpus.get(pmid, {}).get("title", "<missing>")
        print(f"  {rank:2d}  dist={dist:.4f}  pmid={pmid}  {title[:110]}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub_build = sub.add_parser("build")
    sub_build.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    sub_build.add_argument("--force", action="store_true")
    sub_query = sub.add_parser("query")
    sub_query.add_argument("text")
    sub_query.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    if args.cmd == "build":
        build_index(batch_size=args.batch_size, force=args.force)
    elif args.cmd == "query":
        query(args.text, top_k=args.top_k)


if __name__ == "__main__":
    main()
