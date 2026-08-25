# Literature database and reranker code

This folder contains code only. The PubMed corpus, FAISS/BM25 indexes, and
BEEP checkpoints are intentionally excluded.

## Required local assets

```text
database/
├── Retrieval/
│   ├── corpus/brain_tumor_recurrence_shared_10k.jsonl
│   ├── biencoder/biencoder-dense-retriever.pt
│   └── indexes/
│       ├── bm25/bm25.pkl
│       └── dense_beep/
│           ├── dense.faiss
│           ├── embeddings.npy
│           └── mapping.json
└── BEEP/checkpoints/crossencoder-reranker.pt
```

The two BEEP checkpoints were distributed with the BEEP reproduction:

- Dense biencoder: `biencoder-dense-retriever.pt`
- Cross-encoder reranker: `crossencoder-reranker.pt`

The reranker code also downloads MiniLM, MedCPT, ColBERTv2, or BGE-M3 from
their Hugging Face model repositories when selected.

## Build indexes

From the `BrainTumor` root:

```bash
python database/Retrieval/code/sparse_retrieval.py
python database/Retrieval/code/dense_retrieval_beep.py build --batch-size 16
```

Quick retrieval check:

```bash
python database/Retrieval/code/dense_retrieval_beep.py query \
  "IDH-wildtype glioblastoma after radiotherapy" --top-k 5
```

Reranker check:

```bash
python database/Reranker/tests/smoke_test.py --only beep,minilm
```
