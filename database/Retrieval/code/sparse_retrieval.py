import os
import json
import argparse
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent

def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path, 'r') as f:
        for line in f:
            article = json.loads(line.strip())
            corpus.append(article)
    return corpus

def tokenize(text):
    # Simple whitespace tokenizer. For better results, use scispacy or nltk.
    return text.lower().split()

def main():
    parser = argparse.ArgumentParser(description="BM25 Sparse Retrieval")
    parser.add_argument("--corpus", type=str, default=str(PIPELINE_DIR / "corpus/brain_tumor_recurrence_shared_10k.jsonl"), help="Corpus JSONL file")
    parser.add_argument("--index", type=str, default=str(PIPELINE_DIR / "indexes/bm25/bm25.pkl"), help="Output BM25 index path")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"Corpus file {args.corpus} not found.")
        return

    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} articles.")

    tokenized_corpus = [tokenize(doc['title'] + " " + doc['abstract']) for doc in corpus]
    
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"Saving BM25 index to {args.index}...")
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    with open(args.index, 'wb') as f:
        pickle.dump({'bm25': bm25, 'corpus': corpus}, f)
        
    print("BM25 index built successfully!")

if __name__ == "__main__":
    main()
