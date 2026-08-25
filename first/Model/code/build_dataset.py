"""
build_dataset.py — Materialise one (Exp, retriever, reranker) configuration
into the mlx-tune / mlx_lm.lora chat-format JSONL files used for LoRA
fine-tuning of Mistral-7B-Instruct-v0.3.

The generated JSONL is already *Mistral-native*: every line is a two-turn
chat record that `tokenizer.apply_chat_template(..., chat_template="mistral")`
(inside mlx-tune at train time, and inside `infer.py` at inference time)
wraps with Mistral's `[INST] ... [/INST]` instruction format. The tokenizer
handles the turn delimiters — our job is only to produce the content
bodies and `label` field. A rendered, fully Mistral-wrapped sample prompt
is also dumped next to each JSONL as `sample_chat.txt` so reviewers can
see exactly what the LLM receives.

Two prompt styles are supported:

  * **joint** (default, BEEP paper main-result inference mode):
    1 record per patient; the prompt concatenates all top-K retrieved
    docs into one literature block.

  * **per-doc** (`--per-doc`): K records per patient (one per retrieved
    doc, with `doc_rank` and `doc_pmid` fields). Used downstream by
    `infer.py` + `ensemble.py` to reproduce BEEP §3.4's top-K ensemble
    (uniform / softmax / max aggregation).

Output layout:

    Model/prompts/
        Exp1__baseline/                   # baseline (no RAG)
        Exp4__beep__beep/                 # joint top-K
        Exp4__beep__beep__perdoc/         # per-doc variant (same cell)
            train.jsonl  valid.jsonl  test.jsonl
            config.json  sample_chat.txt
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from feature_render import render_split
from prompts import ABSTRACT_TRUNC_DEFAULT, build_chat_record, build_user_message
from retrieval_pipeline import RetrievalConfig, retrieve_and_rerank

ROOT       = Path(__file__).resolve().parents[2]    # First_Recur/
PROMPT_DIR = ROOT / "Model" / "prompts"

SPLITS = ("Train", "Validation", "Test")
SPLIT_FILE = {"Train": "train.jsonl", "Validation": "valid.jsonl", "Test": "test.jsonl"}

ALL_EXPERIMENTS = ("Exp1", "Exp2", "Exp3", "Exp4", "Exp5", "Exp6")
ALL_RETRIEVERS  = ("none", "beep")
ALL_RERANKERS   = ("none", "beep", "minilm", "medcpt", "colbert", "bge_m3")


def config_tag(experiment: str, retriever: str, reranker: str,
               per_doc: bool = False) -> str:
    if retriever == "none" and reranker == "none":
        base = f"{experiment}__baseline"
    else:
        base = f"{experiment}__{retriever}__{reranker}"
    return base + "__perdoc" if per_doc else base


def _dump_sample(outdir: Path, records_per_split: dict[str, list[dict]]) -> None:
    """Write a human-readable Mistral-wrapped sample prompt so reviewers can
    verify the exact string the LLM sees (after apply_chat_template)."""
    sample_rec = None
    for split_name in ("Train", "Validation", "Test"):
        recs = records_per_split.get(split_name) or []
        if recs:
            sample_rec = recs[0]
            break
    if sample_rec is None:
        return

    user_msg = next(m["content"] for m in sample_rec["messages"] if m["role"] == "user")
    assistant = next(m["content"] for m in sample_rec["messages"] if m["role"] == "assistant")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        mistral_prompt = tok.apply_chat_template(
            sample_rec["messages"], tokenize=False, add_generation_prompt=False)
    except Exception as e:
        mistral_prompt = (f"(apply_chat_template unavailable offline: {e!r})\n\n"
                          f"[INST] {user_msg} [/INST] {assistant}")

    bar = "=" * 78
    (outdir / "sample_chat.txt").write_text(
        f"{bar}\n"
        "Sample training record (first example of first non-empty split).\n"
        "This is exactly what tokenizer.apply_chat_template(messages,\n"
        "chat_template='mistral') produces for Mistral-7B-Instruct-v0.3.\n"
        f"{bar}\n\n"
        f"patient_id  : {sample_rec.get('patient_id')}\n"
        f"label       : {sample_rec.get('label')}\n"
        f"doc_rank    : {sample_rec.get('doc_rank', '(joint: all K docs in one prompt)')}\n"
        f"doc_pmid    : {sample_rec.get('doc_pmid', '(joint)')}\n\n"
        "---------- Mistral-native [INST]...[/INST] wrapped prompt ----------\n\n"
        f"{mistral_prompt}\n"
    )


def build_one_config(experiment: str, retriever: str, reranker: str,
                     top_k_sparse: int = 10, top_k_dense: int = 10,
                     final_top_k: int = 3,
                     abstract_max_chars: int = ABSTRACT_TRUNC_DEFAULT,
                     per_doc: bool = False,
                     splits: Iterable[str] = SPLITS) -> Path:
    """Materialise train/valid/test JSONL for one configuration."""
    tag    = config_tag(experiment, retriever, reranker, per_doc=per_doc)
    outdir = PROMPT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = RetrievalConfig(
        retriever    = retriever,
        reranker     = reranker,
        top_k_sparse = top_k_sparse,
        top_k_dense  = top_k_dense,
        final_top_k  = final_top_k,
    )

    summary = {
        "experiment": experiment, "retriever": retriever, "reranker": reranker,
        "top_k_sparse": top_k_sparse, "top_k_dense": top_k_dense,
        "final_top_k":  final_top_k,
        "abstract_max_chars": abstract_max_chars if abstract_max_chars else "full",
        "style": "per-doc" if per_doc else "joint",
        "splits": {},
    }

    per_split_records: dict[str, list[dict]] = {}

    for split in splits:
        df = render_split(experiment, split)
        records: list[dict] = []
        t0 = time.time()
        for _, row in df.iterrows():
            label = int(row["label"])
            pid   = str(row["Patient_ID"])
            if retriever == "none" and reranker == "none":
                records.append(build_chat_record(
                    row["text"], label, docs=None, patient_id=pid,
                    abstract_max_chars=abstract_max_chars))
                continue

            docs = retrieve_and_rerank(row["text"], cfg)

            if not per_doc:
                records.append(build_chat_record(
                    row["text"], label, docs=docs, patient_id=pid,
                    abstract_max_chars=abstract_max_chars))
            else:
                for rank, doc in enumerate(docs, 1):
                    rec = build_chat_record(
                        row["text"], label, docs=[doc], patient_id=pid,
                        abstract_max_chars=abstract_max_chars)
                    rec["doc_rank"] = rank
                    rec["doc_pmid"] = doc.get("pmid", "")
                    records.append(rec)

        elapsed = time.time() - t0
        per_split_records[split] = records

        out_path = outdir / SPLIT_FILE[split]
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        summary["splits"][split] = {
            "n_records":   len(records),
            "n_patients":  len({r["patient_id"] for r in records}),
            "n_pos":       int(sum(r["label"] for r in records)),
            "seconds":     round(elapsed, 2),
        }
        print(f"  [{tag} / {split}]  {len(records):>4d} records  ({elapsed:.1f}s)  -> {out_path}")

    with (outdir / "config.json").open("w") as f:
        json.dump(summary, f, indent=2)

    _dump_sample(outdir, per_split_records)
    return outdir


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=list(ALL_EXPERIMENTS),
                    choices=list(ALL_EXPERIMENTS))
    ap.add_argument("--retrievers", nargs="+", default=["beep"],
                    choices=list(ALL_RETRIEVERS))
    ap.add_argument("--rerankers", nargs="+", default=["beep"],
                    choices=list(ALL_RERANKERS))
    ap.add_argument("--baseline", action="store_true",
                    help="Also build the no-RAG baseline JSONL for each experiment.")
    ap.add_argument("--per-doc", action="store_true",
                    help="Emit one JSONL record per (patient, retrieved-doc) "
                         "instead of one per patient. Enables BEEP top-K ensemble.")
    ap.add_argument("--top-k-sparse", type=int, default=10)
    ap.add_argument("--top-k-dense",  type=int, default=10)
    ap.add_argument("--final-top-k",  type=int, default=3,
                    help="Top-K docs kept after reranking (BEEP default = 3).")
    ap.add_argument("--abstract-max-chars", type=int, default=ABSTRACT_TRUNC_DEFAULT,
                    help="Truncate each abstract to this many chars; "
                         "0 = pass the full abstract.")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS),
                    choices=list(SPLITS))
    args = ap.parse_args()

    pairs: list[tuple[str, str, str]] = []
    if args.baseline:
        for exp in args.experiments:
            pairs.append((exp, "none", "none"))
    for exp, ret, rrk in product(args.experiments, args.retrievers, args.rerankers):
        if ret == "none" and rrk == "none":
            continue
        pairs.append((exp, ret, rrk))

    print(f"Building {len(pairs)} dataset configurations "
          f"(style = {'per-doc' if args.per_doc else 'joint'}):")
    for p in pairs:
        print("   -", p)
    print()

    for exp, ret, rrk in pairs:
        print(f"=== {exp} | retriever={ret} | reranker={rrk} "
              f"| per_doc={args.per_doc} ===")
        build_one_config(exp, ret, rrk,
                         top_k_sparse=args.top_k_sparse,
                         top_k_dense=args.top_k_dense,
                         final_top_k=args.final_top_k,
                         abstract_max_chars=args.abstract_max_chars,
                         per_doc=args.per_doc,
                         splits=args.splits)


if __name__ == "__main__":
    main()
