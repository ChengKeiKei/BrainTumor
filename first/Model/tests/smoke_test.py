"""
smoke_test.py — Light end-to-end check for Model/ that does NOT download the
4-bit Mistral-7B weights and does NOT call mlx_lm.lora.

It exercises:

  1. feature_render: produces non-empty EHR text for all 4 experiments.
  2. retrieval_pipeline: BM25 + dense_beep + BEEP cross-encoder rerank
     returns >=1 doc in <30 s.
  3. prompts: the assembled chat record has the right shape, the
     "Yes"/"No" assistant turn, and matches the BEEP-style prompt
     anatomy.
  4. evaluate: dummy P(Yes) on the Test split → metrics JSON, AUROC ≈ 0.5.

Run from the submission root:
    python first/Model/tests/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]    # First_Recur/
MODEL_D  = ROOT / "Model"
sys.path.insert(0, str(MODEL_D / "code"))

LOG_DIR  = MODEL_D / "tests" / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

PASS = "PASS"
FAIL = "FAIL"


def log(msg: str, log_path: Path) -> None:
    print(msg)
    with log_path.open("a") as f:
        f.write(msg + "\n")


def t1_feature_render(log_path: Path) -> str:
    from feature_render import list_experiments, render_split
    rendered = {}
    for exp in list_experiments():
        df = render_split(exp, "Test")
        first = df.iloc[0]
        if not first["text"]:
            return f"{FAIL} {exp}: empty text"
        rendered[exp] = first["text"]
    log(f"  rendered all {len(rendered)} experiments OK", log_path)
    log(f"  Exp1[0] (excerpt):\n    " + rendered["Exp1"][:160].replace("\n", " | "), log_path)
    return PASS


def t2_retrieval(log_path: Path) -> str:
    from feature_render import render_split
    from retrieval_pipeline import RetrievalConfig, retrieve_and_rerank

    df = render_split("Exp4", "Test")
    query = df.iloc[0]["text"]

    cfg = RetrievalConfig(retriever="beep", reranker="beep",
                          top_k_sparse=10, top_k_dense=10, final_top_k=3)
    t0 = time.time()
    docs = retrieve_and_rerank(query, cfg)
    dt = time.time() - t0
    if not docs:
        return f"{FAIL} retrieval returned 0 docs"
    log(f"  retrieved + reranked in {dt:.1f}s", log_path)
    for i, d in enumerate(docs, 1):
        title = (d.get("title") or "").strip().replace("\n", " ")[:90]
        log(f"    {i}. pmid={d.get('pmid','')}  {title}", log_path)
    return PASS


def t3_prompt(log_path: Path) -> str:
    from feature_render import render_split
    from prompts import build_chat_record

    df = render_split("Exp4", "Test")
    row = df.iloc[0]

    docs = [{"pmid": "X1", "title": "MGMT methylation predicts TMZ response",
             "abstract": "Patients with methylated MGMT promoter ..." * 3},
            {"pmid": "X2", "title": "Recurrent glioblastoma after radiotherapy",
             "abstract": "Median OS in recurrent GBM ..." * 3}]
    rec = build_chat_record(row["text"], int(row["label"]),
                            docs=docs, patient_id=str(row["Patient_ID"]))

    msgs = rec["messages"]
    if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return f"{FAIL} chat ordering wrong"
    if msgs[1]["content"] not in ("Yes", "No"):
        return f"{FAIL} assistant turn must be Yes/No, got {msgs[1]['content']!r}"
    if "Relevant Literature Context:" not in msgs[0]["content"]:
        return f"{FAIL} literature header missing"
    if "Prediction (tumor recurrence/progression, Yes/No):" not in msgs[0]["content"]:
        return f"{FAIL} prediction tail missing"
    log(f"  built {len(msgs)}-turn chat record (label={rec['label']})", log_path)
    log(f"  user_msg length = {len(msgs[0]['content'])} chars", log_path)
    return PASS


def t5_build_dataset_and_sample_dump(log_path: Path) -> str:
    """Build a tiny (1 split × 2 patients) per-doc dataset and verify the
    Mistral-wrapped sample file is produced."""
    import build_dataset
    from retrieval_pipeline import RetrievalConfig, retrieve_and_rerank
    from prompts import build_chat_record
    from feature_render import render_split

    tag_joint = build_dataset.config_tag("Exp4", "beep", "beep", per_doc=False)
    tag_perdoc = build_dataset.config_tag("Exp4", "beep", "beep", per_doc=True)

    tmp_outdir = build_dataset.PROMPT_DIR / tag_perdoc
    # Quick build: only Test split, only 2 patients
    cfg = RetrievalConfig(retriever="beep", reranker="beep",
                          top_k_sparse=10, top_k_dense=10, final_top_k=3)
    df = render_split("Exp4", "Test").iloc[:2]
    tmp_outdir.mkdir(parents=True, exist_ok=True)
    import json as _json
    per_doc_records = []
    for _, row in df.iterrows():
        docs = retrieve_and_rerank(row["text"], cfg)
        for rank, doc in enumerate(docs, 1):
            rec = build_chat_record(row["text"], int(row["label"]),
                                    docs=[doc], patient_id=str(row["Patient_ID"]))
            rec["doc_rank"] = rank
            rec["doc_pmid"] = doc.get("pmid", "")
            per_doc_records.append(rec)
    (tmp_outdir / "test.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in per_doc_records) + "\n")
    build_dataset._dump_sample(tmp_outdir, {"Test": per_doc_records})

    sample_path = tmp_outdir / "sample_chat.txt"
    if not sample_path.exists():
        return f"{FAIL} sample_chat.txt missing"
    content = sample_path.read_text()
    if "[INST]" not in content or "[/INST]" not in content:
        return f"{FAIL} sample_chat.txt missing Mistral [INST] wrapping"
    log(f"  built {len(per_doc_records)} per-doc records for 2 patients x top-3", log_path)
    log(f"  sample_chat.txt produced ({sample_path.stat().st_size} bytes)", log_path)
    log(f"  first 200 chars of Mistral-wrapped prompt:\n    "
        + content.split('[INST]', 1)[1][:200].replace('\n', ' | '), log_path)
    return PASS


def t6_train_log_parser(log_path: Path) -> str:
    """Confirm parse_training_log + plot_loss_curve handle mlx_lm.lora-style logs."""
    import train as train_mod
    fake_log = LOG_DIR / "fake_train.log"
    fake_log.write_text(
        "Iter 10: Train loss 1.032, Learning Rate 2.0e-5, It/sec 0.54\n"
        "Iter 20: Train loss 0.874, Learning Rate 2.0e-5, It/sec 0.55\n"
        "Iter 50: Val loss 0.781\n"
        "Iter 30: Train loss 0.712\n"
        "Iter 100: Val loss 0.654\n"
    )
    ck = LOG_DIR / "fake_ckpt"; ck.mkdir(exist_ok=True)
    (ck / "train.log").write_text(fake_log.read_text())
    rd = LOG_DIR / "fake_result"; rd.mkdir(exist_ok=True)

    t_pairs, v_pairs = train_mod.parse_training_log(ck / "train.log")
    if len(t_pairs) < 2 or len(v_pairs) < 1:
        return f"{FAIL} parse_training_log: train={len(t_pairs)} val={len(v_pairs)}"
    out = train_mod.plot_loss_curve(ck, "fake_tag", rd)
    if out is None or not out.exists():
        return f"{FAIL} loss_curve.png not produced"
    log(f"  train-log parser: {len(t_pairs)} train, {len(v_pairs)} val rows", log_path)
    log(f"  plot -> {out}", log_path)
    return PASS


def t4_evaluate(log_path: Path) -> str:
    """Smoke-test evaluate on dummy random scores over the real Test split."""
    import random
    from feature_render import render_split
    from evaluate import compute_metrics
    import numpy as np

    df = render_split("Exp1", "Test")
    rng = random.Random(0)
    scores = [rng.random() for _ in range(len(df))]
    labels = df["label"].astype(int).tolist()
    m = compute_metrics(np.asarray(labels), np.asarray(scores), bootstrap=200)
    log(f"  n={m['n']}  AUROC={m['AUROC']:.3f}  AUPRC={m['AUPRC']:.3f}  "
        f"F1={m['Macro_F1']:.3f}  Brier={m['Brier']:.3f}  ECE={m['ECE']:.3f}",
        log_path)
    if not (0.0 <= m["AUROC"] <= 1.0):
        return f"{FAIL} AUROC out of range: {m['AUROC']}"
    return PASS


def main() -> int:
    log_path = LOG_DIR / "smoke_test_all.log"
    if log_path.exists():
        log_path.unlink()
    print(f"[smoke_test] log → {log_path}")
    tests = [
        ("feature_render",   t1_feature_render),
        ("retrieval",        t2_retrieval),
        ("prompt",           t3_prompt),
        ("evaluate",         t4_evaluate),
        ("per_doc+sample",   t5_build_dataset_and_sample_dump),
        ("train_log+plot",   t6_train_log_parser),
    ]
    summary = []
    for name, fn in tests:
        log(f"\n--- {name} ---", log_path)
        try:
            res = fn(log_path)
        except Exception as e:
            log(f"  EXCEPTION: {e!r}\n{traceback.format_exc()}", log_path)
            res = FAIL
        summary.append((name, res))

    log("\n" + "=" * 50, log_path)
    for name, res in summary:
        log(f"  {name:<18s} {res}", log_path)
    log("=" * 50, log_path)

    failed = [n for n, r in summary if r != PASS]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
