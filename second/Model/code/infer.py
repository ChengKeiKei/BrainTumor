"""
infer.py — Generate calibrated P(Yes) predictions for a held-out split using
a LoRA-adapted Mistral-7B-Instruct-v0.3.

We compute P(Yes) by comparing the logits of the "Yes" and "No" tokens at
the first response position — a continuous score suitable for AUROC/AUPR/
ECE/Brier downstream. This matches the BEEP paper's outcome-prediction
head (binary classification via constrained decoding).

Usage:

    python code/infer.py --tag Exp4__beep__beep                 # uses test.jsonl
    python code/infer.py --tag Exp4__beep__beep --split valid
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT      = Path(__file__).resolve().parents[2]    # Second_Recur/
MODEL_DIR = ROOT / "Model"
PROMPTS_D = MODEL_DIR / "prompts"
CKPT_D    = MODEL_DIR / "checkpoints"
RESULT_D  = MODEL_DIR / "results"

# Default base model; overridable per-call via --model-id or RAG_MODEL_ID.
# IMPORTANT: must match what train.py used for this tag, or scoring will be
# from a different model than was fine-tuned.  We persist the train-time
# model_id into checkpoints/<tag>/model_id.txt so infer can auto-detect it.
DEFAULT_MODEL_ID = (ROOT / "Model" / "local_models" / "BioMistral-7B-DARE-4bit").as_posix()
MODEL_ID  = os.environ.get("RAG_MODEL_ID", DEFAULT_MODEL_ID)

SPLIT_FILE = {"train": "train.jsonl", "valid": "valid.jsonl",
              "validation": "valid.jsonl", "test": "test.jsonl"}


def resolve_adapter(adapter_path: Path) -> Path:
    """Find the directory containing `adapter_config.json` for a cell.

    Layout history:
      v1 — flat:    checkpoints/<tag>/adapters/adapter_config.json
      v2 — nested:  checkpoints/<family>/<sub>/<tag>/adapters/adapter_config.json
                    e.g. checkpoints/BEEP/PubMedBERT/Exp4__beep__beep/adapters/

    We try the v1 path first (cheap). If that fails, fall back to a
    breadth-first search under `checkpoints/` for `<tag>/adapters/`.
    """
    if (adapter_path / "adapter_config.json").exists():
        return adapter_path
    nested = adapter_path / "adapters"
    if (nested / "adapter_config.json").exists():
        return nested

    tag = adapter_path.parent.name           # …/<tag>/adapters → tag
    ckpt_root = adapter_path.parents[1]      # …/checkpoints
    if ckpt_root.exists():
        for cfg in ckpt_root.rglob("adapter_config.json"):
            if cfg.parent.parent.name == tag:
                return cfg.parent
    return adapter_path


def get_yes_no_ids(tokenizer):
    """Return the token IDs the model **actually emits** for Yes/No after
    the Mistral [INST]...[/INST] turn ends.

    IMPORTANT context-sensitivity bug we previously had
    ----------------------------------------------------
    Mistral's BPE assigns *different* IDs to "Yes" depending on whether
    it appears in isolation (id 6360) or right after `[/INST]` (id 6381).
    Same for "No" (2538 vs 3269). Training uses
    `tokenizer.apply_chat_template`, which produces the *in-context*
    tokenization, so the LoRA adapter learns to maximise logits at the
    in-context IDs (6381 / 3269). If we score on the isolated IDs, we
    are reading from the wrong vocab slots and the calibrated
    probabilities are no longer the model's true predictions.

    This function builds the IDs by tokenising the answer **after** the
    real chat-template prefix and taking the first newly-generated token,
    which is exactly the position infer.py later reads logits at.
    """
    # Build the actual prefix the model sees (with add_generation_prompt=True
    # so we land on the token slot the LM will use to generate its answer).
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    yes_full   = tokenizer.encode(prefix + "Yes", add_special_tokens=False)
    no_full    = tokenizer.encode(prefix + "No",  add_special_tokens=False)
    yes_id = yes_full[len(prefix_ids)]
    no_id  = no_full [len(prefix_ids)]

    yes_decoded = tokenizer.decode([yes_id])
    no_decoded  = tokenizer.decode([no_id])
    print(
        f"[infer.py] Yes id={yes_id} ({yes_decoded!r})  "
        f"No id={no_id} ({no_decoded!r})  "
        f"[in-context tokenization after [/INST]]"
    )

    # Sanity check: if the in-context token doesn't decode to the literal
    # "Yes"/"No", the chat template has changed under us — bail loudly
    # rather than silently produce broken metrics.
    if yes_decoded.strip() != "Yes" or no_decoded.strip() != "No":
        raise RuntimeError(
            f"Yes/No token resolution failed: yes_id={yes_id} -> {yes_decoded!r}, "
            f"no_id={no_id} -> {no_decoded!r}. Inspect tokenizer / chat template."
        )
    return [yes_id], [no_id]


def predict_proba(model, tokenizer, user_prompt: str,
                  yes_ids, no_ids) -> float:
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    logits = model(mx.array([input_ids]))               # (1, L, V)
    last   = logits[0, -1, :]
    mx.eval(last)
    logit_yes = mx.max(mx.array([last[i].item() for i in yes_ids]))
    logit_no  = mx.max(mx.array([last[i].item() for i in no_ids]))
    p_yes = mx.softmax(mx.array([logit_yes, logit_no]))[0].item()
    return float(p_yes)


def infer(tag: str, split: str = "test", max_samples: int = -1,
          model_id: str | None = None,
          prompts_dir: str | None = None,
          output_tag: str | None = None) -> Path:
    """Run inference for one cell.

    The optional ``prompts_dir`` kwarg lets callers (notably
    ``run_cv_grid.py``) pin the prompt source explicitly, instead of relying
    on the default ``PROMPTS_D / tag`` lookup. This is required for the
    5-fold CV driver, which writes per-fold prompts under
    ``prompts/<base_tag>__cv__foldN/...`` and may not match the resolved
    inference tag exactly.
    """
    split = split.lower()
    if split not in SPLIT_FILE:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILE)}")

    prompts_path = Path(prompts_dir) if prompts_dir else (PROMPTS_D / tag)
    test_path = prompts_path / SPLIT_FILE[split]
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found")

    adapter = resolve_adapter(CKPT_D / tag / "adapters")
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json under {adapter}")

    out_dir  = RESULT_D / (output_tag or tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{split}.jsonl"

    # Resolve model_id: explicit arg > sidecar file written by train.py > default.
    sidecar = CKPT_D / tag / "model_id.txt"
    if model_id is None:
        if sidecar.exists():
            model_id = sidecar.read_text().strip()
        else:
            model_id = MODEL_ID
    print(f"[infer.py] loading {model_id}  +  adapter @ {adapter}")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id, adapter_path=str(adapter))
    model.eval()
    load_seconds = time.perf_counter() - t0

    yes_ids, no_ids = get_yes_no_ids(tokenizer)

    records = []
    with test_path.open() as fh:
        for line in fh:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]

    preds = []
    t0 = time.perf_counter()
    for i, rec in enumerate(records, 1):
        user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        score = predict_proba(model, tokenizer, user_msg, yes_ids, no_ids)
        pred_row = {
            "patient_id": rec.get("patient_id"),
            "label":      int(rec.get("label", -1)),
            "score":      score,
        }
        if "doc_rank" in rec:
            pred_row["doc_rank"] = rec["doc_rank"]
            pred_row["doc_pmid"] = rec.get("doc_pmid", "")
        preds.append(pred_row)
        if i % 25 == 0 or i == len(records):
            print(f"  [{i:4d}/{len(records)}]  last p={score:.3f}")
    pred_seconds = time.perf_counter() - t0

    with out_path.open("w") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")

    timing = {
        "tag": tag, "split": split, "n": len(preds),
        "load_seconds":   round(load_seconds, 4),
        "predict_seconds": round(pred_seconds, 4),
        "predict_seconds_per_sample": round(pred_seconds / max(len(preds), 1), 4),
    }
    (out_dir / f"timing_{split}.json").write_text(json.dumps(timing, indent=2))
    print(f"\n[infer.py] {len(preds)} predictions → {out_path}")
    print(f"[infer.py] timing  → {out_dir / f'timing_{split}.json'}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test", choices=list(SPLIT_FILE))
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--model-id", default="",
                    help=f"Override base model. Default = {MODEL_ID}")
    args = ap.parse_args()
    infer(args.tag, args.split, args.max_samples,
          model_id=args.model_id or None)
