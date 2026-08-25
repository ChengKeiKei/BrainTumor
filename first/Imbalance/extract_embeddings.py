"""
Extract frozen Mistral/BioMistral prompt embeddings for First_Recur.

This is the leakage-safe LLM-aware imbalance path:

    RAG prompt -> frozen LLM embedding -> class-weighted XGBoost

Because the encoder is frozen and never trained on First_Recur labels, train
embeddings are not in-sample predictions and do not need OOF generation.
"""
from __future__ import annotations

import argparse
import json
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "Model" / "prompts"
OUT_ROOT = ROOT / "Imbalance" / "embeddings"

ENCODER_HF_ID = {
    "mistral":     "mistralai/Mistral-7B-Instruct-v0.3",
    "biomistral":  "BioMistral/BioMistral-7B-DARE",
    "gemma3-9b":   "google/gemma-3-9b-it",
    "gemma3-27b":  "google/gemma-3-27b-it",
    "qwen3-8b":    "Qwen/Qwen3-8B",
    "llama3-8b":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "medllama3":   "ProbeMedicalYonseiMAILab/medllama3-v20",
}
SPLIT_FILE = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}


def _resolve_encoder(name_or_id: str) -> tuple[str, str]:
    """Map shortname to HF id, or pass through any HF 'org/repo' string.

    Returns (slug_for_directory, hf_repo_id).  If a custom HF id is used,
    the slug is the lowercased basename with non-alphanum replaced by '-'.
    """
    if name_or_id in ENCODER_HF_ID:
        return name_or_id, ENCODER_HF_ID[name_or_id]
    if "/" in name_or_id:
        slug = name_or_id.split("/")[-1].lower()
        slug = "".join(c if c.isalnum() else "-" for c in slug).strip("-")
        return slug, name_or_id
    raise ValueError(
        f"Unknown encoder shortname: {name_or_id!r}. "
        f"Either pass a registered shortname ({list(ENCODER_HF_ID)}) "
        "or a full HuggingFace repo id like 'google/gemma-3-9b-it'."
    )


def _device(requested: str = "auto") -> str:
    """Resolve a device string. 'auto' picks mps > cuda > cpu.

    Explicit 'mps'/'cuda' fall back to cpu (with a warning) if unavailable, so
    a long-running batch never crashes mid-cell on hardware-availability.
    """
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("[device] mps requested but unavailable; falling back to cpu")
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[device] cuda requested but unavailable; falling back to cpu")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device: {requested!r}")


def _load_encoder(encoder: str, device_request: str = "auto"):
    _, hf_id = _resolve_encoder(encoder)
    device = _device(device_request)
    print(f"[encoder] loading {hf_id} on {device}")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    print(f"[encoder] loaded in {time.perf_counter() - t0:.1f}s")
    return tok, model, device


POOLING_CHOICES = ("last", "mean", "max", "all")


@torch.no_grad()
def _embed_one(tok, model, device: str, user_msg: str, max_length: int,
               poolings: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Run one forward pass and return the requested pooled vectors.

    Mean and max pooling honour the attention mask so padding tokens (if any)
    do not contribute. For batch-size 1 with truncation only, the mask is all
    ones, so this is equivalent to pooling over every kept token.
    """
    chat = [{"role": "user", "content": user_msg}]
    formatted = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    enc = tok(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1]                       # [1, T, H]
    mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)  # [1, T, 1]

    result: dict[str, np.ndarray] = {}
    if "last" in poolings:
        # Index of the last non-pad token (robust if padding ever appears).
        last_idx = int(enc["attention_mask"][0].sum().item()) - 1
        result["last"] = h[0, last_idx, :].detach().to("cpu").float().numpy()
    if "mean" in poolings:
        denom = mask.sum(dim=1).clamp(min=1)         # [1, 1]
        mean_vec = (h * mask).sum(dim=1) / denom      # [1, H]
        result["mean"] = mean_vec[0].detach().to("cpu").float().numpy()
    if "max" in poolings:
        very_neg = torch.finfo(h.dtype).min
        masked = h.masked_fill(mask == 0, very_neg)
        max_vec, _ = masked.max(dim=1)                # [1, H]
        result["max"] = max_vec[0].detach().to("cpu").float().numpy()
    return result


def _resolve_poolings(pooling: str) -> tuple[str, ...]:
    if pooling == "all":
        return ("last", "mean", "max")
    if pooling in ("last", "mean", "max"):
        return (pooling,)
    raise ValueError(f"Unknown pooling: {pooling!r}; choose from {POOLING_CHOICES}")


def extract_for_cell(cell_tag: str, encoder: str,
                     splits=("train", "valid", "test"),
                     max_length: int = 3072,
                     pooling: str = "all",
                     device_request: str = "auto") -> Path:
    cell_prompts = PROMPTS_DIR / cell_tag
    if not cell_prompts.exists():
        raise FileNotFoundError(f"No prompts dir for {cell_tag}: {cell_prompts}")

    slug, _ = _resolve_encoder(encoder)
    poolings = _resolve_poolings(pooling)
    # New layout inserts a pooling sub-directory: embeddings/<enc>/<pool>/<cell>/.
    # The old flat layout embeddings/<enc>/<cell>/ is treated as legacy "last"
    # by run_llm_xgb.py, so previously extracted last-token runs are not wasted.
    out_dirs = {p: OUT_ROOT / slug / p / cell_tag for p in poolings}
    for p, d in out_dirs.items():
        d.mkdir(parents=True, exist_ok=True)

    tok, model, device = _load_encoder(encoder, device_request)
    timing = {p: {"per_split_seconds": {}} for p in poolings}
    hidden_dim = None

    for split in splits:
        src = cell_prompts / SPLIT_FILE[split]
        if not src.exists():
            print(f"[{cell_tag}/{split}] missing {src}; skipping")
            continue

        ids: list[str] = []
        labels: list[int] = []
        bucket: dict[str, list[np.ndarray]] = {p: [] for p in poolings}
        t0 = time.perf_counter()
        with src.open() as f:
            for line in f:
                rec = json.loads(line)
                user_msg = next(
                    (m["content"] for m in rec["messages"] if m["role"] == "user"),
                    "",
                )
                vecs = _embed_one(tok, model, device, user_msg, max_length, poolings)
                for p in poolings:
                    bucket[p].append(vecs[p])
                ids.append(str(rec.get("patient_id", "")))
                labels.append(int(rec.get("label", -1)))

        elapsed = time.perf_counter() - t0
        ids_df = pd.DataFrame({"Patient_ID": ids, "label": labels})
        for p in poolings:
            timing[p]["per_split_seconds"][split] = round(elapsed, 2)
            arr = np.stack(bucket[p], axis=0).astype(np.float32)
            hidden_dim = int(arr.shape[1])
            np.save(out_dirs[p] / f"{split}.npy", arr)
            ids_df.to_csv(out_dirs[p] / f"{split}_ids.csv", index=False)
            print(f"[{cell_tag}/{split}/{p:<4}] {len(ids)} prompts -> {arr.shape} "
                  f"({elapsed:.1f}s shared fwd)")

    for p in poolings:
        meta = {
            "cell_tag": cell_tag,
            "encoder": slug,
            "encoder_input": encoder,
            "encoder_hf_id": _resolve_encoder(encoder)[1],
            "device": device,
            "max_length": max_length,
            "pooling": p,
            "uses_lora": False,
            "hidden_dim": hidden_dim,
            "timing": timing[p],
        }
        (out_dirs[p] / "meta.json").write_text(json.dumps(meta, indent=2))

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    # Return the first pooling output dir (informational only).
    return next(iter(out_dirs.values()))


def list_cells_with_prompts() -> list[str]:
    return sorted(
        d.name for d in PROMPTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "test.jsonl").exists()
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--encoder", default="mistral",
                    help="shortname (mistral|biomistral|gemma3-9b|gemma3-27b|qwen3-8b|llama3-8b|medllama3) "
                         "or any HuggingFace repo id 'org/repo'")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                    choices=list(SPLIT_FILE))
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--pooling", default="all", choices=list(POOLING_CHOICES),
                    help="which pooled vector(s) to save; 'all' computes "
                         "last/mean/max in one forward pass (recommended).")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "mps", "cuda", "cpu"],
                    help="force a device. Use 'cpu' to bypass an MPS hang "
                         "during the 14 GB model transfer on macOS.")
    args = ap.parse_args()

    if not args.cell_tag and not args.all:
        ap.error("supply --cell-tag or --all")
    cells = list_cells_with_prompts() if args.all else [args.cell_tag]
    print(f"Extracting {args.encoder} embeddings (pooling={args.pooling}) "
          f"for {len(cells)} cell(s)")
    failures = 0
    for cell in cells:
        try:
            extract_for_cell(cell, args.encoder, args.splits, args.max_length,
                             pooling=args.pooling, device_request=args.device)
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {cell}: {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
