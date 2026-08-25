"""
run_grid.py — Drive the full 4 × 11 = 44-cell experiment grid end-to-end.

For every cell (experiment × retriever × reranker) defined in
`configs/grid.yaml`, this script will:

   1. build_dataset.build_one_config(...)   →  prompts/<tag>/{train,valid,test}.jsonl
   2. train.train(tag=<tag>, ...)           →  checkpoints/<tag>/adapters/
   3. infer.infer(<tag>, "test")            →  results/<tag>/predictions_test.jsonl
   4. evaluate.evaluate(...)                →  results/<tag>/predictions_test.metrics.json
   5. Append a row to results/aggregate.csv with the headline metrics.

Use --dry-run to print the planned cells without launching anything heavy.

Filtering options:
   --experiments Exp1 Exp4
   --retrievers beep
   --rerankers beep minilm
   --skip-train      (just rebuild prompts + run infer/eval against existing adapter)
   --skip-build      (assume prompts/ already populated)
   --skip-infer      (assume predictions/ already populated, just re-eval)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import yaml

import build_dataset
import train as train_mod
import infer as infer_mod
import evaluate as evaluate_mod
from build_dataset import config_tag

ROOT      = Path(__file__).resolve().parents[2]    # First_Recur/
MODEL_DIR = ROOT / "Model"
CFG_DIR   = MODEL_DIR / "configs"
RESULT_D  = MODEL_DIR / "results"

MODEL_ALIAS = {
    "mistral":    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "biomistral": (MODEL_DIR / "local_models" / "BioMistral-7B-DARE-4bit").as_posix(),
}


def _resolve_model_id(alias: str) -> str:
    return MODEL_ALIAS.get(alias, alias)


def _llm_suffix(alias: str) -> str:
    """Tag suffix per LLM. Empty string for the default Mistral so existing
    artifacts (Exp4__beep__beep) keep their tag and aggregate.csv rows."""
    return "" if alias == "mistral" else alias


def _full_tag(exp: str, ret: str, rrk: str, llm_alias: str) -> str:
    base = config_tag(exp, ret, rrk)
    suffix = _llm_suffix(llm_alias)
    return f"{base}__{suffix}" if suffix else base


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _expand_grid(grid_cfg: dict, experiments: list[str] | None = None,
                 retrievers: list[str] | None = None,
                 rerankers:  list[str] | None = None) -> list[tuple[str, str, str]]:
    exps = experiments or grid_cfg.get("experiments", ["Exp1", "Exp2", "Exp3", "Exp4"])
    cells = []
    for c in grid_cfg["cells"]:
        ret = c["retriever"]; rrk = c["reranker"]
        if retrievers and ret not in retrievers and ret != "none":
            continue
        if rerankers  and rrk not in rerankers  and rrk != "none":
            continue
        for exp in exps:
            cells.append((exp, ret, rrk))
    return cells


def _append_aggregate_row(tag: str, metrics: dict, agg_csv: Path) -> None:
    headers = [
        "tag", "n", "n_pos", "n_neg",
        "AUROC", "AUROC_lo", "AUROC_hi",
        "AUPRC", "AUPRC_lo", "AUPRC_hi",
        "Macro_F1", "Accuracy", "Sensitivity", "Specificity",
        "MCC", "Brier", "ECE",
    ]
    new = not agg_csv.exists()
    agg_csv.parent.mkdir(parents=True, exist_ok=True)
    with agg_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(headers)
        w.writerow([
            tag, metrics["n"], metrics["n_pos"], metrics["n_neg"],
            metrics["AUROC"], metrics["AUROC_95CI"][0], metrics["AUROC_95CI"][1],
            metrics["AUPRC"], metrics["AUPRC_95CI"][0], metrics["AUPRC_95CI"][1],
            metrics["Macro_F1"], metrics["Accuracy"],
            metrics["Sensitivity"], metrics["Specificity"],
            metrics["MCC"], metrics["Brier"], metrics["ECE"],
        ])


def _tag_is_done(tag: str) -> bool:
    """Has this cell already produced a valid (non-NaN AUROC) metrics file?"""
    m = RESULT_D / tag / "predictions_test.metrics.json"
    if not m.exists():
        return False
    try:
        d = json.loads(m.read_text())
        auroc = d.get("AUROC")
        return auroc is not None and isinstance(auroc, (int, float)) \
            and auroc == auroc  # False if NaN
    except Exception:
        return False


def run_grid(default_cfg: dict, grid_cfg: dict, *,
             experiments=None, retrievers=None, rerankers=None,
             llm: str = "mistral",
             only_tags: list[str] | None = None,
             skip_existing: bool = False,
             dry_run=False, skip_build=False, skip_train=False, skip_infer=False):

    cells = _expand_grid(grid_cfg, experiments, retrievers, rerankers)

    if only_tags:
        want = set(only_tags)
        cells = [c for c in cells if _full_tag(*c, llm) in want]
    if skip_existing:
        cells = [c for c in cells if not _tag_is_done(_full_tag(*c, llm))]

    print(f"Planning {len(cells)} cells with LLM={llm}:")
    for cell in cells:
        print("  -", _full_tag(*cell, llm))
    if dry_run:
        return

    # Pin the chosen model for this whole run (train.py + infer.py read this env var).
    model_id = _resolve_model_id(llm)
    os.environ["RAG_MODEL_ID"] = model_id
    print(f"[run_grid] RAG_MODEL_ID = {model_id}")

    train_kw = default_cfg["train"]
    retr_kw  = default_cfg["retrieval"]
    eval_kw  = default_cfg["eval"]

    agg_csv  = RESULT_D / "aggregate.csv"
    failures: list[tuple[tuple, str]] = []

    for cell in cells:
        exp, ret, rrk = cell
        base_tag = config_tag(exp, ret, rrk)
        tag      = _full_tag(exp, ret, rrk, llm)
        print(f"\n{'='*70}\n>>> {tag}   [llm={llm}]\n{'='*70}")
        try:
            if not skip_build:
                # Prompt rendering is LLM-agnostic (same EHR text), so we always
                # build under the un-suffixed base_tag and let train/infer reuse
                # those prompts regardless of which model is fine-tuned on them.
                build_dataset.build_one_config(
                    exp, ret, rrk,
                    top_k_sparse=retr_kw["top_k_sparse"],
                    top_k_dense=retr_kw["top_k_dense"],
                    final_top_k=retr_kw["final_top_k"],
                    abstract_max_chars=retr_kw.get("abstract_max_chars", 2000),
                )
            if not skip_train:
                train_mod.train(
                    tag=tag,
                    prompts_dir=str(MODEL_DIR / "prompts" / base_tag),
                    steps=train_kw["steps"],
                    batch_size=train_kw["batch_size"],
                    lr=float(train_kw["lr"]),
                    save_steps=train_kw["save_steps"],
                    eval_steps=train_kw["eval_steps"],
                    logging_steps=train_kw["logging_steps"],
                )
            pred_path = (RESULT_D / tag / "predictions_test.jsonl")
            if not skip_infer:
                pred_path = infer_mod.infer(
                    tag, split="test",
                    prompts_dir=str(MODEL_DIR / "prompts" / base_tag),
                )
            metrics = evaluate_mod.evaluate(
                pred_path, threshold=eval_kw["threshold"],
                bootstrap=eval_kw["bootstrap"])
            _append_aggregate_row(tag, metrics, agg_csv)
        except Exception as exc:
            print(f"[run_grid] FAILED on {tag}: {exc!r}")
            traceback.print_exc()
            failures.append((cell, repr(exc)))

    print("\n" + "=" * 70)
    print(f"Done. Aggregate: {agg_csv}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for c, e in failures:
            print(" ", c, "→", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--default-config", default=str(CFG_DIR / "default.yaml"))
    ap.add_argument("--grid-config",    default=str(CFG_DIR / "grid.yaml"))
    ap.add_argument("--experiments",    nargs="*", default=None,
                    choices=["Exp1", "Exp2", "Exp3", "Exp4", "Exp5", "Exp6"])
    ap.add_argument("--retrievers",     nargs="*", default=None)
    ap.add_argument("--rerankers",      nargs="*", default=None)
    ap.add_argument("--only-tags",      nargs="*", default=None,
                    help="Run only cells whose tag is in this list "
                         "(e.g. Exp4__beep__minilm Exp4__beep__bge_m3).")
    ap.add_argument("--skip-existing",  action="store_true",
                    help="Skip cells that already have a valid "
                         "predictions_test.metrics.json (non-NaN AUROC).")
    ap.add_argument("--dry-run",        action="store_true")
    ap.add_argument("--skip-build",     action="store_true")
    ap.add_argument("--skip-train",     action="store_true")
    ap.add_argument("--skip-infer",     action="store_true")
    ap.add_argument("--llm",            default="mistral",
                    choices=list(MODEL_ALIAS.keys()),
                    help="Which base LLM to fine-tune. mistral=keep legacy "
                         "tag (no suffix). biomistral=domain-pretrained, "
                         "tags get __biomistral suffix so existing artifacts "
                         "stay intact.")
    args = ap.parse_args()

    default_cfg = _load_yaml(Path(args.default_config))
    grid_cfg    = _load_yaml(Path(args.grid_config))
    run_grid(default_cfg, grid_cfg,
             experiments=args.experiments,
             retrievers=args.retrievers,
             rerankers=args.rerankers,
             llm=args.llm,
             only_tags=args.only_tags,
             skip_existing=args.skip_existing,
             dry_run=args.dry_run,
             skip_build=args.skip_build,
             skip_train=args.skip_train,
             skip_infer=args.skip_infer)
