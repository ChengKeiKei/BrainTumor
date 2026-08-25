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
import shutil
import time
import traceback
from pathlib import Path

import yaml

import build_dataset
import train as train_mod
import infer as infer_mod
import evaluate as evaluate_mod
from build_dataset import config_tag

ROOT      = Path(__file__).resolve().parents[2]    # Second_Recur/
MODEL_DIR = ROOT / "Model"
CFG_DIR   = MODEL_DIR / "configs"
RESULT_D  = MODEL_DIR / "results"

# Map of "friendly model alias" → "fully-qualified model id" used by mlx_lm.
# Used so cells in grid.yaml stay readable ("biomistral" vs an HF path).
MODEL_ALIAS = {
    "biomistral": (MODEL_DIR / "local_models" / "BioMistral-7B-DARE-4bit").as_posix(),
    "mistral":    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}


def _model_alias_to_id(alias: str) -> str:
    if alias in MODEL_ALIAS:
        return MODEL_ALIAS[alias]
    return alias  # treat anything else as already-resolved (HF path or local path)


def _suffix_for_model(alias_or_id: str) -> str:
    """Short suffix appended to a tag so we don't overwrite checkpoints when
    running the same (exp,ret,rrk) cell on a different base model."""
    return next((a for a, mid in MODEL_ALIAS.items()
                 if mid == alias_or_id or a == alias_or_id), None) \
           or Path(alias_or_id).name.lower().replace("-", "_")


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _expand_grid(grid_cfg: dict, experiments: list[str] | None = None,
                 retrievers: list[str] | None = None,
                 rerankers:  list[str] | None = None,
                 models:     list[str] | None = None
                 ) -> list[tuple[str, str, str, str]]:
    """Return one tuple (exp, retriever, reranker, model_alias) per planned cell."""
    exps        = experiments or grid_cfg.get("experiments", ["Exp1","Exp2","Exp3","Exp4"])
    grid_models = models      or grid_cfg.get("models",      ["biomistral"])
    cells = []
    for c in grid_cfg["cells"]:
        ret = c["retriever"]; rrk = c["reranker"]
        if retrievers and ret not in retrievers and ret != "none":
            continue
        if rerankers  and rrk not in rerankers  and rrk != "none":
            continue
        # Cell can override the global model list with `models: [...]`.
        cell_models = c.get("models", grid_models)
        for exp in exps:
            for m in cell_models:
                cells.append((exp, ret, rrk, m))
    return cells


def _full_tag(exp: str, ret: str, rrk: str, model_alias: str,
              captions_version: str = "v1") -> str:
    """Cell tag used everywhere on disk (prompts/, checkpoints/, results/).

    A non-default `captions_version` (e.g. "v2_context", "v3_structured")
    is appended as an extra suffix so multiple RadFM-prompt experiments can
    coexist without clobbering each other:
        ExpC_TxVLM__beep__minilm__biomistral                   (v1, default)
        ExpC_TxVLM__beep__minilm__biomistral__v2_context       (v2)
        ExpC_TxVLM__beep__minilm__biomistral__v3_structured    (v3)
    """
    base = f"{config_tag(exp, ret, rrk)}__{_suffix_for_model(model_alias)}"
    if captions_version and captions_version != "v1":
        base = f"{base}__{captions_version}"
    return base


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
             models=None,
             captions_version: str = "v1",
             only_tags: list[str] | None = None,
             skip_existing: bool = False,
             dry_run=False, skip_build=False, skip_train=False, skip_infer=False):

    # Propagate the captions choice to feature_render via env var so
    # build_dataset → render_split picks up the right MRI captions file.
    os.environ["RAG_CAPTIONS_VERSION"] = captions_version
    print(f"[run_grid] RAG_CAPTIONS_VERSION = {captions_version}")

    cells = _expand_grid(grid_cfg, experiments, retrievers, rerankers, models)

    if only_tags:
        want = set(only_tags)
        cells = [c for c in cells if _full_tag(*c, captions_version) in want]
    if skip_existing:
        cells = [c for c in cells if not _tag_is_done(_full_tag(*c, captions_version))]

    print(f"Planning {len(cells)} cells:")
    for cell in cells:
        print(f"  - {cell}  →  tag = {_full_tag(*cell, captions_version)}")
    if dry_run:
        return

    train_kw = default_cfg["train"]
    retr_kw  = default_cfg["retrieval"]
    eval_kw  = default_cfg["eval"]

    agg_csv  = RESULT_D / "aggregate.csv"
    failures: list[tuple[tuple, str]] = []

    # Track which (exp,ret,rrk) prompt dirs we've already built — different
    # `models` cells share the same prompts, so we only need to build once.
    built_prompt_tags: set[str] = set()

    for cell in cells:
        exp, ret, rrk, model_alias = cell
        # Prompt directory carries the captions-version suffix too, because
        # the same (exp, ret, rrk) cell renders DIFFERENT prompts under v1 vs
        # v2/v3 (the VLM block changes). Without this split the v2 run would
        # silently reuse v1 prompts.
        if captions_version != "v1":
            prompt_tag = f"{config_tag(exp, ret, rrk)}__{captions_version}"
        else:
            prompt_tag = config_tag(exp, ret, rrk)
        full_tag   = _full_tag(*cell, captions_version)      # with model + caps suffix
        model_id   = _model_alias_to_id(model_alias)
        print(f"\n{'='*70}\n>>> {full_tag}   [model={model_alias}]\n{'='*70}")
        try:
            if not skip_build and prompt_tag not in built_prompt_tags:
                build_dataset.build_one_config(
                    exp, ret, rrk,
                    top_k_sparse=retr_kw["top_k_sparse"],
                    top_k_dense=retr_kw["top_k_dense"],
                    final_top_k=retr_kw["final_top_k"],
                    abstract_max_chars=retr_kw.get("abstract_max_chars", 2000),
                )
                # build_dataset.build_one_config always writes to
                #   MODEL_DIR/prompts/<config_tag(exp,ret,rrk)>
                # (it doesn't know about captions_version). For non-v1 runs,
                # move the freshly-built dir to the suffixed prompt_tag so
                # (a) train.train(prompts_dir=...) finds it under the suffixed
                # path that we want and (b) a later v1 (or different version)
                # rebuild won't overwrite the data we just built.
                if captions_version != "v1":
                    unsuffixed = MODEL_DIR / "prompts" / config_tag(exp, ret, rrk)
                    target     = MODEL_DIR / "prompts" / prompt_tag
                    if target.exists():
                        shutil.rmtree(target)
                    if unsuffixed.exists():
                        unsuffixed.rename(target)
                built_prompt_tags.add(prompt_tag)

            # Train and infer write under <full_tag> so different base models
            # don't clobber each other.  But our prompts live under
            # <prompt_tag>; reuse them by overriding prompts_dir.
            prompts_path = MODEL_DIR / "prompts" / prompt_tag

            if not skip_train:
                train_mod.train(
                    tag=full_tag,
                    prompts_dir=str(prompts_path),
                    model_id=model_id,
                    steps=train_kw["steps"],
                    batch_size=train_kw["batch_size"],
                    lr=float(train_kw["lr"]),
                    save_steps=train_kw["save_steps"],
                    eval_steps=train_kw["eval_steps"],
                    logging_steps=train_kw["logging_steps"],
                )
            # infer.py looks under prompts/<full_tag> by default; mirror the
            # prompt jsonl files there so we don't duplicate disk usage.
            mirror_dir = MODEL_DIR / "prompts" / full_tag
            if not mirror_dir.exists():
                mirror_dir.symlink_to(prompts_path, target_is_directory=True)

            pred_path = (RESULT_D / full_tag / "predictions_test.jsonl")
            if not skip_infer:
                pred_path = infer_mod.infer(full_tag, split="test", model_id=model_id)
            metrics = evaluate_mod.evaluate(
                pred_path, threshold=eval_kw["threshold"],
                bootstrap=eval_kw["bootstrap"])
            _append_aggregate_row(full_tag, metrics, agg_csv)
        except Exception as exc:
            print(f"[run_grid] FAILED on {full_tag}: {exc!r}")
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
                    choices=["Exp1", "Exp2", "Exp3", "Exp4",
                             "ExpA_TxNoMol", "ExpA_Tx", "ExpB_TxRadiomic",
                             "ExpC_TxVLM", "ExpD_TxRadVLM"])
    ap.add_argument("--retrievers",     nargs="*", default=None)
    ap.add_argument("--rerankers",      nargs="*", default=None)
    ap.add_argument("--models",         nargs="*", default=None,
                    help="Override base-model alias list "
                         "(e.g. --models biomistral mistral)")
    ap.add_argument("--captions-version", default="v1",
                    help="Which RadFM captions to feed the VLM block "
                         "(must match a Dataset/Processed/mri_captions[_<version>].csv "
                         "file produced by VLM/run_radfm_captions.py). "
                         "Non-v1 values are appended to the cell tag so different "
                         "prompt experiments don't overwrite each other. "
                         "Default = v1.")
    ap.add_argument("--only-tags",      nargs="*", default=None,
                    help="Run only cells whose tag is in this list "
                         "(e.g. Exp4__beep__minilm Exp4__nomic__beep).")
    ap.add_argument("--skip-existing",  action="store_true",
                    help="Skip cells that already have a valid "
                         "predictions_test.metrics.json (non-NaN AUROC).")
    ap.add_argument("--dry-run",        action="store_true")
    ap.add_argument("--skip-build",     action="store_true")
    ap.add_argument("--skip-train",     action="store_true")
    ap.add_argument("--skip-infer",     action="store_true")
    args = ap.parse_args()

    default_cfg = _load_yaml(Path(args.default_config))
    grid_cfg    = _load_yaml(Path(args.grid_config))
    run_grid(default_cfg, grid_cfg,
             experiments=args.experiments,
             retrievers=args.retrievers,
             rerankers=args.rerankers,
             models=args.models,
             captions_version=args.captions_version,
             only_tags=args.only_tags,
             skip_existing=args.skip_existing,
             dry_run=args.dry_run,
             skip_build=args.skip_build,
             skip_train=args.skip_train,
             skip_infer=args.skip_infer)
