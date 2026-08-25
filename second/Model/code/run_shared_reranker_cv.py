"""Shared-adapter five-reranker cross-validation for Second_Recur.

Each fold trains one LoRA adapter per LLM on a deterministic, class-balanced
mixture of BEEP, MiniLM, MedCPT, ColBERT, and BGE-M3 RAG contexts. The same
adapter is then held fixed while every reranker is evaluated separately.
This isolates reranker choice from independent adapter-training variation.

All calibration is fold-local: each reranker's Platt scaler is fitted on that
fold's validation patients and applied to only that fold's held-out patients.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
from pathlib import Path

import run_cv_grid

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "Model"
PROMPT_DIR = MODEL_DIR / "prompts"
RESULT_DIR = MODEL_DIR / "results"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

EXPERIMENT = "ExpC_TxVLM"
RETRIEVER = "beep"
CAPTIONS = "v3_structured"
RERANKERS = ("beep", "minilm", "medcpt", "colbert", "bge_m3")
SPLIT_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl")
N_PATIENTS = 147


def prompt_tag(reranker: str, fold_id: int) -> str:
    return f"{EXPERIMENT}__{RETRIEVER}__{reranker}__cv__fold{fold_id}__{CAPTIONS}"


def mixed_prompt_tag(fold_id: int) -> str:
    return f"{EXPERIMENT}__{RETRIEVER}__mixed5__cv__fold{fold_id}__{CAPTIONS}"


def adapter_tag(llm: str, fold_id: int) -> str:
    return f"{EXPERIMENT}__{RETRIEVER}__mixed5__{llm}__sharedcv__fold{fold_id}__{CAPTIONS}"


def eval_tag(reranker: str, llm: str, fold_id: int) -> str:
    return f"{EXPERIMENT}__{RETRIEVER}__{reranker}__{llm}__sharedcv__fold{fold_id}__{CAPTIONS}"


def aggregate_tag(reranker: str, llm: str) -> str:
    return f"{EXPERIMENT}__{RETRIEVER}__{reranker}__{llm}__sharedcv__{CAPTIONS}"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _stable_order(patient_ids: list[str], fold_id: int, split_file: str,
                  label: int) -> list[str]:
    def key(pid: str) -> str:
        raw = f"sharedcv-v1|{fold_id}|{split_file}|{label}|{pid}".encode()
        return hashlib.sha256(raw).hexdigest()
    return sorted(patient_ids, key=key)


def ensure_source_prompts(fold_id: int, *, force: bool = False) -> None:
    for reranker in RERANKERS:
        tag = prompt_tag(reranker, fold_id)
        out = PROMPT_DIR / tag
        complete = all((out / name).exists() for name in SPLIT_FILES)
        if complete and not force:
            print(f"  [reuse prompts] {tag}")
            continue
        built = run_cv_grid._build_one_fold(
            fold_id,
            EXPERIMENT,
            RETRIEVER,
            reranker,
            captions_version=CAPTIONS,
        )
        if built != tag:
            raise RuntimeError(f"prompt-tag mismatch: built={built}, expected={tag}")


def build_mixed_prompts(fold_id: int, *, force: bool = False) -> Path:
    out = PROMPT_DIR / mixed_prompt_tag(fold_id)
    if not force and all((out / name).exists() for name in SPLIT_FILES) \
            and (out / "config.json").exists():
        print(f"  [reuse mixed prompts] {out.name}")
        return out

    out.mkdir(parents=True, exist_ok=True)
    config: dict = {
        "design": "shared adapter with deterministic class-balanced mixed-reranker contexts",
        "experiment": EXPERIMENT,
        "retriever": RETRIEVER,
        "rerankers": list(RERANKERS),
        "captions_version": CAPTIONS,
        "fold": fold_id,
        "assignment_hash": "sha256(sharedcv-v1|fold|split|label|patient_id)",
        "splits": {},
    }

    for split_file in SPLIT_FILES:
        by_reranker: dict[str, dict[str, dict]] = {}
        for reranker in RERANKERS:
            rows = _read_jsonl(PROMPT_DIR / prompt_tag(reranker, fold_id) / split_file)
            by_reranker[reranker] = {str(row["patient_id"]): row for row in rows}

        id_sets = [set(rows) for rows in by_reranker.values()]
        if not id_sets or any(ids != id_sets[0] for ids in id_sets[1:]):
            raise RuntimeError(f"patient mismatch across rerankers: fold={fold_id}, split={split_file}")

        reference = by_reranker[RERANKERS[0]]
        assignments: dict[str, str] = {}
        counts = {reranker: {"n": 0, "n_pos": 0} for reranker in RERANKERS}
        for label in (0, 1):
            ids = [pid for pid, row in reference.items() if int(row["label"]) == label]
            ids = _stable_order(ids, fold_id, split_file, label)
            offset = (fold_id + label) % len(RERANKERS)
            for i, pid in enumerate(ids):
                reranker = RERANKERS[(i + offset) % len(RERANKERS)]
                assignments[pid] = reranker
                counts[reranker]["n"] += 1
                counts[reranker]["n_pos"] += label

        mixed_rows = []
        for pid in sorted(reference):
            reranker = assignments[pid]
            row = dict(by_reranker[reranker][pid])
            row["sharedcv_training_reranker"] = reranker
            mixed_rows.append(row)

        _write_jsonl(out / split_file, mixed_rows)
        config["splits"][split_file] = {
            "n": len(mixed_rows),
            "n_pos": int(sum(int(row["label"]) for row in mixed_rows)),
            "reranker_counts": counts,
            "assignments": assignments,
        }

    source_sample = PROMPT_DIR / prompt_tag(RERANKERS[0], fold_id) / "sample_chat.txt"
    if source_sample.exists():
        shutil.copy2(source_sample, out / "sample_chat.txt")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  [built mixed prompts] {out}")
    return out


def release_retrieval_models() -> None:
    try:
        import retrieval_pipeline
        retrieval_pipeline._RERANKER_CACHE.clear()
        retrieval_pipeline._DENSE_CACHE.clear()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def model_id(llm: str) -> str:
    if llm == "mistral":
        return "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    if llm == "biomistral":
        return (MODEL_DIR / "local_models" / "BioMistral-7B-DARE-4bit").as_posix()
    raise ValueError(llm)


def train_adapter(llm: str, fold_id: int, args: argparse.Namespace) -> str:
    import train

    tag = adapter_tag(llm, fold_id)
    adapter_dir = CHECKPOINT_DIR / tag / "adapters"
    done = (
        (adapter_dir / "adapter_config.json").exists()
        and (adapter_dir / "adapters.safetensors").exists()
        and (RESULT_DIR / tag / "timing_train.json").exists()
    )
    if args.skip_existing and done:
        print(f"  [reuse adapter] {tag}")
        return tag

    train.train(
        tag=tag,
        prompts_dir=str(PROMPT_DIR / mixed_prompt_tag(fold_id)),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        model_id=model_id(llm),
        grad_checkpoint=args.grad_checkpoint,
        max_seq_length=args.max_seq_length,
    )
    return tag


def evaluate_reranker(llm: str, reranker: str, fold_id: int,
                      adapter: str, *, skip_existing: bool) -> Path:
    import evaluate
    import infer

    tag = eval_tag(reranker, llm, fold_id)
    out = RESULT_DIR / tag
    calibrated = out / "predictions_test_platt.jsonl"
    if skip_existing and calibrated.exists():
        print(f"  [reuse evaluation] {tag}")
        return calibrated

    prompts = str(PROMPT_DIR / prompt_tag(reranker, fold_id))
    valid = infer.infer(adapter, split="valid", model_id=model_id(llm),
                        prompts_dir=prompts, output_tag=tag)
    test = infer.infer(adapter, split="test", model_id=model_id(llm),
                       prompts_dir=prompts, output_tag=tag)
    calibrated = run_cv_grid._platt_calibrate(valid, test, out)
    metrics = evaluate.evaluate(calibrated, threshold=0.5, bootstrap=1000)
    print(f"  [{llm}/{reranker}/fold{fold_id}] "
          f"AUROC={metrics['AUROC']:.3f}, Macro-F1={metrics['Macro_F1']:.3f}")
    return calibrated


def aggregate(llm: str, reranker: str, n_folds: int) -> dict:
    import evaluate

    rows: list[dict] = []
    for fold_id in range(n_folds):
        path = RESULT_DIR / eval_tag(reranker, llm, fold_id) / "predictions_test_platt.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(_read_jsonl(path))

    patient_ids = [str(row.get("patient_id")) for row in rows]
    if len(rows) != N_PATIENTS or len(set(patient_ids)) != N_PATIENTS:
        raise RuntimeError(
            f"incomplete OOF pool for {llm}/{reranker}: "
            f"{len(rows)} rows, {len(set(patient_ids))} unique patients"
        )

    out = RESULT_DIR / aggregate_tag(reranker, llm)
    out.mkdir(parents=True, exist_ok=True)
    calibrated_path = out / "oof_predictions_platt.jsonl"
    _write_jsonl(calibrated_path, rows)
    calibrated_metrics = evaluate.evaluate(calibrated_path, threshold=0.5, bootstrap=2000)
    (out / "oof_metrics.json").write_text(json.dumps(calibrated_metrics, indent=2))

    raw_rows = []
    for row in rows:
        raw = dict(row)
        raw["score"] = float(row["score_raw"])
        raw["calibration"] = "None"
        raw_rows.append(raw)
    raw_path = out / "oof_predictions_raw.jsonl"
    _write_jsonl(raw_path, raw_rows)
    raw_metrics = evaluate.evaluate(raw_path, threshold=0.5, bootstrap=2000)
    (out / "oof_metrics_raw.json").write_text(json.dumps(raw_metrics, indent=2))

    row = {
        "tag": aggregate_tag(reranker, llm),
        "llm": llm,
        "reranker": reranker,
        "adapter_design": "shared_mixed5",
        "calibration": "Platt_fold_validation",
        "threshold": 0.5,
        "n_oof": calibrated_metrics["n"],
        "n_pos": calibrated_metrics["n_pos"],
        "n_neg": calibrated_metrics["n_neg"],
        "raw_AUROC": raw_metrics["AUROC"],
        "raw_AUROC_lo": raw_metrics["AUROC_95CI"][0],
        "raw_AUROC_hi": raw_metrics["AUROC_95CI"][1],
        "raw_AUPRC": raw_metrics["AUPRC"],
        "calibrated_AUROC": calibrated_metrics["AUROC"],
        "calibrated_AUPRC": calibrated_metrics["AUPRC"],
        "calibrated_Macro_F1": calibrated_metrics["Macro_F1"],
        "calibrated_Accuracy": calibrated_metrics["Accuracy"],
        "calibrated_Sensitivity": calibrated_metrics["Sensitivity"],
        "calibrated_Specificity": calibrated_metrics["Specificity"],
        "calibrated_MCC": calibrated_metrics["MCC"],
        "calibrated_Brier": calibrated_metrics["Brier"],
    }
    upsert_aggregate(row)
    print(f"  [OOF {llm}/{reranker}] raw AUROC={raw_metrics['AUROC']:.3f}, "
          f"calibrated Macro-F1={calibrated_metrics['Macro_F1']:.3f}")
    return {"raw": raw_metrics, "calibrated": calibrated_metrics}


def upsert_aggregate(row: dict) -> None:
    path = RESULT_DIR / "aggregate_shared_reranker_cv.csv"
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="") as fh:
            existing = [r for r in csv.DictReader(fh) if r.get("tag") != row["tag"]]
    existing.append({key: row.get(key, "") for key in row})
    existing.sort(key=lambda item: (item.get("llm", ""), item.get("reranker", "")))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(existing)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llms", nargs="+", default=["mistral", "biomistral"],
                    choices=["mistral", "biomistral"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-ids", nargs="+", type=int)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force-prompts", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold_ids = args.fold_ids if args.fold_ids is not None else list(range(args.folds))
    if any(fold_id < 0 or fold_id >= args.folds for fold_id in fold_ids):
        raise ValueError(f"fold IDs must be between 0 and {args.folds - 1}")

    if not run_cv_grid.CV_DIR.exists():
        run_cv_grid.make_folds(n_folds=args.folds, seed=run_cv_grid.DEFAULT_SEED)

    if not args.aggregate_only:
        for fold_id in fold_ids:
            print(f"\n{'=' * 78}\nPREPARE SHARED FOLD {fold_id}\n{'=' * 78}")
            ensure_source_prompts(fold_id, force=args.force_prompts)
            build_mixed_prompts(fold_id, force=args.force_prompts)
            release_retrieval_models()
            if args.prepare_only:
                continue

            for llm in args.llms:
                print(f"\n{'=' * 78}\nTRAIN {llm.upper()} SHARED FOLD {fold_id}\n{'=' * 78}")
                adapter = train_adapter(llm, fold_id, args)
                for reranker in RERANKERS:
                    evaluate_reranker(llm, reranker, fold_id, adapter,
                                      skip_existing=args.skip_existing)

    if args.prepare_only or len(fold_ids) != args.folds:
        return

    print(f"\n{'=' * 78}\nAGGREGATE SHARED OOF RESULTS\n{'=' * 78}")
    for llm in args.llms:
        for reranker in RERANKERS:
            aggregate(llm, reranker, args.folds)


if __name__ == "__main__":
    main()
