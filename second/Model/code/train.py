"""
train.py — Fine-tune Mistral-7B-Instruct-v0.3 with LoRA on one
(experiment × retriever × reranker) configuration.

Backend: mlx-tune / mlx_lm.lora (Apple MLX), the same path that worked
in P1. Adapters are saved into `Model/checkpoints/<tag>/adapters/`.

Usage:

    python code/train.py --tag Exp4__beep__beep        # implies prompts/Exp4__beep__beep/
    python code/train.py --prompts-dir prompts/Exp4__beep__beep \
                         --steps 250 --batch-size 4 --lr 2e-5

A loss-curve PNG is written to `Model/results/<tag>/loss.png`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]    # Second_Recur/
MODEL_DIR = ROOT / "Model"
PROMPTS_D = MODEL_DIR / "prompts"
CKPT_D    = MODEL_DIR / "checkpoints"
RESULT_D  = MODEL_DIR / "results"
DEFAULT_CFG = MODEL_DIR / "configs" / "default.yaml"

# Default base model.  Override with `--model-id` (CLI) or `RAG_MODEL_ID` (env).
# We default to BioMistral-7B-DARE on Second_Recur because Exp(i)/(ii) are
# pure-text biomedical prompts where the PMC pre-training pays off.  Pass
# `--model-id mlx-community/Mistral-7B-Instruct-v0.3-4bit` to flip back to
# vanilla Mistral for the A/B comparison cell.
DEFAULT_MODEL_ID = (ROOT / "Model" / "local_models" / "BioMistral-7B-DARE-4bit").as_posix()
MODEL_ID  = os.environ.get("RAG_MODEL_ID", DEFAULT_MODEL_ID)
LORA_RANK = 16
MAX_SEQ_LENGTH = 2048


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def write_lora_yaml(path: Path, rank: int = LORA_RANK) -> None:
    path.write_text(
        "lora_parameters:\n"
        f"  alpha: {rank}\n"
        "  dropout: 0.05\n"
        f"  rank: {rank}\n"
        "  scale: 1.0\n"
    )


# --------------------------------------------------------------------------- #
# Post-training artefacts: loss plot + timing.json                            #
# --------------------------------------------------------------------------- #
_LOSS        = r"([0-9.]+|nan|-?inf)"   # accept NaN / inf so divergence stays visible
_ITER_RE     = re.compile(
    rf"Iter\s*(\d+):\s*Train loss\s*{_LOSS}.*?"
    r"(?:It/sec\s*([0-9.]+))?", re.IGNORECASE)
_VAL_RE      = re.compile(rf"Iter\s*(\d+):\s*Val loss\s*{_LOSS}", re.IGNORECASE)
_ITER_RE_ALT = re.compile(rf"step\s*=?\s*(\d+).*?loss\s*=?\s*{_LOSS}", re.IGNORECASE)
_VAL_RE_ALT  = re.compile(rf"step\s*=?\s*(\d+).*?eval[_ ]loss\s*=?\s*{_LOSS}", re.IGNORECASE)


def _safe_float(s: str) -> float:
    """float() with explicit nan/inf handling (float('nan') works but be explicit)."""
    s = s.lower()
    if s == "nan":   return float("nan")
    if s == "inf":   return float("inf")
    if s == "-inf":  return float("-inf")
    return float(s)


def parse_training_log(log_path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (train_pairs, val_pairs) = [(iter, loss), ...] each."""
    train_pairs: list[tuple[int, float]] = []
    val_pairs:   list[tuple[int, float]] = []
    if not log_path.exists():
        return train_pairs, val_pairs
    text = log_path.read_text(errors="ignore")
    for line in text.splitlines():
        m = _ITER_RE.search(line) or _ITER_RE_ALT.search(line)
        if m and "val loss" not in line.lower():
            train_pairs.append((int(m.group(1)), _safe_float(m.group(2))))
        m2 = _VAL_RE.search(line) or _VAL_RE_ALT.search(line)
        if m2:
            val_pairs.append((int(m2.group(1)), _safe_float(m2.group(2))))
    return train_pairs, val_pairs


def plot_loss_curve(ckpt_dir: Path, tag: str, result_dir: Path) -> Path | None:
    """Render results/<tag>/loss_curve.png from checkpoints/<tag>/train.log."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train.py] matplotlib not installed; skipping loss plot")
        return None

    train_pairs, val_pairs = parse_training_log(ckpt_dir / "train.log")
    if not train_pairs and not val_pairs:
        print("[train.py] no (iter, loss) rows parsed from train.log; skipping plot")
        return None

    out = result_dir / "loss_curve.png"
    result_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    if train_pairs:
        its, ls = zip(*train_pairs)
        plt.plot(its, ls, label="Train loss", color="#2563EB", linewidth=2)
    if val_pairs:
        its, ls = zip(*val_pairs)
        plt.plot(its, ls, label="Val loss", color="#DC2626", linewidth=2,
                 linestyle="--", marker="o", markersize=5)
    plt.xlabel("Iteration"); plt.ylabel("Cross-entropy loss")
    plt.title(f"Mistral-7B LoRA — {tag}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[train.py] loss curve  -> {out}")

    (result_dir / "loss_history.json").write_text(json.dumps({
        "tag": tag,
        "train": [{"iter": i, "loss": l} for i, l in train_pairs],
        "val":   [{"iter": i, "loss": l} for i, l in val_pairs],
    }, indent=2))
    return out


def write_timing(result_dir: Path, tag: str, *, elapsed: float,
                 steps: int, batch_size: int, lr: float,
                 grad_checkpoint: bool, max_seq_length: int) -> Path:
    out = result_dir / "timing_train.json"
    result_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag":               tag,
        "steps":             steps,
        "batch_size":        batch_size,
        "learning_rate":     lr,
        "grad_checkpoint":   grad_checkpoint,
        "max_seq_length":    max_seq_length,
        "wall_seconds":      round(elapsed, 3),
        "wall_minutes":      round(elapsed / 60, 3),
        "seconds_per_step":  round(elapsed / max(steps, 1), 4),
    }, indent=2))
    print(f"[train.py] train timing  -> {out}")
    return out


def run_mlx_lora(prompts_dir: Path, ckpt_dir: Path, *, steps: int,
                 batch_size: int, lr: float, save_steps: int,
                 eval_steps: int, logging_steps: int,
                 model_id: str | None = None,
                 grad_checkpoint: bool = True,
                 max_seq_length: int = 3072) -> int:
    """Invoke `mlx_lm.lora` directly with explicit eval & save cadences."""
    adapters_dir = ckpt_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train.jsonl", "valid.jsonl"):
        src = prompts_dir / split
        if src.exists():
            shutil.copy2(src, ckpt_dir / split)
    stale_test = ckpt_dir / "test.jsonl"
    if stale_test.exists():
        stale_test.unlink()

    yaml_cfg = ckpt_dir / "lora_config.yaml"
    write_lora_yaml(yaml_cfg)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id or MODEL_ID,
        "--train",
        "--data", str(ckpt_dir),
        "--iters", str(steps),
        "--learning-rate", str(lr),
        "--batch-size", str(batch_size),
        "--adapter-path", str(adapters_dir),
        "-c", str(yaml_cfg),
        "--save-every", str(save_steps),
        "--steps-per-report", str(logging_steps),
        "--steps-per-eval", str(eval_steps),
        "--val-batches", "-1",
        # Some rerankers (notably MiniLM, which favours long web-style
        # passages) push the top-3 concatenated prompt past 2048 tokens
        # on Exp4. Right-truncation at 2048 was clipping the `Yes`/`No`
        # response token, leaving zero unmasked targets under
        # --mask-prompt → NaN loss for the whole run. 3072 covers the
        # observed p-max of ~2250 tokens with comfortable headroom.
        "--max-seq-length", str(max_seq_length),
        "--mask-prompt",
    ]
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    print("\n[train.py] running:\n  " + " ".join(cmd) + "\n", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=os.environ.copy())
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def train(tag: str | None = None, prompts_dir: str | None = None,
          steps: int = 250, batch_size: int = 4, lr: float = 2e-5,
          save_steps: int = 250, eval_steps: int = 50,
          logging_steps: int = 10, model_id: str | None = None,
          grad_checkpoint: bool = True,
          max_seq_length: int = 3072) -> Path:
    if prompts_dir:
        prompts_path = Path(prompts_dir)
        if not prompts_path.is_absolute():
            prompts_path = (ROOT / prompts_path).resolve()
        if tag is None:
            tag = prompts_path.name
    elif tag:
        prompts_path = PROMPTS_D / tag
    else:
        raise ValueError("Specify either --tag or --prompts-dir.")

    if not (prompts_path / "train.jsonl").exists():
        raise FileNotFoundError(
            f"train.jsonl missing under {prompts_path}.\n"
            f"Run code/build_dataset.py first.")

    ckpt_dir   = CKPT_D / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path   = ckpt_dir / "train.log"
    # Persist the chosen base model so infer.py can find it later without
    # the caller having to remember which model trained which tag.
    (ckpt_dir / "model_id.txt").write_text((model_id or MODEL_ID) + "\n")

    env_bin = os.path.dirname(sys.executable)
    if env_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([env_bin, os.environ.get("PATH", "")])

    result_dir = RESULT_D / tag
    result_dir.mkdir(parents=True, exist_ok=True)

    wall_t0 = time.perf_counter()
    with log_path.open("w") as log_file:
        tee_out = TeeStream(sys.__stdout__, log_file)
        tee_err = TeeStream(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[train.py] tag = {tag}")
            print(f"[train.py] prompts = {prompts_path}")
            print(f"[train.py] checkpoint = {ckpt_dir}")
            print(f"[train.py] results = {result_dir}")
            effective_model_id = model_id or MODEL_ID
            print(f"[train.py] model  = {effective_model_id}")
            print(f"[train.py] LoRA   r={LORA_RANK}, alpha={LORA_RANK}, dropout=0.05")
            print(f"[train.py] steps={steps}, batch={batch_size}, lr={lr}")
            print(f"[train.py] grad_checkpoint={grad_checkpoint}, "
                  f"max_seq_length={max_seq_length}")
            rc = run_mlx_lora(prompts_path, ckpt_dir,
                              steps=steps, batch_size=batch_size, lr=lr,
                              save_steps=save_steps, eval_steps=eval_steps,
                              logging_steps=logging_steps,
                              model_id=effective_model_id,
                              grad_checkpoint=grad_checkpoint,
                              max_seq_length=max_seq_length)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, "mlx_lm.lora")
            print(f"\n[train.py] adapters -> {ckpt_dir / 'adapters'}")
    wall_elapsed = time.perf_counter() - wall_t0

    write_timing(result_dir, tag, elapsed=wall_elapsed,
                 steps=steps, batch_size=batch_size, lr=lr,
                 grad_checkpoint=grad_checkpoint,
                 max_seq_length=max_seq_length)
    plot_loss_curve(ckpt_dir, tag, result_dir)
    return ckpt_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",         default="", help="Tag under prompts/ (and checkpoints/).")
    ap.add_argument("--prompts-dir", default="", help="Override prompts directory.")
    ap.add_argument("--steps",       type=int,   default=250)
    ap.add_argument("--batch-size",  type=int,   default=4)
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--save-steps",  type=int,   default=250)
    ap.add_argument("--eval-steps",  type=int,   default=50)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="Disable activation checkpointing when memory permits.")
    ap.add_argument("--model-id", default="",
                    help=f"Override base model. Default = {MODEL_ID}")
    args = ap.parse_args()

    train(tag=args.tag or None, prompts_dir=args.prompts_dir or None,
          steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          save_steps=args.save_steps, eval_steps=args.eval_steps,
          logging_steps=args.logging_steps,
          model_id=args.model_id or None,
          grad_checkpoint=not args.no_grad_checkpoint,
          max_seq_length=args.max_seq_length)
