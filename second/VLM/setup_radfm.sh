#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup_radfm.sh — one-shot install for the RadFM (13 B) medical VLM backend.
#
# Steps:
#   1. clone the official RadFM repo (code only; ~5 MB)
#   2. download the LLaMA-13B tokenizer files (~2 MB)
#   3. download `pytorch_model.zip` from HuggingFace (≈ 50 GB)
#   4. unzip to `pytorch_model.bin` (≈ 50 GB more, transient)
#
# Run from `Second_Recur/VLM/` (or anywhere — the script is location-aware).
# Total disk needed: ~100 GB during step 4, ~55 GB after cleanup.
# Resumable: every download uses `huggingface_hub.snapshot_download` /
# `wget -c`, so re-running the script picks up where it stopped.
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RADFM_DIR="$HERE/RadFM_repo"
WEIGHTS_DIR="$HERE/RadFM_weights"
TOK_DIR="$WEIGHTS_DIR/Language_files"
BIN_FILE="$WEIGHTS_DIR/pytorch_model.bin"
ZIP_FILE="$WEIGHTS_DIR/pytorch_model.zip"

mkdir -p "$WEIGHTS_DIR"

# -----------------------------------------------------------------------------
# 1. Clone the RadFM repo (we only need Quick_demo/Model/ for the inference code)
# -----------------------------------------------------------------------------
if [ ! -d "$RADFM_DIR/.git" ]; then
    echo "[1/4] cloning RadFM repo..."
    git clone --depth 1 https://github.com/chaoyi-wu/RadFM.git "$RADFM_DIR"
else
    echo "[1/4] RadFM repo already present — skipping clone"
fi

# -----------------------------------------------------------------------------
# 2. Download LLaMA-13B tokenizer files (small).  We only need the tokenizer,
#    not the LLM weights (those come from `pytorch_model.bin`).
# -----------------------------------------------------------------------------
if [ ! -f "$TOK_DIR/tokenizer.model" ]; then
    echo "[2/4] pulling LLaMA-13B tokenizer files (≈2 MB)..."
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="huggyllama/llama-13b",
    local_dir="$TOK_DIR",
    allow_patterns=["tokenizer*", "special_tokens_map.json"],
    local_dir_use_symlinks=False,
)
PY
else
    echo "[2/4] tokenizer files already present — skipping"
fi

# -----------------------------------------------------------------------------
# 3. Download pytorch_model.zip (50 GB).  Single-file download is simpler than
#    the multi-part .z01..z04 split; HuggingFace serves both, single is enough.
# -----------------------------------------------------------------------------
if [ ! -f "$BIN_FILE" ] && [ ! -f "$ZIP_FILE" ]; then
    echo "[3/4] downloading pytorch_model.zip (≈50 GB) — this is the slow step."
    echo "      You can Ctrl-C and re-run this script; the download resumes."
    python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="chaoyi-wu/RadFM",
    filename="pytorch_model.zip",
    local_dir="$WEIGHTS_DIR",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
elif [ -f "$BIN_FILE" ]; then
    echo "[3/4] pytorch_model.bin already extracted — skipping download"
else
    echo "[3/4] pytorch_model.zip already present — skipping download"
fi

# -----------------------------------------------------------------------------
# 4. Unzip → pytorch_model.bin
# -----------------------------------------------------------------------------
if [ ! -f "$BIN_FILE" ] && [ -f "$ZIP_FILE" ]; then
    echo "[4/4] unzipping pytorch_model.zip..."
    (cd "$WEIGHTS_DIR" && unzip -o pytorch_model.zip)
    echo "      removing zip to reclaim ~50 GB..."
    rm -f "$ZIP_FILE"
else
    echo "[4/4] pytorch_model.bin already extracted — skipping unzip"
fi

# -----------------------------------------------------------------------------
# Sanity
# -----------------------------------------------------------------------------
echo
echo "==================== RadFM install summary ===================="
echo "Repo  : $RADFM_DIR"
echo "Tokens: $TOK_DIR ($(du -sh "$TOK_DIR" 2>/dev/null | cut -f1))"
echo "Weight: $BIN_FILE ($(du -sh "$BIN_FILE" 2>/dev/null | cut -f1))"
echo "==============================================================="
echo
echo "Next: python run_radfm_captions.py --max 1   # smoke test"
