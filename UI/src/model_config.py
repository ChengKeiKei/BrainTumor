"""Selected UI models from the thesis deployment choice.

First Recurrence: BioMistral + Exp3 + MedCPT + last-token hybrid XGBoost.
Second Recurrence: BioMistral + PubMedBERT ExpC_v3 shared-adapter LoRA.

Literature retrieval in this demo is still local TF-IDF over the 10k PubMed
corpus. Training used MedCPT (FR) / PubMedBERT (SR) as the reranker; those
weights are not loaded here unless a later UI path adds them.
"""

from __future__ import annotations

from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = UI_ROOT.parent
RAG_ROOT = HANDOFF_ROOT.parent
FR_ROOT = RAG_ROOT / "First_Recur"
SR_ROOT = RAG_ROOT / "Second_Recur"
MODEL_DIR = UI_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- First Recurrence hybrid ---
FR_CELL_TAG = "Exp3__beep__medcpt"
FR_ENCODER_SLUG = "biomistral"
FR_ENCODER_DISPLAY = "BioMistral-7B-DARE"
FR_ENCODER_HF_ID = "BioMistral/BioMistral-7B-DARE"
FR_POOLING = "last"
FR_CLINICAL_GROUP = "Exp3_metadata_treatment"
FR_RERANKER_DISPLAY = "MedCPT"
# Legacy last-token dir recovered the internal test AUROC 0.978.
# Do not prefer embeddings/biomistral/last/Exp3__beep__medcpt (that re-extract scored lower).
FR_TRAINING_EMB_DIR = FR_ROOT / "Imbalance" / "embeddings" / "biomistral" / "Exp3__beep__medcpt"
FR_ARTIFACT_PATH = MODEL_DIR / "first_recur_hybrid_biomistral_last_exp3_medcpt.joblib"
FR_LORA_ADAPTER = FR_ROOT / "Model" / "checkpoints" / "Exp3__beep__medcpt__biomistral" / "adapters"

FR_INTERNAL_TEST = {
    "AUROC": 0.978,
    "AUPRC": 0.993,
    "Macro_F1": 0.868,
}

# --- Second Recurrence LoRA ---
# Thesis binary UI cell: BioMistral + PubMedBERT, shared-adapter five-fold mean.
# Live scoring loads fold 0 of that shared adapter (one checkpoint, not an ensemble).
SR_CELL_TAG = "ExpC_TxVLM__beep__mixed5__biomistral__sharedcv__fold0__v3_structured"
SR_ENCODER_DISPLAY = "BioMistral-7B-DARE 4-bit"
SR_RERANKER_DISPLAY = "PubMedBERT"
SR_PROMPT_VARIANT = "ExpC_v3 structured"
SR_MODEL_ID = (SR_ROOT / "Model" / "local_models" / "BioMistral-7B-DARE-4bit").as_posix()
SR_ADAPTER_DIR = SR_ROOT / "Model" / "checkpoints" / SR_CELL_TAG / "adapters"

SR_FIVEFOLD_METRICS = {
    "AUROC": 0.720,
    "Macro_F1": 0.700,
    "Sensitivity": 0.718,
    "Specificity": 0.681,
}


def resolve_fr_training_emb_dir() -> Path:
    """Best-effort path to the MU training embeddings.

    These only exist on the training machine and are needed solely to
    (re)build the hybrid artifact. Deployed installs run from the shipped
    .joblib, so never raise here — callers that actually need the files
    check .exists() themselves.
    """
    if FR_TRAINING_EMB_DIR.exists():
        return FR_TRAINING_EMB_DIR
    fallback = FR_ROOT / "Imbalance" / "embeddings" / FR_ENCODER_SLUG / FR_POOLING / FR_CELL_TAG
    if fallback.exists():
        return fallback
    return FR_TRAINING_EMB_DIR
