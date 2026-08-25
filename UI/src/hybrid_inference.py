from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from .encoding import encode_ui_clinical_row, is_missing
from .literature import LiteratureDoc, build_literature_context_beep
from .model_config import (
    FR_ARTIFACT_PATH,
    FR_CELL_TAG,
    FR_CLINICAL_GROUP,
    FR_ENCODER_DISPLAY,
    FR_ENCODER_HF_ID,
    FR_LORA_ADAPTER,
    FR_POOLING,
    FR_RERANKER_DISPLAY,
    FR_ROOT,
    SR_ADAPTER_DIR,
    SR_CELL_TAG,
    SR_ENCODER_DISPLAY,
    SR_FIVEFOLD_METRICS,
    SR_MODEL_ID,
    SR_PROMPT_VARIANT,
    SR_RERANKER_DISPLAY,
    UI_ROOT,
    resolve_fr_training_emb_dir,
)
from .risk_engine import build_first_recurrence_prompt


FEATURE_GROUPS_PATH = UI_ROOT / "configs" / "feature_groups.json"
DEFAULT_ARTIFACT = FR_ARTIFACT_PATH
TRAINING_EMB_DIR = resolve_fr_training_emb_dir()
FR_LORA_CHECKPOINTS = {
    f"BioMistral {FR_CELL_TAG}": FR_LORA_ADAPTER,
}


def _feature_group(feature: str) -> str:
    if any(marker in feature for marker in ["mutation", "methylation", "1p/19q", "amplification", "deletion"]):
        return "Molecular"
    if any(marker in feature for marker in ["Chemo", "Radiation", "Dose", "Fractions", "surgery", "procedure"]):
        return "Treatment"
    if any(marker in feature for marker in ["Diagnosis", "Grade", "Tumor", "Sex", "Race", "Age"]):
        return "Clinical"
    return "Clinical"


def _humanize_feature_name(feature: str) -> tuple[str, str, str]:
    if feature.startswith("LLM_PCA_"):
        number = feature.removeprefix("LLM_PCA_")
        return (
            f"LLM semantic evidence feature {number}",
            "LLM evidence",
            "Compressed signal from the patient prompt and retrieved literature after pooling and PCA.",
        )

    clean = feature.replace("_nan", " missing").replace("_", " ")
    for sep in ["=", ":"]:
        clean = clean.replace(sep, " ")
    clean = " ".join(clean.split())
    return clean, _feature_group(feature), "Structured patient feature used by the XGBoost classifier."


def _dummy_parent(raw_feature: str, clinical_cols: list[str]) -> str | None:
    for col in sorted(clinical_cols, key=len, reverse=True):
        if raw_feature.startswith(f"{col}_"):
            return col
    return None


def _format_xgb_contributions(
    feature_names: list[str],
    contributions: np.ndarray,
    feature_values: np.ndarray,
    clinical_row: pd.DataFrame,
    clinical_cols: list[str],
) -> pd.DataFrame:
    """Doctor-facing XAI: group one-hots and collapse all LLM_PCA_* into one row."""
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    patient_values = clinical_row.iloc[0].to_dict()
    llm_total = 0.0
    llm_count = 0

    for raw_feature, contribution, encoded_value in zip(feature_names, contributions, feature_values):
        # Collapse abstract PCA dims into one interpretable LLM evidence bar.
        if raw_feature.startswith("LLM_PCA_"):
            llm_total += float(contribution)
            llm_count += 1
            continue

        parent = _dummy_parent(raw_feature, clinical_cols)
        if parent:
            raw_value = patient_values.get(parent)
            entered_value = "missing / not entered" if is_missing(raw_value) else str(raw_value)
            if is_missing(raw_value):
                explanation = (
                    "This reflects the model's learned effect of this field being unknown or not tested, "
                    "not a measured molecular or treatment result."
                )
            else:
                explanation = (
                    "Combined contribution of all one-hot indicators for this clinical field. "
                    f"Entered/mapped value: {entered_value}."
                )
            if parent not in grouped:
                grouped[parent] = {
                    "feature": parent,
                    "feature_group": _feature_group(parent),
                    "contribution": 0.0,
                    "raw_feature": f"grouped one-hot: {parent}_*",
                    "encoded_value": entered_value,
                    "explanation": explanation,
                }
            grouped[parent]["contribution"] += float(contribution)
            continue

        feature, group, explanation = _humanize_feature_name(raw_feature)
        patient_value = patient_values.get(raw_feature, encoded_value)
        encoded_display = "missing / not entered" if is_missing(patient_value) else str(patient_value)
        if is_missing(patient_value):
            explanation = (
                "This reflects the model's learned effect of this field being unknown or not tested, "
                "not a measured molecular or treatment result."
            )
        else:
            explanation = f"Structured clinical value used directly by XGBoost. Mapped value: {encoded_display}."

        rows.append(
            {
                "feature": feature,
                "feature_group": group,
                "contribution": float(contribution),
                "raw_feature": raw_feature,
                "encoded_value": encoded_display,
                "explanation": explanation,
            }
        )

    if llm_count:
        rows.append(
            {
                "feature": "LLM evidence (prompt + literature)",
                "feature_group": "LLM evidence",
                "contribution": llm_total,
                "raw_feature": f"sum of {llm_count} LLM_PCA_* features",
                "encoded_value": "combined",
                "explanation": (
                    "Total contribution from all compressed LLM embedding features "
                    "(patient prompt + retrieved PubMed abstracts after PCA)."
                ),
            }
        )

    rows.extend(grouped.values())
    df = pd.DataFrame(rows)
    df["direction"] = np.where(df["contribution"] >= 0, "pushes risk up", "pushes risk down")
    # Prefer named clinical/treatment/molecular features; keep aggregated LLM as one row.
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(12)
    return df[["feature", "feature_group", "encoded_value", "contribution", "direction", "explanation", "raw_feature"]]

@dataclass
class HybridPrediction:
    probability: float
    label: str
    threshold: float
    mode: str
    model_name: str
    evidence_prompt: str
    top_contributions: pd.DataFrame
    retrieved_docs: list[LiteratureDoc]
    warning: str = ""
    checkpoint_paths: dict[str, str] | None = None


def _clinical_feature_cols() -> list[str]:
    if FEATURE_GROUPS_PATH.exists():
        groups = json.loads(FEATURE_GROUPS_PATH.read_text())
        return groups[FR_CLINICAL_GROUP]
    legacy = FR_ROOT / "Dataset" / "Processed" / "feature_groups.json"
    groups = json.loads(legacy.read_text())
    return groups[FR_CLINICAL_GROUP]


def _load_split(emb_dir: Path, split: str):
    X = np.load(emb_dir / f"{split}.npy").astype(np.float32)
    ids = pd.read_csv(emb_dir / f"{split}_ids.csv")
    return X, ids["label"].astype(int).to_numpy(), ids["Patient_ID"].astype(str).tolist()


def _load_clinical(pids: list[str], split: str, cols: list[str]) -> pd.DataFrame:
    file_map = {"train": "Train.csv", "valid": "Validation.csv", "test": "Test.csv"}
    df = pd.read_csv(FR_ROOT / "Dataset" / "splits" / file_map[split])
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    return df.set_index("Patient_ID").loc[pids, cols].copy()


def _fit_clinical_encoder(train_df: pd.DataFrame) -> dict[str, Any]:
    train_raw = train_df.replace({pd.NA: np.nan})
    numeric_cols = [c for c in train_raw.columns if pd.api.types.is_numeric_dtype(train_raw[c])]
    categorical_cols = [c for c in train_raw.columns if c not in numeric_cols]
    medians = train_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True).fillna(-999.0).to_dict()
    cat_columns = list(pd.get_dummies(train_raw[categorical_cols], dummy_na=True).columns)
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_medians": medians,
        "cat_columns": cat_columns,
    }


def _encode_clinical(df: pd.DataFrame, enc: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    raw = df.replace({pd.NA: np.nan})
    numeric_cols = enc["numeric_cols"]
    categorical_cols = enc["categorical_cols"]
    medians = enc["numeric_medians"]
    num = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for c in numeric_cols:
        num[c] = num[c].fillna(float(medians.get(c, -999.0)))
    cat = pd.get_dummies(raw[categorical_cols], dummy_na=True)
    cat = cat.reindex(columns=enc["cat_columns"], fill_value=0.0)
    out = pd.concat([num, cat], axis=1)
    return out.to_numpy(dtype=np.float32), list(out.columns)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().to_dict()
    n = len(y)
    k = len(counts)
    return np.asarray([n / (k * counts[int(v)]) for v in y], dtype=np.float32)


def build_first_recurrence_artifact(force: bool = False) -> Path:
    if DEFAULT_ARTIFACT.exists() and not force:
        return DEFAULT_ARTIFACT

    emb_dir = TRAINING_EMB_DIR
    if not emb_dir.exists():
        raise FileNotFoundError(f"Missing LLM embeddings: {emb_dir}")

    X_tr, y_tr, pid_tr = _load_split(emb_dir, "train")
    X_va, y_va, pid_va = _load_split(emb_dir, "valid")
    X_te, y_te, pid_te = _load_split(emb_dir, "test")

    pca = PCA(n_components=min(64, X_tr.shape[0] - 1, X_tr.shape[1]), random_state=42)
    Z_tr = pca.fit_transform(X_tr)
    Z_va = pca.transform(X_va)
    Z_te = pca.transform(X_te)

    feature_cols = _clinical_feature_cols()
    clin_tr = _load_clinical(pid_tr, "train", feature_cols)
    clin_va = _load_clinical(pid_va, "valid", feature_cols)
    clin_te = _load_clinical(pid_te, "test", feature_cols)
    clinical_encoder = _fit_clinical_encoder(clin_tr)
    C_tr, clinical_names = _encode_clinical(clin_tr, clinical_encoder)
    C_va, _ = _encode_clinical(clin_va, clinical_encoder)
    C_te, _ = _encode_clinical(clin_te, clinical_encoder)

    A_tr = np.concatenate([Z_tr, C_tr], axis=1)
    A_va = np.concatenate([Z_va, C_va], axis=1)
    A_te = np.concatenate([Z_te, C_te], axis=1)
    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=4,
        device="cpu",
    )
    model.fit(A_tr, y_tr, sample_weight=_balanced_sample_weights(y_tr), eval_set=[(A_va, y_va)], verbose=False)

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    p_te = model.predict_proba(A_te)[:, 1]
    y_hat = (p_te >= 0.5).astype(int)
    internal_test = {
        "AUROC": float(roc_auc_score(y_te, p_te)),
        "AUPRC": float(average_precision_score(y_te, p_te)),
        "Macro_F1": float(f1_score(y_te, y_hat, average="macro")),
        "n_test": int(len(y_te)),
    }

    feature_names = [f"LLM_PCA_{i + 1:02d}" for i in range(Z_tr.shape[1])] + clinical_names
    model.get_booster().feature_names = feature_names
    artifact = {
        "model": model,
        "pca": pca,
        "clinical_encoder": clinical_encoder,
        "clinical_feature_cols": feature_cols,
        "feature_names": feature_names,
        "mean_embedding": X_tr.mean(axis=0).astype(np.float32),
        "config": {
            "task": "First Recurrence",
            "encoder": FR_ENCODER_DISPLAY,
            "encoder_hf_id": FR_ENCODER_HF_ID,
            "pooling": FR_POOLING,
            "cell_tag": FR_CELL_TAG,
            "reranker": FR_RERANKER_DISPLAY,
            "clinical_group": FR_CLINICAL_GROUP,
            "threshold": 0.5,
            "embedding_dim": int(X_tr.shape[1]),
            "pca_dim": int(Z_tr.shape[1]),
            "internal_test": internal_test,
            "mode": (
                f"RAG literature + frozen {FR_ENCODER_DISPLAY} last-token embedding "
                f"+ Exp3 clinical/treatment + class-weighted XGBoost"
            ),
        },
    }
    joblib.dump(artifact, DEFAULT_ARTIFACT)
    print(
        f"[FR hybrid] {FR_CELL_TAG} {FR_ENCODER_DISPLAY} last-token "
        f"test AUROC={internal_test['AUROC']:.3f} "
        f"AUPRC={internal_test['AUPRC']:.3f} "
        f"Macro-F1={internal_test['Macro_F1']:.3f} -> {DEFAULT_ARTIFACT}"
    )
    return DEFAULT_ARTIFACT


def _clinical_row_from_ui(data: dict[str, Any], cols: list[str] | None = None) -> pd.DataFrame:
    mapped = encode_ui_clinical_row(data)
    use_cols = cols or _clinical_feature_cols()
    return pd.DataFrame([{c: mapped.get(c) for c in use_cols}])


def _iteration_kwargs(model: XGBClassifier) -> dict[str, Any]:
    best = getattr(model, "best_iteration", None)
    if best is None:
        return {}
    return {"iteration_range": (0, int(best) + 1)}


class LLMEmbedder:
    def __init__(
        self,
        model_id: str = FR_ENCODER_HF_ID,
        device: str = "auto",
        adapter_path: str | None = None,
        local_files_only: bool = True,
    ):
        self.model_id = model_id
        self.device_request = device
        self.adapter_path = adapter_path
        self.local_files_only = local_files_only
        self._tok = None
        self._model = None
        self._device = None

    def _resolve_device(self) -> str:
        if self.device_request != "auto":
            return self.device_request
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self):
        if self._model is not None:
            return
        if self.adapter_path:
            try:
                from mlx_lm import load as mlx_load
            except ModuleNotFoundError as exc:
                raise RuntimeError("mlx_lm is required to load LoRA adapter checkpoints.") from exc

            model, tok = mlx_load(self.model_id, adapter_path=self.adapter_path)
            model.eval()
            self._tok = tok
            self._model = model
            self._device = "mlx"
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )
        model.eval().to(device)
        self._tok = tok
        self._model = model
        self._device = device

    @torch.no_grad()
    def embed(self, prompt: str, pooling: str = FR_POOLING, max_length: int = 3072) -> np.ndarray:
        self.load()
        assert self._tok is not None and self._model is not None and self._device is not None
        if self._device == "mlx":
            raise RuntimeError(
                "MLX LoRA checkpoints expose generation/logits, but this deployment path needs "
                "last-hidden-state embeddings for XGBoost. Use frozen-base embedding checkpoint "
                "for hybrid XGBoost, or LoRA Yes/No inference for direct LLM prediction."
            )
        chat = [{"role": "user", "content": prompt}]
        formatted = self._tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = self._tok(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        out = self._model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]
        mask = enc["attention_mask"].to(h.dtype)
        if pooling == "mean":
            vec = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return vec[0].detach().cpu().float().numpy().astype(np.float32)
        if pooling == "last":
            last_idx = int(mask[0].sum().item()) - 1
            return h[0, last_idx, :].detach().cpu().float().numpy().astype(np.float32)
        raise ValueError(f"Unknown pooling {pooling!r}; expected 'last' or 'mean'.")

    def last_embedding(self, prompt: str, max_length: int = 3072) -> np.ndarray:
        return self.embed(prompt, pooling="last", max_length=max_length)

    def mean_embedding(self, prompt: str, max_length: int = 3072) -> np.ndarray:
        return self.embed(prompt, pooling="mean", max_length=max_length)


def build_first_hybrid_prompt(data: dict[str, Any], docs: list[LiteratureDoc]) -> str:
    literature = build_literature_context_beep(docs)
    clinical = build_first_recurrence_prompt(data)
    return (
        f"Relevant Literature Context:\n{literature}\n\n"
        f"{clinical}"
    )


def predict_first_hybrid(
    data: dict[str, Any],
    docs: list[LiteratureDoc],
    use_real_llm: bool = False,
    embedder: LLMEmbedder | None = None,
) -> HybridPrediction:
    artifact_path = DEFAULT_ARTIFACT
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Hybrid XGBoost checkpoint not found: {artifact_path}. "
            "Run build_deployment_artifacts.py once before launching Streamlit."
        )
    artifact = joblib.load(artifact_path)
    prompt = build_first_hybrid_prompt(data, docs)

    warning = ""
    checkpoint_paths = {
        "xgboost_checkpoint": str(artifact_path),
        "frozen_llm_checkpoint": FR_ENCODER_HF_ID,
        "training_embedding_source": str(TRAINING_EMB_DIR),
        "cell_tag": FR_CELL_TAG,
        "pooling": FR_POOLING,
        "reranker_training": FR_RERANKER_DISPLAY,
        "clinical_group": FR_CLINICAL_GROUP,
    }
    if use_real_llm:
        if embedder is None:
            embedder = LLMEmbedder()
        emb = embedder.embed(prompt, pooling=FR_POOLING)
        mode = (
            f"Checkpoint hybrid: frozen {FR_ENCODER_DISPLAY} last-token embedding "
            "+ saved XGBoost checkpoint"
        )
    else:
        emb = artifact["mean_embedding"]
        warning = (
            "Smoke test only: this uses the saved XGBoost checkpoint but substitutes the training-set "
            "mean LLM embedding. It verifies that the XGBoost checkpoint loads, but it is not a true "
            "new-patient LLM+XGBoost prediction."
        )
        mode = "Checkpoint smoke test: saved XGBoost checkpoint + mean training embedding"

    if emb.shape[0] != int(artifact["config"]["embedding_dim"]):
        raise ValueError(
            f"Embedding dim mismatch: got {emb.shape[0]}, expected {artifact['config']['embedding_dim']}"
        )

    z = artifact["pca"].transform(emb.reshape(1, -1))
    clinical = _clinical_row_from_ui(data, artifact.get("clinical_feature_cols"))
    c, _ = _encode_clinical(clinical, artifact["clinical_encoder"])
    X = np.concatenate([z, c], axis=1)
    model: XGBClassifier = artifact["model"]
    probability = float(model.predict_proba(X)[0, 1])
    threshold = float(artifact["config"].get("threshold", 0.5))
    label = "High recurrence risk" if probability >= threshold else "Lower recurrence risk"

    booster = model.get_booster()
    iter_kwargs = _iteration_kwargs(model)
    dm = xgb.DMatrix(X, feature_names=artifact["feature_names"])
    contrib = booster.predict(dm, pred_contribs=True, **iter_kwargs)
    values = contrib[0][:-1]
    df = _format_xgb_contributions(
        feature_names=artifact["feature_names"],
        contributions=values,
        feature_values=X[0],
        clinical_row=clinical,
        clinical_cols=artifact["clinical_feature_cols"],
    )

    return HybridPrediction(
        probability=probability,
        label=label,
        threshold=threshold,
        mode=mode,
        model_name=artifact["config"]["mode"],
        evidence_prompt=prompt,
        top_contributions=df,
        retrieved_docs=docs,
        warning=warning,
        checkpoint_paths=checkpoint_paths,
    )


def checkpoint_status() -> pd.DataFrame:
    rows = [
        {
            "component": "First Recurrence hybrid XGBoost",
            "path": str(DEFAULT_ARTIFACT),
            "exists": DEFAULT_ARTIFACT.exists(),
            "used_for": f"{FR_ENCODER_DISPLAY} + Exp3 + {FR_RERANKER_DISPLAY} last-token hybrid",
        },
        {
            "component": "Frozen BioMistral checkpoint",
            "path": FR_ENCODER_HF_ID,
            "exists": True,
            "used_for": "Live last-token embedding extraction (local files only)",
        },
        {
            "component": "Training embedding source",
            "path": str(TRAINING_EMB_DIR),
            "exists": TRAINING_EMB_DIR.exists(),
            "used_for": "Artifact construction and smoke-test fallback (legacy last-token)",
        },
        {
            "component": "Feature group config",
            "path": str(FEATURE_GROUPS_PATH),
            "exists": FEATURE_GROUPS_PATH.exists(),
            "used_for": "Exp3 metadata + treatment schema (molecular not in XGBoost)",
        },
        {
            "component": "Second Recurrence BioMistral LoRA",
            "path": str(SR_ADAPTER_DIR),
            "exists": (SR_ADAPTER_DIR / "adapter_config.json").exists()
            and (SR_ADAPTER_DIR / "adapters.safetensors").exists(),
            "used_for": (
                f"{SR_ENCODER_DISPLAY} + {SR_RERANKER_DISPLAY} ({SR_PROMPT_VARIANT}); "
                f"five-fold AUROC {SR_FIVEFOLD_METRICS['AUROC']:.3f}"
            ),
        },
        {
            "component": "Second Recurrence base model",
            "path": SR_MODEL_ID,
            "exists": Path(SR_MODEL_ID).exists(),
            "used_for": f"Live Yes/No LoRA scoring ({SR_CELL_TAG})",
        },
    ]
    for name, path in FR_LORA_CHECKPOINTS.items():
        rows.append(
            {
                "component": f"FR LoRA adapter (status only): {name}",
                "path": str(path),
                "exists": (path / "adapter_config.json").exists() and (path / "adapters.safetensors").exists(),
                "used_for": "Direct LoRA Yes/No inference, not hybrid XGBoost embedding",
            }
        )
    return pd.DataFrame(rows)
