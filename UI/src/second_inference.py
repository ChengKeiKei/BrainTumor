"""Second Recurrence scoring: selected BioMistral+PubMedBERT LoRA, with demo fallback."""

from __future__ import annotations

from typing import Any

from .literature import LiteratureDoc, build_literature_context_beep
from .lora_inference import SecondRecurrenceLoRA
from .model_config import (
    SR_ADAPTER_DIR,
    SR_CELL_TAG,
    SR_ENCODER_DISPLAY,
    SR_FIVEFOLD_METRICS,
    SR_MODEL_ID,
    SR_PROMPT_VARIANT,
    SR_RERANKER_DISPLAY,
)
from .risk_engine import (
    Prediction,
    _confidence,
    _missing,
    _risk_level,
    build_second_recurrence_prompt,
    predict_second_recurrence,
)


def build_second_lora_prompt(data: dict[str, Any], docs: list[LiteratureDoc]) -> str:
    clinical = build_second_recurrence_prompt(data)
    literature = build_literature_context_beep(docs)
    return (
        f"{clinical}\n"
        f"Relevant Literature Context:\n{literature}\n\n"
        "Prediction (tumor recurrence/progression, Yes/No):"
    )


def predict_second(
    data: dict[str, Any],
    required_labels: dict[str, str],
    docs: list[LiteratureDoc] | None = None,
    *,
    use_real_llm: bool = False,
    lora: SecondRecurrenceLoRA | None = None,
) -> Prediction:
    docs = docs or []
    prompt = build_second_lora_prompt(data, docs)
    missing_required = [label for key, label in required_labels.items() if _missing(data.get(key))]
    optional_missing = sum(
        1
        for key in ["enhancing_volume", "edema_volume", "radiomic_summary", "idh1", "codeletion_1p19q", "mgmt"]
        if _missing(data.get(key))
    )

    if not use_real_llm:
        demo = predict_second_recurrence(data, required_labels)
        return Prediction(
            probability=demo.probability,
            risk_level=demo.risk_level,
            evidence_completeness=demo.evidence_completeness,
            drivers=demo.drivers,
            contributions=demo.contributions,
            missing_required=demo.missing_required,
            evidence_prompt=prompt,
            mode="Smoke test: demo engine standing in for BioMistral + PubMedBERT LoRA",
            model_name=(
                f"Selected model is {SR_ENCODER_DISPLAY} + {SR_RERANKER_DISPLAY} "
                f"({SR_PROMPT_VARIANT}). Smoke mode does not load the LoRA adapter."
            ),
            warning=(
                "Smoke test only: the score comes from the transparent demo engine, not from "
                f"the {SR_ENCODER_DISPLAY} LoRA adapter. Turn on live LoRA scoring after the "
                "local 4-bit BioMistral weights and adapter are available."
            ),
            checkpoint_paths={
                "adapter": str(SR_ADAPTER_DIR),
                "base_model": SR_MODEL_ID,
                "cell_tag": SR_CELL_TAG,
                "five_fold_auroc": str(SR_FIVEFOLD_METRICS["AUROC"]),
            },
        )

    if lora is None:
        lora = SecondRecurrenceLoRA()
    probability = max(0.02, min(0.98, lora.predict_proba(prompt)))
    return Prediction(
        probability=probability,
        risk_level=_risk_level(probability),
        evidence_completeness=_confidence(missing_required, optional_missing),
        drivers=["Live BioMistral LoRA Yes/No probability"],
        contributions=[],
        missing_required=missing_required,
        evidence_prompt=prompt,
        mode=f"Live LoRA: {SR_ENCODER_DISPLAY} + {SR_RERANKER_DISPLAY} ({SR_PROMPT_VARIANT})",
        model_name=(
            f"{SR_ENCODER_DISPLAY} shared-adapter fold 0, PubMedBERT evidence, "
            f"{SR_PROMPT_VARIANT}. Thesis five-fold mean AUROC "
            f"{SR_FIVEFOLD_METRICS['AUROC']:.3f}."
        ),
        warning="",
        checkpoint_paths={
            "adapter": str(SR_ADAPTER_DIR),
            "base_model": SR_MODEL_ID,
            "cell_tag": SR_CELL_TAG,
        },
    )
