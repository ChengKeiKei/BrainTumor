"""
prompts.py — BEEP-style prompt assembly.

The format is identical to what worked in P1's mlx_chat training data
so that the existing mlx-tune / mlx_lm.lora pipeline can train the LoRA
adapter on top of Mistral-7B-Instruct-v0.3 without any tokenizer changes.

Anatomy of the user message:

    Relevant Literature Context:
    [1] {title}. - {abstract truncated to 600 chars}...
    [2] ...
    [3] ...

    You are a clinical outcome prediction assistant.
    Given the following post-treatment glioma patient information, predict
    whether the patient will experience recurrence/progression.

    {EHR narrative produced by feature_render.render_patient(...)}

    Prediction (tumor recurrence/progression, Yes/No):

The assistant message is the literal "Yes" or "No" token, and we keep a
machine-readable `label` (0/1) field next to the messages so eval can
compute AUROC/AUPR.

For the no-RAG baseline, the "Relevant Literature Context" block is
omitted entirely — the rest of the prompt is unchanged.
"""
from __future__ import annotations

from typing import Sequence

# Default abstract length cap. The 10K PubMed corpus has median=1590 chars,
# mean=1557, p95=2330. At 2000 we keep the full abstract for ~93% of docs
# and lose only the long tail. 3 x 2000 = 6000 chars ≈ 1500 tokens,
# comfortably inside MAX_SEQ_LENGTH=2048 once EHR text + instruction are
# added. Set to 0 (or pass abstract_max_chars=0) to disable truncation.
ABSTRACT_TRUNC_DEFAULT = 2000

INSTRUCTION = (
    "You are a clinical outcome prediction assistant.\n"
    "Given the following post-treatment glioma patient information, predict "
    "whether the patient will experience recurrence/progression.\n"
)
TAIL = "\nPrediction (tumor recurrence/progression, Yes/No):"


def _format_doc(rank: int, doc: dict, abstract_max_chars: int) -> str:
    title = (doc.get("title") or "").strip().rstrip(".")
    abstract = (doc.get("abstract") or "").strip().replace("\n", " ")
    if abstract_max_chars and len(abstract) > abstract_max_chars:
        abstract = abstract[:abstract_max_chars].rstrip() + "..."
    body = f"{title}. - {abstract}" if abstract else f"{title}."
    return f"[{rank}] {body}"


def build_user_message(patient_text: str, docs: Sequence[dict] | None = None,
                       abstract_max_chars: int = ABSTRACT_TRUNC_DEFAULT) -> str:
    """Assemble the full user-side prompt.

    `abstract_max_chars`:
        >0 : truncate each abstract to this many chars
        0  : pass the full abstract (no truncation)
    """
    parts: list[str] = []
    if docs:
        parts.append("Relevant Literature Context:")
        for i, d in enumerate(docs, 1):
            parts.append(_format_doc(i, d, abstract_max_chars))
        parts.append("")  # blank line before instruction
    parts.append(INSTRUCTION)
    parts.append(patient_text.strip())
    parts.append(TAIL)
    return "\n".join(parts)


def build_chat_record(patient_text: str, label: int, docs: Sequence[dict] | None = None,
                      patient_id: str | None = None,
                      abstract_max_chars: int = ABSTRACT_TRUNC_DEFAULT) -> dict:
    """One JSONL record for mlx-tune / mlx_lm.lora training."""
    user_msg = build_user_message(patient_text, docs, abstract_max_chars=abstract_max_chars)
    record = {
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": "Yes" if int(label) == 1 else "No"},
        ],
        "label": int(label),
    }
    if patient_id is not None:
        record["patient_id"] = str(patient_id)
    return record
