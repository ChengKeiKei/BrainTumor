"""Yes/No LoRA scoring for the selected Second Recurrence UI model."""

from __future__ import annotations

from pathlib import Path

from .model_config import SR_ADAPTER_DIR, SR_MODEL_ID


def _get_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    yes_full = tokenizer.encode(prefix + "Yes", add_special_tokens=False)
    no_full = tokenizer.encode(prefix + "No", add_special_tokens=False)
    yes_id = yes_full[len(prefix_ids)]
    no_id = no_full[len(prefix_ids)]
    yes_decoded = tokenizer.decode([yes_id]).strip()
    no_decoded = tokenizer.decode([no_id]).strip()
    if yes_decoded != "Yes" or no_decoded != "No":
        raise RuntimeError(
            f"Yes/No token resolution failed: yes={yes_decoded!r}, no={no_decoded!r}."
        )
    return [yes_id], [no_id]


class SecondRecurrenceLoRA:
    def __init__(self, model_id: str | None = None, adapter_path: str | Path | None = None):
        self.model_id = model_id or SR_MODEL_ID
        self.adapter_path = Path(adapter_path) if adapter_path else SR_ADAPTER_DIR
        self._model = None
        self._tokenizer = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self.adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"SR LoRA adapter not found: {self.adapter_path}")
        if not Path(self.model_id).exists():
            raise FileNotFoundError(
                f"BioMistral 4-bit base model not found locally: {self.model_id}"
            )
        try:
            import mlx.core as mx
            from mlx_lm import load as mlx_load
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mlx_lm is required for live Second Recurrence BioMistral LoRA scoring."
            ) from exc

        model, tokenizer = mlx_load(self.model_id, adapter_path=str(self.adapter_path))
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._yes_ids, self._no_ids = _get_yes_no_ids(tokenizer)
        self._mx = mx

    def predict_proba(self, user_prompt: str) -> float:
        self.load()
        assert self._model is not None and self._tokenizer is not None
        assert self._yes_ids is not None and self._no_ids is not None
        mx = self._mx
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self._tokenizer.encode(formatted, add_special_tokens=False)
        logits = self._model(mx.array([input_ids]))
        last = logits[0, -1, :]
        mx.eval(last)
        logit_yes = mx.max(mx.array([last[i].item() for i in self._yes_ids]))
        logit_no = mx.max(mx.array([last[i].item() for i in self._no_ids]))
        p_yes = mx.softmax(mx.array([logit_yes, logit_no]))[0].item()
        return float(p_yes)
