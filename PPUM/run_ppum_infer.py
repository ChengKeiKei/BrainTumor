"""
Score the PPUM external prompts with the EXISTING MU-trained LoRA adapters.
Uses infer.py's own model/tokenizer/logit machinery, but reads prompts from
PPUM/generated/prompts/<mu_tag>__ppum/test.jsonl and writes predictions to
PPUM/generated/results/<mu_tag>__ppum/predictions_test.jsonl.

no-RAG uses adapter Exp{i}__baseline ; RAG uses adapter Exp{i}__beep__beep.

Run: python PPUM/run_ppum_infer.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
CODE = SUBMISSION_ROOT / "first" / "Model" / "code"
sys.path.insert(0, str(CODE))
import infer as I                                  # noqa: E402
from mlx_lm import load                            # noqa: E402

PPUM_ROOT = Path(__file__).resolve().parent
PROMPTS = PPUM_ROOT / "generated" / "prompts"
OUTROOT = PPUM_ROOT / "generated" / "results"
CKPT = SUBMISSION_ROOT / "first" / "Model" / "checkpoints"
MISTRAL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
EXPS = ["Exp1", "Exp2", "Exp3", "Exp4"]
ARMS = [("baseline", "no-RAG"), ("beep__beep", "RAG")]


def score_cell(mu_tag: str):
    prompt_file = PROMPTS / f"{mu_tag}__ppum" / "test.jsonl"
    if not prompt_file.exists():
        print(f"  SKIP {mu_tag}: no prompt file"); return
    adapter = I.resolve_adapter(CKPT / mu_tag / "adapters")
    if not (adapter / "adapter_config.json").exists():
        print(f"  SKIP {mu_tag}: no adapter at {adapter}"); return

    print(f"[{mu_tag}] load adapter @ {adapter}")
    model, tok = load(MISTRAL, adapter_path=str(adapter))
    model.eval()
    yes_ids, no_ids = I.get_yes_no_ids(tok)

    recs = [json.loads(l) for l in open(prompt_file)]
    out = []
    t0 = time.time()
    for i, rec in enumerate(recs, 1):
        user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        s = I.predict_proba(model, tok, user, yes_ids, no_ids)
        out.append({"patient_id": rec.get("patient_id"),
                    "label": int(rec.get("label", -1)), "score": s})
    outdir = OUTROOT / f"{mu_tag}__ppum"; outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "predictions_test.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[{mu_tag}] {len(out)} preds ({time.time()-t0:.1f}s) -> {outdir}")


def main():
    for exp in EXPS:
        for arm, _ in ARMS:
            score_cell(f"{exp}__{arm}")
    print("\nALL PPUM INFERENCE DONE")


if __name__ == "__main__":
    main()
