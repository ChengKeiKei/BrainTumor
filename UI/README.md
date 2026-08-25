# Glioma Recurrence Risk Demo UI

Doctor-facing Streamlit demo for the first/second recurrence workflows.

## Location

This UI now lives under the handoff tree:

```text
24083155/UI/
```

## Run

```bash
cd "/Users/ckk/RAG/24083155/UI"
python -m pip install -r requirements.txt
python tests/test_xai_stress.py
streamlit run app.py
```

Optional artifact rebuild (needs `First_Recur` embeddings on disk):

```bash
python build_deployment_artifacts.py
```

## Structure

```text
UI/
├── app.py
├── build_deployment_artifacts.py
├── requirements.txt
├── configs/
│   ├── feature_groups.json      # Exp3 (active) + Exp4 (reference) clinical schemas
│   └── molecular_codes.json     # MU numeric code legend
├── models/                      # runtime joblib artifact (gitignored)
├── cache/                       # TF-IDF retrieval cache (gitignored)
├── src/
│   ├── encoding.py              # UI label → training vocabulary
│   ├── fields.py
│   ├── hybrid_inference.py      # First Recurrence hybrid
│   ├── literature.py
│   ├── lora_inference.py        # Second Recurrence BioMistral Yes/No LoRA
│   ├── model_config.py          # selected UI models
│   ├── risk_engine.py           # Second Recurrence smoke/demo engine
│   ├── second_inference.py
│   └── ui_components.py
└── tests/
    └── test_xai_stress.py
```

## Checkpoints used by First Recurrence

Selected UI model: **BioMistral-7B-DARE + Exp3 + MedCPT + last-token hybrid**.

Internal MU test (legacy last-token embeddings, threshold 0.5): AUROC 0.978, AUPRC 0.993, Macro-F1 0.868.

| Component | Checkpoint / path | Role |
|---|---|---|
| Hybrid classifier | `UI/models/first_recur_hybrid_biomistral_last_exp3_medcpt.joblib` | PCA-64 + Exp3 clinical/treatment + class-weighted XGBoost |
| Frozen LLM | `BioMistral/BioMistral-7B-DARE` | Last non-pad hidden-state embeddings (live mode) |
| Training embeddings | `First_Recur/Imbalance/embeddings/biomistral/Exp3__beep__medcpt` | Legacy last-token dir used for the 0.978 cell |
| Experiment cell | `Exp3__beep__medcpt` | BEEP retrieval + MedCPT rerank prompts |
| Clinical schema | `Exp3_metadata_treatment` | Metadata + treatment; molecular is not in XGBoost |
| LoRA (status only) | `First_Recur/Model/checkpoints/Exp3__beep__medcpt__biomistral/adapters` | Direct Yes/No LoRA path; not used by hybrid XGBoost |

Pipeline:

```text
local PubMed TF-IDF retrieval / optional live PubMed
→ Exp3 patient prompt (no molecular block)
→ frozen BioMistral-7B-DARE last-token embedding
→ train-only PCA-64
→ Exp3 clinical/treatment features
→ class-weighted XGBoost
→ TreeSHAP-style pred_contribs XAI
```

Training ranked literature with MedCPT. This demo retrieves from the same 10k PubMed corpus with TF-IDF unless MedCPT weights are later wired in.

## Checkpoints used by Second Recurrence

Selected UI model: **BioMistral + PubMedBERT** (ExpC_v3 structured, shared-adapter family).

Thesis five-fold mean: AUROC 0.720, Macro-F1 0.700, Sens 0.718, Spec 0.681.

Live scoring loads fold 0 of that shared adapter (`ExpC_TxVLM__beep__mixed5__biomistral__sharedcv__fold0__v3_structured`) plus the local 4-bit BioMistral base. It is one checkpoint, not the five-fold ensemble.

| Component | Checkpoint / path | Role |
|---|---|---|
| LoRA adapter | `Second_Recur/Model/checkpoints/ExpC_TxVLM__beep__mixed5__biomistral__sharedcv__fold0__v3_structured/adapters` | Live Yes/No probability |
| Base LLM | `Second_Recur/Model/local_models/BioMistral-7B-DARE-4bit` | Frozen 4-bit BioMistral |
| Smoke fallback | `src/risk_engine.py` | Transparent demo engine when live LoRA is off |

Training ranked literature with PubMedBERT. This demo uses local TF-IDF for retrieval.

## Modes

- Fast smoke test: FR uses saved XGBoost + training-set mean LLM embedding; SR uses the demo engine.
- Full live scoring: FR loads local BioMistral-7B and embeds the new patient prompt; SR loads BioMistral 4-bit + LoRA and reads Yes/No logits.

Live LLM scoring is **off by default** so Streamlit does not download weights.

## XAI

- **First Recurrence:** SHAP-style XGBoost `pred_contribs` **and** counterfactual what-if flips. Molecular flips do not change the Exp3 hybrid score.
- **Second Recurrence:** counterfactual only (no SHAP), matching the LoRA-first path without a hybrid feature matrix.
- **Second Recurrence imaging:** upload MRI → optional RadFM caption (when `second/VLM/RadFM_weights` exists) → editable report text used by scoring/counterfactuals.
