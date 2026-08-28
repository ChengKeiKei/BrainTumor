# BrainTumor — FYP

## Live demo

Try the hosted Streamlit demo (no install needed):

**https://braintumor-demo.streamlit.app**

Notes for visitors:

- The First Recurrence tab runs the real trained hybrid XGBoost artifact
  (BioMistral last-token embeddings + PCA + XGBoost, internal MU test
  AUROC 0.978). The Second Recurrence tab runs in demo mode on the free
  server — the BioMistral-7B LoRA cannot fit in the free tier's memory, so
  it shows the full workflow and explainability with a rule-based scorer.
- If the app has been idle it may show "get this app back up" — click it
  and wait about a minute.
- Research demo only; not for clinical decision-making.

## Included best-model checkpoints

- **First Recurrence (deployed UI model): BioMistral + Exp3 + MedCPT hybrid.**
  - `first/Model/checkpoints/Exp3__beep__medcpt__biomistral/adapters/` —
    MU-trained MLX LoRA adapter (final `adapters.safetensors` +
    `adapter_config.json`).
  - `UI/models/first_recur_hybrid_biomistral_last_exp3_medcpt.joblib` —
    the hybrid last-token-embedding + PCA + XGBoost artifact used by the UI
    (internal MU test AUROC 0.978, AUPRC 0.993).
- **Second Recurrence (deployed UI model): BioMistral + PubMedBERT
  shared-adapter LoRA (five-fold mean AUROC 0.720).**
  - `second/Model/checkpoints/ExpC_TxVLM__beep__mixed5__biomistral__sharedcv__fold0__v3_structured/adapters/`
    — the fold-0 shared adapter, which is the one the UI loads for live
    scoring (final adapter only). The reported 0.720 is the five-fold mean;
    folds 1-4 remain in the original `Second_Recur` folder if needed.

Intermediate step snapshots, training JSONL files, and logs were not copied.
The BioMistral-7B-DARE base weights (full and 4-bit) are still excluded and
must be downloaded/placed locally as described in the model READMEs.

## Folder map

```text
BrainTumor/
├── dataset/   preprocessing notebooks and leakage/landmark code
├── first/     first-recurrence RAG, LoRA, evaluation, and hybrid XGBoost
├── second/    second-recurrence multimodal, CV, baseline, and VLM code
├── PPUM/      external PPUM mapping, inference, calibration, and evaluation
├── database/  shared PubMed retrieval and five-reranker implementation
└── UI/        Streamlit doctor demo for hybrid first-recurrence inference
```

The main flow is:

```text
raw clinical/MRI files
→ leakage-safe preprocessing and patient splits
→ structured patient narrative
→ PubMed BM25 + BEEP dense retrieval
→ BEEP/MiniLM/MedCPT/ColBERT/BGE-M3 reranking
→ Mistral or BioMistral prompt
→ LoRA probability or frozen-embedding XGBoost
→ calibration and patient-level evaluation
```

## 1. Environment

Python 3.11 is recommended. LoRA training and inference use MLX and therefore
require an Apple-silicon Mac. Dataset preprocessing, conventional baselines,
and most retrieval code can run on Linux or macOS.

```bash
cd BrainTumor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Apple silicon:

```bash
pip install -r requirements-apple-silicon.txt
```

Other systems, for preprocessing/baselines/retrieval only:

```bash
pip install -r requirements.txt
```

Model repositories are downloaded on first use. RadFM is separate and much
heavier; see the VLM section below.

## 2. Restore private and large inputs

Follow [dataset/README.md](dataset/README.md) and
[database/README.md](database/README.md). In summary:

- Put the MU clinical workbook, segmentation workbook, and MRI folders under
  `dataset/first/Raw/` and `dataset/second/Raw/`.
- Put the 10,000-document PubMed JSONL, BEEP biencoder checkpoint, BEEP
  reranker checkpoint, and generated retrieval indexes under `database/`.
- Never commit these files. `.gitignore` already excludes them.

## 3. Preprocess the cohorts

Run the first-recurrence notebook from its directory:

```bash
cd dataset/first
jupyter lab preprocessing.ipynb
```

It generates `Processed/` and the frozen `splits/` files used by every
first-recurrence model.

Run the second-recurrence notebooks in this order:

```bash
cd ../second
jupyter lab Clinical_preprocessing.ipynb
jupyter lab Image_preprocessing.ipynb
jupyter lab Radiomic_preprocessing.ipynb
cd ../..
```

The notebooks in this submission have zero saved outputs. EDA is optional;
the first-recurrence `EDA.ipynb` is included only as reusable analysis code.

## 4. Prepare the literature database

After placing the corpus and BEEP checkpoints:

```bash
python database/Retrieval/code/sparse_retrieval.py
python database/Retrieval/code/dense_retrieval_beep.py build --batch-size 16
```

These commands create ignored index files under
`database/Retrieval/indexes/`.

## 5. Run first-recurrence prediction

Start with a dry run of the configured 24-cell experiment grid:

```bash
python first/Model/code/run_grid.py --dry-run
```

Build and run one no-RAG cell:

```bash
python first/Model/code/build_dataset.py \
  --experiments Exp4 --baseline --retrievers none --rerankers none
python first/Model/code/train.py --tag Exp4__baseline
python first/Model/code/infer.py --tag Exp4__baseline --split valid
python first/Model/code/infer.py --tag Exp4__baseline --split test
python first/Model/code/evaluate.py \
  --predictions first/Model/results/Exp4__baseline/predictions_test.jsonl
```

Build one RAG cell:

```bash
python first/Model/code/build_dataset.py \
  --experiments Exp4 --retrievers beep --rerankers beep
python first/Model/code/train.py --tag Exp4__beep__beep
python first/Model/code/infer.py --tag Exp4__beep__beep --split test
```

Run the configured full grid only after the one-cell check succeeds:

```bash
python first/Model/code/run_grid.py --skip-existing
```

Recommended class-imbalance improvement path:

```bash
python first/Imbalance/extract_embeddings.py \
  --cell-tag Exp4__beep__beep --encoder mistral --pooling last
python first/Imbalance/run_llm_xgb.py \
  --cell-tag Exp4__beep__beep --encoder mistral \
  --pooling last --pca-dim 64 --include-clinical
```

Use `last` or `max` pooling first. The old experiments found float16 overflow
in a small fraction of mean-pooled embeddings.

## 6. Run second-recurrence prediction

Generate the stratified five-fold split structure:

```bash
python second/Model/code/run_cv_grid.py --make-folds-only
```

Run conventional pooled five-fold baselines:

```bash
python second/Model/code/run_final_baselines_cv.py
```

Run one LLM cross-validation cell:

```bash
python second/Model/code/run_cv_grid.py \
  --exp ExpC_TxVLM --ret beep --rrk colbert \
  --llm mistral --captions-version v3_structured
```

Check the standard grid before launching it:

```bash
python second/Model/code/run_grid.py --dry-run
```

For the locked five-level no-RAG comparison:

```bash
python second/Model/code/run_grid.py \
  --grid-config second/Model/configs/final_norag_grid.yaml --dry-run
# Remove --dry-run only after reviewing the five planned cells.
```

The original second-recurrence landmark depends on observed outcome/follow-up
status, so it supports retrospective association only. For a prospective
sensitivity analysis, run:

```bash
python dataset/second/audit_followup_censoring.py
python second/Model/code/run_fixed_landmark_baselines_cv.py
```

Use pooled patient-level five-fold results as the main second-recurrence
estimate. The inherited fixed split has substantial class-prior drift.

### Optional RadFM captions

RadFM requires its own repository and approximately 50 GB of weights:

```bash
bash second/VLM/setup_radfm.sh
python second/VLM/run_radfm_captions.py \
  --max 1 --prompt-version v3_structured
python second/VLM/run_radfm_captions.py \
  --prompt-version v3_structured
python second/VLM/evaluate_captions.py
```

Always run the one-scan smoke test before processing the full MRI set.

## 7. Run PPUM external validation

Place the private PPUM workbook under `PPUM/input/`. The scripts default to
`PPUM/input/Collect Data (4).xlsx`; the latest cohort used in this project was
`Collect Data (6).xlsx` (n=64). Either rename your workbook to the default, or
pass your actual filename with `--xlsx`. The sheet must be named
`PPUM_First_Recurrence`.

```text
PPUM/input/<your PPUM workbook>.xlsx
```

Then run (add `--xlsx "PPUM/input/<your file>.xlsx"` to the first command if
you did not use the default name):

```bash
python PPUM/build_ppum_external.py
python PPUM/build_ppum_prompts.py
python PPUM/run_ppum_infer.py
python PPUM/clinical_external_ppum.py
python PPUM/eval_ppum.py
```

The PPUM LoRA inference step expects matching MU-trained adapters under
`first/Model/checkpoints/`. All PPUM-derived files are written below
`PPUM/generated/` and are ignored.

PPUM is a small, shifted external cohort. Treat confidence intervals and
missing molecular/treatment fields explicitly; do not claim institution-wide
superiority from point estimates alone.

## 8. Run the doctor demo UI

```bash
cd UI
python -m pip install -r requirements.txt
python tests/test_xai_stress.py
streamlit run app.py
```

Open http://localhost:8501. First Recurrence uses the saved hybrid artifact
`UI/models/first_recur_hybrid_biomistral_last_exp3_medcpt.joblib`
(Exp3__beep__medcpt + BioMistral-7B-DARE last-token embeddings + PCA +
class-weighted XGBoost). Second Recurrence uses the BioMistral + PubMedBERT
shared-adapter LoRA (fold 0) for live scoring, with a transparent demo
engine as the smoke-test fallback. See [UI/README.md](UI/README.md).

### How to enter a patient

Do not type “days from diagnosis”. Pick a **diagnosis date**, then pick
calendar dates for treatments, recurrence, and MRI. The UI converts each
event into days after diagnosis for the model.

1. In the sidebar, choose **First Recurrence** or **Second Recurrence**.
2. Leave **Use live LLM scoring** off unless local BioMistral weights are
   installed. Smoke test is the default demo path.
3. Set **Diagnosis date** first. This is required.
4. Fill demographics, histology, molecular, and treatment fields using the
   dropdowns. Blank/unknown is allowed; the encoder maps it to missing.
5. For treatment windows, enter **start date** and **end date** (chemo,
   RT, salvage, immunotherapy). Leave a date empty if that treatment was
   not given or the date is unknown.
6. Enter **First recurrence date** when known. Second Recurrence also has
   an **MRI date**.
7. Second Recurrence only: optionally **upload MRI**. PNG/JPG get an
   editable fallback caption. NIfTI 4-modality (`t1c`/`t1n`/`t2f`/`t2w`)
   is required for real RadFM. You can edit the imaging report before
   scoring.
8. Click **Run prediction**.

The UI then shows the risk probability, a short clinical interpretation,
and explainability:

- **First Recurrence:** SHAP-style feature contributions plus a
  counterfactual what-if table (and an interactive “Try your own” panel).
- **Second Recurrence:** counterfactual what-if only (no SHAP). Green
  means the change lowered predicted risk; red means it raised it.

### Example A — First Recurrence, typical GBM

Use this as a first smoke-test click-through.

| Field | Example value |
|---|---|
| Diagnosis date | 2023-01-15 |
| Age | 62 |
| Sex | Male |
| Race | White |
| Histology | Glioblastoma |
| Grade | 4 |
| IDH | Wildtype |
| MGMT | Unmethylated |
| 1p/19q | Intact |
| Surgery | Gross total resection |
| Chemo start / end | 2023-02-01 / 2023-07-15 |
| RT start / end | 2023-02-01 / 2023-03-15 |
| RT dose | 60 Gy |
| First recurrence date | 2023-11-10 |

What the UI computes internally (you do not type these):

- Chemo start = 17 days after diagnosis
- RT start = 17 days; RT end = 59 days
- First recurrence = 299 days

Then click **Run prediction**. In smoke-test mode the hybrid score stays
in a fairly tight band because the LLM embedding is the training-set
mean; clinical/treatment fields still move the probability a few points.
Open **What would change this prediction?** to see flips such as MGMT
methylated vs unmethylated. Molecular flips do not change the Exp3 hybrid
score (molecular is not in that XGBoost schema).

### Example B — First Recurrence, lower-risk contrast

Keep the same diagnosis date `2023-01-15`, then change:

| Field | Example value |
|---|---|
| Age | 34 |
| Histology | Oligodendroglioma |
| Grade | 2 |
| IDH | Mutant |
| 1p/19q | Codeleted |
| MGMT | Methylated |
| First recurrence date | leave empty (or a much later date) |

Run again and compare the probability and the counterfactual table with
Example A. Age, histology, grade, and treatment timing are the fields
that can move the Exp3 hybrid score.

### Example C — Second Recurrence, with optional MRI

| Field | Example value |
|---|---|
| Diagnosis date | 2022-06-01 |
| Age | 58 |
| Histology | Glioblastoma |
| Grade | 4 |
| IDH | Wildtype |
| MGMT | Unmethylated |
| First recurrence date | 2023-03-15 |
| MRI date | 2023-04-01 |
| Salvage start / end | 2023-03-20 / 2023-08-01 |
| MRI upload | optional PNG/JPG, or skip |

After upload, edit the imaging report if needed (for example add
“progressive enhancing mass” or “no progression / stable”). Click
**Run prediction**. Use the counterfactual panel to test flips such as
MGMT, extent of resection, or a milder MRI report. In smoke-test mode
this uses the transparent demo engine; turn on live LLM scoring only
when the BioMistral 4-bit base and LoRA adapter are present locally.

## 9. Rebuild second-recurrence summaries

After model outputs exist:

```bash
python second/Evaluation/compare_final_oof.py
python second/Evaluation/build_shared_reranker_summary.py
python second/Evaluation/build_final_completion_results.py
python second/Evaluation/verify_results_unchanged.py
```

Generated summaries go to `second/Evaluation/generated/`.

## 10. Guidance for the next researcher

Highest-priority improvements:

1. Replace the outcome-dependent second-recurrence landmark with a
   pre-specified landmark and prediction horizon, and preserve censoring.
2. Use nested or repeated patient-level cross-validation for model selection;
   fit preprocessing, PCA, calibration, and thresholds inside training folds.
3. Compare the frozen-embedding hybrid against LoRA using identical folds and
   paired patient-level bootstrap tests.
4. Rebuild retrieval indexes from a dated, documented PubMed snapshot and
   record corpus/checkpoint hashes.
5. Add automated tests using synthetic rows so data leakage guards can be
   checked without private data.
6. Treat RadFM captions as optional until anatomy and leakage audits pass on a
   larger reviewed sample.

## 11. Generated-file policy

Keep this folder code-plus-best-checkpoints only. Before submission, verify:

```bash
find . -type f | sort
find . -type f \( -name '*.png' -o -name '*.csv' -o -name '*.xlsx' \
  -o -name '*.pt' -o -name '*.bin' -o -name '*.faiss' -o -name '*.log' \)
```

The second command should print nothing in a clean submission. The only
binary artifacts expected are the shipped best-model checkpoints:
`*.safetensors` under `first/Model/checkpoints/` and
`second/Model/checkpoints/`, and the single `.joblib` under `UI/models/`.
