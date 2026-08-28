from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.fields import (
    FIRST_RECURRENCE_FIELDS,
    FIRST_RECURRENCE_GUIDE,
    SECOND_RECURRENCE_FIELDS,
    SECOND_RECURRENCE_GUIDE,
)
from src.hybrid_inference import LLMEmbedder, checkpoint_status, predict_first_hybrid
from src.imaging import generate_caption_from_uploads, radfm_available
from src.literature import LiteratureDoc, docs_to_frame, retrieve_live_pubmed, retrieve_local_literature
from src.lora_inference import SecondRecurrenceLoRA
from src.model_config import (
    FR_ENCODER_DISPLAY,
    FR_INTERNAL_TEST,
    FR_RERANKER_DISPLAY,
    SR_ADAPTER_DIR,
    SR_ENCODER_DISPLAY,
    SR_FIVEFOLD_METRICS,
    SR_MODEL_ID,
    SR_RERANKER_DISPLAY,
)
from src.risk_engine import (
    Prediction,
    build_first_recurrence_prompt,
    build_second_recurrence_prompt,
)
from src.second_inference import predict_second
from src.ui_components import (
    apply_page_style,
    guide_box,
    render_fields,
    render_prediction,
    render_required_progress,
    validate_required_inputs,
)


ROOT = Path(__file__).resolve().parent

# Widget-key presets for the one-click demo patients. Date widgets use the
# "_cal" suffix; select values must match the Field options exactly.
EXAMPLE_FR_PATIENT: dict[str, object] = {
    "patient_id": "DEMO-GBM-01",
    "diagnosis_date_cal": date(2024, 1, 10),
    "age": 62,
    "sex": "Male",
    "race": "Malay",
    "primary_diagnosis": "Glioblastoma",
    "grade": "4",
    "biopsy_before_resection": "No",
    "previous_brain_tumor": "No",
    "first_surgery_day_cal": date(2024, 1, 15),
    "initial_chemo": "Yes",
    "initial_chemo_name": "Temozolomide",
    "initial_chemo_start_day_cal": date(2024, 2, 20),
    "initial_chemo_end_day_cal": date(2024, 8, 20),
    "radiotherapy": "Yes",
    "rt_start_day_cal": date(2024, 2, 20),
    "rt_end_day_cal": date(2024, 4, 5),
    "rt_dose": 60,
    "rt_fractions": 30,
    "idh1": "Wildtype / negative",
    "mgmt": "Unmethylated",
}

EXAMPLE_SR_PATIENT: dict[str, object] = {
    "patient_id": "DEMO-SR-01",
    "diagnosis_date_cal": date(2023, 6, 1),
    "age": 58,
    "sex": "Female",
    "primary_diagnosis": "Glioblastoma",
    "grade": "4",
    "time_to_first_progression_cal": date(2024, 4, 15),
    "type_first_progression": "Local",
    "multiple_surgeries": "No",
    "additional_therapy": "Yes",
    "additional_therapy_start_cal": date(2024, 5, 1),
    "additional_therapy_end_cal": date(2024, 8, 1),
    "additional_cycles": 4,
    "immunotherapy": "No",
    "latest_mri_day_cal": date(2024, 9, 10),
    "mri_report": (
        "Interval increase in enhancing lesion along the resection cavity margin with new "
        "nodular enhancement and increased surrounding FLAIR edema, suspicious for progression."
    ),
    "idh1": "Wildtype / negative",
    "mgmt": "Unmethylated",
}


def load_example_patient(namespace: str, values: dict[str, object]) -> None:
    """Must be called before the field widgets are instantiated in this rerun."""
    for key, value in values.items():
        st.session_state[f"{namespace}_{key}"] = value


@st.cache_resource(show_spinner=False)
def get_cached_embedder() -> LLMEmbedder:
    return LLMEmbedder()


@st.cache_resource(show_spinner=False)
def get_cached_sr_lora() -> SecondRecurrenceLoRA:
    return SecondRecurrenceLoRA()


def export_payload(task: str, data: dict, prediction_probability: float, extra: dict | None = None) -> bytes:
    payload = {
        "task": task,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": data,
        "demo_prediction_probability": round(prediction_probability, 4),
        "note": "Demo UI payload. Not for clinical use.",
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def render_xai_contribution_chart(contrib) -> None:
    chart_df = contrib.copy()
    chart_df["sign"] = chart_df["contribution"].apply(lambda value: "Pushes risk up" if value >= 0 else "Pushes risk down")
    chart_df["display_feature"] = chart_df["feature"]
    order = chart_df.reindex(chart_df["contribution"].abs().sort_values(ascending=True).index)["display_feature"].tolist()

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("contribution:Q", title="XGBoost contribution on model margin, not probability"),
            y=alt.Y("display_feature:N", sort=order, title=None),
            color=alt.Color(
                "sign:N",
                scale=alt.Scale(
                    domain=["Pushes risk up", "Pushes risk down"],
                    range=["#2fa66a", "#d84a4a"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("feature_group:N", title="Group"),
                alt.Tooltip("encoded_value:N", title="Patient value"),
                alt.Tooltip("contribution:Q", title="Contribution", format=".3f"),
                alt.Tooltip("direction:N", title="Direction"),
                alt.Tooltip("explanation:N", title="Meaning"),
            ],
        )
        .properties(height=max(360, 34 * len(chart_df)))
    )
    rule = alt.Chart(chart_df).mark_rule(color="#334155", strokeWidth=1).encode(x=alt.datum(0))
    st.altair_chart(chart + rule, use_container_width=True)


def render_first_recurrence_tab() -> None:
    st.subheader("First Recurrence prediction")
    st.caption(
        "Prediction point: before the first recurrence/progression event. "
        "Do not enter post-event MRI, salvage treatment, death date, or recurrence-confirmation notes."
    )
    guide_box("Input checklist for accurate First Recurrence prediction", FIRST_RECURRENCE_GUIDE)
    with st.expander("Advanced: model and literature settings", expanded=False):
        st.caption(
            f"Selected model: {FR_ENCODER_DISPLAY} + Exp3 + {FR_RERANKER_DISPLAY} last-token hybrid "
            f"(internal test AUROC {FR_INTERNAL_TEST['AUROC']:.3f}, AUPRC {FR_INTERNAL_TEST['AUPRC']:.3f}, "
            f"Macro-F1 {FR_INTERNAL_TEST['Macro_F1']:.3f}). Molecular fields are not in the XGBoost schema."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            top_k = st.slider("Local PubMed abstracts", 1, 10, 5, key="fr_local_topk")
        with c2:
            use_live_pubmed = st.checkbox("Add latest PubMed refresh", value=False, key="fr_live_pubmed")
        with c3:
            use_real_llm = st.checkbox(
                f"Use live {FR_ENCODER_DISPLAY} last-token embedding",
                value=False,
                key="fr_real_llm",
                help="Off by default to prevent unwanted HuggingFace downloads. Turn on only after the matching BioMistral base model is available locally.",
            )
        st.markdown("**Checkpoint registry**")
        st.caption(
            f"Training used {FR_RERANKER_DISPLAY} to rank PubMed evidence. This demo retrieves from the local "
            "10k corpus with TF-IDF so the UI can run without loading reranker weights."
        )
        st.dataframe(checkpoint_status(), use_container_width=True, hide_index=True)
    ex_col, ex_note = st.columns([1, 3])
    with ex_col:
        if st.button("Load example patient", key="fr_load_example"):
            load_example_patient("fr", EXAMPLE_FR_PATIENT)
    with ex_note:
        st.caption(
            "Fills the form with a typical newly diagnosed GBM case (62-year-old, TMZ + radiotherapy) "
            "so you can see a full prediction in one click. You can edit any value afterwards."
        )

    data, _required_labels = render_fields(FIRST_RECURRENCE_FIELDS, "fr")

    render_required_progress(data, FIRST_RECURRENCE_FIELDS)
    if st.button("Predict First Recurrence risk", type="primary", key="predict_fr"):
        missing, unknown = validate_required_inputs(data, FIRST_RECURRENCE_FIELDS)
        if missing:
            st.error(
                "Please fill in these required fields before predicting: "
                + ", ".join(missing)
            )
            return
        if unknown:
            st.warning(
                "These required questions are still 'Unknown' and will be treated as missing "
                "data by the model: " + ", ".join(unknown)
            )
        query = build_first_recurrence_prompt(data)
        with st.status("Retrieving biomedical literature and running hybrid model...", expanded=True) as status:
            st.write("Searching local brain-tumor PubMed corpus...")
            docs = retrieve_local_literature(query, top_k=top_k)
            live_warning = ""
            if use_live_pubmed:
                st.write("Refreshing latest PubMed records from NCBI...")
                live_docs, live_warning = retrieve_live_pubmed(
                    "glioma recurrence progression treatment molecular prognostic prediction",
                    top_k=3,
                )
                docs = live_docs + docs
            st.write("Running hybrid inference...")
            embedder = get_cached_embedder() if use_real_llm else None
            try:
                hybrid = predict_first_hybrid(data, docs, use_real_llm=use_real_llm, embedder=embedder)
            except OSError as exc:
                st.error(
                    "The matching frozen BioMistral base model is not available locally, so live embedding was stopped before any download. "
                    "Either use smoke-test mode, or place BioMistral-7B-DARE on disk."
                )
                st.exception(exc)
                return
            status.update(label="Hybrid prediction complete", state="complete")

        st.session_state["fr_result"] = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data": data,
            "docs": [(d.pmid, d.title, d.abstract, d.source, d.score, d.year) for d in docs],
            "live_warning": live_warning,
            "warning": hybrid.warning,
            "checkpoint_paths": hybrid.checkpoint_paths,
            "probability": hybrid.probability,
            "threshold": hybrid.threshold,
            "label": hybrid.label,
            "mode": hybrid.mode,
            "model_name": hybrid.model_name,
            "evidence_prompt": hybrid.evidence_prompt,
            "contrib": hybrid.top_contributions.to_dict(orient="records"),
        }

    result = st.session_state.get("fr_result")
    if not result:
        return

    docs = [LiteratureDoc(*row) for row in result["docs"]]

    if result.get("live_warning"):
        st.warning(result["live_warning"])
    if result.get("warning"):
        st.warning(result["warning"])
    if result.get("checkpoint_paths"):
        with st.expander("Checkpoint paths used for this prediction", expanded=False):
            st.json(result["checkpoint_paths"])

    prob_pct = result["probability"] * 100
    result_pid = result["data"].get("patient_id") or "(no patient ID)"
    st.markdown(
        f"""
        <div class="info-card {'risk-high' if result['probability'] >= result['threshold'] else 'risk-low'}">
            <span class="label-pill">Connected model: {result['mode']}</span>
            <h3 style="margin-top:0;">{result['label']}</h3>
            <p style="font-size:2rem; font-weight:700; margin:0.2rem 0;">{prob_pct:.1f}%</p>
            <p class="small-muted">{result['model_name']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Result for patient **{result_pid}**, generated {result.get('created_at', 'earlier')}. "
        "If you change any input above, click Predict again to refresh this result."
    )
    st.caption(
        "This percentage is the model's raw score, **not a clinically calibrated probability**. "
        "Use it to compare and rank patients, not as the absolute chance of recurrence."
    )

    st.markdown("#### XAI: SHAP-style feature contributions")
    st.caption(
        "These are TreeSHAP Shapley-value contributions for each feature on the XGBoost margin "
        "(via `pred_contribs`), not a separate 'SHAP feature set'. "
        "Positive/green bars push toward recurrence-positive; negative/red bars push downward. "
        "They explain this patient locally and are not causal effects. "
        "The Exp3 hybrid does not include molecular columns, so molecular inputs are recorded but do not move the score."
    )
    contrib = pd.DataFrame(result["contrib"])
    render_xai_contribution_chart(contrib)
    st.dataframe(contrib, use_container_width=True, hide_index=True)

    st.markdown("#### Retrieved literature used by the RAG stage")
    st.dataframe(docs_to_frame(docs), use_container_width=True, hide_index=True)
    with st.expander("Full prompt passed into LLM / hybrid feature extractor"):
        st.code(result["evidence_prompt"], language="text")
    st.download_button(
        "Download case payload",
        data=export_payload("First Recurrence", result["data"], result["probability"]),
        file_name=f"first_recurrence_payload_{result['data'].get('patient_id') or 'patient'}.json",
        mime="application/json",
        key="download_fr",
    )


def _render_second_imaging_section(data: dict) -> str:
    st.markdown("#### MRI upload → RadFM / editable caption")
    st.caption(
        "Preferred: upload four pre-landmark NIfTI volumes named with t1c / t1n / t2f / t2w. "
        "PNG/JPG preview uploads are accepted for UI demo, but RadFM captioning needs NIfTI. "
        f"RadFM weights available: {'yes' if radfm_available() else 'no'}."
    )
    uploads = st.file_uploader(
        "Upload MRI (NIfTI .nii/.nii.gz preferred; PNG/JPG allowed for draft caption)",
        accept_multiple_files=True,
        key="sr_mri_uploads",
        help="RadFM path expects four NIfTI volumes. Preview images generate an editable draft caption only.",
    )
    try_radfm = st.checkbox("Try RadFM caption generation when NIfTI + weights are available", value=True, key="sr_try_radfm")

    clinical_context = (
        f"Age={data.get('age')}; Sex={data.get('sex')}; "
        f"Diagnosis={data.get('primary_diagnosis')}; Grade={data.get('grade')}; "
        f"TTP1={data.get('time_to_first_progression')}"
    )

    if st.button("Generate caption from upload", key="sr_gen_caption"):
        result = generate_caption_from_uploads(
            uploads,
            clinical_context=clinical_context,
            try_radfm=try_radfm,
        )
        st.session_state["sr_caption_result"] = {
            "caption": result.caption,
            "source": result.source,
            "warning": result.warning,
            "details": result.details,
        }
        if result.caption:
            st.session_state["sr_mri_report"] = result.caption

    caption_meta = st.session_state.get("sr_caption_result")
    if caption_meta:
        st.info(f"Caption source: {caption_meta.get('source')}")
        if caption_meta.get("warning"):
            st.warning(caption_meta["warning"])
        if caption_meta.get("details"):
            with st.expander("Upload / caption details"):
                st.json(caption_meta["details"])

    if "sr_mri_report" not in st.session_state:
        st.session_state["sr_mri_report"] = ""
    mri_report = st.text_area(
        "MRI / RadFM report text before second-recurrence prediction *",
        key="sr_mri_report",
        height=140,
        help="Editable caption used by the scorer.",
    )
    return mri_report


def render_second_recurrence_tab() -> None:
    st.subheader("Second Recurrence prediction")
    st.caption(
        "Prediction point: after first recurrence/progression, before possible second recurrence/further progression. "
        "Only enter pre-second-event salvage treatment and imaging evidence."
    )
    if not (Path(SR_MODEL_ID).exists() and SR_ADAPTER_DIR.exists()):
        st.warning(
            "**Demo mode.** The trained Second Recurrence model (BioMistral-7B LoRA, "
            "five-fold AUROC 0.720) is too large for this free server, so this tab uses a "
            "rule-based scorer to demonstrate the full workflow, inputs, and explainability. "
            "Run the app locally with the model weights to get real LoRA predictions."
        )
    guide_box("Input checklist for accurate Second Recurrence prediction", SECOND_RECURRENCE_GUIDE)

    with st.expander("Advanced: model and literature settings", expanded=False):
        st.caption(
            f"Selected model: {SR_ENCODER_DISPLAY} + {SR_RERANKER_DISPLAY} "
            f"(five-fold AUROC {SR_FIVEFOLD_METRICS['AUROC']:.3f}, Macro-F1 {SR_FIVEFOLD_METRICS['Macro_F1']:.3f}, "
            f"Sens {SR_FIVEFOLD_METRICS['Sensitivity']:.3f}, Spec {SR_FIVEFOLD_METRICS['Specificity']:.3f})."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            sr_top_k = st.slider("Local PubMed abstracts", 1, 10, 5, key="sr_local_topk")
        with c2:
            sr_live_pubmed = st.checkbox("Add latest PubMed refresh", value=False, key="sr_live_pubmed")
        with c3:
            use_real_sr = st.checkbox(
                f"Use live {SR_ENCODER_DISPLAY} LoRA",
                value=False,
                key="sr_real_llm",
                help="Off by default. Turn on only after the local 4-bit BioMistral weights and shared-adapter checkpoint are available.",
            )
        st.caption(
            f"Training used {SR_RERANKER_DISPLAY} to rank PubMed evidence. This demo retrieves from the local "
            "10k corpus with TF-IDF unless you later load the reranker weights."
        )
        st.dataframe(checkpoint_status(), use_container_width=True, hide_index=True)

    ex_col, ex_note = st.columns([1, 3])
    with ex_col:
        if st.button("Load example patient", key="sr_load_example"):
            load_example_patient("sr", EXAMPLE_SR_PATIENT)
    with ex_note:
        st.caption(
            "Fills the form with a typical post-first-recurrence GBM case (local progression, "
            "salvage therapy, progressing MRI) so you can see a full prediction in one click."
        )

    # Handle imaging separately so upload → caption can populate the report field.
    data, required_labels = render_fields(SECOND_RECURRENCE_FIELDS, "sr", skip_keys={"mri_report"})
    data["mri_report"] = _render_second_imaging_section(data)

    render_required_progress(data, SECOND_RECURRENCE_FIELDS)
    if st.button("Predict Second Recurrence risk", type="primary", key="predict_sr"):
        missing, unknown = validate_required_inputs(data, SECOND_RECURRENCE_FIELDS)
        if missing:
            st.error(
                "Please fill in these required fields before predicting: "
                + ", ".join(missing)
            )
            return
        if unknown:
            st.warning(
                "These required questions are still 'Unknown' and will be treated as missing "
                "data by the model: " + ", ".join(unknown)
            )
        query = build_second_recurrence_prompt(data)
        with st.status("Retrieving literature and scoring Second Recurrence...", expanded=True) as status:
            st.write("Searching local brain-tumor PubMed corpus...")
            docs = retrieve_local_literature(query, top_k=sr_top_k)
            live_warning = ""
            if sr_live_pubmed:
                st.write("Refreshing latest PubMed records from NCBI...")
                live_docs, live_warning = retrieve_live_pubmed(
                    "glioma recurrence progression salvage treatment MRI prognostic prediction",
                    top_k=3,
                )
                docs = live_docs + docs
            st.write("Running selected Second Recurrence model...")
            lora = get_cached_sr_lora() if use_real_sr else None
            try:
                prediction = predict_second(
                    data,
                    required_labels,
                    docs,
                    use_real_llm=use_real_sr,
                    lora=lora,
                )
            except (OSError, FileNotFoundError, RuntimeError) as exc:
                st.error(
                    "Live BioMistral LoRA scoring could not start. Use smoke-test mode, or place the "
                    "4-bit BioMistral weights and the selected shared-adapter checkpoint on disk."
                )
                st.exception(exc)
                return
            status.update(label="Second Recurrence scoring complete", state="complete")

        st.session_state["sr_result"] = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data": data,
            "required_labels": required_labels,
            "docs": [(d.pmid, d.title, d.abstract, d.source, d.score, d.year) for d in docs],
            "live_warning": live_warning,
            "use_real_llm": use_real_sr,
            "probability": prediction.probability,
            "risk_level": prediction.risk_level,
            "evidence_completeness": prediction.evidence_completeness,
            "missing_required": prediction.missing_required,
            "evidence_prompt": prediction.evidence_prompt,
            "drivers": prediction.drivers,
            "contributions": prediction.contributions,
            "mode": prediction.mode,
            "model_name": prediction.model_name,
            "warning": prediction.warning,
            "checkpoint_paths": prediction.checkpoint_paths,
        }

    result = st.session_state.get("sr_result")
    if not result:
        return

    docs = [LiteratureDoc(*row) for row in result["docs"]]
    if result.get("live_warning"):
        st.warning(result["live_warning"])
    sr_pid = result["data"].get("patient_id") or "(no patient ID)"
    st.caption(
        f"Result for patient **{sr_pid}**, generated {result.get('created_at', 'earlier')}. "
        "If you change any input above, click Predict again to refresh this result."
    )
    prediction = Prediction(
        probability=result["probability"],
        risk_level=result["risk_level"],
        evidence_completeness=result["evidence_completeness"],
        drivers=result["drivers"],
        contributions=result["contributions"],
        missing_required=result["missing_required"],
        evidence_prompt=result["evidence_prompt"],
        mode=result.get("mode", ""),
        model_name=result.get("model_name", ""),
        warning=result.get("warning", ""),
        checkpoint_paths=result.get("checkpoint_paths"),
    )
    render_prediction(prediction, show_rule_xai=False)

    st.markdown("#### Retrieved literature context")
    st.caption(
        "Literature is shown for RAG context. Training used PubMedBERT as the reranker; this demo "
        "uses local TF-IDF."
    )
    st.dataframe(docs_to_frame(docs), use_container_width=True, hide_index=True)
    st.download_button(
        "Download case payload",
        data=export_payload(
            "Second Recurrence",
            result["data"],
            result["probability"],
            extra={"caption_meta": st.session_state.get("sr_caption_result")},
        ),
        file_name=f"second_recurrence_payload_{result['data'].get('patient_id') or 'patient'}.json",
        mime="application/json",
        key="download_sr",
    )


def main() -> None:
    st.set_page_config(
        page_title="Glioma Recurrence Risk Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_page_style()

    st.title("Literature-Augmented Glioma Recurrence Risk Demo")
    st.markdown(
        """
        Enter the patient's details, click **Predict**, and the app returns a recurrence-risk
        estimate with an explanation of which inputs pushed the risk up or down.
        Fields marked with `*` are required. Treatment dates are entered as calendar dates —
        the app converts them to days-from-diagnosis for the model automatically.
        """
    )
    st.info(
        "Research demo — not for clinical decision-making. Live LLM scoring is off by default; "
        "expand 'Advanced: model and literature settings' inside each tab to turn it on or see model details."
    )

    with st.sidebar:
        st.header("How to use")
        st.markdown(
            """
            1. Pick a tab: **First Recurrence** (newly diagnosed, pre-recurrence) or
               **Second Recurrence** (already recurred once).
            2. Fill in the patient details. Enter **calendar dates**; days are calculated for you.
            3. Answer *Unknown* when a result is genuinely unavailable — the model handles missing data.
            4. Click **Predict**, then review the risk card and (for First Recurrence)
               the feature-contribution chart below it.
            """
        )
        st.divider()
        st.markdown("**Do not enter future information**")
        st.write("First Recurrence: exclude follow-up MRI and salvage treatment after the first recurrence.")
        st.write("Second Recurrence: exclude MRI/treatment after the second recurrence.")
        st.divider()
        st.caption(
            f"Models — FR: {FR_ENCODER_DISPLAY} + Exp3 + {FR_RERANKER_DISPLAY} hybrid XGBoost. "
            f"SR: {SR_ENCODER_DISPLAY} + {SR_RERANKER_DISPLAY} LoRA. "
            "FR explains with SHAP-style feature contributions."
        )

    tab_first, tab_second = st.tabs(["First Recurrence", "Second Recurrence"])
    with tab_first:
        render_first_recurrence_tab()
    with tab_second:
        render_second_recurrence_tab()


if __name__ == "__main__":
    main()
