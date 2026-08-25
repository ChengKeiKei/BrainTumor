from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from .fields import Field, apply_date_offsets, days_from_diagnosis
from .risk_engine import Prediction


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #0b1730;
            --ink: #111827;
            --muted: #64748b;
            --teal: #16a596;
            --blue: #2d8ee3;
            --paper: #f6f8fa;
            --line: #d9e2ec;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            color: var(--navy);
            letter-spacing: 0 !important;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
        }
        .info-card {
            background: white;
            border: 1px solid var(--line);
            border-left: 5px solid var(--teal);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0 0.8rem 0;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }
        .label-pill {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            background: #e8f6f3;
            color: #0f766e;
            font-weight: 700;
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }
        .risk-high {
            border-left-color: #d64a4a;
        }
        .risk-mid {
            border-left-color: #e9941a;
        }
        .risk-low {
            border-left-color: #16a596;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def guide_box(title: str, items: dict[str, list[str]]) -> None:
    st.markdown(f"#### {title}")
    cols = st.columns(3)
    colors = ["#16a596", "#2d8ee3", "#d64a4a"]
    for col, (heading, values), color in zip(cols, items.items(), colors):
        with col:
            st.markdown(
                f"""
                <div class="info-card" style="border-left-color:{color}">
                    <strong>{heading}</strong>
                    <ul>
                        {''.join(f'<li>{item}</li>' for item in values)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction(prediction: Prediction, *, show_rule_xai: bool = False) -> None:
    probability_pct = prediction.probability * 100
    risk_class = "risk-high" if probability_pct >= 75 else "risk-mid" if probability_pct >= 45 else "risk-low"

    st.markdown(
        f"""
        <div class="info-card {risk_class}">
            <span class="label-pill">{prediction.mode or "Current connected model"}</span>
            <h3 style="margin-top:0;">{prediction.risk_level}</h3>
            <p style="font-size:2rem; font-weight:700; margin:0.2rem 0;">{probability_pct:.1f}%</p>
            <p class="small-muted">{prediction.model_name or "Second Recurrence XAI uses counterfactuals only (no SHAP), matching the LoRA-first deployment path."}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if prediction.warning:
        st.warning(prediction.warning)
    if prediction.checkpoint_paths:
        with st.expander("Checkpoint paths used for this prediction", expanded=False):
            st.json(prediction.checkpoint_paths)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Input / evidence completeness</strong>
                <p style="font-size:1.05rem; margin:0.35rem 0 0 0;">{prediction.evidence_completeness}</p>
                <p class="small-muted">This means whether the required clinical fields are present. It is not a guarantee that the model is correct.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if prediction.missing_required:
            st.error("Missing required input: " + ", ".join(prediction.missing_required))
        else:
            st.success("All required inputs are present.")
    with c2:
        st.markdown("#### What is passed to the framework")
        st.code(prediction.evidence_prompt, language="text")

    if show_rule_xai and prediction.contributions:
        st.markdown("#### Rule contributions (debug)")
        df = pd.DataFrame(prediction.contributions)
        df["effect"] = df["effect"].astype(float)
        st.dataframe(
            df[["feature", "effect", "explanation"]].sort_values("effect", key=lambda s: s.abs(), ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_counterfactual_table(df: pd.DataFrame, *, title: str = "XAI: Counterfactual what-if analysis") -> None:
    st.markdown(f"#### {title}")
    st.caption(
        "Each row changes one or a few inputs and re-runs the scorer. "
        "Green = lower predicted risk than baseline; red = higher predicted risk. "
        "These are local what-if explanations, not causal treatment effects."
    )
    if df is None or df.empty:
        st.info("No counterfactual scenarios were generated for the current inputs.")
        return
    flips = df[df["flips_decision"] == True]  # noqa: E712
    if not flips.empty:
        st.success(
            f"{len(flips)} counterfactual(s) flip the decision threshold. "
            f"Largest |swing|: {flips.iloc[0]['counterfactual']} ({flips.iloc[0]['delta_pp']:+.2f} pp)."
        )

    view = df.copy()
    # Keep a signed helper for styling; display friendly columns.
    def _style_row(row: pd.Series) -> list[str]:
        delta = float(row.get("delta_pp", 0.0))
        if delta < 0:
            color = "background-color:#e8f6f3; color:#0f766e;"
        elif delta > 0:
            color = "background-color:#fde8e8; color:#b91c1c;"
        else:
            color = ""
        return [color] * len(row)

    styled = view.style.apply(_style_row, axis=1).format(
        {
            "baseline_prob_%": "{:.2f}",
            "new_prob_%": "{:.2f}",
            "delta_pp": "{:+.2f}",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_interactive_counterfactual(
    *,
    namespace: str,
    baseline_data: dict[str, Any],
    baseline_prob: float,
    threshold: float,
    predict_fn,
    mode: str,
) -> None:
    """Let the user edit key fields and immediately see the what-if probability."""
    st.markdown("#### Try your own counterfactual")
    st.caption("Change values below and click Apply. Green means risk went down; red means risk went up.")

    patch: dict[str, Any] = {}
    if mode == "first":
        c1, c2, c3 = st.columns(3)
        with c1:
            patch["idh1"] = st.selectbox(
                "What-if IDH1",
                ("Keep current", "Mutant / positive", "Wildtype / negative", "Unknown / not tested"),
                key=f"{namespace}_cf_idh1",
            )
            patch["mgmt"] = st.selectbox(
                "What-if MGMT",
                ("Keep current", "Methylated", "Unmethylated", "Unknown / not tested"),
                key=f"{namespace}_cf_mgmt",
            )
        with c2:
            patch["grade"] = st.selectbox(
                "What-if WHO grade",
                ("Keep current", "1", "2", "3", "4"),
                key=f"{namespace}_cf_grade",
            )
            patch["primary_diagnosis"] = st.selectbox(
                "What-if diagnosis",
                ("Keep current", "Glioblastoma", "Oligodendroglioma", "Astrocytoma", "Diffuse glioma"),
                key=f"{namespace}_cf_dx",
            )
        with c3:
            patch["radiotherapy"] = st.selectbox(
                "What-if radiotherapy",
                ("Keep current", "Yes", "No", "Unknown / not available"),
                key=f"{namespace}_cf_rt",
            )
            age_now = baseline_data.get("age")
            default_age = float(age_now) if isinstance(age_now, (int, float)) else 60.0
            patch["age"] = st.number_input(
                "What-if age",
                min_value=0.0,
                max_value=100.0,
                value=default_age,
                key=f"{namespace}_cf_age",
            )
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            ttp = baseline_data.get("time_to_first_progression")
            default_ttp = float(ttp) if isinstance(ttp, (int, float)) else 365.0
            patch["time_to_first_progression"] = st.number_input(
                "What-if days to first progression",
                min_value=0.0,
                max_value=36500.0,
                value=default_ttp,
                key=f"{namespace}_cf_ttp",
            )
            patch["additional_therapy"] = st.selectbox(
                "What-if salvage therapy",
                ("Keep current", "Yes", "No", "Unknown / not available"),
                key=f"{namespace}_cf_salvage",
            )
        with c2:
            patch["idh1"] = st.selectbox(
                "What-if IDH1",
                ("Keep current", "Mutant / positive", "Wildtype / negative", "Unknown / not tested"),
                key=f"{namespace}_cf_idh1",
            )
            patch["mgmt"] = st.selectbox(
                "What-if MGMT",
                ("Keep current", "Methylated", "Unmethylated", "Unknown / not tested"),
                key=f"{namespace}_cf_mgmt",
            )
        with c3:
            mri_choice = st.selectbox(
                "What-if MRI caption preset",
                (
                    "Keep current",
                    "Progressing (new lesion / increase / edema)",
                    "Stable (no progression / unchanged)",
                ),
                key=f"{namespace}_cf_mri_preset",
            )
            patch["multiple_surgeries"] = st.selectbox(
                "What-if multiple surgeries",
                ("Keep current", "Yes", "No", "Unknown / not available"),
                key=f"{namespace}_cf_multi",
            )

    if st.button("Apply custom what-if", key=f"{namespace}_cf_apply"):
        candidate = dict(baseline_data)
        applied: dict[str, Any] = {}
        for key, value in patch.items():
            if value == "Keep current":
                continue
            if key == "age" and mode == "first":
                if value != baseline_data.get("age"):
                    applied[key] = value
                continue
            if key == "time_to_first_progression" and mode == "second":
                if value != baseline_data.get("time_to_first_progression"):
                    applied[key] = value
                continue
            applied[key] = value

        if mode == "second":
            mri_preset = st.session_state.get(f"{namespace}_cf_mri_preset", "Keep current")
            if str(mri_preset).startswith("Progressing"):
                applied["mri_report"] = (
                    "Clear progression with new enhancing lesion, increase in edema and mass effect."
                )
            elif str(mri_preset).startswith("Stable"):
                applied["mri_report"] = (
                    "Stable disease with no progression, unchanged residual, decrease in edema."
                )

        if not applied:
            st.info("No changes selected versus the baseline case.")
            return

        candidate.update(applied)
        if mode == "first" and applied.get("radiotherapy") == "No":
            candidate.update({"rt_dose": None, "rt_fractions": None, "rt_start_day": None, "rt_end_day": None})
        if mode == "first" and applied.get("radiotherapy") == "Yes":
            candidate.setdefault("rt_dose", 60)
            candidate.setdefault("rt_fractions", 30)

        new_prob = float(predict_fn(candidate))
        delta = (new_prob - baseline_prob) * 100
        color = "#0f766e" if delta < 0 else "#b91c1c" if delta > 0 else "#64748b"
        base_label = "High" if baseline_prob >= threshold else "Lower"
        new_label = "High" if new_prob >= threshold else "Lower"
        flipped = base_label != new_label

        st.markdown(
            f"""
            <div class="info-card" style="border-left-color:{color}">
                <strong>Custom counterfactual result</strong>
                <p style="margin:0.35rem 0 0.2rem 0;">
                    Baseline <b>{baseline_prob*100:.1f}%</b>
                    → New <b style="color:{color}">{new_prob*100:.1f}%</b>
                    (<b style="color:{color}">{delta:+.2f} pp</b>)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Applied changes: {applied}")
        st.write(
            f"Decision: **{base_label} → {new_label}**"
            + (" *(FLIPPED)*" if flipped else "")
        )


def render_fields(
    fields: tuple[Field, ...],
    namespace: str,
    *,
    skip_keys: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    skip_keys = skip_keys or set()
    grouped: dict[str, list[Field]] = defaultdict(list)
    for field in fields:
        if field.key in skip_keys:
            continue
        grouped[field.group].append(field)

    values: dict[str, Any] = {}
    required_labels = {field.key: field.label for field in fields if field.required}

    for group, group_fields in grouped.items():
        with st.expander(
            group,
            expanded=group
            in {
                "Patient information",
                "Required clinical features",
                "Required treatment features",
                "Required baseline features",
                "Required first-recurrence landmark",
                "Required salvage treatment features",
                "Required imaging evidence",
            },
        ):
            cols = st.columns(2)
            visible_idx = 0
            for field in group_fields:
                if field.depends_on is not None:
                    dep_key, dep_value = field.depends_on
                    current = values.get(dep_key, st.session_state.get(f"{namespace}_{dep_key}"))
                    if current != dep_value:
                        values[field.key] = None
                        continue
                idx = visible_idx
                visible_idx += 1
                with cols[idx % 2]:
                    widget_key = f"{namespace}_{field.key}_cal" if field.field_type == "date" else f"{namespace}_{field.key}"
                    if field.field_type == "select":
                        values[field.key] = st.selectbox(
                            field.display_label, field.options, key=widget_key, help=field.help_text or None
                        )
                    elif field.field_type == "number":
                        values[field.key] = st.number_input(
                            field.display_label,
                            min_value=field.min_value,
                            max_value=field.max_value,
                            value=None,
                            placeholder="Leave blank if unavailable",
                            key=widget_key,
                            help=field.help_text or None,
                        )
                    elif field.field_type == "date":
                        values[field.key] = st.date_input(
                            field.display_label,
                            value=None,
                            format="YYYY-MM-DD",
                            key=widget_key,
                            help=field.help_text or "Leave blank if unavailable.",
                        )
                        diagnosis = values.get("diagnosis_date")
                        event = values.get(field.key)
                        if field.key != "diagnosis_date" and diagnosis and event:
                            st.caption(f"Calculated: {days_from_diagnosis(diagnosis, event)} days from diagnosis")
                    elif field.field_type == "textarea":
                        values[field.key] = st.text_area(
                            field.display_label, key=widget_key, height=110, help=field.help_text or None
                        )
                    else:
                        values[field.key] = st.text_input(
                            field.display_label, key=widget_key, help=field.help_text or None
                        )
    return apply_date_offsets(values), required_labels


def validate_required_inputs(
    data: dict[str, Any], fields: tuple[Field, ...]
) -> tuple[list[str], list[str]]:
    """Split required fields into hard-missing and still-'Unknown' before predicting.

    Hard-missing (empty text/number/date) should block prediction; 'Unknown'
    select answers are legitimate clinical answers and only deserve a warning.
    """
    missing: list[str] = []
    unknown: list[str] = []
    for field in fields:
        if not field.required:
            continue
        if field.depends_on is not None:
            dep_key, dep_value = field.depends_on
            if data.get(dep_key) != dep_value:
                continue
        value = data.get(field.key)
        if field.field_type == "select":
            if value is None or str(value).strip().lower().startswith("unknown"):
                unknown.append(field.label)
        elif value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field.label)
    return missing, unknown
