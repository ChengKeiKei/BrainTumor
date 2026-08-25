"""Counterfactual XAI helpers for First and Second Recurrence demos."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pandas as pd


def _changed(base: dict[str, Any], patch: dict[str, Any]) -> str:
    parts = []
    for key, new_value in patch.items():
        old = base.get(key)
        parts.append(f"{key}: {old!r} → {new_value!r}")
    return "; ".join(parts)


def run_counterfactuals(
    base_data: dict[str, Any],
    scenarios: list[tuple[str, dict[str, Any]]],
    predict_fn: Callable[[dict[str, Any]], float],
    *,
    baseline_prob: float | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate one-change (or bundled) counterfactual scenarios.

    predict_fn must return a probability in [0, 1].
    """
    if baseline_prob is None:
        baseline_prob = float(predict_fn(base_data))

    rows: list[dict[str, Any]] = []
    base_label = "High" if baseline_prob >= threshold else "Lower"
    for name, patch in scenarios:
        candidate = deepcopy(base_data)
        candidate.update(patch)
        new_prob = float(predict_fn(candidate))
        new_label = "High" if new_prob >= threshold else "Lower"
        rows.append(
            {
                "counterfactual": name,
                "change": _changed(base_data, patch),
                "baseline_prob_%": round(baseline_prob * 100, 2),
                "new_prob_%": round(new_prob * 100, 2),
                "delta_pp": round((new_prob - baseline_prob) * 100, 2),
                "baseline_label": base_label,
                "new_label": new_label,
                "flips_decision": base_label != new_label,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reindex(df["delta_pp"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def first_recurrence_scenarios(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scenarios: list[tuple[str, dict[str, Any]]] = []

    if data.get("idh1") != "Mutant / positive":
        scenarios.append(("Set IDH1 to mutant / positive", {"idh1": "Mutant / positive"}))
    if data.get("idh1") != "Wildtype / negative":
        scenarios.append(("Set IDH1 to wildtype / negative", {"idh1": "Wildtype / negative"}))

    if data.get("mgmt") != "Methylated":
        scenarios.append(("Set MGMT to methylated", {"mgmt": "Methylated"}))
    if data.get("mgmt") != "Unmethylated":
        scenarios.append(("Set MGMT to unmethylated", {"mgmt": "Unmethylated"}))

    if data.get("codeletion_1p19q") != "Mutant / positive":
        scenarios.append(("Set 1p/19q to codeleted", {"codeletion_1p19q": "Mutant / positive"}))

    if data.get("radiotherapy") != "No":
        scenarios.append(
            (
                "Remove radiotherapy",
                {
                    "radiotherapy": "No",
                    "rt_dose": None,
                    "rt_fractions": None,
                    "rt_start_day": None,
                    "rt_end_day": None,
                },
            )
        )
    if data.get("radiotherapy") != "Yes":
        scenarios.append(
            (
                "Add radiotherapy 60 Gy / 30 fx",
                {
                    "radiotherapy": "Yes",
                    "rt_dose": 60,
                    "rt_fractions": 30,
                    "rt_start_day": data.get("rt_start_day") or 30,
                    "rt_end_day": data.get("rt_end_day") or 72,
                },
            )
        )

    if data.get("initial_chemo") != "No":
        scenarios.append(("Remove initial chemotherapy", {"initial_chemo": "No"}))
    if data.get("initial_chemo") != "Yes":
        scenarios.append(
            (
                "Add initial chemotherapy (TMZ)",
                {"initial_chemo": "Yes", "initial_chemo_name": "Temozolomide"},
            )
        )

    if str(data.get("grade")) != "2":
        scenarios.append(("Lower WHO grade to 2", {"grade": "2"}))
    if str(data.get("grade")) != "4":
        scenarios.append(("Raise WHO grade to 4", {"grade": "4"}))

    if data.get("primary_diagnosis") != "Oligodendroglioma":
        scenarios.append(
            (
                "Change diagnosis to Oligodendroglioma",
                {"primary_diagnosis": "Oligodendroglioma", "grade": "2"},
            )
        )
    if data.get("primary_diagnosis") != "Glioblastoma":
        scenarios.append(
            (
                "Change diagnosis to Glioblastoma",
                {"primary_diagnosis": "Glioblastoma", "grade": "4"},
            )
        )

    age = data.get("age")
    if isinstance(age, (int, float)):
        scenarios.append(("Age −15 years", {"age": max(0, float(age) - 15)}))
        scenarios.append(("Age +15 years", {"age": min(100, float(age) + 15)}))

    # Favorable molecular bundle
    scenarios.append(
        (
            "Favorable molecular bundle (IDHmut + codeleted + MGMTm)",
            {
                "idh1": "Mutant / positive",
                "codeletion_1p19q": "Mutant / positive",
                "mgmt": "Methylated",
            },
        )
    )
    scenarios.append(
        (
            "Unfavorable molecular bundle (IDHwt + MGMTun)",
            {
                "idh1": "Wildtype / negative",
                "codeletion_1p19q": "Wildtype / negative",
                "mgmt": "Unmethylated",
            },
        )
    )
    return scenarios


def second_recurrence_scenarios(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scenarios: list[tuple[str, dict[str, Any]]] = []

    ttp = data.get("time_to_first_progression")
    if isinstance(ttp, (int, float)):
        scenarios.append(("First progression much earlier (180 days)", {"time_to_first_progression": 180}))
        scenarios.append(("First progression later (1200 days)", {"time_to_first_progression": 1200}))

    if data.get("additional_therapy") != "Yes":
        scenarios.append(
            (
                "Add salvage / additional therapy (6 cycles)",
                {"additional_therapy": "Yes", "additional_cycles": 6},
            )
        )
    if data.get("additional_therapy") != "No":
        scenarios.append(("Remove salvage / additional therapy", {"additional_therapy": "No", "additional_cycles": 0}))

    if data.get("multiple_surgeries") != "Yes":
        scenarios.append(("Set multiple surgeries = Yes", {"multiple_surgeries": "Yes"}))
    if data.get("multiple_surgeries") != "No":
        scenarios.append(("Set multiple surgeries = No", {"multiple_surgeries": "No"}))

    scenarios.append(
        (
            "MRI caption → progressing",
            {
                "mri_report": (
                    "Clear progression with new enhancing lesion, increase in edema and mass effect."
                )
            },
        )
    )
    scenarios.append(
        (
            "MRI caption → stable",
            {
                "mri_report": (
                    "Stable disease with no progression, unchanged residual, decrease in edema."
                )
            },
        )
    )

    if data.get("idh1") != "Mutant / positive":
        scenarios.append(("Set IDH1 to mutant / positive", {"idh1": "Mutant / positive"}))
    if data.get("idh1") != "Wildtype / negative":
        scenarios.append(("Set IDH1 to wildtype / negative", {"idh1": "Wildtype / negative"}))

    if data.get("mgmt") != "Methylated":
        scenarios.append(("Set MGMT to methylated", {"mgmt": "Methylated"}))
    if data.get("mgmt") != "Unmethylated":
        scenarios.append(("Set MGMT to unmethylated", {"mgmt": "Unmethylated"}))

    enh = data.get("enhancing_volume")
    if not isinstance(enh, (int, float)) or enh <= 20:
        scenarios.append(("Raise enhancing volume to 35", {"enhancing_volume": 35}))
    else:
        scenarios.append(("Lower enhancing volume to 8", {"enhancing_volume": 8}))

    if data.get("primary_diagnosis") != "Glioblastoma":
        scenarios.append(("Change diagnosis to Glioblastoma G4", {"primary_diagnosis": "Glioblastoma", "grade": "4"}))
    if data.get("primary_diagnosis") != "Oligodendroglioma":
        scenarios.append(
            (
                "Change diagnosis to Oligodendroglioma G2",
                {"primary_diagnosis": "Oligodendroglioma", "grade": "2"},
            )
        )
    return scenarios
