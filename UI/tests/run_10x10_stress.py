"""Run 10 First Recurrence + 10 Second Recurrence stress cases, including date→day conversion."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))
sys.path.insert(0, str(UI_ROOT / "tests"))

from src.counterfactual import first_recurrence_scenarios, run_counterfactuals, second_recurrence_scenarios
from src.fields import apply_date_offsets
from src.hybrid_inference import predict_first_hybrid
from src.literature import LiteratureDoc
from src.second_inference import predict_second
from test_second_recurrence_stress import CASES as SR_CASES
from test_second_recurrence_stress import REQUIRED_LABELS
from test_second_recurrence_stress import (
    run_second_battery,
    test_feature_flip_directions,
    test_probability_matches_full_rule_logit,
    test_ten_cases_vary_and_ordered,
    test_xai_effects_are_consistent,
)
from test_xai_stress import CASES as FR_CASES
from test_xai_stress import run_smoke_battery, test_real_embeddings_move_probability, test_ten_cases_not_hardcoded


DOCS = [LiteratureDoc("1", "GBM prognosis", "Glioblastoma IDH-wildtype recurrence.", "stub", 0.9, "2023")]
DX = date(2023, 1, 1)


def _with_dates(numeric: dict, day_keys: tuple[str, ...]) -> dict:
    out = dict(numeric)
    out["diagnosis_date"] = DX
    for key in day_keys:
        days = numeric.get(key)
        if isinstance(days, (int, float)):
            out[key] = DX + timedelta(days=int(days))
    return apply_date_offsets(out)


def first_date_path_matches_numeric() -> pd.DataFrame:
    day_keys = (
        "first_surgery_day",
        "initial_chemo_start_day",
        "initial_chemo_end_day",
        "rt_start_day",
        "rt_end_day",
    )
    rows = []
    for name, data in FR_CASES.items():
        numeric = dict(data)
        numeric["diagnosis_date"] = DX
        p_num = predict_first_hybrid(numeric, DOCS, use_real_llm=False).probability
        dated = _with_dates(data, day_keys)
        p_date = predict_first_hybrid(dated, DOCS, use_real_llm=False).probability
        rows.append(
            {
                "case": name,
                "numeric_%": round(p_num * 100, 2),
                "from_dates_%": round(p_date * 100, 2),
                "match": abs(p_num - p_date) < 1e-9,
            }
        )
    return pd.DataFrame(rows)


def second_date_path_matches_numeric() -> pd.DataFrame:
    day_keys = (
        "time_to_first_progression",
        "additional_therapy_start",
        "additional_therapy_end",
        "immunotherapy_start",
        "latest_mri_day",
    )
    rows = []
    for name, data in SR_CASES.items():
        numeric = dict(data)
        numeric["diagnosis_date"] = DX
        p_num = predict_second(numeric, REQUIRED_LABELS, DOCS, use_real_llm=False).probability
        dated = _with_dates(data, day_keys)
        p_date = predict_second(dated, REQUIRED_LABELS, DOCS, use_real_llm=False).probability
        rows.append(
            {
                "case": name,
                "numeric_%": round(p_num * 100, 2),
                "from_dates_%": round(p_date * 100, 2),
                "match": abs(p_num - p_date) < 1e-9,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 72)
    print("FIRST RECURRENCE — 10 hybrid smoke cases")
    print("=" * 72)
    fr = run_smoke_battery()
    print(fr.to_string(index=False))
    fr_probs = fr["prob_%"].tolist()
    print(f"Unique={len(set(fr_probs))}  range={min(fr_probs):.2f}%..{max(fr_probs):.2f}%  span={max(fr_probs)-min(fr_probs):.2f} pp")
    print(f"XAI reconstructs proba: {bool(fr['xai_ok'].all())}")

    print("\nFIRST — date inputs must match numeric day offsets")
    fr_dates = first_date_path_matches_numeric()
    print(fr_dates.to_string(index=False))
    assert fr_dates["match"].all(), fr_dates

    gbm = FR_CASES["01_GBM_G4_IDHwt_MGMTun_RT"]
    p0 = predict_first_hybrid(gbm, DOCS, False).probability
    cf = run_counterfactuals(gbm, first_recurrence_scenarios(gbm), lambda d: predict_first_hybrid(d, DOCS, False).probability, baseline_prob=p0)
    print(f"\nFIRST counterfactuals: {len(cf)} scenarios, {int(cf['flips_decision'].sum())} flip threshold")

    test_ten_cases_not_hardcoded()
    test_real_embeddings_move_probability()

    print("\n" + "=" * 72)
    print("SECOND RECURRENCE — 10 UI-path cases (smoke demo standing in for LoRA)")
    print("=" * 72)
    sr = run_second_battery()
    print(sr.to_string(index=False))
    sr_probs = sr["prob_%"].tolist()
    print(f"Unique={len(set(sr_probs))}  range={min(sr_probs):.2f}%..{max(sr_probs):.2f}%  span={max(sr_probs)-min(sr_probs):.2f} pp")

    print("\nSECOND — date inputs must match numeric day offsets")
    sr_dates = second_date_path_matches_numeric()
    print(sr_dates.to_string(index=False))
    assert sr_dates["match"].all(), sr_dates

    base = SR_CASES["03_mid_astro_within_2y"]
    p0s = predict_second(base, REQUIRED_LABELS, DOCS, use_real_llm=False).probability
    cf2 = run_counterfactuals(
        base,
        second_recurrence_scenarios(base),
        lambda d: predict_second(d, REQUIRED_LABELS, DOCS, use_real_llm=False).probability,
        baseline_prob=p0s,
    )
    print(f"\nSECOND counterfactuals: {len(cf2)} scenarios, {int(cf2['flips_decision'].sum())} flip threshold")

    test_ten_cases_vary_and_ordered()
    test_feature_flip_directions()
    test_xai_effects_are_consistent()
    test_probability_matches_full_rule_logit()

    print("\nAll First (10) and Second (10) stress checks passed, including date→day conversion.")
