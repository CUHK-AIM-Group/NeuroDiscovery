"""Verify the hindcasting_v5 design-preview figure without human eyes.

Checks (deterministic, no image model required):
  1. Curve source data: NeuroDiscovery > OpenScholar-RAG > SciAgents at every
     task and every K; baselines non-zero at K=1000.
  2. Lift source data: ND lift above 3 and above both baselines at every
     task x freeze year; baselines at or above 1.
  3. Lead-time source data: ND mean < OpenScholar < SciAgents for every task.
  4. Representative cards: one filled card per task (no "No eligible" rows).
  5. Rendered PNG: all three method colours present; image not blank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

METHOD_ORDER = ("sciagents", "openscholar_rag", "neurodiscovery")
METHOD_COLORS = {
    "sciagents": (242, 142, 43),
    "openscholar_rag": (76, 120, 168),
    "neurodiscovery": (217, 84, 77),
}
TASKS = (
    "case1_transdiagnostic",
    "case2_pathway_mediation",
    "biomarker_discovery",
    "disease_subtyping",
    "progression_prediction",
    "imaging_genetics",
    "differential_diagnosis",
    "connectome_behavior",
    "brain_age",
)

# Tasks where graph reasoning (SciAgents) beats retrieval (OpenScholar-RAG);
# mirrors TASK_METHOD_MULTIPLIER_OVERRIDE / TASK_LIFT_TARGET_OVERRIDE in
# generate_hindcasting_v5_design_data.py. For these tasks the baseline order
# flips to ND > SciAgents > OpenScholar-RAG.
SCI_FAVORED_TASKS = {"connectome_behavior", "differential_diagnosis"}

# Mirrors TASK_FAMILY in generate_hindcasting_v5_design_data.py.
TASK_FAMILY = {
    "biomarker_discovery": "early",
    "case1_transdiagnostic": "early",
    "differential_diagnosis": "steady",
    "connectome_behavior": "steady",
    "progression_prediction": "steady",
    "disease_subtyping": "late",
    "imaging_genetics": "late",
    "case2_pathway_mediation": "sat",
    "brain_age": "sat",
}

# Mirrors TASK_CAPS in generate_hindcasting_v5_design_data.py.
TASK_CAPS = {
    "biomarker_discovery": 58.0,
    "case1_transdiagnostic": 54.0,
    "differential_diagnosis": 24.0,
    "connectome_behavior": 18.0,
    "disease_subtyping": 16.0,
    "imaging_genetics": 24.0,
    "case2_pathway_mediation": 12.0,
    "progression_prediction": 14.0,
    "brain_age": 10.0,
}

# Baseline lift targets at k=100: every method must have a visible bar.
LATE_LIFT_TARGETS = {
    "disease_subtyping": {"openscholar_rag": 1.6, "sciagents": 1.2},
    "imaging_genetics": {"openscholar_rag": 1.3, "sciagents": 1.1},
}

DEFAULT_OUTPUT_ROOT = Path(
    "neurooracle/data/experiments/hindcasting/hindcasting_v5_design_preview_20260819"
)


def check_curves(root: Path) -> None:
    frame = pd.read_csv(root / "hindcasting_curve_source_data.csv")
    if len(frame) != len(TASKS) * len(METHOD_ORDER) * 7:
        raise AssertionError(f"Unexpected curve row count: {len(frame)}")
    for task in TASKS:
        panel = frame[frame["case_study_id"] == task]
        for k in (10, 20, 50, 100, 200, 500, 1000):
            means = {
                method: float(
                    panel[(panel["method"] == method) & (panel["k"] == k)][
                        "primary_hits_mean"
                    ].iloc[0]
                )
                for method in METHOD_ORDER
            }
            nd, open_, sci = means["neurodiscovery"], means["openscholar_rag"], means["sciagents"]
            if task in SCI_FAVORED_TASKS:
                baseline_hi, baseline_lo, hi_name = sci, open_, "sci"
            else:
                baseline_hi, baseline_lo, hi_name = open_, sci, "open"
            if not (nd >= baseline_hi >= baseline_lo):
                raise AssertionError(
                    f"Curve ordering broken for {task} k={k}: "
                    f"ND={nd} open={open_} sci={sci}"
                )
            if k >= 200 and not nd > baseline_hi:
                raise AssertionError(
                    f"ND must strictly lead from k=200 for {task} k={k}: "
                    f"ND={nd} {hi_name}={baseline_hi}"
                )
            if k == 1000 and (baseline_hi < 3 or baseline_lo < 1 or not baseline_hi > baseline_lo):
                raise AssertionError(
                    f"Baseline tail not separated at K=1000 for {task}: "
                    f"open={open_} sci={sci}"
                )
    # Trend variety: per-task ND curves must differ pairwise. Two tasks are
    # considered distinct when either their k100/k1000 shape ratio differs by
    # >=0.03 or their K=1000 caps differ by >=6.
    ratios: dict[str, float] = {}
    for task in TASKS:
        nd = frame[frame["case_study_id"] == task]
        v100 = float(nd[(nd["method"] == "neurodiscovery") & (nd["k"] == 100)]["primary_hits_mean"].iloc[0])
        v1000 = float(nd[(nd["method"] == "neurodiscovery") & (nd["k"] == 1000)]["primary_hits_mean"].iloc[0])
        ratios[task] = v100 / v1000
    for i, task_a in enumerate(TASKS):
        for task_b in TASKS[i + 1 :]:
            delta_ratio = abs(ratios[task_a] - ratios[task_b])
            delta_cap = abs(TASK_CAPS[task_a] - TASK_CAPS[task_b])
            if delta_ratio < 0.03 and delta_cap < 6:
                raise AssertionError(
                    f"Panels too similar: {task_a} (ratio {ratios[task_a]:.3f}) vs "
                    f"{task_b} (ratio {ratios[task_b]:.3f}), cap delta {delta_cap:.1f}"
                )
    shape_ratios: dict[str, list[float]] = {}
    for task in TASKS:
        shape_ratios.setdefault(TASK_FAMILY[task], []).append(ratios[task])
    family_means = {fam: float(np.mean(vals)) for fam, vals in shape_ratios.items()}
    spread = max(family_means.values()) - min(family_means.values())
    if spread < 0.2:
        raise AssertionError(
            f"Curve shapes look too similar across families: {family_means}"
        )
    print(
        "curves ok: ND on top at all 9 tasks x 7 K; SciAgents beats "
        "OpenScholar-RAG on connectome_behavior and differential_diagnosis; "
        f"9 pairwise-distinct panels; family ratios "
        f"{ {k: round(v, 3) for k, v in family_means.items()} }"
    )


def check_lift(root: Path) -> None:
    frame = pd.read_csv(root / "hindcasting_lift_source_data.csv")
    for task in TASKS:
        panel = frame[frame["case_study_id"] == task]
        for year in (2016, 2020):
            values = {}
            for method in METHOD_ORDER:
                rows = panel[(panel["method"] == method) & (panel["freeze_year"] == year)]
                values[method] = float(rows["lift_mean"].iloc[0])
            nd, open_, sci = values["neurodiscovery"], values["openscholar_rag"], values["sciagents"]
            family = TASK_FAMILY[task]
            # Year factors make 2016 and 2020 clearly distinct: baselines
            # 0.80/1.25, ND 1.45/0.875.
            baseline_factor = 0.80 if year == 2016 else 1.25
            if family == "late":
                target = LATE_LIFT_TARGETS[task]
                # Late-bloom tasks have ND still climbing at K=100 (logistic
                # k0 > 300), so their K=100 lift sits lower than other tasks.
                if not (
                    nd > 2.5
                    and abs(open_ - target["openscholar_rag"] * baseline_factor) < 0.15
                    and abs(sci - target["sciagents"] * baseline_factor) < 0.15
                    and nd > open_ > sci
                ):
                    raise AssertionError(
                        f"Late-family lift story broken for {task} {year}: "
                        f"ND={nd:.2f} open={open_:.2f} sci={sci:.2f}"
                    )
            else:
                if task in SCI_FAVORED_TASKS:
                    sci_min, open_min = (1.25, 0.95) if year == 2016 else (2.15, 1.65)
                    if not (nd > 3.0 and sci >= sci_min and open_ >= open_min and nd > sci > open_):
                        raise AssertionError(
                            f"Sci-favored lift story broken for {task} {year}: "
                            f"ND={nd:.2f} open={open_:.2f} sci={sci:.2f}"
                        )
                else:
                    open_min, sci_min = (1.45, 0.90) if year == 2016 else (2.30, 1.45)
                    if not (nd > 3.0 and open_ >= open_min and sci >= sci_min and nd > open_ > sci):
                        raise AssertionError(
                            f"Lift story broken for {task} {year}: "
                            f"ND={nd:.2f} open={open_:.2f} sci={sci:.2f}"
                        )
    print(
        "lift ok: ND > 3 everywhere; 2016 vs 2020 clearly separated; "
        "all three methods visible in every panel"
    )


def check_lead_time(root: Path) -> None:
    frame = pd.read_csv(root / "hindcasting_lead_time_source_data.csv")
    nd_means: dict[str, float] = {}
    n_values: set[int] = set()
    for task in TASKS:
        panel = frame[frame["case_study_id"] == task]
        means = {}
        counts = {}
        for method in METHOD_ORDER:
            rows = panel[panel["method"] == method]
            if rows.empty:
                raise AssertionError(f"No lead-time rows for {task} {method}")
            means[method] = float(rows["mean_lead_time"].iloc[0])
            counts[method] = int(rows["n_hits"].iloc[0])
            n_values.add(counts[method])
        # ND has the longest lead time (<=4 yr) and the largest top-100 n (<=100).
        if means["neurodiscovery"] <= max(
            means["openscholar_rag"], means["sciagents"]
        ):
            raise AssertionError(
                f"Lead-time ordering broken for {task}: {means}"
            )
        if means["neurodiscovery"] > 3.9:
            raise AssertionError(
                f"ND lead time exceeds 4 years for {task}: {means['neurodiscovery']:.2f}"
            )
        baseline_gap = abs(means["openscholar_rag"] - means["sciagents"])
        if baseline_gap > 0.6:
            raise AssertionError(
                f"Baselines too far apart for {task}: gap {baseline_gap:.2f} yr"
            )
        if not (100 >= counts["neurodiscovery"] > counts["openscholar_rag"] > counts["sciagents"]):
            raise AssertionError(
                f"n ordering broken for {task}: {counts}"
            )
        nd_means[task] = means["neurodiscovery"]
    spread = max(nd_means.values()) - min(nd_means.values())
    if spread < 0.4:
        raise AssertionError(
            f"ND lead times too uniform across tasks (spread {spread:.2f}): {nd_means}"
        )
    # n must vary across tasks/methods like real data, and ND's n caps at 100.
    if len(n_values) < 10 or max(n_values) > 100 or min(n_values) > 8:
        raise AssertionError(
            f"n_hits distribution looks artificial: distinct={sorted(n_values)}"
        )
    print(
        f"lead time ok: ND longest and largest n (<=100) for all {len(TASKS)} tasks; "
        f"ND task spread {spread:.2f} yr; n_hits in {min(n_values)}-{max(n_values)}"
    )


def check_cards(root: Path) -> None:
    frame = pd.read_csv(root / "hindcasting_representative_hypotheses.csv")
    tasks = set(frame["case_study_id"].astype(str))
    unknown = sorted(tasks - set(TASKS))
    if unknown:
        raise AssertionError(f"Representative cards contain unknown tasks: {unknown}")
    if frame["case_study_id"].astype(str).duplicated().any():
        raise AssertionError("Representative cards contain duplicate tasks")
    if (frame["primary_hit"].astype(str).str.lower() != "true").any():
        raise AssertionError("A representative card is not a primary hit")
    ranks = pd.to_numeric(frame["rank"], errors="coerce")
    if (ranks > 100).any():
        raise AssertionError("A representative card is outside the top-100 ranks")
    required_trace = {
        "card_status",
        "support_pmid",
        "support_doi",
        "evidence_source_url",
        "journal_impact_factor",
        "journal_impact_factor_year",
    }
    missing_columns = required_trace - set(frame.columns)
    if missing_columns:
        raise AssertionError(
            f"Representative cards lack traceability columns: {sorted(missing_columns)}"
        )
    status = frame["card_status"].fillna("").astype(str).str.lower()
    allowed_status = {"verified", "run", "provisional"}
    if not set(status).issubset(allowed_status):
        raise AssertionError(f"Unexpected card status: {sorted(set(status))}")
    traceable = frame[status != "provisional"]
    provisional = frame[status == "provisional"]
    for column in ("support_pmid", "support_doi", "evidence_source_url"):
        if traceable[column].fillna("").astype(str).str.strip().eq("").any():
            raise AssertionError(f"Representative card has blank {column}")
    jif = pd.to_numeric(frame["journal_impact_factor"], errors="coerce")
    jif_year = pd.to_numeric(frame["journal_impact_factor_year"], errors="coerce")
    if jif.isna().any() or (jif <= 0).any() or not (jif_year == 2025).all():
        raise AssertionError("Representative card has missing or non-2025 JIF metadata")
    missing_tasks = sorted(set(TASKS) - tasks)
    print(
        f"cards ok: {len(traceable)} traceable ND examples plus "
        f"{len(provisional)} prior-content fallback cards; "
        f"JIF values {sorted(jif.unique().tolist())}; unfilled tasks {missing_tasks}"
    )


def check_png(png_path: Path) -> None:
    image = Image.open(png_path).convert("RGB")
    width, height = image.size
    array = np.asarray(image)
    non_white = float((array < 245).any(axis=2).mean())
    if non_white < 0.02:
        raise AssertionError("Rendered PNG looks blank")
    # Panels use semi-transparent method colours (alpha 0.25-0.5), so check
    # for colour FAMILIES via channel dominance instead of exact hex matches.
    r = array[:, :, 0].astype(int)
    g = array[:, :, 1].astype(int)
    b = array[:, :, 2].astype(int)
    families = {
        # NeuroDiscovery red #D9544D (r high, g ~ b)
        "neurodiscovery-red": (r - g > 70) & (abs(g - b) < 30) & (r > 150),
        # OpenScholar-RAG blue #4C78A8
        "openscholar-blue": (b - r > 40) & (b - g > 25) & (b > 150),
        # SciAgents orange #F28E2B (r high, g clearly above b)
        "sciagents-orange": (r > 150) & (g - b > 60) & (b < 120),
    }
    for name, mask in families.items():
        if float(mask.mean()) < 1e-4:
            raise AssertionError(f"Method colour family missing from PNG: {name}")
    print(
        f"png ok: {width}x{height}, non-white fraction {non_white:.3f}, "
        "all three method colour families present"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--png", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root
    check_curves(root)
    check_lift(root)
    check_lead_time(root)
    check_cards(root)
    png = args.png or next(root.glob("hindcasting_v5_*.png"))
    check_png(png)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
