"""Plot the manuscript-style multi-task hindcasting figure.

Layout, fonts, colours and export follow materials/figure_style_spec.md:
15.8 in fixed canvas width, 12.24 in body width, Times New Roman 14/12 pt,
NeuroDiscovery red #D9544D, bar alpha 0.7 with white edges, #E7E9EB grid,
panel ids at the top-left of each whole panel block. Panels a-i are the
per-task discovery curves and 2016-vs-2020 lift bars; panel k is the
full-width representative-hypothesis card grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEURODISCOVERY_ROOT = (
    REPO_ROOT
    / "neurooracle/data/experiments/hindcasting/"
    "neurodiscovery_20260811_all_seed0_9_eval_semantic_v2"
)
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT
    / "neurooracle/data/experiments/hindcasting/"
    "frozen_baselines_20260811_all_seed0_9_eval_semantic_v2_testyears"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "neurooracle/data/experiments/hindcasting/"
    "semantic_v2_selected_case_studies_original_layout_20260812"
)
DEFAULT_JOURNAL_METRICS_PATH = (
    REPO_ROOT / "materials/hindcasting_journal_impact_factors_2025.csv"
)
DEFAULT_VERIFIED_REPRESENTATIVE_EXAMPLES_PATH = (
    REPO_ROOT
    / "materials/hindcasting_verified_representative_examples_semantic_v2_20260812.csv"
)
DEFAULT_PROVISIONAL_REPRESENTATIVE_EXAMPLES_PATH = (
    REPO_ROOT / "materials/hindcasting_provisional_representative_examples_20260827.csv"
)

K_VALUES = (10, 20, 50, 100, 200, 500, 1000)
METHOD_ORDER = ("sciagents", "openscholar_rag", "neurodiscovery")
METHOD_LABELS = {
    "sciagents": "SciAgents",
    "openscholar_rag": "OpenScholar-RAG",
    "neurodiscovery": "NeuroDiscovery",
}
# Compact labels for the in-panel legend (the full labels above are too wide
# for the small curve panels and overlap the discovery curves).
LEGEND_LABELS = {
    "sciagents": "SciAgents",
    "openscholar_rag": "OpenScholar",
    "neurodiscovery": "NeuroDiscovery",
}
METHOD_COLORS = {
    "sciagents": "#F28E2B",
    "openscholar_rag": "#4C78A8",
    "neurodiscovery": "#D9544D",
}
METHOD_MARKERS = {
    "sciagents": "^",
    "openscholar_rag": "s",
    "neurodiscovery": "o",
}

# Short journal display names for the panel k cards (avoids text truncation).
JOURNAL_SHORT = {
    "brain and behavior": "Brain Behav",
    "human brain mapping": "Hum Brain Mapp",
    "journal of affective disorders": "J Affect Disord",
    "nature communications": "Nat Commun",
    "molecular psychiatry": "Mol Psychiatry",
    "biological psychiatry": "Biol Psychiatry",
    "schizophrenia (heidelberg, germany)": "Schizophrenia",
    "the american journal of psychiatry": "Am J Psychiatry",
    "proceedings of the national academy of sciences of the united states of america": "PNAS",
    "neuropsychopharmacology : official publication of the american college of neuropsychopharmacology": "Neuropsychopharmacology",
    "european archives of psychiatry and clinical neuroscience": "Eur Arch Psychiatry",
}

SELECTED_CASE_STUDIES = (
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
CASE_STUDY_LABELS = {
    "case1_transdiagnostic": "Topic 1: Transdiagnostic atlas",
    "case2_pathway_mediation": "Topic 2: Pathway mediation",
    "biomarker_discovery": "Topic 3: Biomarker discovery",
    "disease_subtyping": "Topic 4: Disease subtyping",
    "progression_prediction": "Topic 5: Progression prediction",
    "imaging_genetics": "Topic 6: Imaging genetics",
    "differential_diagnosis": "Topic 7: Differential diagnosis",
    "connectome_behavior": "Topic 8: Connectome-behaviour",
    "brain_age": "Topic 9: Brain age",
}

COL_GRID = "#E7E9EB"
COL_TEXT = "#000000"
COL_STRICT = "#000000"
FONT_BODY = 12.0
FONT_TITLE = 14.0
FONT_PANEL = 14.0


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": FONT_BODY,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_TITLE,
            "xtick.labelsize": FONT_BODY,
            "ytick.labelsize": FONT_BODY,
            "legend.fontsize": FONT_BODY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_journal_impact_factors(
    path: Path = DEFAULT_JOURNAL_METRICS_PATH,
) -> pd.DataFrame:
    """Load publisher/JCR-verified Journal Impact Factors with provenance."""
    if not path.is_file():
        raise FileNotFoundError(f"Journal-metrics source file not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "journal_key",
        "journal_display_name",
        "impact_factor",
        "metric_year",
        "source_url",
        "verified_on",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Journal-metrics source is missing columns: {sorted(missing)}"
        )
    frame = frame.copy()
    frame["journal_key"] = frame["journal_key"].astype(str).str.strip().str.lower()
    frame["impact_factor"] = pd.to_numeric(frame["impact_factor"], errors="raise")
    frame["metric_year"] = pd.to_numeric(
        frame["metric_year"], errors="raise"
    ).astype(int)
    if frame["journal_key"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["journal_key"].duplicated(), "journal_key"].unique()
        )
        raise ValueError(f"Duplicate journal-metrics rows: {duplicates}")
    if (frame["impact_factor"] <= 0).any():
        raise ValueError("Journal Impact Factors must be positive")
    return frame


def _seed_from_path(path: Path) -> int:
    seed_part = next(part for part in path.parts if part.startswith("seed_"))
    return int(seed_part.split("_", 1)[1])


def _freeze_from_path(path: Path) -> int:
    match = re.search(r"kg(\d{4})_to_", str(path))
    if not match:
        raise ValueError(f"Cannot parse freeze year from {path}")
    return int(match.group(1))


def collect_metrics(roots: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for root in roots:
        pattern = "*/seed_*/*/kg*_to_*_*/hindcasting/metrics.json"
        for path in sorted(root.glob(pattern)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for raw_k, result in (payload.get("topk") or {}).items():
                observed = result.get("observed") or {}
                random_row = result.get("random_same_hypothesis_pool") or {}
                records.append(
                    {
                        "method": str(payload.get("method") or "unknown"),
                        "seed": _seed_from_path(path),
                        "case_study_id": str(payload.get("case_study_id") or ""),
                        "freeze_year": int(payload.get("freeze_year")),
                        "future_start_year": int(payload.get("future_start_year")),
                        "future_end_year": int(payload.get("future_end_year")),
                        "k": int(raw_k),
                        "n_hypotheses": int(payload.get("n_hypotheses") or 0),
                        "primary_hits": int(observed.get("primary_hits") or 0),
                        "endpoint_hits": int(observed.get("endpoint_hits") or 0),
                        "any_future_hits": int(observed.get("any_future_hits") or 0),
                        "random_applicable": bool(random_row.get("applicable")),
                        "random_mean_primary_hits": random_row.get("mean_primary_hits"),
                        "metrics_path": str(path),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("No semantic-v2 metrics.json files were found.")
    return frame


def select_balanced_test_set(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[int]]:
    frame = frame[
        frame["case_study_id"].isin(SELECTED_CASE_STUDIES)
        & frame["method"].isin(METHOD_ORDER)
        & frame["k"].isin(K_VALUES)
    ].copy()
    common_windows: set[int] | None = None
    common_seeds: set[int] | None = None
    for method in METHOD_ORDER:
        method_rows = frame[frame["method"] == method]
        windows = set(int(value) for value in method_rows["freeze_year"].unique())
        seeds = set(int(value) for value in method_rows["seed"].unique())
        common_windows = windows if common_windows is None else common_windows & windows
        common_seeds = seeds if common_seeds is None else common_seeds & seeds

    freeze_years = sorted(common_windows or [])
    seeds = sorted(common_seeds or [])
    if not freeze_years or not seeds:
        raise RuntimeError("The three methods do not share a balanced temporal test set.")

    balanced = frame[
        frame["freeze_year"].isin(freeze_years) & frame["seed"].isin(seeds)
    ].copy()
    key = ["case_study_id", "method", "freeze_year", "seed", "k"]
    expected = (
        len(SELECTED_CASE_STUDIES)
        * len(METHOD_ORDER)
        * len(freeze_years)
        * len(seeds)
        * len(K_VALUES)
    )
    if len(balanced) != expected or balanced.duplicated(key).any():
        raise RuntimeError(
            f"Incomplete balanced test set: expected {expected}, found {len(balanced)}."
        )
    return balanced.sort_values(key).reset_index(drop=True), freeze_years, seeds


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    draws: int = 5000,
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return mean, float(low), float(high)


def summarize_curves(balanced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        balanced.groupby(["case_study_id", "method", "k", "seed"], as_index=False)
        .agg(primary_hits=("primary_hits", "sum"))
    )
    rng = np.random.default_rng(20260812)
    records: list[dict[str, Any]] = []
    for keys, part in per_seed.groupby(["case_study_id", "method", "k"], sort=False):
        case_study_id, method, k = keys
        mean, low, high = _bootstrap_mean_ci(part["primary_hits"].to_numpy(float), rng)
        records.append(
            {
                "case_study_id": case_study_id,
                "method": method,
                "k": int(k),
                "n_seeds": int(part["seed"].nunique()),
                "primary_hits_mean": mean,
                "primary_hits_ci95_low": low,
                "primary_hits_ci95_high": high,
            }
        )
    return per_seed, pd.DataFrame.from_records(records)


def summarize_freeze_lifts(balanced: pd.DataFrame) -> pd.DataFrame:
    rows = balanced[balanced["k"] == 100].copy()
    random_mean = pd.to_numeric(rows["random_mean_primary_hits"], errors="coerce")
    rows["lift_over_random"] = np.where(
        random_mean > 0,
        rows["primary_hits"] / random_mean,
        np.where(rows["primary_hits"] == 0, 0.0, np.nan),
    )
    rng = np.random.default_rng(20260813)
    records: list[dict[str, Any]] = []
    for keys, part in rows.groupby(
        ["case_study_id", "method", "freeze_year"], sort=False
    ):
        case_study_id, method, freeze_year = keys
        mean, low, high = _bootstrap_mean_ci(part["lift_over_random"].to_numpy(float), rng)
        records.append(
            {
                "case_study_id": case_study_id,
                "method": method,
                "freeze_year": int(freeze_year),
                "n_seeds": int(part["seed"].nunique()),
                "lift_mean": mean,
                "lift_ci95_low": low,
                "lift_ci95_high": high,
            }
        )
    return pd.DataFrame.from_records(records)


def collect_recovered_examples(
    roots: list[Path],
    freeze_years: list[int],
    seeds: list[int],
    journal_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "rank",
        "method",
        "id",
        "source_name",
        "target_name",
        "composite_score",
        "primary_hit",
        "primary_lead_time",
        "support_title",
        "support_journal",
        "support_year",
    ]
    frames: list[pd.DataFrame] = []
    for root in roots:
        pattern = "*/seed_*/*/kg*_to_*_*/hindcasting/recovered_examples.csv"
        for path in sorted(root.glob(pattern)):
            task = path.parents[2].name
            freeze_year = _freeze_from_path(path)
            seed = _seed_from_path(path)
            if (
                task not in SELECTED_CASE_STUDIES
                or freeze_year not in freeze_years
                or seed not in seeds
            ):
                continue
            frame = pd.read_csv(path, usecols=columns, low_memory=False)
            frame["case_study_id"] = task
            frame["freeze_year"] = freeze_year
            frame["seed"] = seed
            frames.append(frame)
    metric_columns = [
        "journal_display_name",
        "journal_impact_factor",
        "journal_impact_factor_year",
        "journal_impact_factor_source_url",
        "journal_impact_factor_verified_on",
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                *columns,
                "case_study_id",
                "freeze_year",
                "seed",
                "journal_key",
                *metric_columns,
            ]
        )
    examples = pd.concat(frames, ignore_index=True)
    examples["rank"] = pd.to_numeric(examples["rank"], errors="coerce")
    examples["primary_lead_time"] = pd.to_numeric(
        examples["primary_lead_time"], errors="coerce"
    )
    examples["composite_score"] = pd.to_numeric(
        examples["composite_score"], errors="coerce"
    )
    examples["primary_hit"] = examples["primary_hit"].astype(str).str.lower().eq("true")
    examples["journal_key"] = (
        examples["support_journal"].fillna("").astype(str).str.strip().str.lower()
    )
    if journal_metrics is None:
        journal_metrics = load_journal_impact_factors()
    metric_lookup = journal_metrics.rename(
        columns={
            "impact_factor": "journal_impact_factor",
            "metric_year": "journal_impact_factor_year",
            "source_url": "journal_impact_factor_source_url",
            "verified_on": "journal_impact_factor_verified_on",
        }
    )
    return examples.merge(
        metric_lookup[
            [
                "journal_key",
                "journal_display_name",
                "journal_impact_factor",
                "journal_impact_factor_year",
                "journal_impact_factor_source_url",
                "journal_impact_factor_verified_on",
            ]
        ],
        on="journal_key",
        how="left",
        validate="many_to_one",
    )


def summarize_lead_times(examples: pd.DataFrame) -> pd.DataFrame:
    rows = examples[
        examples["method"].isin(METHOD_ORDER)
        & (examples["rank"] <= 100)
        & examples["primary_hit"]
        & examples["primary_lead_time"].notna()
    ].drop_duplicates(["method", "case_study_id", "freeze_year", "seed", "id"])
    rng = np.random.default_rng(20260814)
    records: list[dict[str, Any]] = []
    for task in SELECTED_CASE_STUDIES:
        for method in METHOD_ORDER:
            values = rows.loc[
                (rows["case_study_id"] == task) & (rows["method"] == method),
                "primary_lead_time",
            ].to_numpy(float)
            if not len(values):
                continue
            mean, low, high = _bootstrap_mean_ci(values, rng)
            records.append(
                {
                    "case_study_id": task,
                    "method": method,
                    "n_hits": int(len(values)),
                    "mean_lead_time": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame.from_records(records)


def select_representative_examples(examples: pd.DataFrame) -> pd.DataFrame:
    # Representative card per task: the earliest-ranked strict
    # NeuroDiscovery hit in the top 100. JIF is deliberately excluded from
    # selection and is attached only as journal-level metadata afterward.
    rows = examples[
        (examples["method"] == "neurodiscovery")
        & (examples["rank"] <= 100)
        & examples["primary_hit"]
    ].copy()
    rows = rows.sort_values(
        [
            "case_study_id",
            "rank",
            "composite_score",
        ],
        ascending=[True, True, False],
    )
    selected = rows.drop_duplicates("case_study_id", keep="first")
    order = {task: index for index, task in enumerate(SELECTED_CASE_STUDIES)}
    selected = selected.assign(
        task_order=selected["case_study_id"].map(order)
    ).sort_values("task_order")
    return selected.reset_index(drop=True)


def load_verified_representative_examples(
    path: Path,
    journal_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Load traceable cards from a completed run export and attach JIF.

    The snapshot is intentionally allowed to cover fewer than all nine topics:
    an absent row is rendered as no verified strict example instead of being
    filled from design-preview data.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Verified representative-example source not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "case_study_id",
        "rank",
        "method",
        "id",
        "source_name",
        "target_name",
        "composite_score",
        "primary_hit",
        "primary_lead_time",
        "support_title",
        "support_journal",
        "support_year",
        "support_pmid",
        "support_doi",
        "evidence_source_url",
        "verified_on",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Verified representative-example source is missing columns: {sorted(missing)}"
        )
    frame = frame.copy()
    frame["case_study_id"] = frame["case_study_id"].astype(str)
    unknown_tasks = sorted(set(frame["case_study_id"]) - set(SELECTED_CASE_STUDIES))
    if unknown_tasks:
        raise ValueError(f"Verified card source contains unknown topics: {unknown_tasks}")
    if frame["case_study_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["case_study_id"].duplicated(), "case_study_id"].unique()
        )
        raise ValueError(f"Verified card source contains duplicate topics: {duplicates}")
    frame["primary_hit"] = frame["primary_hit"].astype(str).str.lower().eq("true")
    if not frame["primary_hit"].all():
        raise ValueError("Verified card source contains a non-primary-hit row")
    if not frame["method"].astype(str).str.lower().eq("neurodiscovery").all():
        raise ValueError("Verified card source must contain only NeuroDiscovery rows")
    for column in ("rank", "primary_lead_time", "support_year", "composite_score"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["journal_key"] = (
        frame["support_journal"].fillna("").astype(str).str.strip().str.lower()
    )
    metric_lookup = journal_metrics.rename(
        columns={
            "impact_factor": "journal_impact_factor",
            "metric_year": "journal_impact_factor_year",
            "source_url": "journal_impact_factor_source_url",
            "verified_on": "journal_impact_factor_verified_on",
        }
    )
    frame = frame.merge(
        metric_lookup[
            [
                "journal_key",
                "journal_display_name",
                "journal_impact_factor",
                "journal_impact_factor_year",
                "journal_impact_factor_source_url",
                "journal_impact_factor_verified_on",
            ]
        ],
        on="journal_key",
        how="left",
        validate="many_to_one",
    )
    missing_jif = frame.loc[
        frame["journal_impact_factor"].isna(), "support_journal"
    ].astype(str)
    if not missing_jif.empty:
        raise ValueError(
            "No verified JIF mapping for card journals: "
            f"{sorted(missing_jif.unique().tolist())}"
        )
    frame["card_status"] = "verified"
    order = {task: index for index, task in enumerate(SELECTED_CASE_STUDIES)}
    return (
        frame.assign(task_order=frame["case_study_id"].map(order))
        .sort_values("task_order")
        .reset_index(drop=True)
    )


def load_provisional_representative_examples(
    path: Path,
    journal_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Load prior layout-card content used only as an explicit placeholder."""
    if not path.is_file():
        raise FileNotFoundError(f"Provisional representative-example source not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "case_study_id",
        "rank",
        "method",
        "id",
        "source_name",
        "target_name",
        "composite_score",
        "primary_hit",
        "primary_lead_time",
        "support_title",
        "support_journal",
        "support_year",
        "card_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Provisional representative-example source is missing columns: {sorted(missing)}"
        )
    frame = frame.copy()
    frame["case_study_id"] = frame["case_study_id"].astype(str)
    unknown_tasks = sorted(set(frame["case_study_id"]) - set(SELECTED_CASE_STUDIES))
    if unknown_tasks:
        raise ValueError(f"Provisional card source contains unknown topics: {unknown_tasks}")
    if frame["case_study_id"].duplicated().any():
        raise ValueError("Provisional card source contains duplicate topics")
    if not frame["card_status"].astype(str).str.lower().eq("provisional").all():
        raise ValueError("Every fallback card must be marked provisional")
    frame["primary_hit"] = frame["primary_hit"].astype(str).str.lower().eq("true")
    if not frame["primary_hit"].all():
        raise ValueError("Provisional card source contains a non-primary-hit row")
    for column in ("rank", "primary_lead_time", "support_year", "composite_score"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["journal_key"] = (
        frame["support_journal"].fillna("").astype(str).str.strip().str.lower()
    )
    metric_lookup = journal_metrics.rename(
        columns={
            "impact_factor": "journal_impact_factor",
            "metric_year": "journal_impact_factor_year",
            "source_url": "journal_impact_factor_source_url",
            "verified_on": "journal_impact_factor_verified_on",
        }
    )
    frame = frame.merge(
        metric_lookup[
            [
                "journal_key",
                "journal_display_name",
                "journal_impact_factor",
                "journal_impact_factor_year",
                "journal_impact_factor_source_url",
                "journal_impact_factor_verified_on",
            ]
        ],
        on="journal_key",
        how="left",
        validate="many_to_one",
    )
    missing_jif = frame.loc[
        frame["journal_impact_factor"].isna(), "support_journal"
    ].astype(str)
    if not missing_jif.empty:
        raise ValueError(
            "No JIF mapping for provisional card journals: "
            f"{sorted(missing_jif.unique().tolist())}"
        )
    order = {task: index for index, task in enumerate(SELECTED_CASE_STUDIES)}
    return (
        frame.assign(task_order=frame["case_study_id"].map(order))
        .sort_values("task_order")
        .reset_index(drop=True)
    )


def draw_task_hit_curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    band: pd.DataFrame,
    task: str,
    *,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(K_VALUES))
    panel = summary[summary["case_study_id"] == task]
    panel_band = band[band["case_study_id"] == task]
    maximum = 0.0
    for method in METHOD_ORDER:
        rows = panel[panel["method"] == method].set_index("k").reindex(K_VALUES)
        rows_band = panel_band[panel_band["method"] == method].set_index("k").reindex(K_VALUES)
        mean = rows["primary_hits_mean"].to_numpy(float)
        low = rows_band["band_low"].to_numpy(float)
        high = rows_band["band_high"].to_numpy(float)
        maximum = max(maximum, float(np.nanmax(high)))
        # figureS8 style: same-hue shaded band spanning the per-seed range.
        ax.fill_between(
            x,
            low,
            high,
            color=METHOD_COLORS[method],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            lw=2.0 if method == "neurodiscovery" else 1.6,
            marker=METHOD_MARKERS[method],
            markersize=3.6,
            markeredgewidth=0,
            alpha=1.0,
            label=LEGEND_LABELS[method],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["10", "", "50", "", "200", "", "1,000"])
    ax.set_xlabel("Hypotheses evaluated")
    if show_ylabel:
        ax.set_ylabel("Future-supported hits")
    else:
        # Non-leftmost panels share the row scale: hide y tick labels so the
        # panel id has clean space between neighbouring panels.
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="y", color=COL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if maximum <= 0:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.text(
            0.5,
            0.54,
            "No strict future-supported hit",
            ha="center",
            va="center",
            color="#666666",
            fontsize=FONT_BODY,
            transform=ax.transAxes,
        )
    else:
        ax.set_ylim(0, maximum * 1.12)
def draw_task_freeze_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    task: str,
    freeze_years: list[int],
    lift_points: pd.DataFrame,
    *,
    show_ylabel: bool,
) -> None:
    panel = summary[summary["case_study_id"] == task]
    x = np.arange(len(freeze_years))
    width = 0.23
    maximum = 0.0
    rng = np.random.default_rng(20260819)
    for method_index, method in enumerate(METHOD_ORDER):
        rows = panel[panel["method"] == method].set_index("freeze_year").reindex(freeze_years)
        mean = rows["lift_mean"].fillna(0).to_numpy(float)
        low = rows["lift_ci95_low"].fillna(0).to_numpy(float)
        high = rows["lift_ci95_high"].fillna(0).to_numpy(float)
        maximum = max(maximum, float(np.nanmax(high)))
        ax.bar(
            x + (method_index - 1) * width,
            mean,
            width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.7,
            yerr=np.vstack([np.maximum(mean - low, 0), np.maximum(high - mean, 0)]),
            error_kw={"elinewidth": 0.75, "capsize": 2.0, "capthick": 0.75, "ecolor": "#333333"},
        )
        # figure5 style: jittered per-seed points over each bar group.
        point_rows = lift_points[
            (lift_points["case_study_id"] == task) & (lift_points["method"] == method)
        ]
        for year_index, freeze_year in enumerate(freeze_years):
            year_rows = point_rows[point_rows["freeze_year"] == freeze_year]
            if year_rows.empty:
                continue
            center = x[year_index] + (method_index - 1) * width
            jitter = rng.uniform(-0.09, 0.09, size=len(year_rows))
            ax.plot(
                center + jitter,
                year_rows["lift"].to_numpy(float),
                "o",
                color=METHOD_COLORS[method],
                markersize=1.5,
                alpha=0.6,
                zorder=4,
            )
    ax.axhline(1.0, color="#808080", lw=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(freeze_years)
    ax.set_xlabel("Freeze year (next 5 years)")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="y", color=COL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(1.15, maximum * 1.12))


def draw_lead_time_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    y_base = np.arange(len(SELECTED_CASE_STUDIES))[::-1]
    offsets = {"sciagents": -0.20, "openscholar_rag": 0.0, "neurodiscovery": 0.20}
    for method in METHOD_ORDER:
        method_rows = summary[summary["method"] == method].set_index("case_study_id")
        for task, y in zip(SELECTED_CASE_STUDIES, y_base, strict=True):
            if task not in method_rows.index:
                continue
            row = method_rows.loc[task]
            mean = float(row["mean_lead_time"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            y_pos = y + offsets[method]
            ax.errorbar(
                mean,
                y_pos,
                xerr=np.array([[mean - low], [high - mean]]),
                fmt=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                ecolor=METHOD_COLORS[method],
                elinewidth=1.3,
                capsize=2.6,
                markersize=6.5,
                label=METHOD_LABELS[method] if task == SELECTED_CASE_STUDIES[0] else None,
                zorder=3,
            )
    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [f"Case {index}" for index in range(1, len(SELECTED_CASE_STUDIES) + 1)]
    )
    ax.set_xticks([2, 3, 4])
    ax.set_xlim(1.5, 4.6)
    ax.set_xlabel("Mean lead time (years; 95% bootstrap CI)")
    ax.set_title("Lead time of top-100 future-supported hypotheses", fontweight="bold")
    ax.grid(axis="x", color=COL_GRID, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=FONT_BODY, handlelength=1.2, markerscale=1.0)


def draw_violin_summary(ax: plt.Axes, examples: pd.DataFrame) -> None:
    """Panel j: per-hypothesis lead-time distributions by method.

    Mirrors the R alternative selected for the final layout: violins filled
    with the method colours (alpha 0.25), a black mean +/- SD error bar, and
    jittered individual hypotheses (black, alpha 0.6).
    """
    rows = examples[
        examples["method"].isin(METHOD_ORDER)
        & (examples["rank"] <= 100)
        & examples["primary_hit"]
        & examples["primary_lead_time"].notna()
    ].drop_duplicates(["method", "case_study_id", "freeze_year", "seed", "id"])
    rng = np.random.default_rng(20260819)
    for x, method in enumerate(METHOD_ORDER):
        values = rows.loc[rows["method"] == method, "primary_lead_time"].to_numpy(float)
        if not len(values):
            continue
        parts = ax.violinplot(
            [values],
            positions=[x],
            widths=0.85,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(METHOD_COLORS[method])
            body.set_alpha(0.45)
        mean = float(np.mean(values))
        sd = float(np.std(values))
        ax.errorbar(
            [x],
            [mean],
            yerr=np.array([[sd], [sd]]),
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=2.2,
            capthick=0.6,
            zorder=4,
        )
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.plot(
            x + jitter,
            values,
            "o",
            color="black",
            markersize=1.5,
            alpha=0.6,
            zorder=3,
        )
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.set_ylim(0.5, 4.6)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylabel("Lead time (years)")
    # Title placed above the axes at the same height as panel k's title so
    # the two bottom panels are vertically aligned.
    ax.text(
        0.5,
        1.03,
        "Lead time: proposal to confirmation",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONT_TITLE,
        fontweight="bold",
        color=COL_TEXT,
    )
    ax.grid(axis="y", color=COL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_discovery_count_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Panel k: top-100 discovery counts per task as horizontal grouped bars.

    Encodes n graphically (bar length) so the forest plot next to it needs no
    per-point ``n=...`` annotations.
    """
    y_base = np.arange(len(SELECTED_CASE_STUDIES))[::-1]
    width = 0.24
    rows = summary.set_index(["case_study_id", "method"])
    for method_index, method in enumerate(METHOD_ORDER):
        values = [
            int(rows.loc[(task, method), "n_hits"])
            if (task, method) in rows.index
            else 0
            for task in SELECTED_CASE_STUDIES
        ]
        ax.barh(
            y_base + (method_index - 1) * width,
            values,
            width,
            color=METHOD_COLORS[method],
            edgecolor="#333333",
            linewidth=0.4,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [f"Case {index}" for index in range(1, len(SELECTED_CASE_STUDIES) + 1)]
    )
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Top-100 discoveries (n)")
    ax.set_title("Top-100 future-supported discoveries", fontweight="bold")
    ax.grid(axis="x", color=COL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=FONT_BODY, handlelength=1.2)


def draw_examples_table(ax: plt.Axes, selected: pd.DataFrame) -> None:
    ax.axis("off")
    ax.text(
        0.50,
        1.03,
        "Representative NeuroDiscovery future-supported hypotheses",
        fontsize=FONT_TITLE,
        fontweight="bold",
        color=COL_TEXT,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )
    x_positions = (0.00, 0.34, 0.68)
    y_positions = (0.685, 0.3775, 0.070)
    card_width = 0.315
    card_height = 0.292
    selected_by_task = {
        str(row["case_study_id"]): row for _, row in selected.iterrows()
    }
    for index, task_id in enumerate(SELECTED_CASE_STUDIES):
        x0 = x_positions[index % 3]
        y0 = y_positions[index // 3]
        task = CASE_STUDY_LABELS[task_id]
        row = selected_by_task.get(task_id)
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x0, y0),
                card_width,
                card_height,
                boxstyle="round,pad=0.006,rounding_size=0.008",
                transform=ax.transAxes,
                facecolor="#FAFAFA",
                edgecolor="#D5D5D5",
                linewidth=0.7,
            )
        )
        ax.text(
            x0 + 0.014,
            y0 + card_height - 0.022,
            task,
            fontsize=FONT_BODY,
            fontweight="bold",
            color=METHOD_COLORS["neurodiscovery"],
            transform=ax.transAxes,
            va="top",
        )
        if row is None:
            ax.text(
                x0 + 0.014,
                y0 + card_height - 0.080,
                "No verified strict example",
                fontsize=FONT_BODY,
                fontweight="bold",
                color="#666666",
                transform=ax.transAxes,
                va="top",
            )
            ax.text(
                x0 + 0.014,
                y0 + card_height - 0.145,
                "No future-supported card in the\narchived semantic-v2 export",
                fontsize=FONT_BODY,
                color="#666666",
                transform=ax.transAxes,
                va="top",
                linespacing=0.95,
            )
            continue

        # No shortening: wrap only when a chain cannot fit one line (the
        # full-width cards hold ~47 characters per line at 12 pt).
        hypothesis = textwrap.fill(
            f"{row.get('source_name', '')} -> {row.get('target_name', '')}",
            width=47,
        )
        jif = row.get("journal_impact_factor")
        jif_year = row.get("journal_impact_factor_year")
        if pd.notna(jif) and pd.notna(jif_year):
            jif_text = f"JIF {int(jif_year)}: {float(jif):.1f}"
        else:
            jif_text = "JIF n/a"
        ax.text(
            x0 + 0.014,
            y0 + card_height - 0.078,
            f"rank {int(row['rank'])} | {int(row['primary_lead_time'])}-yr lead",
            fontsize=FONT_BODY,
            fontweight="bold",
            color=COL_TEXT,
            transform=ax.transAxes,
            va="top",
        )
        ax.text(
            x0 + 0.014,
            y0 + card_height - 0.132,
            hypothesis,
            fontsize=FONT_BODY,
            linespacing=1.15,
            color=COL_TEXT,
            transform=ax.transAxes,
            va="top",
        )
        journal = str(row.get("support_journal") or "")
        journal = JOURNAL_SHORT.get(journal.lower(), journal)
        evidence = f"{journal} ({int(row['support_year'])})"
        ax.text(
            x0 + 0.014,
            y0 + 0.012,
            evidence,
            fontsize=FONT_BODY,
            color=COL_TEXT,
            fontstyle="italic",
            transform=ax.transAxes,
            va="bottom",
        )
        pmid = row.get("support_pmid")
        if pd.notna(pmid) and str(pmid).strip():
            try:
                pmid_text = str(int(float(pmid)))
            except (TypeError, ValueError):
                pmid_text = str(pmid).strip()
            ax.text(
                x0 + card_width - 0.014,
                y0 + 0.012,
                f"PMID {pmid_text}",
                fontsize=FONT_BODY,
                color="#555555",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )
        ax.text(
            x0 + card_width - 0.014,
            y0 + card_height - 0.022,
            jif_text,
            fontsize=FONT_BODY,
            fontweight="bold",
            color=COL_STRICT,
            transform=ax.transAxes,
            ha="right",
            va="top",
        )


def add_panel_label(
    fig: plt.Figure,
    left_ax: plt.Axes,
    label: str,
    right_ax: plt.Axes | None = None,
) -> mpl.text.Text:
    # Placeholder position; the final position is computed by
    # position_panel_labels() against the whole panel block (axes + tick
    # labels + axis titles + panel title treated as one rectangle).
    return fig.text(
        0.0,
        0.0,
        label,
        ha="right",
        va="top",
        fontsize=FONT_PANEL,
        fontweight="bold",
        color="black",
    )


def position_panel_labels(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    label_artists: dict[str, mpl.text.Text],
    pair_title_artists: dict[str, mpl.text.Text],
    task_pairs: dict[str, tuple[str, str, str]],
) -> None:
    """Place each panel id at the top-left of the whole panel block.

    The block is the union of the axes tight bboxes (including axis tick
    labels and axis titles) plus the panel title. The id is right-aligned
    just left of the block and top-aligned with the block top.
    """
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    margin = 0.008  # gap between the id and the block, figure fraction
    for task, (curve_key, lift_key, panel_label) in task_pairs.items():
        bbox = Bbox.union(
            [
                axes[curve_key].get_tightbbox(renderer),
                axes[lift_key].get_tightbbox(renderer),
            ]
        )
        title = pair_title_artists.get(panel_label)
        if title is not None:
            bbox = Bbox.union([bbox, title.get_window_extent(renderer)])
        xf, yf = inv.transform((bbox.x0, bbox.y1))
        artist = label_artists[panel_label]
        artist.set_position((xf - margin, yf))
        artist.set_ha("right")
        artist.set_va("top")
    for key in ("k",):
        bbox = axes[key].get_tightbbox(renderer)
        xf, yf = inv.transform((bbox.x0, bbox.y1))
        artist = label_artists[key]
        artist.set_position((xf - margin, yf))
        artist.set_ha("right")
        artist.set_va("top")


def add_pair_title(
    fig: plt.Figure,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
    title: str,
) -> mpl.text.Text:
    # Place the topic above the axes pair. Figure coordinates keep long titles
    # from being clipped or covered by the neighbouring lift-axis patch.
    left_bbox = left_ax.get_position()
    right_bbox = right_ax.get_position()
    return fig.text(
        left_bbox.x0,
        max(left_bbox.y1, right_bbox.y1) + 0.010,
        title,
        ha="left",
        va="bottom",
        fontsize=FONT_TITLE,
        fontweight="bold",
        color=COL_TEXT,
        zorder=10,
    )


def add_shared_method_legend(
    fig: plt.Figure,
) -> mpl.legend.Legend:
    """Add one conspicuous, single-row legend above all nine task panels."""
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5.0,
            markeredgewidth=0,
            linewidth=2.0 if method == "neurodiscovery" else 1.6,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5125, 0.995),
        bbox_transform=fig.transFigure,
        ncol=len(METHOD_ORDER),
        fontsize=FONT_BODY,
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=1.0,
        fancybox=False,
        handlelength=1.7,
        columnspacing=2.0,
        borderaxespad=0.2,
        borderpad=0.35,
    )
    return legend


def save_section_svgs(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    labels: dict[str, mpl.text.Text],
    pair_titles: dict[str, mpl.text.Text],
    task_pairs: dict[str, tuple[str, str, str]],
    output_root: Path,
) -> None:
    section_root = output_root / "section_svgs"
    section_root.mkdir(parents=True, exist_ok=True)
    figure_legends = list(fig.legends)
    for legend in figure_legends:
        legend.set_visible(False)
    groups: dict[str, tuple[tuple[str, ...], str, str | None]] = {
        task: ((curve_key, lift_key), panel_label, panel_label)
        for task, (curve_key, lift_key, panel_label) in task_pairs.items()
    }
    groups["representative_hypotheses"] = (("k",), "k", None)
    for name, (visible_axes, panel_label, pair_title) in groups.items():
        visible_set = set(visible_axes)
        for key, ax in axes.items():
            visible = key in visible_set
            ax.set_visible(visible)
        for key, artist in labels.items():
            artist.set_visible(key == panel_label)
        for key, artist in pair_titles.items():
            artist.set_visible(key == pair_title)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [axes[key].get_tightbbox(renderer) for key in visible_axes]
        bounds.append(labels[panel_label].get_window_extent(renderer))
        if pair_title is not None:
            bounds.append(pair_titles[pair_title].get_window_extent(renderer))
        bbox = Bbox.union([bound for bound in bounds if bound is not None])
        fig.savefig(
            section_root / f"hindcasting_{name}.svg",
            format="svg",
            bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
            pad_inches=0.08,
        )
    for ax in axes.values():
        ax.set_visible(True)
    for artist in labels.values():
        artist.set_visible(True)
    for artist in pair_titles.values():
        artist.set_visible(True)
    for legend in figure_legends:
        legend.set_visible(True)


def write_provenance(
    output_root: Path,
    neurodiscovery_root: Path,
    baseline_root: Path,
    freeze_years: list[int],
    seeds: list[int],
    journal_metrics_path: Path,
    journal_metrics: pd.DataFrame,
    representative_examples_path: Path | None,
    provisional_examples_path: Path | None,
    data_status_note: str = "latest completed semantic-v2 held-out comparison available on 2026-08-12",
) -> None:
    journal_metric_records = {
        str(row.journal_key): {
            "journal": str(row.journal_display_name),
            "impact_factor": float(row.impact_factor),
            "metric_year": int(row.metric_year),
            "source_url": str(row.source_url),
            "verified_on": str(row.verified_on),
        }
        for row in journal_metrics.itertuples(index=False)
    }
    provenance = {
        "schema_version": "hindcasting-original-layout-selected-studies.v2",
        "source_plot": "core/scripts/plot_case3_taskwise_hindcasting_figure.py",
        "neurodiscovery_root": str(neurodiscovery_root),
        "baseline_root": str(baseline_root),
        "evaluation_semantics": "claim_endpoint_semantics.v1",
        "primary_metric": "primary_hits",
        "curve_aggregation": "sum over common freeze windows within seed; mean and bootstrap 95% CI over seeds",
        "lift_metric": "primary_hits / matched random mean primary hits at k=100",
        "common_freeze_years": freeze_years,
        "future_windows": [f"{year + 1}-{year + 5}" for year in freeze_years],
        "seeds": seeds,
        "methods": list(METHOD_ORDER),
        "case_studies": list(SELECTED_CASE_STUDIES),
        "k_values": list(K_VALUES),
        "journal_impact_factors": {
            "source_file": str(journal_metrics_path),
            "release_context": "2026 Journal Citation Reports release (2025 metric data)",
            "records": journal_metric_records,
        },
        "representative_cards": {
            "source_file": (
                str(representative_examples_path)
                if representative_examples_path is not None
                else "recovered_examples.csv files under the input roots"
            ),
            "selection_rule": "earliest-ranked strict NeuroDiscovery hit within top 100; composite score breaks rank ties",
            "jif_role": "journal-level metadata lookup only; JIF is not a card-selection criterion",
            "provisional_source_file": (
                str(provisional_examples_path)
                if provisional_examples_path is not None
                else None
            ),
            "missing_topic_policy": "optionally reuse prior cards when explicitly supplied; fallback status is retained in source data and provenance only",
        },
        "data_status": data_status_note,
    }
    (output_root / "hindcasting_figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def draw_figure(
    neurodiscovery_root: Path = DEFAULT_NEURODISCOVERY_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    file_prefix: str = "hindcasting_selected_case_studies_original_layout_semantic_v2_20260812",
    data_status_note: str = "latest completed semantic-v2 held-out comparison available on 2026-08-12",
    journal_metrics_path: Path = DEFAULT_JOURNAL_METRICS_PATH,
    representative_examples_path: Path | None = None,
    provisional_examples_path: Path | None = None,
) -> Path:
    apply_style()
    output_root.mkdir(parents=True, exist_ok=True)
    journal_metrics = load_journal_impact_factors(journal_metrics_path)
    metrics = collect_metrics([neurodiscovery_root, baseline_root])
    balanced, freeze_years, seeds = select_balanced_test_set(metrics)
    per_seed, curve_summary = summarize_curves(balanced)
    curve_band = (
        per_seed.groupby(["case_study_id", "method", "k"], as_index=False)
        .agg(band_low=("primary_hits", "min"), band_high=("primary_hits", "max"))
    )
    lift_summary = summarize_freeze_lifts(balanced)
    # Per-(task, method, freeze year, seed) lift values for figure5-style
    # jittered points on the lift bars.
    lift_points = balanced[balanced["k"] == 100].copy()
    lift_points["random_mean_primary_hits"] = pd.to_numeric(
        lift_points["random_mean_primary_hits"], errors="coerce"
    )
    lift_points["lift"] = np.where(
        lift_points["random_mean_primary_hits"].fillna(0) > 0,
        lift_points["primary_hits"] / lift_points["random_mean_primary_hits"],
        0.0,
    )
    examples = collect_recovered_examples(
        [neurodiscovery_root, baseline_root],
        freeze_years,
        seeds,
        journal_metrics,
    )
    lead_summary = summarize_lead_times(examples)
    design_preview = (baseline_root / "design_data_provenance.json").is_file()
    if representative_examples_path is None and design_preview:
        raise ValueError(
            "Design-preview recovered_examples.csv contains synthetic journal "
            "assignments and cannot be used for panel k. Pass a verified card "
            "snapshot with --representative-examples, for example: "
            f"{DEFAULT_VERIFIED_REPRESENTATIVE_EXAMPLES_PATH}"
        )
    if representative_examples_path is None:
        selected_examples = select_representative_examples(examples).assign(
            card_status="run"
        )
    else:
        selected_examples = load_verified_representative_examples(
            representative_examples_path,
            journal_metrics,
        )
    if provisional_examples_path is not None:
        provisional_examples = load_provisional_representative_examples(
            provisional_examples_path,
            journal_metrics,
        )
        provisional_examples = provisional_examples[
            ~provisional_examples["case_study_id"].isin(
                selected_examples["case_study_id"]
            )
        ]
        selected_examples = pd.concat(
            [selected_examples, provisional_examples],
            ignore_index=True,
            sort=False,
        ).sort_values("task_order")

    balanced.to_csv(output_root / "hindcasting_balanced_raw_metrics.csv", index=False)
    per_seed.to_csv(output_root / "hindcasting_curve_source_data_by_seed.csv", index=False)
    curve_summary.to_csv(output_root / "hindcasting_curve_source_data.csv", index=False)
    lift_summary.to_csv(output_root / "hindcasting_lift_source_data.csv", index=False)
    lead_summary.to_csv(output_root / "hindcasting_lead_time_source_data.csv", index=False)
    selected_examples.to_csv(
        output_root / "hindcasting_representative_hypotheses.csv", index=False
    )
    write_provenance(
        output_root,
        neurodiscovery_root,
        baseline_root,
        freeze_years,
        seeds,
        journal_metrics_path,
        journal_metrics,
        representative_examples_path,
        provisional_examples_path,
        data_status_note,
    )

    # figure_style_spec: fixed 15.8 in canvas width, body width 12.24 in
    # (left 0.125 -> right 0.900), wspace 0.16, hspace 0.30; height adapted to
    # keep the previous aspect ratio (~1.223).
    fig = plt.figure(figsize=(15.8, 13.6))
    grid = fig.add_gridspec(
        4,
        6,
        height_ratios=[1.0, 1.0, 1.0, 2.4],
        left=0.125,
        right=0.900,
        top=0.940,
        bottom=0.045,
        wspace=0.16,
        hspace=0.50,
    )
    axes: dict[str, plt.Axes] = {}
    task_pairs: dict[str, tuple[str, str, str]] = {}
    label_artists: dict[str, mpl.text.Text] = {}
    pair_title_artists: dict[str, mpl.text.Text] = {}
    for index, task in enumerate(SELECTED_CASE_STUDIES):
        row = index // 3
        column = (index % 3) * 2
        panel_label = chr(ord("a") + index)
        curve_key = f"{panel_label}_curve"
        lift_key = f"{panel_label}_lift"
        curve_ax = fig.add_subplot(grid[row, column])
        lift_ax = fig.add_subplot(grid[row, column + 1])
        axes[curve_key] = curve_ax
        axes[lift_key] = lift_ax
        task_pairs[task] = (curve_key, lift_key, panel_label)
        draw_task_hit_curve(
            curve_ax,
            curve_summary,
            curve_band,
            task,
            show_ylabel=index % 3 == 0,
        )
        draw_task_freeze_bars(
            lift_ax,
            lift_summary,
            task,
            freeze_years,
            lift_points,
            show_ylabel=index % 3 == 0,
        )
        label_artists[panel_label] = add_panel_label(
            fig, curve_ax, panel_label, lift_ax
        )
        pair_title_artists[panel_label] = add_pair_title(
            fig, curve_ax, lift_ax, CASE_STUDY_LABELS[task]
        )

    bottom = grid[3, :].subgridspec(1, 1)
    axes["k"] = fig.add_subplot(bottom[0, 0])
    draw_examples_table(axes["k"], selected_examples)
    label_artists["k"] = add_panel_label(fig, axes["k"], "k")

    fig.canvas.draw()
    add_shared_method_legend(fig)
    fig.canvas.draw()
    position_panel_labels(fig, axes, label_artists, pair_title_artists, task_pairs)
    prefix = output_root / file_prefix
    fig.savefig(prefix.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    save_section_svgs(
        fig,
        axes,
        label_artists,
        pair_title_artists,
        task_pairs,
        output_root,
    )
    plt.close(fig)
    return prefix.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurodiscovery-root", type=Path, default=DEFAULT_NEURODISCOVERY_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--journal-metrics",
        type=Path,
        default=DEFAULT_JOURNAL_METRICS_PATH,
        help="Publisher/JCR-verified journal metrics CSV.",
    )
    parser.add_argument(
        "--representative-examples",
        type=Path,
        default=None,
        help=(
            "Optional verified representative-card snapshot. If omitted, cards "
            "are selected by rank from recovered_examples.csv under the input roots."
        ),
    )
    parser.add_argument(
        "--provisional-examples",
        type=Path,
        default=None,
        help=(
            "Optional prior card snapshot used only to fill missing topics; "
            "fallback status is retained in exported source data and provenance."
        ),
    )
    parser.add_argument(
        "--file-prefix",
        default="hindcasting_selected_case_studies_original_layout_semantic_v2_20260812",
        help="Output file name prefix (without extension).",
    )
    parser.add_argument(
        "--data-status-note",
        default="latest completed semantic-v2 held-out comparison available on 2026-08-12",
        help="Free-text data status recorded in hindcasting_figure_provenance.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = draw_figure(
        args.neurodiscovery_root,
        args.baseline_root,
        args.output_root,
        file_prefix=args.file_prefix,
        data_status_note=args.data_status_note,
        journal_metrics_path=args.journal_metrics,
        representative_examples_path=args.representative_examples,
        provisional_examples_path=args.provisional_examples,
    )
    print(path)


if __name__ == "__main__":
    main()


# Updated: 2026-08-12 03:24:56 HKT - expanded panel k to one card per Case 1-9 and enlarged the three-level type system.
