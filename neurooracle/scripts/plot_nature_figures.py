"""Generate Nature-style NeuroOracle figure mockups.

The script creates synthetic result tables and publication-style mock figures
for Fig.2-Fig.6. Fig.1 is intentionally omitted because it is already drafted as
the framework figure.

Later, replace the ``make_mock_data`` outputs with real result tables while
keeping the plotting functions unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap


RNG = np.random.default_rng(42)
FIGURE_SIZE = (16.8, 10.0)
FIGURE_SIZE_TALL = (16.8, 10.6)

COLORS = {
    "green_dark": "#006D2C",
    "green": "#238B45",
    "green_mid": "#74C476",
    "green_light": "#C7E9C0",
    "purple_dark": "#3F007D",
    "purple": "#6A51A3",
    "purple_mid": "#9E9AC8",
    "purple_light": "#DADAEB",
    "blue": "#2B8CBE",
    "blue_mid": "#6BAED6",
    "blue_light": "#D9EAF7",
    "orange": "#FDAE61",
    "orange_dark": "#E6550D",
    "salmon": "#FB6A4A",
    "grey": "#6B6B6B",
    "light_grey": "#E6E6E6",
    "dark": "#111111",
    "off_white": "#FAFAFA",
}

DOMAIN_COLORS = {
    "Disease": COLORS["blue"],
    "Brain region": COLORS["green"],
    "Imaging marker": COLORS["blue_mid"],
    "Gene/pathway": COLORS["purple"],
    "Outcome": COLORS["orange"],
    "Dataset": COLORS["grey"],
    "Claim": "#EBCB8B",
}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "neurooracle_seq", ["#FFFFFF", "#D8ECF7", "#8EC1DE", "#3C94C5", "#075A8C"]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "neurooracle_div", ["#2166AC", "#F7F7F7", "#D6604D"]
)
GREEN_CMAP = LinearSegmentedColormap.from_list(
    "neurooracle_green", ["#FFFFFF", "#C7E9C0", "#74C476", "#238B45", "#005A32"]
)
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "neurooracle_purple", ["#FFFFFF", "#DADAEB", "#9E9AC8", "#6A51A3", "#3F007D"]
)


@dataclass(frozen=True)
class MockData:
    figure2: dict[str, pd.DataFrame]
    figure3: dict[str, pd.DataFrame]
    figure4: dict[str, pd.DataFrame]
    figure5: dict[str, pd.DataFrame]
    figure6: dict[str, pd.DataFrame]


def configure_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    sans_stack = ["Arial", "Helvetica", "DejaVu Sans"]
    primary_sans = next((font for font in sans_stack if font in available_fonts), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [primary_sans, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.titleweight": "normal",
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
            "axes.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.7,
            "ytick.major.width": 1.7,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="right",
    )


def write_tables(data: MockData, out_dir: Path) -> None:
    tables_dir = out_dir / "sample_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for group_name, group in data.__dict__.items():
        for table_name, table in group.items():
            table.to_csv(tables_dir / f"{group_name}_{table_name}.csv", index=False)


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    cmap: str | LinearSegmentedColormap = SEQ_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
) -> mpl.image.AxesImage:
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    if cbar_label:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel(cbar_label, rotation=90)
        cbar.ax.tick_params(length=0)
    return im


def wrapped_network_labels(labels: Iterable[str]) -> dict[str, str]:
    """Short labels for dense network panels."""
    mapping = {
        "Temporal cortex": "Temporal\ncortex",
        "Posterior cingulate": "Posterior\ncingulate",
        "fMRI connectivity": "fMRI\nconnectivity",
        "Cortical thickness": "Cortical\nthickness",
        "Cognitive decline": "Cognitive\ndecline",
        "Psychosis severity": "Psychosis\nseverity",
        "Inflammatory pathway": "Inflam.",
        "microglial PET signal": "PET\nsignal",
        "insula connectivity": "Insula\nFC",
        "negative symptoms": "Neg.\nsymptoms",
        "Complement cascade": "Complement",
        "Dopamine pathway": "Dopamine",
        "hippocampal volume": "Hippo.\nvolume",
        "frontostriatal FC": "FStr\nFC",
        "cognitive decline": "Cognition",
        "Synaptic plasticity": "Synaptic",
        "memory impairment": "Memory",
    }
    return {label: mapping.get(label, label) for label in labels}


def make_mock_data() -> MockData:
    domains = [
        "Disease",
        "Brain region",
        "Imaging marker",
        "Gene/pathway",
        "Outcome",
        "Dataset",
    ]
    base = np.array(
        [
            [0.95, 0.76, 0.80, 0.45, 0.69, 0.36],
            [0.76, 0.92, 0.84, 0.32, 0.52, 0.44],
            [0.80, 0.84, 0.90, 0.51, 0.74, 0.58],
            [0.45, 0.32, 0.51, 0.88, 0.64, 0.28],
            [0.69, 0.52, 0.74, 0.64, 0.91, 0.40],
            [0.36, 0.44, 0.58, 0.28, 0.40, 0.82],
        ]
    )
    domain_connectivity = pd.DataFrame(base, index=domains, columns=domains).reset_index(
        names="domain"
    )

    scale = pd.DataFrame(
        {
            "entity": ["Concepts", "Relations", "Claims", "Papers", "Datasets"],
            "curated_db": [17200, 36800, 9800, 0, 42],
            "paper_derived": [69100, 117600, 43200, 28600, 0],
            "case_study": [7400, 15800, 12100, 1640, 9],
        }
    )

    diseases = ["SCZ", "BD", "MDD", "OCD", "ADHD", "AN", "AD"]
    markers = ["CortThick", "CortSurf", "fMRI FC", "PET amyloid", "FDG PET", "DTI FA"]
    rows = []
    for disease in diseases:
        for marker in markers:
            claim_count = int(RNG.integers(30, 520))
            rows.append(
                {
                    "disease": disease,
                    "marker": marker,
                    "claim_count": claim_count,
                    "support_fraction": float(RNG.uniform(0.42, 0.86)),
                    "contradiction_fraction": float(RNG.uniform(0.02, 0.18)),
                }
            )
    disease_marker = pd.DataFrame(rows)

    evidence_state_rows = []
    state_profiles = {
        "Disease": [0.58, 0.10, 0.17, 0.15],
        "Brain region": [0.63, 0.08, 0.19, 0.10],
        "Imaging marker": [0.52, 0.12, 0.18, 0.18],
        "Gene/pathway": [0.36, 0.09, 0.26, 0.29],
        "Outcome": [0.45, 0.11, 0.24, 0.20],
        "Dataset": [0.67, 0.03, 0.19, 0.11],
    }
    for domain, values in state_profiles.items():
        for state, value in zip(["Supported", "Contradicted", "Sparse", "Emerging"], values):
            evidence_state_rows.append({"domain": domain, "state": state, "fraction": value})
    evidence_state = pd.DataFrame(evidence_state_rows)

    years = np.arange(2000, 2027)
    claims_over_time = pd.DataFrame(
        {
            "year": years,
            "curated_database": np.cumsum(RNG.poisson(65, len(years))),
            "paper_claims": np.cumsum(np.linspace(40, 900, len(years)) + RNG.normal(0, 35, len(years))).astype(int),
            "case_study_claims": np.maximum(
                0, np.cumsum(np.where(years >= 2022, RNG.poisson(420, len(years)), 0))
            ),
        }
    )

    fig2 = {
        "scale": scale,
        "domain_connectivity": domain_connectivity,
        "disease_marker": disease_marker,
        "claims_over_time": claims_over_time,
        "evidence_state": evidence_state,
    }

    hypotheses = [
        "H1 temporal cortex",
        "H2 inferior parietal",
        "H3 posterior cingulate",
        "H4 insula",
        "H5 anterior cingulate",
        "H6 orbitofrontal",
        "H7 precentral",
        "H8 fusiform",
    ]
    components = ["Evidence", "Novelty", "Plausibility", "Feasibility", "Critic"]
    rows = []
    for hyp in hypotheses:
        center = RNG.uniform(0.55, 0.88)
        for comp in components:
            rows.append(
                {
                    "hypothesis": hyp,
                    "component": comp,
                    "score": float(np.clip(RNG.normal(center, 0.11), 0.2, 0.98)),
                }
            )
    score_components = pd.DataFrame(rows)
    score_components["total_score"] = score_components.groupby("hypothesis")[
        "score"
    ].transform("mean")

    ablation = pd.DataFrame(
        {
            "setting": ["LLM only", "Curated DB", "Paper claims", "Claims + metadata", "Full NeuroOracle"],
            "precision_at_10": [0.22, 0.38, 0.51, 0.63, 0.74],
            "dataset_anchor_rate": [0.18, 0.26, 0.43, 0.71, 0.82],
        }
    )
    seed_rows = []
    for seed in range(1, 13):
        for case in ["case1", "case2", "case3"]:
            seed_rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "overlap_at_20": float(np.clip(RNG.normal(0.62, 0.12), 0.25, 0.95)),
                    "mean_score": float(np.clip(RNG.normal(0.72, 0.08), 0.45, 0.94)),
                }
            )
    seed_stability = pd.DataFrame(seed_rows)
    rejection = pd.DataFrame(
        {
            "reason": [
                "Too broad outcome",
                "No dataset anchor",
                "Weak evidence path",
                "Atom mismatch",
                "Contradicted evidence",
            ],
            "count": [46, 38, 31, 19, 13],
        }
    )
    fig3 = {
        "score_components": score_components,
        "ablation": ablation,
        "seed_stability": seed_stability,
        "rejection_reasons": rejection,
    }

    regions = [
        "Transverse temporal",
        "Inferior parietal",
        "Middle temporal",
        "Posterior cingulate",
        "Fusiform",
        "Inferior temporal",
        "Precentral",
        "Orbitofrontal",
        "Insula",
        "Anterior cingulate",
    ]
    diseases_cs1 = ["SCZ", "BD", "MDD", "OCD", "ADHD", "AN"]
    rows = []
    region_centers = np.linspace(-0.36, -0.12, len(regions))
    for region, center in zip(regions, region_centers):
        for disease in diseases_cs1:
            n_case = int(RNG.integers(18, 65))
            n_control = int(RNG.integers(70, 120))
            d = float(RNG.normal(center, 0.11))
            se = float(np.sqrt((n_case + n_control) / (n_case * n_control) + d**2 / (2 * (n_case + n_control - 2))))
            rows.append(
                {
                    "region": region,
                    "disease": disease,
                    "cohens_d": d,
                    "se": se,
                    "ci_low": d - 1.96 * se,
                    "ci_high": d + 1.96 * se,
                    "n_case": n_case,
                    "n_control": n_control,
                }
            )
    region_effects = pd.DataFrame(rows)
    shared = (
        region_effects.assign(abs_d=lambda x: x["cohens_d"].abs())
        .groupby("region")
        .agg(
            mean_d=("cohens_d", "mean"),
            mean_abs_d=("abs_d", "mean"),
            direction_consistency=("cohens_d", lambda s: float(np.mean(s < 0))),
        )
        .reset_index()
    )
    shared["i2"] = np.clip(RNG.normal(0.32, 0.18, len(shared)), 0.02, 0.78)
    shared["shared_strength"] = (
        shared["mean_abs_d"] * shared["direction_consistency"] * (1 - shared["i2"] / 1.25)
    )
    shared = shared.sort_values("shared_strength", ascending=False)

    fig4 = {"region_effects": region_effects, "shared_scores": shared}

    chains = [
        ("Complement cascade", "microglial PET signal", "cognitive decline"),
        ("Dopamine pathway", "frontostriatal FC", "psychosis severity"),
        ("Inflammatory pathway", "insula connectivity", "negative symptoms"),
        ("Synaptic plasticity", "hippocampal volume", "memory impairment"),
        ("Myelination pathway", "white-matter FA", "executive dysfunction"),
        ("Stress response", "amygdala reactivity", "anxiety burden"),
    ]
    chain_rows = []
    for i, (gene, marker, outcome) in enumerate(chains, 1):
        chain_rows.append(
            {
                "chain": f"C{i}",
                "gene_pathway": gene,
                "imaging_marker": marker,
                "outcome": outcome,
                "paper_count": int(RNG.integers(18, 130)),
                "claim_count": int(RNG.integers(25, 260)),
                "novelty": float(RNG.uniform(0.42, 0.86)),
                "support": float(RNG.uniform(0.50, 0.91)),
            }
        )
    chain_table = pd.DataFrame(chain_rows)
    med_rows = []
    for chain in chain_table["chain"]:
        indirect = float(RNG.normal(0.22, 0.08))
        se = float(RNG.uniform(0.04, 0.09))
        med_rows.append(
            {
                "chain": chain,
                "effect": indirect,
                "ci_low": indirect - 1.96 * se,
                "ci_high": indirect + 1.96 * se,
                "p_value": float(10 ** RNG.uniform(-4, -1)),
            }
        )
    mediation = pd.DataFrame(med_rows)
    fig5 = {"chains": chain_table, "mediation": mediation}

    ks = np.arange(1, 51)
    precision = pd.DataFrame(
        {
            "rank_k": ks,
            "NeuroOracle": 0.18 + 0.56 * (1 - np.exp(-ks / 13)),
            "Degree heuristic": 0.12 + 0.31 * (1 - np.exp(-ks / 17)),
            "Keyword co-occurrence": 0.10 + 0.24 * (1 - np.exp(-ks / 19)),
            "Random path": 0.05 + 0.10 * (1 - np.exp(-ks / 24)),
        }
    )
    freeze_years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    horizons = [1, 2, 3, 4, 5]
    rows = []
    for fy in freeze_years:
        for h in horizons:
            rows.append(
                {
                    "freeze_year": fy,
                    "horizon_years": h,
                    "hit_rate": float(np.clip(0.18 + 0.07 * h + 0.015 * (fy - 2016) + RNG.normal(0, 0.035), 0.05, 0.9)),
                }
            )
    freeze_heatmap = pd.DataFrame(rows)
    lead_time = pd.DataFrame(
        {
            "hypothesis": [f"H{i}" for i in range(1, 31)],
            "rank": np.arange(1, 31),
            "lead_time_years": RNG.integers(1, 7, 30),
            "future_support_claims": RNG.integers(1, 18, 30),
            "score": np.clip(RNG.normal(0.68, 0.12, 30), 0.35, 0.95),
        }
    )
    baseline = pd.DataFrame(
        {
            "method": ["Random path", "Keyword", "Degree", "LLM only", "NeuroOracle"],
            "precision_at_20": [0.14, 0.25, 0.33, 0.41, 0.63],
            "recall_at_50": [0.11, 0.20, 0.28, 0.37, 0.58],
        }
    )
    fig6 = {
        "precision_curve": precision,
        "freeze_heatmap": freeze_heatmap,
        "lead_time": lead_time,
        "baseline": baseline,
    }

    return MockData(figure2=fig2, figure3=fig3, figure4=fig4, figure5=fig5, figure6=fig6)


def plot_figure2(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    scale = data["scale"]
    domain_conn = data["domain_connectivity"]
    disease_marker = data["disease_marker"]
    claims_over_time = data["claims_over_time"]
    evidence_state = data["evidence_state"]

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 1.1], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    x = np.arange(len(scale))
    bottom = np.zeros(len(scale))
    for col, color, label in [
        ("curated_db", COLORS["green"], "Curated databases"),
        ("paper_derived", COLORS["blue_mid"], "Paper-derived"),
        ("case_study", COLORS["orange"], "Case-study injection"),
    ]:
        ax_a.bar(x, scale[col] / 1000, bottom=bottom, color=color, label=label, width=0.7)
        bottom += scale[col] / 1000
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(scale["entity"], rotation=35, ha="right")
    ax_a.set_ylabel("Count, thousands")
    ax_a.set_ylim(0, 215)
    ax_a.set_title("Evidence-map scale")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.52, 0.98), handlelength=1.4, borderaxespad=0)
    panel_label(ax_a, "a")

    matrix = domain_conn.drop(columns="domain").to_numpy()
    labels = domain_conn["domain"].tolist()
    heatmap(ax_b, matrix, labels, labels, cmap=SEQ_CMAP, vmin=0, vmax=1, cbar_label="Connectivity")
    ax_b.set_title("Domain connectivity")
    panel_label(ax_b, "b")

    marker_order = ["CortThick", "CortSurf", "fMRI FC", "PET amyloid", "FDG PET", "DTI FA"]
    disease_order = ["SCZ", "BD", "MDD", "OCD", "ADHD", "AN", "AD"]
    for _, row in disease_marker.iterrows():
        ax_c.scatter(
            marker_order.index(row["marker"]),
            disease_order.index(row["disease"]),
            s=row["claim_count"] / 2.2,
            c=row["support_fraction"],
            cmap=GREEN_CMAP,
            vmin=0.35,
            vmax=0.9,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_c.set_xticks(np.arange(len(marker_order)))
    ax_c.set_xticklabels(marker_order, rotation=45, ha="right")
    ax_c.set_yticks(np.arange(len(disease_order)))
    ax_c.set_yticklabels(disease_order)
    ax_c.set_title("Disease-marker evidence")
    ax_c.grid(color="#EDEDED", linewidth=0.8)
    panel_label(ax_c, "c")

    ax_d.plot(claims_over_time["year"], claims_over_time["curated_database"], color=COLORS["green"], label="Curated databases", lw=3)
    ax_d.plot(claims_over_time["year"], claims_over_time["paper_claims"], color=COLORS["blue"], label="Paper-derived claims", lw=3)
    ax_d.plot(claims_over_time["year"], claims_over_time["case_study_claims"], color=COLORS["orange"], label="Case-study claims", lw=3)
    ax_d.fill_between(claims_over_time["year"], claims_over_time["paper_claims"], alpha=0.08, color=COLORS["blue"])
    ax_d.set_ylabel("Cumulative claims")
    ax_d.set_xlabel("Publication year")
    ax_d.set_title("Temporal growth of structured evidence")
    ax_d.legend(frameon=False, ncol=3, loc="upper left")
    panel_label(ax_d, "d")

    state_order = ["Supported", "Contradicted", "Sparse", "Emerging"]
    state_colors = {
        "Supported": COLORS["green"],
        "Contradicted": COLORS["salmon"],
        "Sparse": COLORS["grey"],
        "Emerging": COLORS["orange"],
    }
    domain_order = ["Disease", "Brain region", "Imaging marker", "Gene/pathway", "Outcome", "Dataset"]
    left = np.zeros(len(domain_order))
    y = np.arange(len(domain_order))
    for state in state_order:
        vals = (
            evidence_state[evidence_state["state"] == state]
            .set_index("domain")
            .loc[domain_order, "fraction"]
            .to_numpy()
        )
        ax_e.barh(y, vals, left=left, height=0.68, color=state_colors[state], label=state)
        left += vals
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(domain_order)
    ax_e.set_xlim(0, 1)
    ax_e.set_xlabel("Fraction of claims")
    ax_e.set_title("Evidence state by domain")
    ax_e.legend(
        loc="center left",
        bbox_to_anchor=(1.03, 0.50),
        ncol=1,
        handlelength=1.0,
        handletextpad=0.45,
        borderaxespad=0,
    )
    ax_e.invert_yaxis()
    panel_label(ax_e, "e")

    save_fig(fig, out_dir, "fig2_evidence_landscape")


def draw_simple_sankey(ax: plt.Axes, left: Iterable[str], mid: Iterable[str], right: Iterable[str]) -> None:
    ax.axis("off")
    left = list(left)
    mid = list(mid)
    right = list(right)
    levels = [left, mid, right]
    xs = [0.14, 0.50, 0.86]
    box_w = 0.24
    box_h = 0.095
    rects: dict[str, tuple[float, float]] = {}
    for x, labels in zip(xs, levels):
        ys = np.linspace(0.82, 0.18, len(labels))
        for y, label in zip(ys, labels):
            rect = patches.FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor="white",
                edgecolor=COLORS["dark"],
                linewidth=0.9,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", fontsize=12.5, zorder=4)
            rects[label] = (x, y)
    links = [
        (left[0], mid[0], 4),
        (left[1], mid[1], 3),
        (left[2], mid[2], 3),
        (mid[0], right[0], 3),
        (mid[1], right[1], 4),
        (mid[2], right[2], 2),
        (left[0], mid[1], 1.8),
        (mid[1], right[0], 1.5),
    ]
    for src, dst, width in links:
        x0, y0 = rects[src]
        x1, y1 = rects[dst]
        con = patches.FancyArrowPatch(
            (x0 + box_w / 2, y0),
            (x1 - box_w / 2, y1),
            connectionstyle="arc3,rad=0.08",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            color=COLORS["green"],
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(con)


def plot_figure3(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    scores = data["score_components"]
    ablation = data["ablation"]
    stability = data["seed_stability"]
    rejection = data["rejection_reasons"]

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.25, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    draw_simple_sankey(
        ax_a,
        ["Evidence\npaths", "Dataset\nanchors", "Critic\nreview"],
        ["Atom chains", "Candidate\nhypotheses", "Ranked\nhypotheses"],
        ["Executable\nanalysis", "Supported\ndiscovery", "Graph\nupdate"],
    )
    ax_a.set_title("Evidence-to-hypothesis flow")
    panel_label(ax_a, "a")

    pivot = scores.pivot(index="hypothesis", columns="component", values="score")
    ordered = scores.groupby("hypothesis")["total_score"].first().sort_values(ascending=False).index
    pivot = pivot.loc[ordered]
    heatmap(
        ax_b,
        pivot.to_numpy(),
        pivot.columns.tolist(),
        pivot.index.tolist(),
        cmap=SEQ_CMAP,
        vmin=0,
        vmax=1,
        cbar_label="Score",
    )
    ax_b.set_title("Scored hypothesis components")
    panel_label(ax_b, "b")

    total = pivot.mean(axis=1).sort_values()
    ax_c.barh(total.index, total.values, color=COLORS["green"])
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("Composite score")
    ax_c.set_title("Top hypothesis ranking")
    panel_label(ax_c, "c")

    x = np.arange(len(ablation))
    w = 0.36
    ax_d.bar(x - w / 2, ablation["precision_at_10"], width=w, color=COLORS["blue"], label="Precision@10")
    ax_d.bar(x + w / 2, ablation["dataset_anchor_rate"], width=w, color=COLORS["orange"], label="Dataset anchor rate")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(ablation["setting"], rotation=35, ha="right")
    ax_d.set_ylim(0, 1)
    ax_d.set_title("Generator ablation")
    ax_d.legend()
    panel_label(ax_d, "d")

    cases = ["case1", "case2", "case3"]
    positions = np.arange(len(cases))
    box_data = [stability.loc[stability["case"] == case, "overlap_at_20"] for case in cases]
    ax_e.boxplot(box_data, patch_artist=True, tick_labels=cases, boxprops={"facecolor": "#B9D8EE", "edgecolor": COLORS["blue"]}, medianprops={"color": COLORS["dark"]})
    ax_e.scatter(
        np.repeat(positions + 1, [len(x) for x in box_data]),
        np.concatenate([x.to_numpy() for x in box_data]),
        s=14,
        color=COLORS["dark"],
        alpha=0.45,
    )
    ax_e.set_ylim(0, 1)
    ax_e.set_ylabel("Overlap@20")
    ax_e.set_title("Multi-seed stability")
    panel_label(ax_e, "e")

    ax_f.barh(rejection["reason"], rejection["count"], color=COLORS["orange_dark"], alpha=0.78)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Rejected hypotheses")
    ax_f.set_title("Critic rejection reasons")
    panel_label(ax_f, "f")

    save_fig(fig, out_dir, "fig3_generation_ranking")


def plot_forest(ax: plt.Axes, effects: pd.DataFrame, title: str) -> None:
    effects = effects.sort_values("cohens_d")
    y = np.arange(len(effects))
    ax.errorbar(
        effects["cohens_d"],
        y,
        xerr=[effects["cohens_d"] - effects["ci_low"], effects["ci_high"] - effects["cohens_d"]],
        fmt="o",
        color=COLORS["green"],
        ecolor=COLORS["grey"],
        elinewidth=1.8,
        capsize=3,
        markersize=5.5,
    )
    ax.axvline(0, color=COLORS["dark"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(effects["disease"])
    ax.set_xlabel("Cohen's d")
    ax.set_title(title)


def draw_mock_brain(ax: plt.Axes, shared: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ellipse = patches.Ellipse((0, 0), 4.5, 2.7, facecolor="#F2F4F6", edgecolor=COLORS["dark"], linewidth=1.0)
    ax.add_patch(ellipse)
    sulci = [
        [(-1.8, 0.4), (-1.2, 0.8), (-0.4, 0.65), (0.5, 0.9), (1.6, 0.45)],
        [(-1.9, -0.1), (-0.9, 0.15), (0.2, -0.15), (1.3, 0.05), (2.0, -0.25)],
        [(-1.3, -0.75), (-0.4, -0.45), (0.6, -0.7), (1.6, -0.45)],
    ]
    for line in sulci:
        ax.plot([p[0] for p in line], [p[1] for p in line], color="#B9C2C9", lw=1)

    coords = {
        "Transverse temporal": (-1.2, -0.65),
        "Inferior parietal": (0.4, 0.7),
        "Middle temporal": (-0.9, -0.2),
        "Posterior cingulate": (0.25, 0.05),
        "Fusiform": (-0.25, -0.8),
        "Inferior temporal": (-1.15, -0.45),
        "Precentral": (1.1, 0.45),
        "Orbitofrontal": (1.45, -0.65),
        "Insula": (0.05, -0.2),
        "Anterior cingulate": (0.65, 0.15),
    }
    vmax = shared["shared_strength"].max()
    cmap = SEQ_CMAP
    for _, row in shared.iterrows():
        x, y = coords[row["region"]]
        ax.scatter(x, y, s=80 + 650 * row["shared_strength"] / vmax, color=cmap(row["shared_strength"] / vmax), edgecolor="white", linewidth=0.7)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.7, 1.7)


def plot_figure4(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    effects = data["region_effects"]
    shared = data["shared_scores"]

    fig = plt.figure(figsize=FIGURE_SIZE_TALL, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.05, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_e = fig.add_subplot(gs[1, 2])

    pivot = effects.pivot(index="region", columns="disease", values="cohens_d").loc[shared["region"]]
    heatmap(
        ax_a,
        pivot.to_numpy(),
        pivot.columns.tolist(),
        pivot.index.tolist(),
        cmap=DIVERGING_CMAP,
        vmin=-0.65,
        vmax=0.35,
        cbar_label="Cohen's d",
    )
    ax_a.set_title("Disease-by-region case-control effects")
    panel_label(ax_a, "a")

    plot_forest(
        ax_b,
        effects[effects["region"] == "Inferior parietal"],
        "Inferior parietal",
    )
    panel_label(ax_b, "b")

    plot_forest(
        ax_c,
        effects[effects["region"] == "Transverse temporal"],
        "Transverse temporal",
    )
    panel_label(ax_c, "c")

    top = shared.sort_values("shared_strength", ascending=True).tail(8)
    ax_d.barh(top["region"], top["shared_strength"], color=COLORS["green"])
    ax_d.set_xlabel("Shared alteration score")
    ax_d.set_title("Cross-disorder shared strength")
    panel_label(ax_d, "d")

    draw_mock_brain(ax_e, shared)
    ax_e.set_title("Mapped shared regions")
    panel_label(ax_e, "e")

    save_fig(fig, out_dir, "fig4_case1_transdiagnostic")


def plot_figure5(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    chains = data["chains"]
    mediation = data["mediation"].merge(chains[["chain", "gene_pathway", "imaging_marker", "outcome"]], on="chain")

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    draw_simple_sankey(
        ax_a,
        ["Complement", "Dopamine", "Inflammation"],
        ["PET\nsignal", "Frontostriatal\nFC", "Insula\nFC"],
        ["Cognition", "Psychosis", "Symptoms"],
    )
    ax_a.set_title("Pathway-imaging-outcome chains")
    panel_label(ax_a, "a")

    x = np.arange(len(chains))
    ax_b.scatter(chains["support"], chains["novelty"], s=chains["claim_count"] * 1.35, color=COLORS["purple"], alpha=0.72, edgecolor="white")
    for i, row in chains.iterrows():
        ax_b.text(row["support"] + 0.006, row["novelty"] + 0.006, row["chain"], fontsize=12)
    ax_b.set_xlim(0.45, 0.95)
    ax_b.set_ylim(0.35, 0.95)
    ax_b.set_xlabel("Evidence support")
    ax_b.set_ylabel("Novelty")
    ax_b.set_title("Chain evidence profile")
    panel_label(ax_b, "b")

    mediation = mediation.sort_values("effect")
    y = np.arange(len(mediation))
    ax_c.errorbar(
        mediation["effect"],
        y,
        xerr=[mediation["effect"] - mediation["ci_low"], mediation["ci_high"] - mediation["effect"]],
        fmt="o",
        color=COLORS["orange"],
        ecolor=COLORS["grey"],
        capsize=3,
        elinewidth=1.8,
        markersize=5.5,
    )
    ax_c.axvline(0, color=COLORS["dark"], lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(mediation["chain"])
    ax_c.set_xlabel("Indirect effect")
    ax_c.set_title("Mediation effect estimates")
    panel_label(ax_c, "c")

    evidence_matrix = chains.set_index("chain")[["paper_count", "claim_count", "support", "novelty"]]
    evidence_matrix["paper_count"] = evidence_matrix["paper_count"] / evidence_matrix["paper_count"].max()
    evidence_matrix["claim_count"] = evidence_matrix["claim_count"] / evidence_matrix["claim_count"].max()
    heatmap(
        ax_d,
        evidence_matrix.to_numpy(),
        ["Papers", "Claims", "Support", "Novelty"],
        evidence_matrix.index.tolist(),
        cmap=SEQ_CMAP,
        vmin=0,
        vmax=1,
        cbar_label="Normalized value",
    )
    ax_d.set_title("Chain ranking evidence matrix")
    panel_label(ax_d, "d")

    G = nx.DiGraph()
    chain_subset = chains.head(4).reset_index(drop=True)
    for _, row in chain_subset.iterrows():
        G.add_edge(row["gene_pathway"], row["imaging_marker"], weight=row["support"])
        G.add_edge(row["imaging_marker"], row["outcome"], weight=row["support"])
    y_positions = np.linspace(0.78, -0.78, len(chain_subset))
    pos = {}
    for y_pos, (_, row) in zip(y_positions, chain_subset.iterrows()):
        pos[row["gene_pathway"]] = (-1.15, y_pos)
        pos[row["imaging_marker"]] = (0.0, y_pos)
        pos[row["outcome"]] = (1.15, y_pos)
    node_colors = []
    for node in G.nodes:
        if any(node == x for x in chains["gene_pathway"]):
            node_colors.append(DOMAIN_COLORS["Gene/pathway"])
        elif any(node == x for x in chains["imaging_marker"]):
            node_colors.append(DOMAIN_COLORS["Imaging marker"])
        else:
            node_colors.append(DOMAIN_COLORS["Outcome"])
    nx.draw_networkx_edges(G, pos, ax=ax_e, arrows=True, arrowstyle="-|>", width=1.8, alpha=0.42, edge_color=COLORS["grey"])
    nx.draw_networkx_nodes(G, pos, ax=ax_e, node_color=node_colors, node_size=680, edgecolors="white", linewidths=1.0)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=wrapped_network_labels(G.nodes),
        ax=ax_e,
        font_size=10.8,
        font_family="DejaVu Sans",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.2},
    )
    xs = np.array([xy[0] for xy in pos.values()])
    ys = np.array([xy[1] for xy in pos.values()])
    ax_e.set_xlim(-1.65, 1.65)
    ax_e.set_ylim(-1.05, 1.05)
    ax_e.axis("off")
    ax_e.set_title("Claim-backed chain subgraph")
    panel_label(ax_e, "e")

    save_fig(fig, out_dir, "fig5_case2_mediation")


def plot_figure6(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    precision = data["precision_curve"]
    freeze = data["freeze_heatmap"]
    lead_time = data["lead_time"]
    baseline = data["baseline"]

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    for col, color in [
        ("NeuroOracle", COLORS["green"]),
        ("Degree heuristic", COLORS["blue"]),
        ("Keyword co-occurrence", COLORS["orange"]),
        ("Random path", COLORS["grey"]),
    ]:
        ax_a.plot(precision["rank_k"], precision[col], lw=2, label=col, color=color)
    ax_a.set_xlabel("Rank K")
    ax_a.set_ylabel("Future-supported fraction")
    ax_a.set_ylim(0, 0.85)
    ax_a.set_title("Future evidence enrichment")
    ax_a.legend()
    panel_label(ax_a, "a")

    pivot = freeze.pivot(index="freeze_year", columns="horizon_years", values="hit_rate")
    heatmap(
        ax_b,
        pivot.to_numpy(),
        [str(c) for c in pivot.columns],
        [str(i) for i in pivot.index],
        cmap=SEQ_CMAP,
        vmin=0,
        vmax=0.75,
        cbar_label="Hit rate",
    )
    ax_b.set_xlabel("Future window, years")
    ax_b.set_ylabel("Freeze year")
    ax_b.set_title("Freeze-year robustness")
    panel_label(ax_b, "b")

    x = np.arange(len(baseline))
    ax_c.bar(x - 0.18, baseline["precision_at_20"], width=0.36, color=COLORS["green"], label="Precision@20")
    ax_c.bar(x + 0.18, baseline["recall_at_50"], width=0.36, color=COLORS["blue_mid"], label="Recall@50")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(baseline["method"], rotation=35, ha="right")
    ax_c.set_ylim(0, 0.75)
    ax_c.set_title("Baseline comparison")
    ax_c.legend()
    panel_label(ax_c, "c")

    ax_d.scatter(
        lead_time["rank"],
        lead_time["lead_time_years"],
        s=lead_time["future_support_claims"] * 15,
        c=lead_time["score"],
        cmap=SEQ_CMAP,
        vmin=0.35,
        vmax=0.95,
        edgecolor="white",
    )
    ax_d.set_xlabel("Hypothesis rank")
    ax_d.set_ylabel("Lead time, years")
    ax_d.set_title("Years gained before future evidence")
    panel_label(ax_d, "d")

    examples = lead_time.sort_values(["score", "future_support_claims"], ascending=False).head(6)
    ax_e.axis("off")
    ax_e.set_title("Recovered future-supported examples")
    for i, row in enumerate(examples.itertuples(index=False)):
        y = 0.92 - i * 0.14
        ax_e.text(0.02, y, row.hypothesis, fontweight="bold", va="center")
        ax_e.text(0.18, y, f"rank {row.rank}", va="center")
        ax_e.text(0.39, y, f"{row.lead_time_years} yr lead", va="center")
        ax_e.text(0.68, y, f"{row.future_support_claims} claims", va="center")
    panel_label(ax_e, "e")

    failure = pd.DataFrame(
        {
            "reason": ["No future paper", "Too broad", "Wrong direction", "Dataset mismatch", "Atom mismatch"],
            "count": [34, 22, 15, 12, 8],
        }
    )
    ax_f.barh(failure["reason"], failure["count"], color=COLORS["orange_dark"], alpha=0.78)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Failed candidates")
    ax_f.set_title("Hindcasting failure modes")
    panel_label(ax_f, "f")

    save_fig(fig, out_dir, "fig6_case3_hindcasting")


def write_plan(out_dir: Path) -> None:
    plan = """# NeuroOracle Nature Figure Plan

Fig.1 Framework overview: keep the existing system schematic.

Fig.2 Evidence landscape: evidence scale, domain connectivity, disease-marker evidence density, temporal claim growth, and a claim-backed subgraph.

Fig.3 Hypothesis generation and ranking: evidence-to-hypothesis flow, scoring heatmap, ranked hypotheses, generator ablation, multi-seed stability, and critic rejection reasons.

Fig.4 Case Study 1: disease-by-region effect-size heatmap, forest plots, shared alteration ranking, and mapped transdiagnostic brain regions.

Fig.5 Case Study 2: pathway-imaging-outcome chains, chain evidence profile, mediation effects, ranking matrix, and claim-backed chain subgraph.

Fig.6 Case Study 3: hindcasting precision curves, freeze-year robustness, baseline comparison, lead-time distribution, recovered examples, and failure modes.
"""
    (out_dir / "FIGURE_PLAN.md").write_text(plan, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("neurooracle/data/figures/nature_mock"),
        help="Directory for mock figures and sample tables.",
    )
    args = parser.parse_args()

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = make_mock_data()
    write_tables(data, args.out_dir)
    write_plan(args.out_dir)
    plot_figure2(data.figure2, args.out_dir)
    plot_figure3(data.figure3, args.out_dir)
    plot_figure4(data.figure4, args.out_dir)
    plot_figure5(data.figure5, args.out_dir)
    plot_figure6(data.figure6, args.out_dir)
    print(f"Wrote figures and sample tables to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
