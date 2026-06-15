"""Generate Nature-style NeuroOracle figure mockups.

The script creates synthetic result tables and publication-style mock figures
for four main Nature-style figures: KG, Case Study 1, Case Study 2, and Case
Study 3.

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
        -0.18,
        1.20,
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
    included_groups = {"figure2", "figure4", "figure5", "figure6"}
    for group_name, group in data.__dict__.items():
        if group_name not in included_groups:
            continue
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


def draw_result_brain(ax: plt.Axes, center: tuple[float, float], scale: float, activation: float) -> None:
    x0, y0 = center
    brain = patches.Ellipse(
        (x0, y0),
        1.25 * scale,
        0.78 * scale,
        facecolor="#F1F3F5",
        edgecolor="#333333",
        linewidth=0.8,
        zorder=1,
    )
    ax.add_patch(brain)
    rng = np.random.default_rng(int(activation * 1000))
    for _ in range(9):
        xs = np.linspace(-0.48, 0.48, 30) * scale + x0
        phase = rng.uniform(0, 2 * np.pi)
        ys = (0.11 * np.sin(np.linspace(0, 2.8 * np.pi, 30) + phase) + rng.uniform(-0.18, 0.18)) * scale + y0
        ax.plot(xs, ys, color="#9AA3AA", lw=0.55, alpha=0.75, clip_path=brain, zorder=2)
    for dx, dy, val in [(-0.24, 0.12, activation), (0.10, -0.03, 0.62), (0.25, 0.16, 0.42)]:
        color = DIVERGING_CMAP(0.5 + 0.5 * val)
        ax.scatter(x0 + dx * scale, y0 + dy * scale, s=110 * scale, color=color, edgecolor="white", linewidth=0.4, zorder=3)


def draw_kg_result_module(ax: plt.Axes, domain_conn: pd.DataFrame, evidence_state: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Evidence-map result motif", pad=10)
    heat_ax = ax.inset_axes([0.06, 0.37, 0.62, 0.57], transform=ax.transAxes)
    matrix = domain_conn.drop(columns="domain").to_numpy()
    labels = ["Clin", "Brain", "IM", "Gene", "Out", "Data"]
    heat_ax.imshow(matrix, aspect="auto", cmap=DIVERGING_CMAP, vmin=0, vmax=1)
    heat_ax.set_xticks(np.arange(len(labels)))
    heat_ax.set_yticks(np.arange(len(labels)))
    heat_ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.4)
    heat_ax.set_yticklabels(labels, fontsize=7.4)
    heat_ax.tick_params(length=0)
    heat_ax.set_title("domain connectivity", fontsize=10, pad=3)
    for spine in heat_ax.spines.values():
        spine.set_linewidth(0.8)

    src_ax = ax.inset_axes([0.08, 0.06, 0.36, 0.22], transform=ax.transAxes)
    src_ax.bar([0], [1.45], color=COLORS["blue_mid"], width=0.52)
    src_ax.bar([0], [0.55], bottom=[1.45], color=COLORS["green"], width=0.52)
    state_mean = evidence_state.groupby("state")["fraction"].mean()
    bottom = 0
    for state, color in [
        ("Emerging", COLORS["blue"]),
        ("Sparse", "#BFA98A"),
        ("Contradicted", COLORS["salmon"]),
        ("Supported", COLORS["green"]),
    ]:
        val = float(state_mean.loc[state] * 2.0)
        src_ax.bar([1.0], [val], bottom=[bottom], color=color, width=0.52)
        bottom += val
    src_ax.set_xticks([0, 1.0])
    src_ax.set_xticklabels(["Sources", "State"], fontsize=8.4)
    src_ax.set_ylim(0, 2.1)
    src_ax.set_ylabel("Curated db", fontsize=8.2)
    src_ax.tick_params(axis="y", labelsize=7.4)

    legend_items = [
        ("Supported", COLORS["green"]),
        ("Contradicted", COLORS["salmon"]),
        ("Sparse", "#BFA98A"),
        ("Emerging", COLORS["blue"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        y = 0.26 - 0.055 * i
        ax.scatter(0.58, y, s=32, marker="s", color=color, transform=ax.transAxes, clip_on=False)
        ax.text(0.64, y, label, va="center", fontsize=8.2, transform=ax.transAxes, clip_on=False)


def draw_cross_disorder_result_module(ax: plt.Axes, shared: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-disorder neuroimaging clusters", pad=16)
    labels = [
        ("salience", (0.28, 0.76), 0.95),
        ("hippocampus-\namygdala", (0.72, 0.76), 0.72),
        ("striatal / DRD2", (0.30, 0.36), 0.84),
        ("fronto-striatal", (0.72, 0.36), 0.68),
    ]
    for label, center, act in labels:
        draw_result_brain(ax, center, 0.34, act)
        ax.text(center[0], center[1] - 0.19, label, ha="center", va="top", fontsize=9.6)
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    cb_ax = ax.inset_axes([0.26, 0.03, 0.48, 0.045], transform=ax.transAxes)
    cb_ax.imshow(grad, aspect="auto", cmap=DIVERGING_CMAP)
    cb_ax.set_xticks([0, 64, 127])
    cb_ax.set_xticklabels(["blue", "white", "red"], fontsize=8.4)
    cb_ax.set_yticks([])
    cb_ax.set_title("shared alteration direction", fontsize=8.8, pad=2)


def draw_case2_result_module(ax: plt.Axes, chains: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Gene-IM-outcome mediation motif", pad=10)
    y = 0.76
    nodes = [
        (0.16, "piR\nAPOE e4", COLORS["green_light"], COLORS["green_dark"]),
        (0.50, "Cortical\nthickness", "#F2F2F2", COLORS["dark"]),
        (0.84, "Clin\ncognitive\ndecline", COLORS["purple_light"], COLORS["purple_dark"]),
    ]
    for x, text, face, edge in nodes:
        circ = patches.Circle((x, y), 0.085, facecolor=face, edgecolor=edge, linewidth=1.2)
        ax.add_patch(circ)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.8, fontweight="bold")
    for x0, x1, color, lw in [(0.245, 0.415, COLORS["dark"], 1.9), (0.585, 0.755, COLORS["dark"], 1.9)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=14))
    ax.annotate(
        "",
        xy=(0.75, y + 0.10),
        xytext=(0.25, y + 0.10),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color=COLORS["grey"], linestyle="--", connectionstyle="arc3,rad=0.15"),
    )
    ax.text(0.50, 0.92, "direct genetic effect", ha="center", va="center", fontsize=8.2, color=COLORS["grey"])
    ax.text(0.50, 0.60, "IM-mediated path", ha="center", va="center", fontsize=10.2, fontweight="bold", color=COLORS["green_dark"])

    forest_ax = ax.inset_axes([0.16, 0.07, 0.68, 0.36], transform=ax.transAxes)
    rows = chains.sort_values("indirect_effect", ascending=False).head(4).iloc[::-1]
    yy = np.arange(len(rows))
    forest_ax.errorbar(
        rows["indirect_effect"],
        yy,
        xerr=[rows["indirect_effect"] - rows["ci_low"], rows["ci_high"] - rows["indirect_effect"]],
        fmt="s",
        color=COLORS["green"],
        ecolor=COLORS["dark"],
        elinewidth=1.2,
        capsize=2,
        markersize=5,
    )
    forest_ax.axvline(0, color=COLORS["dark"], lw=0.8)
    forest_ax.set_yticks(yy)
    forest_ax.set_yticklabels(["indirect", "IM", "IM", "95% CI"], fontsize=8.0)
    forest_ax.set_xlabel("normal x-y proportions", fontsize=8.2)
    forest_ax.tick_params(axis="x", labelsize=7.8)
    forest_ax.set_xlim(-0.05, 0.52)
    forest_ax.set_title("mediation forest plot -> outcome", fontsize=9.4, pad=2)


def draw_case3_result_module(ax: plt.Axes, precision: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Future evidence after freeze year", pad=10)
    ax.plot([0.08, 0.92], [0.75, 0.75], color=COLORS["dark"], lw=1.8, transform=ax.transAxes, clip_on=False)
    ax.annotate("", xy=(0.94, 0.75), xytext=(0.88, 0.75), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["dark"], lw=1.8))
    for x, label in [(0.22, "Past\nfrozen KG"), (0.52, "Present\nmodel state"), (0.82, "Future\nevidence")]:
        ax.plot([x, x], [0.71, 0.79], color=COLORS["dark"], lw=1.4, transform=ax.transAxes)
        ax.text(x, 0.64, label, ha="center", va="top", fontsize=9.4, transform=ax.transAxes)
    future_box = patches.Rectangle((0.69, 0.69), 0.22, 0.18, transform=ax.transAxes, facecolor="#BDE3E2", edgecolor="none", alpha=0.9)
    ax.add_patch(future_box)
    ax.scatter([0.52], [0.75], s=72, color="#77B7B2", edgecolor=COLORS["dark"], transform=ax.transAxes, zorder=3)

    curve_ax = ax.inset_axes([0.17, 0.08, 0.70, 0.40], transform=ax.transAxes)
    curve_ax.plot(precision["rank_k"] / 5, precision["NeuroOracle"], color="#0B7A75", lw=2.6, label="NeuroDiscovery")
    curve_ax.plot(precision["rank_k"] / 5, precision["Random path"], color=COLORS["grey"], lw=2.1, label="baseline")
    curve_ax.set_xlim(0, 10)
    curve_ax.set_ylim(0, 1.0)
    curve_ax.set_xlabel("Years after freeze", fontsize=8.8)
    curve_ax.set_ylabel("Future-supported\nhypotheses", fontsize=8.8)
    curve_ax.tick_params(labelsize=8.2)
    curve_ax.text(0.05, 0.86, "NeuroDiscovery", color="#0B7A75", transform=curve_ax.transAxes, fontsize=9.4)
    curve_ax.annotate(
        "future\nliterature\nsupport",
        xy=(5.2, 0.32),
        xytext=(6.1, 0.58),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["dark"], lw=1.0),
        fontsize=8.3,
    )


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

    chain_rows = [
        ("AD", "APOE/lipid transport", "amyloid PET burden", "cognitive decline", 26, 142, 0.34, 0.22, 0.46, 0.62, 5),
        ("AD", "Complement cascade", "microglial PET signal", "memory impairment", 19, 118, 0.27, 0.15, 0.40, 0.74, 4),
        ("SCZ", "Dopamine pathway", "frontostriatal FC", "psychosis severity", 22, 136, 0.31, 0.18, 0.43, 0.81, 5),
        ("SCZ", "Synaptic plasticity", "DMN RSFC", "functional disability", 17, 96, 0.22, 0.10, 0.34, 0.69, 5),
        ("BD", "Calcium signaling", "amygdala reactivity", "mood instability", 14, 82, 0.19, 0.06, 0.32, 0.65, 4),
        ("MDD", "Inflammatory pathway", "insula connectivity", "negative affect", 16, 128, 0.29, 0.17, 0.41, 0.79, 5),
        ("MDD", "Stress response", "frontolimbic FC", "anxiety burden", 12, 92, 0.21, 0.08, 0.33, 0.77, 4),
        ("OCD", "Glutamate signaling", "cortico-striatal FC", "compulsivity", 9, 74, 0.18, 0.05, 0.30, 0.83, 3),
        ("ADHD", "Catecholamine pathway", "frontoparietal FC", "executive dysfunction", 11, 88, 0.20, 0.07, 0.32, 0.80, 4),
        ("BD", "Myelination pathway", "white-matter FA", "executive dysfunction", 10, 76, 0.17, 0.04, 0.29, 0.58, 4),
    ]
    chain_table = pd.DataFrame(
        chain_rows,
        columns=[
            "disease",
            "gene_pathway",
            "imaging_marker",
            "outcome",
            "direct_claims",
            "mediated_claims",
            "indirect_effect",
            "ci_low",
            "ci_high",
            "targetability",
            "disease_coverage",
        ],
    )
    chain_table["total_claims"] = chain_table["direct_claims"] + chain_table["mediated_claims"]
    chain_table["mediated_fraction"] = chain_table["mediated_claims"] / chain_table["total_claims"]
    chain_table["support_score"] = np.clip(
        0.35
        + 0.22 * chain_table["mediated_fraction"]
        + 0.18 * (chain_table["mediated_claims"] / chain_table["mediated_claims"].max())
        + 0.18 * chain_table["targetability"],
        0,
        1,
    )

    pathway_mode = (
        chain_table.groupby("disease")[["direct_claims", "mediated_claims"]]
        .sum()
        .reset_index()
        .sort_values("mediated_claims", ascending=True)
    )
    pathway_mode["mediated_fraction"] = pathway_mode["mediated_claims"] / (
        pathway_mode["direct_claims"] + pathway_mode["mediated_claims"]
    )

    marker_short = {
        "amyloid PET burden": "Amyloid\nPET",
        "microglial PET signal": "Microglial\nPET",
        "frontostriatal FC": "FStr\nFC",
        "DMN RSFC": "DMN\nRSFC",
        "amygdala reactivity": "Amygdala\nreact.",
        "insula connectivity": "Insula\nFC",
        "frontolimbic FC": "FLimbic\nFC",
        "cortico-striatal FC": "CStr\nFC",
        "frontoparietal FC": "FPar\nFC",
        "white-matter FA": "WM\nFA",
    }
    bridge = chain_table[["disease", "imaging_marker", "mediated_claims", "support_score"]].copy()
    bridge["marker_short"] = bridge["imaging_marker"].map(marker_short)
    bridge["bridge_score"] = np.clip(
        0.30 + 0.45 * bridge["support_score"] + 0.25 * (bridge["mediated_claims"] / bridge["mediated_claims"].max()),
        0,
        1,
    )

    target_priority = (
        chain_table.groupby("imaging_marker")
        .agg(
            support_score=("support_score", "mean"),
            mediated_claims=("mediated_claims", "sum"),
            targetability=("targetability", "mean"),
            disease_coverage=("disease_coverage", "max"),
        )
        .reset_index()
    )
    target_priority["priority_score"] = np.clip(
        0.30 * target_priority["support_score"]
        + 0.25 * (target_priority["mediated_claims"] / target_priority["mediated_claims"].max())
        + 0.30 * target_priority["targetability"]
        + 0.15 * (target_priority["disease_coverage"] / target_priority["disease_coverage"].max()),
        0,
        1,
    )
    target_priority["marker_short"] = target_priority["imaging_marker"].map(marker_short)
    target_priority = target_priority.sort_values("priority_score", ascending=True)

    fig5 = {
        "chains": chain_table,
        "pathway_mode": pathway_mode,
        "im_bridge": bridge,
        "target_priority": target_priority,
    }

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

    fig = plt.figure(figsize=FIGURE_SIZE_TALL, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.02, 1.08, 1.02, 1.18], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_f = fig.add_subplot(gs[0, 3])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3])

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
    ax_e.legend(loc="lower right", ncol=1, fontsize=9.2, handlelength=1.0, handletextpad=0.45)
    ax_e.invert_yaxis()
    panel_label(ax_e, "e")

    draw_kg_result_module(ax_f, domain_conn, evidence_state)
    panel_label(ax_f, "f")

    save_fig(fig, out_dir, "fig1_knowledge_graph")


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

    draw_cross_disorder_result_module(ax_e, shared)
    panel_label(ax_e, "e")

    save_fig(fig, out_dir, "fig2_case1_transdiagnostic")


def plot_figure5(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    chains = data["chains"]
    pathway_mode = data["pathway_mode"]
    bridge = data["im_bridge"]
    priority = data["target_priority"]

    fig = plt.figure(figsize=FIGURE_SIZE_TALL, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.05, 1.05], height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    def fig3_label(ax: plt.Axes, label: str) -> None:
        ax.text(
            -0.12,
            1.075,
            label,
            transform=ax.transAxes,
            fontsize=28,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    gene_short = {
        "APOE/lipid transport": "APOE",
        "Complement cascade": "Complement",
        "Dopamine pathway": "Dopamine",
        "Inflammatory pathway": "Inflammation",
        "Synaptic plasticity": "Synaptic",
        "Calcium signaling": "Calcium",
        "Stress response": "Stress",
        "Catecholamine pathway": "Catecholamine",
        "Glutamate signaling": "Glutamate",
        "Myelination pathway": "Myelination",
    }
    marker_short = {
        "amyloid PET burden": "Amyloid PET",
        "microglial PET signal": "Microglial PET",
        "frontostriatal FC": "FStr FC",
        "DMN RSFC": "DMN RSFC",
        "insula connectivity": "Insula FC",
        "amygdala reactivity": "Amygdala",
        "frontolimbic FC": "FLimbic FC",
        "frontoparietal FC": "FPar FC",
        "cortico-striatal FC": "CStr FC",
        "white-matter FA": "WM FA",
    }
    outcome_short = {
        "cognitive decline": "cognition",
        "memory impairment": "memory",
        "psychosis severity": "psychosis",
        "functional disability": "disability",
        "negative affect": "negative affect",
        "mood instability": "mood",
        "anxiety burden": "anxiety",
        "compulsivity": "compulsivity",
        "executive dysfunction": "executive",
    }

    draw_case2_result_module(ax_a, chains)
    fig3_label(ax_a, "a")

    y = np.arange(len(pathway_mode))
    totals = pathway_mode["direct_claims"] + pathway_mode["mediated_claims"]
    direct_frac = pathway_mode["direct_claims"] / totals
    mediated_frac = pathway_mode["mediated_claims"] / totals
    ax_b.barh(y, direct_frac, color=COLORS["grey"], height=0.72, label="Direct")
    ax_b.barh(y, mediated_frac, left=direct_frac, color=COLORS["green"], height=0.72, label="IM-mediated")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(pathway_mode["disease"])
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("Fraction of claim evidence")
    ax_b.set_title("Path composition by disease", pad=12)
    ax_b.legend(frameon=False, loc="lower right", fontsize=10.5)
    ax_b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    fig3_label(ax_b, "b")

    disease_order = ["AD", "SCZ", "MDD", "BD", "ADHD", "OCD"]
    marker_order = [x.replace("\n", "") for x in priority.sort_values("priority_score", ascending=False)["marker_short"].tolist()]
    for _, row in bridge.iterrows():
        marker_plot = row["marker_short"].replace("\n", "")
        ax_c.scatter(
            marker_order.index(marker_plot),
            disease_order.index(row["disease"]),
            s=34 + row["mediated_claims"] * 2.1,
            c=row["bridge_score"],
            cmap=GREEN_CMAP,
            vmin=0.45,
            vmax=0.92,
            edgecolor="white",
            linewidth=0.75,
        )
    ax_c.set_xticks(np.arange(len(marker_order)))
    ax_c.set_xticklabels(marker_order, rotation=55, ha="right", fontsize=9.4)
    ax_c.set_yticks(np.arange(len(disease_order)))
    ax_c.set_yticklabels(disease_order)
    ax_c.set_title("Disease-IM bridge strength", pad=12)
    ax_c.grid(color="#EAEAEA", linewidth=0.8)
    fig3_label(ax_c, "c")

    forest = chains.sort_values("indirect_effect", ascending=True).tail(8)
    y = np.arange(len(forest))
    labels = [
        f"{gene_short[row.gene_pathway]} -> {marker_short[row.imaging_marker]} -> {outcome_short[row.outcome]}"
        for row in forest.itertuples()
    ]
    ax_d.errorbar(
        forest["indirect_effect"],
        y,
        xerr=[forest["indirect_effect"] - forest["ci_low"], forest["ci_high"] - forest["indirect_effect"]],
        fmt="o",
        color=COLORS["green_dark"],
        ecolor=COLORS["grey"],
        capsize=3.5,
        elinewidth=2.0,
        markersize=7,
        zorder=3,
    )
    ax_d.axvline(0, color=COLORS["dark"], lw=1.0)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(labels, fontsize=10.2)
    ax_d.set_xlabel("Standardized indirect effect via IM")
    ax_d.set_title("Claim-supported mediation effects", pad=12)
    ax_d.set_xlim(-0.05, 0.52)
    ax_d.grid(axis="x", color="#EAEAEA", linewidth=0.8)
    fig3_label(ax_d, "d")

    p = priority.tail(8)
    y = np.arange(len(p))
    colors = [COLORS["green_dark"] if v > 0.72 else COLORS["green"] for v in p["priority_score"]]
    ax_e.barh(y, p["priority_score"], color=colors, height=0.62)
    ax_e.scatter(p["targetability"], y, color=COLORS["orange_dark"], s=64, zorder=3, edgecolor="white", linewidth=0.6, label="Targetability")
    for i, row in enumerate(p.itertuples()):
        ax_e.text(1.015, i, f"{int(row.disease_coverage)} diseases", va="center", ha="left", fontsize=10.5)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(p["marker_short"])
    ax_e.set_xlim(0, 1.18)
    ax_e.set_xlabel("IM target priority score")
    ax_e.set_title("Treatment-facing IM priorities", pad=12)
    ax_e.legend(frameon=False, loc="lower right", fontsize=10.5)
    fig3_label(ax_e, "e")

    save_fig(fig, out_dir, "fig3_case2_pathway_mediation")


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

    draw_case3_result_module(ax_a, precision)
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

    save_fig(fig, out_dir, "fig4_case3_hindcasting")


def write_plan(out_dir: Path) -> None:
    plan = """# NeuroOracle Nature Figure Plan

Fig.1 Knowledge graph: evidence-map scale, domain connectivity, disease-marker evidence density, temporal claim growth, evidence state by domain, and compact evidence-map result motif.

Fig.2 Case Study 1: disease-by-region effect-size heatmap, forest plots, shared alteration ranking, and cross-disorder neuroimaging cluster motif.

Fig.3 Case Study 2: gene-IM-outcome mediation motif, direct genetic effects versus IM-mediated paths, disease-level path composition, disease-IM bridge strength, mediated effect estimates, and treatment-facing IM prioritization.

Fig.4 Case Study 3: freeze-year-to-future-evidence motif, hindcasting precision curves, freeze-year robustness, baseline comparison, lead-time distribution, recovered examples, and failure modes.
"""
    (out_dir / "FIGURE_PLAN.md").write_text(plan, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("neurooracle/data/figures/nature_four"),
        help="Directory for mock figures and sample tables.",
    )
    args = parser.parse_args()

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = make_mock_data()
    write_tables(data, args.out_dir)
    write_plan(args.out_dir)
    plot_figure2(data.figure2, args.out_dir)
    plot_figure4(data.figure4, args.out_dir)
    plot_figure5(data.figure5, args.out_dir)
    plot_figure6(data.figure6, args.out_dir)
    print(f"Wrote figures and sample tables to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
