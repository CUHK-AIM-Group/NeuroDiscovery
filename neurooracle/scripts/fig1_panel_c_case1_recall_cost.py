"""Draw the Fig. 1 panel-c case-study 1 recall-cost result plot.

This script intentionally does not modify or import the original case-study
comparison plotting script. It rebuilds only the compact main-figure panel from
the saved case-study summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot"
    r"\method_comparison\case1_method_summary_by_trial.csv"
)
DEFAULT_OUT_DIR = ROOT / "neurooracle/data/figures/fig1_panel_c_case_studies/case1_recall_cost"

PALETTE = {
    "neurodiscovery": "#D9544D",
}
MARKERS = {
    "ai_scientist_v2": "o",
    "co_scientist_style": "s",
    "data_to_paper_style": "^",
    "sciagents_style": "D",
    "virtual_lab_style": "v",
    "openscholar_rag": "X",
}
BASELINE_METHODS = (
    "ai_scientist_v2",
    "co_scientist_style",
    "data_to_paper_style",
    "sciagents_style",
    "virtual_lab_style",
    "openscholar_rag",
)
BEST_BASELINE = "sciagents_style"
TARGETS = [1, 5, 10, 20, 30, 50]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
        }
    )


def mean_value(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def recall_cost_values(trial_summary: pd.DataFrame, method: str) -> np.ndarray:
    sub = trial_summary[trial_summary["method"].eq(method)]
    return np.array(
        [mean_value(sub[f"experiments_for_recall_{target}pct"]) for target in TARGETS],
        dtype=float,
    )


def draw_panel(ax: plt.Axes, trial_summary: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = []
    for method in BASELINE_METHODS:
        vals = recall_cost_values(trial_summary, method)
        baseline_rows.append(pd.DataFrame({"method": method, "target": TARGETS, "experiments": vals}))
    baseline_table = pd.concat(baseline_rows, ignore_index=True)

    best_vals = recall_cost_values(trial_summary, BEST_BASELINE)
    nd_vals = recall_cost_values(trial_summary, "neurodiscovery")
    y_max = float(np.nanmax([baseline_table["experiments"].max(), np.nanmax(nd_vals)])) * 1.12

    for method, sub in baseline_table.groupby("method", sort=False):
        is_best = method == BEST_BASELINE
        ax.plot(
            sub["target"],
            sub["experiments"],
            color="#595959" if is_best else "#BDBDBD",
            marker=MARKERS.get(method, "o"),
            markersize=7.2 if is_best else 5.7,
            markerfacecolor="#595959" if is_best else "#D8D8D8",
            markeredgecolor="#595959" if is_best else "#BDBDBD",
            linewidth=3.0 if is_best else 2.1,
            alpha=0.92 if is_best else 0.64,
            zorder=3 if is_best else 1,
        )

    ax.plot(
        TARGETS,
        nd_vals,
        color=PALETTE["neurodiscovery"],
        marker="o",
        markersize=9.0,
        linewidth=3.4,
        zorder=5,
    )

    ax.set_xlim(0.8, 58)
    ax.set_ylim(0, y_max)
    ax.set_xticks(TARGETS)
    ax.set_yticks(np.linspace(0, y_max / 1.12, 6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.yaxis.get_offset_text().set_visible(False)
    ax.tick_params(axis="both", length=5.5, width=1.1)
    ax.set_xlabel("Validated-discovery recall target", fontsize=20, labelpad=12)
    ax.set_ylabel("Experiments required", fontsize=20, labelpad=14)
    ax.grid(axis="y", color="#E2E2E2", linewidth=1.05)
    ax.text(
        51.8,
        nd_vals[-1],
        "Ours",
        color=PALETTE["neurodiscovery"],
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )

    source = pd.concat(
        [
            baseline_table,
            pd.DataFrame({"method": "neurodiscovery", "target": TARGETS, "experiments": nd_vals}),
        ],
        ignore_index=True,
    )
    return source


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)


def path_arg(raw: str) -> Path:
    return Path(raw).expanduser()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=path_arg, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=path_arg, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="fig1_panel_c_case1_recall_cost")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.summary.exists():
        raise FileNotFoundError(f"Missing case-study summary table: {args.summary}")

    apply_style()
    trial_summary = pd.read_csv(args.summary)
    fig, ax = plt.subplots(figsize=(5.2, 3.45), facecolor="white")
    source = draw_panel(ax, trial_summary)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.25, top=0.96)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.out_dir / "source_fig1_panel_c_case1_recall_cost.csv", index=False)
    save_all(fig, args.out_dir, args.stem)
    plt.close(fig)
    print(args.out_dir / f"{args.stem}.png")


if __name__ == "__main__":
    main()
