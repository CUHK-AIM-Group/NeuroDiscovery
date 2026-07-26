"""Draw the Fig. 1 panel-c case-study 2 future-supported-hit result plot.

This script is a compact main-figure version of the replicated-methods
hindcasting panel. It does not modify or import the original plotting script.
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
DEFAULT_SOURCE = ROOT / "neurooracle/data/figures/case3_replicated_methods/source_replicated_methods_topk_mean.csv"
DEFAULT_OUT_DIR = ROOT / "neurooracle/data/figures/fig1_panel_c_case_studies/case2_future_hits"

TOP_KS = [10, 20, 50, 100, 200, 300, 500, 750, 1000]
METHOD_ORDER = ["SciAgents", "OpenScholar-RAG", "NeuroDiscovery"]
METHOD_COLORS = {
    "SciAgents": "#F28E2B",
    "OpenScholar-RAG": "#76B7B2",
    "NeuroDiscovery": "#D9544D",
}


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


def draw_panel(ax: plt.Axes, topk: pd.DataFrame) -> pd.DataFrame:
    x_pos = np.arange(len(TOP_KS))
    source_rows = []
    for method in METHOD_ORDER:
        df = topk[topk["method_label"].eq(method)].set_index("k").reindex(TOP_KS).reset_index()
        y = pd.to_numeric(df["hits_mean"], errors="coerce").to_numpy(float)
        err = pd.to_numeric(df["hits_sem"], errors="coerce").fillna(0).to_numpy(float)
        lw = 3.8 if method == "NeuroDiscovery" else 2.8
        ms = 7.6 if method == "NeuroDiscovery" else 6.8
        z = 5 if method == "NeuroDiscovery" else 3
        ax.plot(x_pos, y, marker="o", lw=lw, ms=ms, color=METHOD_COLORS[method], zorder=z)
        ax.fill_between(x_pos, y - err, y + err, color=METHOD_COLORS[method], alpha=0.13, lw=0, zorder=z - 1)
        source_rows.append(pd.DataFrame({"method_label": method, "k": TOP_KS, "future_supported_hits": y, "sem": err}))

    nd = topk[topk["method_label"].eq("NeuroDiscovery")].set_index("k").reindex(TOP_KS).reset_index()
    nd_y = pd.to_numeric(nd["hits_mean"], errors="coerce").to_numpy(float)
    ax.text(
        x_pos[-1] + 0.25,
        nd_y[-1],
        "Ours",
        color=METHOD_COLORS["NeuroDiscovery"],
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )

    ax.set_xlim(-0.45, len(TOP_KS) - 0.25)
    ax.set_ylim(0, max(170, float(np.nanmax(topk["hits_mean"])) * 1.08))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_yticks([0, 50, 100, 150])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=5.5, width=1.1)
    ax.set_xlabel("Number of hypotheses", fontsize=20, labelpad=12)
    ax.set_ylabel("Future-supported discoveries", fontsize=20, labelpad=14)
    ax.yaxis.set_label_coords(-0.24, 0.43)
    ax.grid(axis="both", color="#E4E4E4", linewidth=1.0)
    return pd.concat(source_rows, ignore_index=True)


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.14}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)


def path_arg(raw: str) -> Path:
    return Path(raw).expanduser()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk", type=path_arg, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=path_arg, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="fig1_panel_c_case2_future_hits")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.topk.exists():
        raise FileNotFoundError(f"Missing case-study top-k table: {args.topk}")

    apply_style()
    topk = pd.read_csv(args.topk)
    fig, ax = plt.subplots(figsize=(5.55, 4.15), facecolor="white")
    source = draw_panel(ax, topk)
    fig.subplots_adjust(left=0.30, right=0.88, bottom=0.22, top=0.93)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.out_dir / "source_fig1_panel_c_case2_future_hits.csv", index=False)
    save_all(fig, args.out_dir, args.stem)
    plt.close(fig)
    print(args.out_dir / f"{args.stem}.svg")


if __name__ == "__main__":
    main()
