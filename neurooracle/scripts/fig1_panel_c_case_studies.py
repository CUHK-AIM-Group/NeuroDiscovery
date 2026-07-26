"""Draw Fig. 1 panel c: two compact NeuroDiscovery case-study rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CASE3_TOPK = ROOT / "neurooracle/data/figures/case3_core_result/source_case3_core_topk_summary.csv"
DEFAULT_CASE3_REQUIRED = (
    ROOT / "neurooracle/data/figures/case3_core_result/source_case3_core_required_hypotheses.csv"
)
DEFAULT_OUT_DIR = ROOT / "neurooracle/data/figures/fig1_panel_c_case_studies"
DEFAULT_ASSET_DIR = DEFAULT_OUT_DIR / "source_assets"
DEFAULT_CASE1_SUMMARY = DEFAULT_ASSET_DIR / "fig_cs1_neurodiscovery_core_surface_summary.csv"


INK = "#111111"
GRID = "#E7E7E7"
BLUE = "#4C78A8"
BLUE_DARK = "#204B6D"
RED = "#DD5148"
GREEN = "#238B45"
GOLD = "#F2C75C"
BOX_EDGE = "#18384A"
GOAL_FILL = "#F4F9FC"
EXEC_FILL = "#FFF4EC"
STEP_FILL = "#F8FBFA"
BOX_ROUNDING = 0.027
BOX_LINEWIDTH = 2.0


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def case1_metrics(summary_path: Path) -> tuple[float, int, int]:
    rows = read_csv_rows(summary_path)
    total_gt = sum(int(float(row["gt_rois"])) for row in rows)
    total_recovered = sum(int(float(row["recovered_rois"])) for row in rows)
    coverage = 100.0 * total_recovered / total_gt
    return coverage, total_recovered, total_gt


def compact_percent(value: float) -> str:
    return f"{value:.0f}"


def case3_topk(topk_path: Path) -> dict[str, list[tuple[int, float]]]:
    rows = read_csv_rows(topk_path)
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        series.setdefault(row["method_label"], []).append((int(row["k"]), float(row["mean"])))
    for values in series.values():
        values.sort()
    return series


def case3_required_hypotheses(required_path: Path, target: int = 50) -> float | None:
    for row in read_csv_rows(required_path):
        if row["method_label"] == "NeuroDiscovery" and int(row["target_future_supported_hits"]) == target:
            raw = row["hypotheses_required"]
            return float(raw) if raw else None
    return None


def axes_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 9,
    weight: str = "normal",
    color: str = INK,
    ha: str = "left",
    va: str = "center",
    **kwargs,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        **kwargs,
    )


def draw_step_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    body: str,
    *,
    fill: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={BOX_ROUNDING}",
        linewidth=BOX_LINEWIDTH,
        edgecolor=BOX_EDGE,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    axes_text(
        ax,
        x + width * 0.5,
        y + height - 0.043,
        label,
        fontsize=10.8,
        weight="bold",
        ha="center",
    )
    axes_text(
        ax,
        x + width * 0.5,
        y + height - 0.086,
        body,
        fontsize=8.55,
        color=INK,
        ha="center",
        va="top",
        linespacing=1.04,
    )


def draw_result_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    metric: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={BOX_ROUNDING}",
        linewidth=BOX_LINEWIDTH,
        edgecolor=BOX_EDGE,
        facecolor="white",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    axes_text(
        ax,
        x + width * 0.5,
        y + height - 0.035,
        "Result",
        fontsize=10.2,
        weight="bold",
        color=BLUE_DARK,
        ha="center",
    )
    axes_text(
        ax,
        x + width * 0.5,
        y + height - 0.064,
        metric,
        fontsize=8.15,
        weight="bold",
        ha="center",
        va="top",
        linespacing=0.98,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=1.65,
        color="#405866",
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arrow)


def draw_runtime_flow(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    step_w: float,
    step_h: float,
    gap: float,
    steps: list[tuple[str, str]],
) -> None:
    for idx, (label, body) in enumerate(steps):
        sx = x + idx * (step_w + gap)
        draw_step_box(
            ax,
            (sx, y),
            step_w,
            step_h,
            label,
            body,
            fill=GOAL_FILL if idx == 0 else STEP_FILL if idx % 2 else EXEC_FILL,
        )
        if idx < len(steps) - 1:
            draw_arrow(
                ax,
                (sx + step_w + 0.004, y + step_h * 0.52),
                (sx + step_w + gap - 0.004, y + step_h * 0.52),
            )


def plot_case1_result(
    fig: plt.Figure,
    ax_bg: plt.Axes,
    rect: tuple[float, float, float, float],
    summary_path: Path,
) -> None:
    x, y, w, h = rect
    coverage, _recovered, _total = case1_metrics(summary_path)
    tests_run = 100.0 - 76.5
    draw_result_card(ax_bg, (x, y), w, h, f"{coverage:.1f}% ROI | 76.5% saved")
    ax_chart = fig.add_axes([x + 0.026, y + 0.030, w - 0.048, h - 0.116])
    xs = [0, 1]
    ax_chart.bar(
        xs,
        [100, 100],
        color="#E7EBEE",
        width=0.62,
        edgecolor="#91A0A7",
        linewidth=0.45,
        zorder=1,
    )
    ax_chart.bar(
        xs,
        [coverage, tests_run],
        color=[GREEN, GOLD],
        width=0.40,
        edgecolor=INK,
        linewidth=0.45,
        zorder=2,
    )
    for xpos, value, color in zip(xs, [coverage, tests_run], [GREEN, GOLD]):
        ax_chart.text(
            xpos,
            min(value + 6, 104),
            compact_percent(value),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=color,
            fontweight="bold",
        )
    ax_chart.set_ylim(0, 112)
    ax_chart.set_xticks([0, 1])
    ax_chart.set_xticklabels(["ROI", "Tests"], fontsize=5.8)
    ax_chart.set_yticks([0, 50, 100])
    ax_chart.set_yticklabels(["0", "50", "100"], fontsize=5.5)
    ax_chart.tick_params(axis="both", length=2, pad=1, width=0.6)
    ax_chart.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax_chart.spines["left"].set_color(INK)
    ax_chart.spines["bottom"].set_color(INK)
    ax_chart.spines["left"].set_linewidth(0.65)
    ax_chart.spines["bottom"].set_linewidth(0.65)


def plot_case3_result(
    fig: plt.Figure,
    ax_bg: plt.Axes,
    rect: tuple[float, float, float, float],
    topk_path: Path,
    required_path: Path,
) -> None:
    x, y, w, h = rect
    series = case3_topk(topk_path)
    nd = series["NeuroDiscovery"]
    baseline_name = next(name for name in series if name != "NeuroDiscovery")
    baseline = series[baseline_name]
    nd_top10 = dict(nd)[10]
    req50 = case3_required_hypotheses(required_path, target=50)

    metric = f"{nd_top10:.0f} top-10 hits"
    if req50 is not None:
        metric = f"{nd_top10:.0f} top-10 hits | {req50:.0f} tests"
    draw_result_card(ax_bg, (x, y), w, h, metric)

    ax = fig.add_axes([x + 0.026, y + 0.030, w - 0.048, h - 0.116])
    for name, values, color, marker, zorder in (
        (baseline_name, baseline, BLUE, "o", 2),
        ("NeuroDiscovery", nd, RED, "o", 3),
    ):
        xs = [k for k, _ in values]
        ys = [mean for _, mean in values]
        ax.fill_between(xs, ys, 0, color=color, alpha=0.10 if name == "NeuroDiscovery" else 0.06)
        ax.plot(xs, ys, color=color, linewidth=2.0, marker=marker, markersize=3.4, zorder=zorder)
        ax.text(
            xs[-1] * 1.05,
            ys[-1],
            "ND" if name == "NeuroDiscovery" else "AI",
            fontsize=5.8,
            fontweight="bold",
            color=color,
            va="center",
            clip_on=False,
        )
    ax.axhline(50, color="#A7A7A7", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    if req50 is not None:
        ax.axvline(req50, color="#A7A7A7", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
        ax.scatter([req50], [50], s=20, marker="D", color=RED, edgecolor=INK, linewidth=0.45, zorder=5)
    ax.scatter([10], [nd_top10], s=22, color=RED, edgecolor=INK, linewidth=0.55, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(8, 1650)
    ax.set_ylim(0, 180)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1k"], fontsize=5.5)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_yticklabels(["0", "50", "100", "150"], fontsize=5.5)
    ax.tick_params(axis="both", length=2, pad=1, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.65)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)


def path_arg(raw: str) -> Path:
    return Path(raw).expanduser()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case1-summary", type=path_arg, default=DEFAULT_CASE1_SUMMARY)
    parser.add_argument("--case3-topk", type=path_arg, default=DEFAULT_CASE3_TOPK)
    parser.add_argument("--case3-required", type=path_arg, default=DEFAULT_CASE3_REQUIRED)
    parser.add_argument("--out-dir", type=path_arg, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="fig1_panel_c_case_study_results")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    missing = [
        path
        for path in (args.case1_summary, args.case3_topk, args.case3_required)
        if not path.exists()
    ]
    if missing:
        missing_lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required panel-C input(s):\n{missing_lines}")

    apply_style()
    fig = plt.figure(figsize=(16.9, 4.85), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    axes_text(ax, 0.012, 0.965, "c. Case-study results", fontsize=21, weight="bold")
    ax.plot([0.012, 0.988], [0.505, 0.505], color="#9C9C9C", lw=0.8, ls=(0, (3, 2)))

    row_specs = [
        {
            "y": 0.555,
            "title": "Case study 1: Cross-diagnostic cortical map",
            "steps": [
                (
                    "Goal",
                    "Map shared cortical ROIs across\npsychiatric disorders without\nexhaustive disease x ROI testing.",
                ),
                (
                    "Plan",
                    "NeuroDiscovery selects high-value\nKG candidates across disease,\nROI, and imaging features.",
                ),
                (
                    "Run",
                    "NeuroRuntime runs selected\nROI-level statistics and stores\nsupported effects with provenance.",
                ),
                (
                    "Refine",
                    "The agent updates evidence,\nremoves weak paths, and focuses\nnew tests on uncovered cortex.",
                ),
            ],
            "result": "case1",
        },
        {
            "y": 0.080,
            "title": "Case study 2: Future-supported hypothesis forecast",
            "steps": [
                (
                    "Goal",
                    "Predict which KG hypotheses\nfuture papers will support\nbefore they appear.",
                ),
                (
                    "Freeze",
                    "Freeze the historical KG and\nhide later claims for a\nprospective replay.",
                ),
                (
                    "Rank",
                    "NeuroDiscovery ranks candidates\nusing KGE paths, evidence strength,\nand critic consensus.",
                ),
                (
                    "Validate",
                    "Replay later claims; count a hit\nonly when new papers support\nthe ranked hypothesis.",
                ),
            ],
            "result": "case3",
        },
    ]

    row_h = 0.365
    flow_x = 0.064
    step_w = 0.140
    step_gap = 0.043
    step_h = 0.205
    result_gap = 0.044
    result_w = step_w * 1.10

    for spec in row_specs:
        y = spec["y"]
        axes_text(ax, 0.024, y + row_h - 0.010, spec["title"], fontsize=14.2, weight="bold")
        box_y = y + 0.050
        axes_text(
            ax,
            flow_x,
            y + row_h - 0.058,
            "NeuroDiscovery / NeuroRuntime workflow",
            fontsize=11.0,
            weight="bold",
            color=BLUE_DARK,
        )
        draw_runtime_flow(
            ax,
            x=flow_x,
            y=box_y,
            step_w=step_w,
            step_h=step_h,
            gap=step_gap,
            steps=spec["steps"],
        )
        flow_end = flow_x + len(spec["steps"]) * step_w + (len(spec["steps"]) - 1) * step_gap
        result_x = flow_end + result_gap
        draw_arrow(ax, (flow_end + 0.006, box_y + step_h * 0.52), (result_x - 0.006, box_y + step_h * 0.52))

        result_rect = (result_x, box_y, result_w, step_h)
        if spec["result"] == "case1":
            plot_case1_result(fig, ax, result_rect, args.case1_summary)
        else:
            plot_case3_result(fig, ax, result_rect, args.case3_topk, args.case3_required)

    save_all(fig, args.out_dir, args.stem)
    plt.close(fig)
    print(args.out_dir / f"{args.stem}.png")


if __name__ == "__main__":
    main()
