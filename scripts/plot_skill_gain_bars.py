from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib import colors as mcolors
from matplotlib import font_manager

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DEFAULT_COLORS = {
    "noskill": "#cfcfd4",
    "withskill": "#4c78a8",
}

# New color scheme based on the second reference image
MODEL_COLORS = {
    "claude-opus-4-6": "#5B8A8A",      # Teal/Dark cyan
    "claude-sonnet-4-6": "#7A9B5B",    # Olive green
    "gpt-5.4": "#6B4C3E",              # Dark brown
    "qwen3.6-plus": "#B8A04C",         # Golden yellow
    "gemini-3.1-pro-preview": "#A8C5DD", # Light blue
    "gemini-3-flash-preview": "#6B9B6B", # Green
    "gpt-5.4-mini": "#D97B3E",         # Orange
    "deepseek-v3.2": "#B8C44C",        # Yellow-green
    "minimax-m2.7": "#8BC4B4",         # Mint green
    "grok-4-20-non-reasoning": "#9B9B9B", # Gray
}

SHORT_NAMES = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "qwen3.6-plus": "Qwen3.6+",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "deepseek-v3.2": "DeepSeek V3.2",
    "minimax-m2.7": "MiniMax M2.7",
    "grok-4-20-non-reasoning": "Grok-4.20",
}

# Global font size - increased significantly
GLOBAL_FONT_SIZE = 38


def setup_fonts():
    """Setup Times New Roman font and verify it's available."""
    font_names = [f.name for f in font_manager.fontManager.ttflist]
    
    times_fonts = [name for name in font_names if 'Times' in name or 'Liberation Serif' in name]
    
    if times_fonts:
        print(f"Available Times fonts: {times_fonts[:5]}")
        font_to_use = 'Times New Roman'
    else:
        print("Warning: Times New Roman not found, using serif fallback")
        font_to_use = 'serif'
    
    font_manager._load_fontmanager(try_read_cache=False)
    
    plt.rcParams.update({
        'font.family': font_to_use,
        'font.size': GLOBAL_FONT_SIZE,
        'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    })
    
    return font_to_use


@dataclass
class ModelPair:
    base_model: str
    no_skills: float
    with_skills: float
    absolute_gain: float
    normalized_gain: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paired no-skill/with-skill benchmark bars."
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Path to a benchmark_scores_*.json file. Defaults to the latest one in output/.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_skill_gain_bars.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--logo-dir", type=Path, default=Path("scripts"),
        help="Directory containing logo assets for each model.",
    )
    parser.add_argument(
        "--title", default="NeuroBench Skill Gain by Model",
        help="Chart title.",
    )
    return parser.parse_args()


def find_latest_json() -> Path:
    candidates = sorted(Path("output").glob("benchmark_scores_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "No benchmark_scores_*.json files found under output/."
        )
    return candidates[-1]


def load_pairs(json_path: Path) -> List[ModelPair]:
    data = json.loads(json_path.read_text())
    comparisons = data["skill_gain_analysis"]["comparisons"]
    pairs = [
        ModelPair(
            base_model=item["base_model"],
            with_skills=float(item["with_skills_average"]),
            no_skills=float(item["no_skills_average"]),
            absolute_gain=float(item["absolute_improvement"]),
            normalized_gain=float(item["normalized_gain"]),
        )
        for item in comparisons
    ]
    # Sort descending by with_skills score
    pairs.sort(key=lambda pair: pair.with_skills, reverse=True)
    return pairs


def lighten_color(color: str, amount: float = 0.40) -> Tuple[float, float, float]:
    """Lighten a hex color by blending toward white."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def add_gain_bracket(
    ax: plt.Axes,
    y_bottom: float,
    y_top: float,
    x_bottom: float,
    x_top: float,
    gain: float,
    font_name: str,
    x_offset: float = 9.0,  # Increased from 6.0 to 9.0
    tick_width: float = 1.0,
) -> None:
    """Draw a horizontal bracket with gain text connecting ends of two bars."""
    bracket_x = max(x_bottom, x_top) + x_offset
    # Horizontal ticks and vertical connector
    ax.plot(
        [bracket_x - tick_width, bracket_x, bracket_x, bracket_x - tick_width],
        [y_bottom, y_bottom, y_top, y_top],
        color="#444444",
        linewidth=2.2,
        clip_on=False,
        zorder=5,
    )
    # Gain text - moved further right
    sign = "+" if gain >= 0 else ""
    ax.text(
        bracket_x + 1.8,  # Increased from 1.2 to 1.8
        (y_bottom + y_top) / 2,
        f"{sign}{gain:.2f}",
        ha="left", va="center",
        fontsize=GLOBAL_FONT_SIZE, color="#333333", weight="bold",
        fontfamily=font_name,
        zorder=5,
    )


def plot_pairs(
    pairs: List[ModelPair],
    output_path: Path,
    logo_dir: Path,
    title: str,
    font_name: str,
) -> None:
    n = len(pairs)
    # Square aspect ratio: 1:1
    fig_size = 20  # Use same size for both width and height
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9fb")

    # ---------- layout constants ----------
    bar_height = 0.85
    pair_inner_gap = 0.0   # No gap between bars in the same pair
    group_gap = 0.50

    # ---------- compute y positions (reversed from x positions) ----------
    y_no_list: List[float] = []
    y_with_list: List[float] = []
    y_center_list: List[float] = []

    cursor = 1.0
    for _ in pairs:
        y_no = cursor
        y_with = cursor + bar_height + pair_inner_gap
        center = (y_no + y_with) / 2.0
        y_no_list.append(y_no)
        y_with_list.append(y_with)
        y_center_list.append(center)
        cursor = y_with + bar_height + group_gap

    # ---------- colours ----------
    with_colors = [
        MODEL_COLORS.get(p.base_model, DEFAULT_COLORS["withskill"])
        for p in pairs
    ]
    no_colors = [lighten_color(c, amount=0.40) for c in with_colors]

    # ---------- draw horizontal bars with black borders ----------
    no_widths = [p.no_skills for p in pairs]
    with_widths = [p.with_skills for p in pairs]

    bars_no = ax.barh(
        y_no_list, no_widths,
        height=bar_height, color=no_colors,
        edgecolor="black", linewidth=2.0, zorder=3,
    )
    bars_with = ax.barh(
        y_with_list, with_widths,
        height=bar_height, color=with_colors,
        edgecolor="black", linewidth=2.0, zorder=3,
    )

    # ---------- add score labels on bars ----------
    for bar, val in zip(bars_no, no_widths):
        ax.text(
            val + 0.5,  # Slightly to the right of bar end
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            ha="left", va="center",
            fontsize=GLOBAL_FONT_SIZE - 2, color="#606060", weight="semibold",
            fontfamily=font_name,
            zorder=5,
        )
    
    for bar, val in zip(bars_with, with_widths):
        ax.text(
            val + 0.5,  # Slightly to the right of bar end
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            ha="left", va="center",
            fontsize=GLOBAL_FONT_SIZE - 2, color="#222222", weight="bold",
            fontfamily=font_name,
            zorder=5,
        )

    # ---------- gain brackets ----------
    for pair, y_no, y_with in zip(pairs, y_no_list, y_with_list):
        y_bottom_center = y_no
        y_top_center = y_with
        add_gain_bracket(
            ax,
            y_bottom_center, y_top_center,
            pair.no_skills, pair.with_skills,
            pair.absolute_gain,
            font_name,
            x_offset=9.0,  # Increased offset
            tick_width=1.0,
        )

    # ---------- axes styling with explicit font ----------
    # Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_xlabel(
        "Benchmark Score (0-100)",
        fontsize=GLOBAL_FONT_SIZE + 14,  # Increased from +8 to +14
        weight="bold",
        labelpad=18,  # Increased padding
        fontfamily=font_name,
    )
    ax.set_yticks([])
    ax.set_xlim(20, 90)

    # Explicitly set x-axis tick label fonts
    ax.tick_params(axis="x", colors="#555555", labelsize=GLOBAL_FONT_SIZE, width=1.6, length=8)
    for label in ax.get_xticklabels():
        label.set_fontfamily(font_name)
        label.set_fontsize(GLOBAL_FONT_SIZE)
        label.set_fontweight('normal')

    ax.grid(axis="x", color="#e0e0e0", linewidth=1.1, zorder=0)
    
    # Show all four spines with black color
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    
    ax.spines["top"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["right"].set_linewidth(1.6)

    # ---------- save ----------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    
    # Setup fonts first
    font_name = setup_fonts()
    
    json_path = args.json or find_latest_json()
    pairs = load_pairs(json_path)
    plot_pairs(pairs, args.output, args.logo_dir, args.title, font_name)
    print(f"Saved chart to {args.output}")
    print(f"Using score file: {json_path}")
    print(f"Logo directory: {args.logo_dir}")


if __name__ == "__main__":
    main()